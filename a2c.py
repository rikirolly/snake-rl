import numpy as np
import tensorflow as tf
import variables


class A2CModel(tf.keras.Model):
    def __init__(self, env_width, env_height):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv2D(100, (4, 4), activation='relu')
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv2D(200, (3, 3), activation='relu')
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.flatten = tf.keras.layers.Flatten()
        self.policy_fc = tf.keras.layers.Dense(200, activation=tf.nn.leaky_relu)
        self.policy_logits = tf.keras.layers.Dense(4)
        self.value_fc = tf.keras.layers.Dense(200, activation=tf.nn.leaky_relu)
        self.value = tf.keras.layers.Dense(1)

    def call(self, inputs, training=False):
        # inputs: [batch, 1, H, W] (NCHW) -> NHWC
        x = tf.transpose(inputs, [0, 2, 3, 1])
        x = self.conv1(x)
        x = self.bn1(x, training=training)
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = self.flatten(x)

        p = self.policy_fc(x)
        logits = self.policy_logits(p)
        probs = tf.nn.softmax(logits)

        v = self.value_fc(x)
        value = self.value(v)

        return probs, logits, value


class A2C:
    def __init__(self, id, n_gpu=4):
        self.id = id

        # GPU assignment: 1 GPU for training (id 0-5), n-1 for playing
        if n_gpu > 1:
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                gpu_id = 0 if self.id <= 5 else 1 + (self.id % (n_gpu - 1))
                if gpu_id < len(gpus):
                    tf.config.set_visible_devices(gpus[gpu_id], 'GPU')
                    tf.config.experimental.set_memory_growth(gpus[gpu_id], True)

        self.model = A2CModel(variables.env_width, variables.env_height)
        self.global_step = tf.Variable(0, name="global_step", trainable=False, dtype=tf.int64)

        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            0.0003, decay_steps=1000, decay_rate=0.92, staircase=True)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

        self.save_dir = "./trained_agents/a2c/"
        self.checkpoint = tf.train.Checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            global_step=self.global_step,
        )
        self.checkpoint_manager = tf.train.CheckpointManager(
            self.checkpoint, self.save_dir, max_to_keep=5)

        self._load_model()

    def _load_model(self):
        try:
            latest = self.checkpoint_manager.latest_checkpoint
            if latest:
                self.checkpoint.restore(latest).expect_partial()
                print("Loaded model: {}".format(latest), flush=True)
            else:
                print("No saved model to load, starting a new model from scratch.", flush=True)
        except Exception as e:
            print(e, flush=True)
            print("No saved model to load, starting a new model from scratch.", flush=True)

    def __call__(self, state):
        p = self.get_probs(state)
        action = np.random.choice(4, 1, p=p)[0]
        return action

    def get_probs(self, state):
        probs, _, _ = self.model(np.array([state], dtype=np.float32), training=False)
        p = probs.numpy()[0]
        if np.isnan(p[0]):
            p = np.ones(len(p)) / len(p)
        return p

    @tf.function
    def _train_step(self, states, actions, advantages, values):
        with tf.GradientTape() as tape:
            _, logits, predicted_values = self.model(states, training=True)

            ce = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=logits, labels=actions)
            action_loss = tf.reduce_mean(ce * advantages)

            value_loss = tf.reduce_mean(
                (values - tf.squeeze(predicted_values, axis=-1)) ** 2)

            total_loss = action_loss + 0.5 * value_loss

        grads = tape.gradient(total_loss, self.model.trainable_variables)
        clipped_grads = [
            tf.clip_by_norm(g, 20.0) if g is not None else tf.zeros_like(v)
            for g, v in zip(grads, self.model.trainable_variables)
        ]
        self.optimizer.apply_gradients(
            zip(clipped_grads, self.model.trainable_variables))
        self.global_step.assign_add(1)
        return total_loss

    def train_with_batchs(self, batch):
        batch_states = []
        batch_actions = []
        batch_values = []
        for x in batch:
            batch_states += x[0]
            batch_actions += x[1]
            batch_values += x[2]

        batch_states = np.array(batch_states, dtype=np.float32)
        batch_actions = np.array(batch_actions, dtype=np.int32)
        batch_values = np.array(batch_values, dtype=np.float32)

        batch_size = 7000
        for i in range(0, len(batch_states), batch_size):
            end = min(i + batch_size, len(batch_states))
            s = batch_states[i:end]
            a = batch_actions[i:end]
            v = batch_values[i:end]

            _, _, predicted_v = self.model(s, training=False)
            advantages = v - tf.squeeze(predicted_v, axis=-1).numpy()

            self._train_step(s, a, advantages.astype(np.float32), v)

    def load_model(self):
        latest = self.checkpoint_manager.latest_checkpoint
        if latest:
            self.checkpoint.restore(latest).expect_partial()

    def save_model(self):
        self.checkpoint_manager.save()

    @property
    def train_itr(self):
        return self.global_step.numpy()
