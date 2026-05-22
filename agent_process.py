from multiprocessing import Process
from a2c import A2C
import numpy as np
import variables
import time

from env import SnakeEnv

GAMMA = 0.99


class AgentProcess(Process):
    def __init__(self, conn, id, n_games):
        super(AgentProcess,self).__init__()
        self.conn = conn
        self.n_games = n_games
        self.id = id
        self.msg_queue = []
        np.random.seed(self.id*100)

    def run(self):
        self.agent = A2C(self.id)

        def treatQueue():
            msg = self.conn.recv()
            if msg == "load":
                self.agent.load_model()
                print("Process "+str(self.id)+" loaded the master (0) model.")

            if msg[0] == "train_with_batchs":
                print("Master process is training ...")
                t0 = time.time()
                self.agent.train_with_batchs(msg[1])
                self.agent.save_model()
                print("Master process finished training. Time : "+str(time.time()-t0)+" \n")
                self.conn.send("saved")

        while True:
            if(self.id != 0):
                batch_values = []
                batch_states = []
                batch_actions = []
                print("Process "+str(self.id)+" starts playing "+str(self.n_games)+" games.")
                scores = []
                env = SnakeEnv()
                overall_data = 0
                for i in range(self.n_games):
                    state = env.init()
                    episode_states = []
                    episode_actions = []
                    episode_rewards = []
                    t = 0
                    lastScoring = -1
                    while True:
                        action = self.agent([state])
                        newState, reward, done = env.step(action)

                        episode_states.append([state])
                        episode_actions.append(action)
                        episode_rewards.append(reward)

                        t += 1
                        if reward == 1:
                            lastScoring = t

                        if done or (t - lastScoring >= 100):
                            break
                        state = newState

                    # Compute discounted returns
                    T = len(episode_rewards)
                    discounted_returns = np.zeros(T)
                    running_return = 0.0
                    for k in range(T - 1, -1, -1):
                        running_return = episode_rewards[k] + GAMMA * running_return
                        discounted_returns[k] = running_return

                    batch_states += episode_states
                    batch_actions += episode_actions
                    batch_values += discounted_returns.tolist()

                    scores.append(env.score)
                    overall_data += t

                    if overall_data >= 10000:
                        break
                print("Process "+str(self.id)+" finished playing.")
                batch = (batch_states, batch_actions, batch_values)
                self.conn.send((np.mean(scores), batch))
            treatQueue()
