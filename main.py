import multiprocessing

from master_process import MasterProcess

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    master = MasterProcess()
    master.train_agents()
