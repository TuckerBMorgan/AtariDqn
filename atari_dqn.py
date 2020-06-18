from multiprocessing import Lock
class WeightTrainStop:
    def __init__(self):
        self.read_counter = 0
        self.read_lock = Lock()
        self.write_lock = Lock()
    def set_weights(self, package):
        self.write_lock.acquire()
        (dqn_weights, target_dqn_weights, v_network_weights) = package
        self.dqn_weights = dqn_weights
        self.target_dqn_weights = target_dqn_weights
        self.v_network_weights = v_network_weights
        self.write_lock.release()
    def get_weights(self):
        self.read_lock.acquire()
        self.read_counter += 1
        if self.read_counter == 1:
            self.write_lock.acquire()
        self.read_lock.release()
        package = (self.dqn_weights, self.target_dqn_weights, self.v_network_weights)
        self.read_lock.acquire()
        self.read_counter -= 1
        if self.read_counter == 0:
            self.write_lock.release()
        self.read_lock.release()
        return (package)

def controller():
    import cv2
    import time
    import tensorflow as tf
    import numpy as numpy

    import CONST
    from multiprocessing import Pool, Queue, Process, cpu_count, Value
    from utils import build_v_network, build_q_network
    counter = Value("i", 0)
    gpus= tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)

    from game_wrapper import GameWrapper
    from replay_buffer import ReplayBuffer, Memory
    from learner_agent import LearnerAgent

    runs = 100
    number_of_workers = 4#cpu_count() - 2# - 2 is 1 for learning, 1 is for memory
    game_wrapper = GameWrapper(CONST.ENV_NAME, CONST.MAX_NOOP_STEPS)
    target_network = build_q_network(game_wrapper.env.action_space.n)
    network = build_q_network(game_wrapper.env.action_space.n)
    v_network = build_v_network()
    per_workers = runs // number_of_workers
    memory_buffer = ReplayBuffer(size=100000, use_per=True)
    agent = LearnerAgent(network, target_network, v_network, game_wrapper.env.action_space.n)
    weight_train_station = WeightTrainStop()
    package = (network.get_weights(), target_network.get_weights(), v_network.get_weights())
    weight_train_station.set_weights(package)

    collected_memories = 0
    workers = []
    memory_line = []
    for i in range(0, number_of_workers):
        q = Queue()
        counter.value += 1
        p = Process(target=runner, args=(per_workers, memory_buffer, weight_train_station))
        p.start()
        workers.append(p)
        memory_line.append(q)
    lt = Process(target=learner_thread, args=(agent, memory_buffer,weight_train_station))
    lt.start()
    for w in workers:
        w.join()
    lt.join()

def learner_thread(agent, membuffer, weight_train_station):
    while True:
        (next_batch, frame_number) = membuffer.get_minbatch()
        agent.learn(next_batch)
        package = (agent.DQN.get_weights(), agent.target_dqn.get_weights(), agent.v_network.get_weights())
        weight_train_station.add_new_weights(package)
    
def runner(runs, memory_buffer, weight_train_stop):
    import cv2
    import time
    import tensorflow as tf
    import numpy as np

    from multiprocessing import Pool, Queue, Process

    from game_wrapper import GameWrapper
    from replay_buffer import ReplayBuffer, Memory
    from runner_agent import RunnerAgent

    from tensorflow.keras.initializers import VarianceScaling
    from tensorflow.keras.layers import (Add, Conv2D, Dense, Flatten, Input, Lambda, Subtract)
    from tensorflow.keras.models import Model
    from tensorflow.keras.optimizers import Adam, RMSprop
    from utils import build_q_network, build_v_network
    import CONST

    gpus= tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)

    game_wrapper = GameWrapper(CONST.ENV_NAME, CONST.MAX_NOOP_STEPS)

    target_network = build_q_network(game_wrapper.env.action_space.n)
    network = build_q_network(game_wrapper.env.action_space.n)
    v_network = build_v_network()

    agent = RunnerAgent(network, target_network, v_network, 4, input_shape=CONST.INPUT_SHAPE)
    generation = 0
    while True:
        print("Starting genreation: ", generation)
        (dqn_weight, target_dqn_weights, v_network) = weight_train_stop.get_weights()
        agent.DQN.set_weights(dqn_weight)
        agent.target_dqn.set_weights(target_dqn_weights)
        agent.v_network.set_weights(v_network)
        frame_number = 0
        for i in range(0, runs):
            terminal = False
            game_wrapper.reset()
            while not terminal:
                action = agent.get_action(frame_number, game_wrapper.state)
                processed_frame, reward, terminal, life_lost = game_wrapper.step(action)
                a_memory = Memory(action, processed_frame[:,:, 0], reward, terminal, True)
                memory_buffer.add_experience(a_memory)
                frame_number+=1
        generation += 1
    

if __name__ == '__main__':
    from multiprocessing import Pool, Queue, Process
    p = Process(target=controller, args=())
    d = p.start()
    p.join()