ENV_NAME = 'BreakoutDeterministic-v4'
LOAD_FROM = './breakout-saves/save-01329871'
SAVE_PATH = 'breakout-saves'
LOAD_REPLAY_BUFFER = True
WRITE_TENSORBOARD = False
USE_PER = False
PRIORITY_SCALE = 0.7
CLIP_REWARD = True
TOTAL_FRAMES = 30000000
MAX_EPISODE_LENGTH = 18000
FRAME_BETWEEN_EVAL = 100000
EVAL_LENGTH = 10000
UPDATE_FREQ = 10000

DISCOUNT_FACTOR = 0.99
MIN_REPLAY_BUFFER_SIZE = 50000
MEM_SIZE = 1000000

MAX_NOOP_STEPS = 20
UPDATE_FREQ = 4

INPUT_SHAPE = (84, 84)
BATCH_SIZE = 32
LEARNING_RATE = 0.000001

#let us create an actor critic version of this, should not be THAT hard
def build_v_network(learning_rate=0.00001, input_shape=(84, 84), history_length=4):
    import tensorflow as tf
    from tensorflow.keras.initializers import VarianceScaling
    from tensorflow.keras.layers import (Add, Conv2D, Dense, Flatten, Input, Lambda, Subtract)
    from tensorflow.keras.models import Model
    from tensorflow.keras.optimizers import Adam, RMSprop


    model_input = Input(shape=(input_shape[0], input_shape[1], history_length)) 
    x = Lambda(lambda  layer : layer / 255)(model_input)
    x = Conv2D(32, (8, 8), strides=4, kernel_initializer=VarianceScaling(scale=.2), activation='relu', use_bias=False)(x)
    x = Conv2D(64, (4, 4), strides=2, kernel_initializer=VarianceScaling(scale=.2), activation='relu', use_bias=False)(x)
    x = Conv2D(64, (3, 3), strides=1, kernel_initializer=VarianceScaling(scale=.2), activation='relu', use_bias=False)(x)
    x = Conv2D(1024, (7, 7), strides=1, kernel_initializer=VarianceScaling(scale=.2), activation='relu', use_bias=False)(x)
    x = Flatten()(x)
    x = Dense(1, kernel_initializer=VarianceScaling(scale=2.0))(x)
    model = Model(model_input, x)
    model.compile(Adam(learning_rate), loss=tf.keras.losses.Huber())
    return model
    
def build_q_network(n_actions, learning_rate=0.00001, input_shape=(84, 84), history_length=4):
    import tensorflow as tf
    from tensorflow.keras.initializers import VarianceScaling
    from tensorflow.keras.layers import (Add, Conv2D, Dense, Flatten, Input, Lambda, Subtract)
    from tensorflow.keras.models import Model
    from tensorflow.keras.optimizers import Adam, RMSprop


    model_input = Input(shape=(input_shape[0], input_shape[1], history_length))
    x = Lambda(lambda  layer : layer / 255)(model_input)
    x = Conv2D(32, (8, 8), strides=4, kernel_initializer=VarianceScaling(scale=.2), activation='relu', use_bias=False)(x)
    x = Conv2D(64, (4, 4), strides=2, kernel_initializer=VarianceScaling(scale=.2), activation='relu', use_bias=False)(x)
    x = Conv2D(64, (3, 3), strides=1, kernel_initializer=VarianceScaling(scale=.2), activation='relu', use_bias=False)(x)
    x = Conv2D(1024, (7, 7), strides=1, kernel_initializer=VarianceScaling(scale=.2), activation='relu', use_bias=False)(x)
    val_stream, adv_stream = Lambda(lambda w: tf.split(w, 2, 3))(x)
    
    val_stream = Flatten()(val_stream)
    val = Dense(1, kernel_initializer=VarianceScaling(scale=2.0))(val_stream)
    
    adv_stream = Flatten()(adv_stream)
    adv = Dense(n_actions, kernel_initializer=VarianceScaling(scale=2.0))(adv_stream)

    reduce_mean = Lambda(lambda  w: tf.reduce_mean(w, axis=1, keepdims=True))
    q_vals = Add()([val, Subtract()([adv, reduce_mean(adv)])])
    model = Model(model_input, q_vals)
    model.compile(Adam(learning_rate), loss=tf.keras.losses.Huber())
    return model
NUMBER_OF_RUNNERS = 1
networks = []
target_networks = []
v_network = []
num_workers = 0

#class RunnerConfig:
#    def __init__():

def controller():
    import cv2
    import time
    import tensorflow as tf
    import numpy as numpy

    from multiprocessing import Pool, Queue, Process, cpu_count, Value
    counter = Value("i", 0)
    gpus= tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)

    from game_wrapper import GameWrapper
    from replay_buffer import ReplayBuffer, Memory
    from learner_agent import LearnerAgent

    INPUT_SHAPE = (84, 84)
    BATCH_SIZE = 32
    LEARNING_RATE = 0.000001

    runs = 100
    number_of_workers = 4#cpu_count() - 2# - 2 is 1 for learning, 1 is for memory
    game_wrapper = GameWrapper(ENV_NAME, MAX_NOOP_STEPS)
    target_network = build_q_network(game_wrapper.env.action_space.n)
    network = build_q_network(game_wrapper.env.action_space.n)
    v_network = build_v_network()
    per_workers = runs // number_of_workers
    memory_buffer = ReplayBuffer(size=100000)
    agent = LearnerAgent(network, target_network, v_network, game_wrapper.env.action_space.n)

    collected_memories = 0
    while True:
        workers = []
        memory_line = []
        for i in range(0, number_of_workers):
            q = Queue()
            counter.value += 1
            p = Process(target=runner, \
                args=(agent.target_dqn.get_weights(), agent.DQN.get_weights(), agent.v_network.get_weights(),per_workers,q, counter) \
                )
            p.start()
            workers.append(p)
            memory_line.append(q)
        total_joined = 0
        workers_done = False
        while not workers_done:
            for i in range(0, number_of_workers):
                if counter.value == 0:
                        workers_done = True
                        break
                try:
                    memory = memory_line[i].get(block=False)
                    collected_memories += 1
                    memory_buffer.add_experience(memory)
                except Exception as e:
                    #print(repr(e))
                    continue
        for i in range(0, number_of_workers):
            workers[i].join()
        if memory_buffer.faux_len() > MIN_REPLAY_BUFFER_SIZE:
            agent.learn(BATCH_SIZE, gamma, collected_memories)
            agent.update_target_network()

    print(memory_buffer.faux_len())

def memory_thread(from_runner_thread, from_learner_thread, to_learner_thread):
    from replay_buffer import ReplayBuffer
    replay_buffer = ReplayBuffer(size=MEM_SIZE)
    while True:
        try:
            new_memory = from_learner_thread.get(block=False)
            replay_buffer.add_experience(new_memory)
        except Exception as e:
            noop()
        try:
            request = from_learner_thread.get(block=False)
            if request:
                mini_batch = replay_buffer.get_minibatch()
                to_learner_thread.put(mini_batch)
        except Exception as e:
            noop()

def learner_thread(agent, to_memory_thread, from_memory_thread):
    to_memory_thread.put(True)
    (next_batch, frame_number) = from_memory_thread.get()
    while True:
        to_memory_thread.put(True)
        agent.learn(next_batch)
        (next_batch, frame_number) = from_memory_thread.get()
    
def runner(network_weights, target_network_weights, v_network_weights, runs, q, counter):
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

    gpus= tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)

    INPUT_SHAPE = (84, 84)
    BATCH_SIZE = 32
    LEARNING_RATE = 0.000001
    game_wrapper = GameWrapper(ENV_NAME, MAX_NOOP_STEPS)
    target_network = build_q_network(game_wrapper.env.action_space.n)
    network = build_q_network(game_wrapper.env.action_space.n)
    v_network = build_v_network()
    network.set_weights(network_weights)
    target_network.set_weights(target_network_weights)
    v_network.set_weights(v_network_weights)
    agent = RunnerAgent(network, target_network, v_network, 4, input_shape=INPUT_SHAPE)
    for i in range(0, runs):
        print(i)
        frame_number = 0
        terminal = False
        game_wrapper.reset()
        while not terminal:
            action = agent.get_action(frame_number, game_wrapper.state)
            processed_frame, reward, terminal, life_lost = game_wrapper.step(action)
            a_memory = Memory(action, processed_frame[:,:, 0], reward, terminal, True)
            q.put(a_memory)
            frame_number+=1
    counter.value -= 1

if __name__ == '__main__':
    from multiprocessing import Pool, Queue, Process
    p = Process(target=controller, args=())
    d = p.start()
    p.join()