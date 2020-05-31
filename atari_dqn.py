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

def controller():
    import cv2
    import time
    import tensorflow as tf
    import numpy as np

    from multiprocessing import Pool, Queue, Process

    gpus= tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)

    from game_wrapper import GameWrapper
    from replay_buffer import ReplayBuffer, Memory
    from learner_agent import LearnerAgent

    INPUT_SHAPE = (84, 84)
    BATCH_SIZE = 32
    LEARNING_RATE = 0.000001

    runs = 10
    number_of_workers = 1
    game_wrapper = GameWrapper(ENV_NAME, MAX_NOOP_STEPS)
    target_network = build_q_network(game_wrapper.env.action_space.n)
    network = build_q_network(game_wrapper.env.action_space.n)
    v_network = build_v_network()
    per_workers = runs // number_of_workers
    workers = []
    memory_line = []
    memory_buffer = ReplayBuffer(size=10000)

    for i in range(0, number_of_workers):
        q = Queue()
        p = Process(target=runner, \
            args=(target_network.get_weights(), network.get_weights(), v_network.get_weights(),per_workers,q) \
            )
        p.start()
        workers.append(p)
        memory_line.append(q)
    total_joined = 0
    done = False
    while not done:
        for i in range(0, number_of_workers):
            if total_joined == len(workers):
                done = True
                break
            try:
                memory = memory_line[i].get(block=False)
                memory_buffer.add_experience(memory)
                if not workers[i].is_alive:
                    workers[i].join()
                    total_joined += 1
            except Exception as e:
                continue
    print(memory_buffer.faux_len())

def runner(network_weights, target_network_weights, v_network_weights, runs, q):
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
            a_memory = Memory(action, processed_frame, reward, terminal, True)
            q.put(a_memory)
            frame_number+=1

if __name__ == '__main__':
    from multiprocessing import Pool, Queue, Process
    p = Process(target=controller, args=())
    d = p.start()
    p.join()