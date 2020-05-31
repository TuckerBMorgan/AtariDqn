import cv2
import time
import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt

from game_wrapper import GameWrapper
from replay_buffer import ReplayBuffer
from agent import Agent

from tensorflow.keras.initializers import VarianceScaling
from tensorflow.keras.layers import (Add, Conv2D, Dense, Flatten, Input, Lambda, Subtract)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, RMSprop

ENV_NAME = 'BreakoutDeterministic-v4'
LOAD_FROM = './breakout-saves/save-01129428'
SAVE_PATH = 'breakout-saves'
LOAD_REPLAY_BUFFER = True
WRITE_TENSORBOARD = False
TENSORBOARD_DIR = 'tensorboard/'
writer = tf.summary.create_file_writer(TENSORBOARD_DIR)
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
game_wrapper = GameWrapper(ENV_NAME, MAX_NOOP_STEPS)
network = build_q_network(game_wrapper.env.action_space.n)
target_network = build_q_network(game_wrapper.env.action_space.n)
v_network = build_v_network()
memory = ReplayBuffer(size=MEM_SIZE, input_shape=INPUT_SHAPE, use_per=USE_PER)
agent = Agent(network, target_network, v_network, memory, 4, input_shape=INPUT_SHAPE, batch_size=BATCH_SIZE, use_per=USE_PER)

meta = agent.load(LOAD_FROM, LOAD_REPLAY_BUFFER)
frame_number = meta['frame_number']
rewards = meta['rewards']
loss_list = meta['loss_list']

last_hundred = np.empty(100, dtype=np.float32)
last_hundred_count = 0
#plt.ion()
while True:
    game_wrapper.reset()
    last_hundred = np.empty(100, dtype=np.float32)
    last_hundred_count = 0

    for _ in range(MAX_EPISODE_LENGTH):
        (action, value) = agent.get_action(frame_number, game_wrapper.state, give_q=True)
        '''
        if last_hundred_count < 100:
            last_hundred[last_hundred_count % 100] = value
            last_hundred_count += 1
        else:
            last_hundred = last_hundred[1:]
            np.append(last_hundred, [value])
        '''
        processed_frame, reward, terminal, life_lost = game_wrapper.step(action, render_mode='human')
        if terminal:
            break