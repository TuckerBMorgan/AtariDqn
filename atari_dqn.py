import cv2
import time
import tensorflow as tf
import numpy as np

from game_wrapper import GameWrapper
from replay_buffer import ReplayBuffer
from agent import Agent

from tensorflow.keras.initializers import VarianceScaling
from tensorflow.keras.layers import (Add, Conv2D, Dense, Flatten, Input, Lambda, Subtract)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, RMSprop

ENV_NAME = 'BreakoutDeterministic-v4'
LOAD_FROM = None
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
if LOAD_FROM is None:
    frame_number = 0
    rewards = []
    loss_list = []
else:
    meta = agent.load(LOAD_FROM, LOAD_REPLAY_BUFFER)
    frame_number = meta['frame_number']
    rewards = meta['rewards']
    loss_list = meta['loss_list']

try:
    with writer.as_default():
        while frame_number < TOTAL_FRAMES:
            epoch_frame = 0
            while epoch_frame < FRAME_BETWEEN_EVAL:
                start_time = time.time()
                game_wrapper.reset()
                life_lost = True
                episode_reward_sum = 0
                for _ in range(MAX_EPISODE_LENGTH):
                    action = agent.get_action(frame_number, game_wrapper.state)
                    processed_frame, reward, terminal, life_lost = game_wrapper.step(action)
                    frame_number +=1 
                    epoch_frame += 1
                    episode_reward_sum += reward
                    agent.add_experience(action=action,
                                         frame=processed_frame[:,:, 0],
                                         reward=reward,clip_reward=CLIP_REWARD,
                                         terminal=life_lost)
                    if frame_number % UPDATE_FREQ == 0 and agent.replay_buffer.count > MIN_REPLAY_BUFFER_SIZE:
                        loss, _ = agent.learn(BATCH_SIZE, gamma=DISCOUNT_FACTOR, frame_number=frame_number, priority_scale=PRIORITY_SCALE)
                        loss_list.append(loss)
                    if frame_number % UPDATE_FREQ == 0 and frame_number> MIN_REPLAY_BUFFER_SIZE:
                        agent.update_target_network()
                    
                    if terminal:
                        terminal = False
                        break
                rewards.append(episode_reward_sum)
                if len(rewards) % 10 == 0:
                    if WRITE_TENSORBOARD:
                        tf.summary.scalar('Reward', np.mean(rewards[-10:]), frame_number)
                        tf.summary.scalar('Loss', np.mean(loss[-100:]), frame_number)
                        writer.flush()
                print(f'Game number: {str(len(rewards)).zfill(6)}  Frame number: {str(frame_number).zfill(8)}  Average reward: {np.mean(rewards[-10:]):0.1f}  Time taken: {(time.time() - start_time):.1f}s')

            terminal = True
            eval_rewards = []
            evaluate_frame_number = 0
            for _ in range(EVAL_LENGTH):
                if terminal:
                    game_wrapper.reset(evaluation=True)
                    life_lost = True
                    episode_reward_sum = 0
                    terminal = False
                action = 1 if life_lost else agent.get_action(frame_number, game_wrapper.state, evaluation=True)
                _, reward, terminal, life_lost = game_wrapper.step(action)
                evaluate_frame_number += 1
                episode_reward_sum += reward

                if terminal:
                    eval_rewards.append(episode_reward_sum)
            
            if len(eval_rewards) > 0:
                final_score = np.mean(eval_rewards)
            else:
                final_score = episode_reward_sum
            
            print('Evaluation score:', final_score)
            if WRITE_TENSORBOARD:
                tf.summary.scalar('Evaluation score', final_score, frame_number)
                writer.flush()
            if len(rewards) > 300 and SAVE_PATH is not None:
                agent.save(f'{SAVE_PATH}/save-{str(frame_number).zfill(8)}', frame_number=frame_number, rewards=rewards, loss_list=loss_list)
except KeyboardInterrupt:
    print('\n Traning exited early')
    writer.close()

    if SAVE_PATH is None:
        try:
            SAVE_PATH = input('Would you like to save the trained model? If so, type in a save path, otherwise, interrupt with ctrl+c. ')
        except KeyboardInterrupt:
            print('\nExiting...')

    if SAVE_PATH is not None:
        print('Saving...')
        agent.save(f'{SAVE_PATH}/save-{str(frame_number).zfill(8)}', frame_number=frame_number, rewards=rewards, loss_list=loss_list)
        print('Saved.')