import json
import os

import numpy as np
import tensorflow as tf

class RunnerAgent(object):
    def __init__(self,
        dqn,
        target_dqn,
        v_network,
        n_actions,
        input_shape=(84, 84),
        history_length=4,
        eps_initial=1,
        eps_final=0.1,
        eps_final_frame=0.01,
        eps_evaluation=0.0,
        eps_annealing_frames=100000,
        max_frames=25000000):
        self.n_actions = n_actions
        self.input_shape = input_shape
        self.history_length = history_length
        
        self.replay_buffer_start_size = 50000
        self.max_frames = max_frames
        self.eps_evaluation = eps_evaluation
        self.eps_initial = eps_initial
        self.eps_final = eps_final
        self.eps_final_frame = eps_final_frame
        self.eps_annealing_frames = eps_annealing_frames
        self.slope = -(self.eps_initial - self.eps_final) / self.eps_annealing_frames
        self.intercept = self.eps_initial - self.slope * self.replay_buffer_start_size
        self.slope_2 = -(self.eps_final - self.eps_final_frame) / (self.max_frames - self.eps_annealing_frames -self.replay_buffer_start_size)
        self.intercept_2 = self.eps_final_frame - self.slope_2*self.max_frames

        self.DQN = dqn
        self.target_dqn = target_dqn
        self.v_network = v_network

    def calc_epsilon(self, frame_number, evaluation=False):
        if evaluation:
            return self.eps_evaluation
        elif frame_number < self.replay_buffer_start_size:
            return self.eps_initial
        elif frame_number >= self.replay_buffer_start_size and frame_number < self.replay_buffer_start_size + self.eps_annealing_frames:
            return self.slope * frame_number + self.intercept
        elif frame_number >= self.replay_buffer_start_size + self.eps_annealing_frames:
            return self.slope_2*frame_number + self.intercept_2
    
    def get_action(self, frame_number, state, evaluation=False, give_q=False):
        eps = self.calc_epsilon(frame_number, evaluation)
        q_value = self.v_network.predict(state.reshape((-1, self.input_shape[0], self.input_shape[1], self.history_length)))[0]
        if np.random.rand(1) < eps:
            if give_q:
                return np.random.randint(0, self.n_actions), q_value
            else:
                return np.random.randint(0, self.n_actions)
        q_vals = self.DQN.predict(state.reshape((-1, self.input_shape[0], self.input_shape[1], self.history_length)))[0]
        if give_q:
            return q_vals.argmax(), q_value
        return q_vals.argmax()
    def save(self, folder_name, **kwargs):
        if not os.path.isdir(folder_name):
            os.makedirs(folder_name)
        self.DQN.save(folder_name + "/dqn.h5")
        self.v_network.save(folder_name + "/v.h5")
        self.target_dqn.save(folder_name + "/target_dqn.h5")
    def load(self, folder_name, load_replay_buffer=True):
        if not os.path.isdir(folder_name):
            raise ValueError(f'{folder_name} is not valid directoy')
        self.DQN = tf.keras.models.load_model(folder_name + '/dqn.h5')
        self.v_network = tf.keras.models.load_model(folder_name + '/v.h5')
        self.target_dqn = tf.keras.models.load_model(folder_name + '/target_dqn.h5')
        #self.optimzer = self.DQN.optimzer
        with open(folder_name + "/meta.json", 'r') as f:
            meta = json.load(f)
        del meta['buff_count'], meta['buff_curr']
        return meta