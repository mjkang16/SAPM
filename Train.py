# -*- coding: utf-8 -*-

import numpy as np

GAMMA = 0.99

def train(Q_Network, train_batch, w_batch, s_scale, input_size, num_actions, size_action_batch):
    
    state_t_batch, state_t_1_batch, action_batch, reward_batch, done_batch = zip(*train_batch)
    
    state_t_batch = np.array(state_t_batch)
    state_t_1_batch = np.array(state_t_1_batch)
    action_batch = np.array(action_batch)
    reward_batch = np.array(reward_batch)
    done_batch = np.array(done_batch)
    
    q_t_1_batch = Q_Network.get_softV(state_t_1_batch, s_scale)
    q_t_1_batch = reward_batch + GAMMA*q_t_1_batch*(1-done_batch)
    
    q_t_1_batch = np.reshape(q_t_1_batch,[-1,1])
    w_batch = np.reshape(w_batch,[-1,1])
    
    errors, cost, _ = Q_Network.train_critic(state_t_batch, action_batch, q_t_1_batch, w_batch)
    errors = np.sum(errors, axis=1)
    
    return errors, cost, state_t_batch

