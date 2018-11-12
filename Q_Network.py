# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
import math

TAU = 0.001

class Q_Network:
    """ Critic Q value model of the DDPG algorithm """
    def __init__(self, seed, dim_state, dim_action, action_size, action_scale, batch_size, layer_size_1, layer_size_2, lr_Q, lr_A):
        
        self.seed = seed
        self.dim_state, self.dim_action = dim_state, dim_action
        self.batch_size = batch_size
        self.action_size = action_size
        self.action_scale = action_scale
        
        self.learning_rate_Q = lr_Q
        self.learning_rate_A = lr_A
        self.N_HIDDEN_1 = layer_size_1
        self.N_HIDDEN_2 = layer_size_2
        
        #self.saver = tf.train.Saver()
        
        self.g = tf.Graph()
        with self.g.as_default():
            self.create_placeholder()
            self.create_action_batch()
            self.create_net()
            self.create_optimizer()
            self.create_softmax()
            self.create_update_operation()
            self.init_session()
            
    def create_placeholder(self):
        self.A_batch = tf.placeholder("float32", [None, self.dim_action])
        self.t_A_batch = tf.placeholder("float32", [None, self.dim_action])
        
        self.state_in = tf.placeholder("float",[None, self.dim_state])
        self.action_in = tf.placeholder("float",[None, self.dim_action])
        self.weighted_action_in = tf.placeholder("float", [self.batch_size, self.action_size])
        
        self.q_value_in = tf.placeholder("float",[None,1])
        self.prioritized_value = tf.placeholder("float",[None,1])
        
        self.scale = tf.placeholder("float")
    
    # create the random action set
    def create_action_batch(self):
        self.action_batch = tf.Variable(tf.random_uniform([self.action_size, self.dim_action], seed=self.seed, minval = -self.action_scale, maxval = self.action_scale), name = "action_batch")
        self.t_action_batch = tf.Variable(tf.random_uniform([self.action_size, self.dim_action], seed=self.seed, minval = -self.action_scale, maxval = self.action_scale), name = "t_action_batch")
    
    def create_net(self):
        N_HIDDEN_1 = self.N_HIDDEN_1
        N_HIDDEN_2 = self.N_HIDDEN_2
        
        with tf.variable_scope('q_critic'):
            h1 = tf.layers.dense(self.state_in, N_HIDDEN_1, tf.nn.softplus,
                                  kernel_initializer=tf.random_normal_initializer(stddev=0.01,seed=self.seed), name='H1')
            h2 = tf.layers.dense(h1, N_HIDDEN_2,
                                  kernel_initializer=tf.random_normal_initializer(stddev=0.01,seed=self.seed), name='H2')
            h3 = tf.layers.dense(self.action_in, N_HIDDEN_2,
                                  kernel_initializer=tf.random_normal_initializer(stddev=0.01,seed=self.seed), name='H3')
            self.q_predict = tf.layers.dense(tf.nn.tanh(h2+h3), 1,
                                  kernel_initializer=tf.random_normal_initializer(stddev=0.01,seed=self.seed), name='critic')
            
            state = tf.tile(h2, [1, self.action_size])
            state = tf.reshape(state, [-1, N_HIDDEN_2])
            action = tf.tile(h3, [self.batch_size, 1])
            stateXaction = state + action
            
            self.stateXaction = tf.layers.dense(tf.nn.tanh(stateXaction), 1, reuse=True, name='critic')
            
            Q = tf.reshape(self.stateXaction, [self.batch_size, -1])
            Q = tf.reduce_sum(Q * self.weighted_action_in, axis = 0)
            self.w_predict = tf.reshape(Q, [-1, 1])
            
        with tf.variable_scope('q_target'):
            h1_t = tf.layers.dense(self.state_in, N_HIDDEN_1, tf.nn.softplus,
                                  kernel_initializer=tf.random_normal_initializer(stddev=0.01,seed=self.seed), name='H1_t')
            h2_t = tf.layers.dense(h1_t, N_HIDDEN_2,
                                  kernel_initializer=tf.random_normal_initializer(stddev=0.01,seed=self.seed), name='H2_t')
            h3_t = tf.layers.dense(self.action_in, N_HIDDEN_2,
                                  kernel_initializer=tf.random_normal_initializer(stddev=0.01,seed=self.seed), name='H3_t')
            self.q_target = tf.layers.dense(tf.nn.tanh(h2_t+h3_t), 1,
                                  kernel_initializer=tf.random_normal_initializer(stddev=0.01,seed=self.seed), name='target')
            
            state_t = tf.tile(h2_t, [1, self.action_size])
            state_t = tf.reshape(state_t, [-1, N_HIDDEN_2])
            action_t = tf.tile(h3_t, [self.batch_size, 1])
            stateXaction_t = state_t + action_t
            
            self.stateXaction_t = tf.layers.dense(tf.nn.tanh(stateXaction_t), 1, reuse=True, name='target')
            """
            Q_t = tf.reshape(self.stateXaction_t, [self.batch_size, -1])
            Q_t = Q_t * self.weighted_action_in
            Q_t = tf.reduce_sum(Q_t, axis = 0)
            self.w_target = tf.reshape(Q_t, [-1, 1])
            """
        
        self.weights_c = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='q_critic')
        self.weights_t = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='q_target')
        
    def create_optimizer(self):
        self.error = self.q_predict-self.q_value_in
        #self.l2_regularizer_loss = 0.0001*tf.reduce_sum(tf.pow(self.W2_c,2))+ 0.0001*tf.reduce_sum(tf.pow(self.B2_c,2)) 
        
        self.cost = self.prioritized_value*tf.pow(self.error,2)#/self.batch_size + tf.reduce_mean(self.prioritized_value)*self.l2_regularizer_loss
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_Q).minimize(self.cost)
        
        self.w_actor = [self.action_in]
        self.w_actions_gradients = tf.gradients(self.w_predict, self.w_actor, tf.fill((self.action_size, 1), -1.))
        self.w_actions_optimizer =\
                    tf.train.AdamOptimizer(self.learning_rate_A).apply_gradients(zip(self.w_actions_gradients,[self.action_batch]))
    
    def create_softmax(self):
        action_Q = tf.reshape(self.q_predict, [-1])/self.scale
        max_x = tf.reduce_max(action_Q, axis=0)
        e_x = tf.exp(action_Q - max_x)
        p = e_x/tf.reduce_sum(e_x, axis=0)
        self.softmax = p/tf.reduce_sum(p, axis=0)
        
        z = tf.reshape(self.stateXaction_t, [self.batch_size, -1])/self.scale
        max_z = tf.reduce_max(z, axis=1)
        e_z = tf.exp(z - max_z[:, tf.newaxis])
        e_sum = tf.reduce_sum(e_z, axis=1)
        self.softV = self.scale * (tf.log(e_sum) + max_z)
    
    def create_update_operation(self):
        copy_net_ops = []
        for var, var_old in zip(self.weights_c, self.weights_t):
            copy_net_ops.append(var_old.assign(var))
        self.copy_net_ops = copy_net_ops
        
        update_net_ops = []
        for var, var_old in zip(self.weights_c, self.weights_t):
            update_net_ops.append(var_old.assign(TAU*var+(1-TAU)*var_old))
        self.update_net_ops = update_net_ops
        
        self.copy_action_ops = [self.t_action_batch.assign(self.action_batch)]
        self.update_action_ops = [self.t_action_batch.assign(TAU*self.action_batch+(1-TAU)*self.t_action_batch)]
        
        self.update_action_batch_op = [
            self.action_batch.assign(self.A_batch),
            self.t_action_batch.assign(self.t_A_batch)
        ]
        
    def init_session(self):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config,graph=self.g)
        self.sess.run(tf.global_variables_initializer())
        self.sess.run([self.copy_net_ops, self.copy_action_ops])
        
    def realign_action_batch(self, A_batch, t_A_batch):
        self.sess.run(self.update_action_batch_op, feed_dict={self.A_batch : A_batch, self.t_A_batch : t_A_batch})
        
    def update_target_critic(self):
        self.sess.run(self.update_net_ops)
    
    def update_action_target_critic(self):
        self.sess.run(self.update_action_ops)
    
    def train_critic(self, state_t_batch, action_t_batch, y_batch, w_batch):
        return self.sess.run([self.error, self.cost, self.optimizer],
                     feed_dict={self.state_in:state_t_batch, self.action_in:action_t_batch, self.q_value_in:y_batch, self.prioritized_value: w_batch})
    
    # 한 개의 state가 입력으로 주어지면, 각 state에 대해 action set에 있는 모든 action에 대한 q값을 반환
    # Q네트워크 사용
    def get_q_batch(self, state_t):
        action_t_1 = self.sess.run(self.action_batch)
        q_batch = self.sess.run(self.q_predict,\
                     feed_dict={self.state_in: state_t, self.action_in: action_t_1})
        return q_batch
    
    # batch_size만큼의 state가 입력으로 주어지면, 각 state에 대해 action set에 있는 모든 action에 대한 q값을 반환
    # w_네트워크 사용
    def get_target_q_batch(self, state_t):
        action_t = self.sess.run(self.t_action_batch)
        t_q_batch = self.sess.run(self.stateXaction_t,\
                     feed_dict={self.state_in: state_t, self.action_in: action_t})
        return t_q_batch
    
    def get_action_batch(self):
        return self.sess.run(self.action_batch)
        #return self.action_batch
    
    def get_target_action_batch(self):
        return self.sess.run(self.t_action_batch)
    
    # action set을 training (using weights), w_네트워크 사용
    def train_weighted_actor(self, state_t, weight_t):
        action_t = self.sess.run(self.action_batch)
        self.sess.run(self.w_actions_optimizer,\
              feed_dict={self.state_in: state_t, self.action_in: action_t, self.weighted_action_in: weight_t})
     
    def get_softmax(self, state, scale=1.):
        action_t = self.sess.run(self.action_batch)
        return self.sess.run(self.softmax, feed_dict={self.state_in:np.reshape(state, [1,-1]), self.action_in:action_t, self.scale:scale})
    
    def get_softV(self, state, scale=1.):
        action_t = self.sess.run(self.t_action_batch)
        return self.sess.run(self.softV, feed_dict={self.state_in:state, self.action_in:action_t, self.scale:scale})
    
    """
    # 네트워크 파라미터 저장 함수
    def save_network(self, game_name, episode, save_epi):
        ep = str(episode)
        self.saver.save(self.sess, "/home/minjae/Reinforcement\Learning/Network/model"+ep+"_"+game_name+".ckpt")
    # 저장된 네트워크 파라미터 불러오는 함수
    def load_network(self, game_name, saved_num):
        ep = str(saved_num)
        self.saver.restore(self.sess, "/home/minjae/Desktop/Network/model"+ep+"_"+game_name+".ckpt")
    """
    