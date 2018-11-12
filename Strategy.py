# -*- coding: utf-8 -*-

import numpy as np
import random
from collections import deque
import gym
from Q_Network import Q_Network
import Train
import DPP
import Exploration
import replaymemory
import matplotlib.pyplot as plt
import pickle
import time

class RL:
    def __init__(self, dis = 0.99, REPLAY_MEMORY = 100000, batch_size = 64, max_steps = 10000000, max_episodes = 500000,
                 
                 layer_size_Q1 = 300, layer_size_Q2 = 400, learning_rate_Q = 0.0001, learning_rate_A = 0.01,
                 training_step = 1, copy_step = 1, action_copy_step = 1, repu_num = 1, beta_max_step = 1500000,
                 
                 ending_cond_epis = 100, ending_cond_reward = 195, min_distance = 0.1,
                 alpha = 0.6, beta_init = 0.4, eps = 0.01, scale = 1, size_action_batch = 500,
                 
                 seed_n = 0, Game = 'CartPole-v0', file_name = 'steps', save_epi = 100, save_network = False):
        
        env = gym.make(Game)
        rng = np.random.RandomState(seed_n)
        
        input_size = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        output_size = self.action_dim
        
        action_map = []
        action_scale = env.action_space.high[0]
        """
        for o in range(self.action_dim):
            print(env.action_space.low[o])
            print(env.action_space.high[o])
        """
        replay_memory = replaymemory.ReplayMemory(rng=rng, memory_size=REPLAY_MEMORY, per_alpha=alpha, per_beta0=beta_init)
        
        ############### parameter 복사 ################
        self.dis = dis
        self.REPLAY_MEMORY = REPLAY_MEMORY
        self.replay_memory_size = REPLAY_MEMORY
        self.replay_memory = replay_memory
        self.batch_size = batch_size
        self.size_action_batch = size_action_batch
        
        self.alpha = alpha
        self.beta_init = beta_init
        self.beta_max_step = beta_max_step
        self.min_distance = min_distance
        self.eps = eps
        self.scale = scale
        self.action_scale = action_scale
        
        self.layer_size_Q1 = layer_size_Q1
        self.layer_size_Q2 = layer_size_Q2
        self.learning_rate_Q = learning_rate_Q
        self.learning_rate_A = learning_rate_A
        
        self.training_step = training_step
        self.copy_step = copy_step
        self.action_copy_step = action_copy_step
        self.repu_num = repu_num
        
        self.seed_n = seed_n
        self.Game = Game
        self.save_epi = save_epi
        self.max_episodes = max_episodes
        self.max_steps = max_steps
        self.env = env
        self.file_name = file_name
        self.save_network = save_network
        
        self.input_size = input_size
        self.output_size = output_size
        self.ending_cond_epis = ending_cond_epis
        self.ending_cond_reward = ending_cond_reward
        #################################################
        
        self.Q_Network = Q_Network(seed_n, input_size, output_size, size_action_batch, action_scale, batch_size, layer_size_Q1, layer_size_Q2, learning_rate_Q, learning_rate_A)
        
        # run_DQN 실행
        self.run_DQN(seed_n = seed_n)
    
    
    # Calculate the action distance
    # 모든 행동 간의 거리를 측정한 뒤, 모든 행동 간의 평균 거리와 특정거리보다 가까운 쌍의 개수를 세는 함수
    def get_action_variance(self, A_batch):
        A_batch = np.array(A_batch)
        num_actions = np.shape(A_batch)[0]
        dim_actions = np.shape(A_batch)[1]
        
        distance = 0.
        num = 0.
        num2 = 0.
        for i in range(num_actions):
            A = np.square(A_batch - A_batch[i])
            A = np.sum(A, axis = 1)
            
            num = num + (A < self.min_distance).sum()
            num2 = num2 + (A < self.min_distance*3).sum()
            
            A = np.sqrt(A)
            distance = distance + np.sum(A)
        
        num = (num-num_actions)*100/(num_actions*(num_actions-1))
        num = round(num, 3)
        
        num2 = (num2-num_actions)*100/(num_actions*(num_actions-1))
        num2 = round(num2, 3)
        
        distance = distance / (num_actions*(num_actions-1))
        distance = round(distance, 4)
        
        # distance : 모든 행동간 평균 거리
        # num : 전체 행동 쌍 중에서 거리가 루트(0.3)미만인 행동 쌍의 비율
        # num2 : 전체 행동 쌍 중에서 거리가 루트(1)미만인 행동 쌍의 개수
        return distance, num, num2
    
    # resampling the action batch
    def realign_action_batch(self, A_batch, t_A_batch):
        A_batch = np.array(A_batch)
        t_A_batch = np.array(t_A_batch)
        num_actions = np.shape(A_batch)[0]
        dim_actions = np.shape(A_batch)[1]
        """
        for j in range(num_actions):
            if np.max(A_batch[j]) > 1 or np.min(A_batch[j]) < -1:
                k = max(np.max(A_batch[j]), -np.min(A_batch[j]))
                A_batch[j] = A_batch[j] / k
        
        """
        A_max = np.max(np.hstack((A_batch, -A_batch)), axis=1, keepdims=True)
        A_batch = A_batch/np.tile(A_max,(1,self.action_dim))*(A_max>0)
        
        for i in range(num_actions):
            A = np.square(A_batch - A_batch[i])
            A = np.sum(A, axis = 1)
            
            # A_batch[i]와 거리가 루트(0.3)미만인 행동(A_batch[i]를 제외한)이 존재할 시,
            if (A < self.min_distance).sum() > 1:
                A_batch[i] = np.random.random(dim_actions)*2*self.action_scale - self.action_scale # resampling
                t_A_batch[i] = A_batch[i]                                                          # t_A_batch[i]도 교체
            
        return A_batch, t_A_batch
    
    
    def resampling_dpp(self, A_batch, t_A_batch):
        A_batch = np.array(A_batch)
        t_A_batch = np.array(t_A_batch)
        num_actions = np.shape(A_batch)[0]
        dim_actions = np.shape(A_batch)[1]
        
        A_max = np.max(np.hstack((A_batch, -A_batch)), axis=1, keepdims=True)
        A_batch = A_batch/np.tile(A_max,(1,self.action_dim))*(A_max>0)
        
        idx = [True]*num_actions
        k = 0
        for i in range(num_actions):
            A = np.square(A_batch[i:,:] - A_batch[i])
            A = np.sum(A, axis = 1)
            if (A < self.min_distance).sum() > 1:
                idx[i] = False
                k += 1
        
        if k > 0:
            A_batch = np.array(A_batch)[idx]
            t_A_batch = np.array(t_A_batch)[idx]
            
            c_A_batch = np.random.random([k*10, dim_actions])*2*self.action_scale - self.action_scale
            A_batch = DPP.sample_k(A_batch, c_A_batch, 0.1, k)
            t_A_batch = np.concatenate((t_A_batch, A_batch[(num_actions-k):,:]))
            
        return A_batch, t_A_batch
        
        
    def format_experience(self, experience):
        states_b, actions_b, rewards_b, states_n_b, done_b = zip(*experience)
        
        minibatch = []
        for num in range(len(states_b)):
            minibatch.append((states_b[num],states_n_b[num],actions_b[num],rewards_b[num],done_b[num]))
        return minibatch
    
    def run_DQN(self, seed_n):
        ############## parameter 복사 ##############
        dis = self.dis
        REPLAY_MEMORY = self.REPLAY_MEMORY
        replay_memory = self.replay_memory
        batch_size = self.batch_size
        size_action_batch = self.size_action_batch
        
        Game = self.Game
        save_epi = self.save_epi
        save_network = self.save_network
        max_episodes = self.max_episodes
        max_steps = self.max_steps
        env = self.env
        
        input_size = self.input_size
        output_size = self.output_size
        
        alpha = self.alpha
        beta_init = self.beta_init
        beta_max_step = self.beta_max_step
        eps = self.eps
        scale = self.scale
        
        training_step = self.training_step
        copy_step = self.copy_step
        action_copy_step = self.action_copy_step
        repu_num = self.repu_num
        
        ending_cond_epis = self.ending_cond_epis
        ending_cond_reward = self.ending_cond_reward
        
        env.seed(seed_n)
        np.random.seed(seed_n)
        random.seed(seed_n)
        #############################################
        
        Q_Network = self.Q_Network
        A_batch = Q_Network.get_action_batch()
        
        case_n = seed_n + 1
        end_episode = 0
        step_count_total = 0
        global_step = 0
        loss = 0.
        
        replay_buffer = deque()
        Q_list = []
        TD_buffer = deque()
        steps_list = []
        step_avg_list = []
        global_step_list = []
        
        t0 = time.time()
        
        print("")
        print("CASE {}".format(case_n))
        print("  STATE DIM : {}, ACTION DIM : {}".format(input_size, self.action_dim))
        print("  Action Particle Method ::: Exp : Softmax")
        print("")
        
        for episode in range(1, max_episodes+1):
            
            done = False
            step_count = 0
            current_step = 0
            cost = 0
            range_out = 0
            state = env.reset()
            
            while not done:
                action_soft = Q_Network.get_softmax(state, scale)
                action0 = np.random.choice(len(action_soft),size=1,p=action_soft)[0]
                action = A_batch[action0]
                
                if np.max(action)>1 or np.min(action)<-1:
                    #print(action)
                    range_out += 1
                
                next_state, reward, done, _ = env.step(action)
                step_count += reward
                global_step += 1
                current_step += 1
                
                replay_memory.save_experience(state, action, reward, next_state, done)
                state = next_state
                
                if global_step <= beta_max_step:
                    replay_memory.anneal_per_importance_sampling(global_step, beta_max_step)
                
                # training step마다 traing 실행
                if global_step > batch_size and global_step % training_step == 0:
                    for re in range(repu_num): # repu_num만큼 반복 training. 거의 1로 사용.
                        # replay_memory로부터 batch를 추출
                        idx, priorities, w_batch, experience = replay_memory.retrieve_experience(batch_size)
                        minibatch = self.format_experience(experience)
                        
                        errors, cost, state_t_batch = Train.train(Q_Network, minibatch, w_batch, scale, input_size, output_size, size_action_batch)
                        replay_memory.update_experience_weight(idx, errors)
                        
                        # action_copy_step 마다 action set을 training
                        if global_step % action_copy_step == 0:
                            action_weight = []

                            # weight 계산
                            for k in range(batch_size):
                                state_t = np.reshape(state_t_batch[k], [1,-1])

                                q_batch = Q_Network.get_q_batch(state_t)
                                q_batch = np.reshape(q_batch, [1,-1])[0]
                                q_batch = q_batch*10.
                                max_q = np.max(q_batch)
                                q_batch = np.exp(q_batch - max_q)
                                action_weight.append(q_batch)

                            # weight 값을 이용한 Q네트워크 training
                            Q_Network.train_weighted_actor(state_t_batch, action_weight)
                            
                            # target-action set을 update
                            Q_Network.update_action_target_critic()
                            A_batch = Q_Network.get_action_batch()
                            t_A_batch = Q_Network.get_target_action_batch()

                            #resampling
                            #A_batch, t_A_batch = self.realign_action_batch(A_batch, t_A_batch)
                            A_batch, t_A_batch = self.resampling_dpp(A_batch, t_A_batch)
                            
                            
                            Q_Network.realign_action_batch(A_batch, t_A_batch)
                            A_batch = Q_Network.get_action_batch()
                            t_A_batch = Q_Network.get_target_action_batch()
                          
                # copy_step 마다 Q네트워크 업데이트
                if global_step % copy_step == 0:
                    Q_Network.update_target_critic()
                    Q_Network.update_action_target_critic()
                            
            steps_list.append(step_count)
            global_step_list.append(global_step)
            
            # Print the average of result 
            if episode < ending_cond_epis:
                step_count_total += steps_list[episode - 1]
                step_avg_list.append(step_count_total / episode)

            if episode == ending_cond_epis:
                step_count_total += steps_list[episode - 1]
                step_avg_list.append(step_count_total / ending_cond_epis)

            if episode > ending_cond_epis:
                step_count_total += steps_list[episode - 1]
                step_count_total -= steps_list[episode - 1 - ending_cond_epis]
                step_avg_list.append(step_count_total / ending_cond_epis)
            
            print("{}           {}".format(episode, round(step_avg_list[episode - 1], 3)))
            print ("                   ( Result : {},  Loss : {},  Steps : {},  Global Steps : {} )"
                               #.format(round(step_count, 3), round(cost, 5), current_step, global_step))
                               .format(round(step_count, 3), 0, current_step, global_step))
            
            distance, per_of_sim, per_of_sim2 = self.get_action_variance(A_batch)
            print ("                   ( Action Batch  ::::  Distance : {},  Percent : {}%({}%),  range_out : {}% )"
                                   .format(distance, per_of_sim, per_of_sim2, range_out/float(current_step)*100))
            
            
            # Save the networks 
            if episode % save_epi == 0:
                file_case = str(case_n)
                if save_network:
                    Q_Network.save_network(game_name = self.file_name+'_seed'+file_case, episode = episode, save_epi = save_epi)
                
                t1 = time.time()
                print("")
                print("running time ::: {}s,  Average ::: {}s".format(round((t1-t0),3), round((t1-t0)/global_step*1000,3)))
                
                with open('/home/minjae/Desktop/JOLP/APM/'+self.file_name+'_seed'+file_case, 'wb') as fout:
                    pickle.dump(step_avg_list, fout)
                with open('/home/minjae/Desktop/JOLP/APM/'+self.file_name+'_global_'+'_seed'+file_case, 'wb') as fout2:
                    pickle.dump(global_step_list, fout2)

                x_values = list(range(1, episode+1))
                y_values = step_avg_list[:]
                plt.plot(x_values, y_values, c='green')
                plt.title(self.file_name)
                plt.grid(True)
                plt.show()
                
            end_episode += 1
            
            # 결과가 목표치를 달성하면 학습 중단
            if step_avg_list[episode - 1] > ending_cond_reward:
                break
            # max_steps 만큼 학습되었으면 학습 중단    
            if global_step > max_steps:
                break

        print("--------------------------------------------------")
        print("--------------------------------------------------")
        
        # 목표치를 달성하여 학습 중단 시, 남은 episode 만큼 실행
        for episode in range(end_episode + 1, max_episodes+1):
            if global_step > max_steps:
                break
            
            s = env.reset()
            reward_sum = 0
            done = False
            while not done :
                # 최대 Q 값을 나타내는 행동 선택
                action_Q = np.reshape(Q_Network.get_q_batch(np.reshape(state,[1,-1])),[1,-1])[0]
                action0 = np.argmax(action_Q)
                action = A_batch[action0]
                
                state, reward, done, _ = env.step(action)
                reward_sum += reward
                global_step += 1

                if done :
                    steps_list.append(reward_sum)
                    global_step_list.append(global_step)
                    step_count_total += steps_list[episode - 1]
                    step_count_total -= steps_list[episode - 1 - ending_cond_epis]
                    step_avg_list.append(step_count_total / ending_cond_epis)
                    print("{}           {}".format(episode, round(step_avg_list[episode - 1], 3)))
                    print ("                   ( Result : {} )".format(reward_sum))
        
            if episode % save_epi == 0:
                file_case = str(case_n)
                if save_network:
                    Q_Network.save_network(game_name = self.file_name+'_seed'+file_case, episode = episode, save_epi = save_epi)
                with open('/home/minjae/Desktop/JOLP/APM/'+self.file_name+'_seed'+file_case, 'wb') as fout:
                    pickle.dump(step_avg_list, fout)
                with open('/home/minjae/Desktop/JOLP/APM/'+self.file_name+'_global_'+'_seed'+file_case, 'wb') as fout2:
                    pickle.dump(global_step_list, fout2)

                x_values = list(range(1, episode+1))
                y_values = step_avg_list[:]
                plt.plot(x_values, y_values, c='green')
                plt.title(self.file_name)
                plt.grid(True)
                plt.show()
        
        t1 = time.time()
        print("")
        print("running time ::: {}s,  Average ::: {}s".format(round((t1-t0),3), round((t1-t0)/global_step*1000,3)))
        
        # parameter 저장
        file_case = str(case_n)
        with open('/home/minjae/Desktop/JOLP/APM/'+self.file_name+'_seed'+file_case, 'wb') as fout:
            pickle.dump(step_avg_list, fout)
        with open('/home/minjae/Desktop/JOLP/APM/'+self.file_name+'_global_'+'_seed'+file_case, 'wb') as fout2:
            pickle.dump(global_step_list, fout2)
        
        # 그래프 출력
        x_values = list(range(1, len(step_avg_list)+1))
        y_values = step_avg_list[:]
        plt.plot(x_values, y_values, c='green')
        plt.title(self.file_name)
        plt.grid(True)
        plt.show()
        
        