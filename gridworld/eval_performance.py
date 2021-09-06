#!/usr/bin/env python3
import os
import numpy as np
import gym
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import pickle 
from gridworld.common.replaybuffer import ReplayBuffer
from gridworld.model.betaforRL import BetaEstimator
import torch
from gridworld.environments.GridWorld import GridWorld, save_model, collect_data
import statistics
import tikzplotlib

LENGTH = 100
GAMMA = 0.99
SIZE = 10

P_paramters = 0.9
Q_parameters = 0.7

def main(epsilon, eval_batch):
    Batches =  [eval_batch]#[1000, 10000, 20000, 50000, 100000]#, 50000, 100000]
    algorithm = {}
    algorithm['true'] = {}
    algorithm['OEE'] = {}
    algorithm['oracle'] = {}
    algorithm['oracle_prod'] = {}
    print ("calculating true rewards now ...")
    env_train = GridWorld(size=[SIZE, SIZE], action_probability=0.7)
    
    env_oracle = GridWorld(size=[SIZE, SIZE], action_probability=0.9)
    avg_rewards = 0.0
    total_steps = 0
    for index in range(LENGTH):
            algorithm['true'][index] = []
            state = env_oracle.reset(seed = 10*index)
            row, col = state
            done = False
            steps = 0
            total_rewards = 0
            while (steps < 200):
                if not done:
                    if np.random.uniform() < epsilon:
                        if env_oracle.state[0]  < env_oracle.rows-1:
                            action = 2#np.random.choice(np.arange(4))
                        else:
                            action = 1#np.random.choice(np.arange(4))#2
                    else:
                        action = np.random.choice(np.arange(4))
                    next_state, reward, done, info = env_oracle.step(action)
                    row, col = next_state
                    reward = 1*(-0.2*(abs(row-(SIZE-1))) - 0.2*(abs(col-(SIZE-1))))
                    state = next_state
    
                    total_rewards += (GAMMA**steps)*reward
                    algorithm['true'][index].append(total_rewards)
                    steps += 1 
                else:
                    total_rewards += (GAMMA ** steps) * 0.0
                    algorithm['true'][index].append(total_rewards)
                    steps += 1 
            avg_rewards += total_rewards
    print ("True Average", (1-GAMMA)*avg_rewards/LENGTH)
    
    print ("calculating oracle rewards now ...")
    avg_rewards = 0.0
    total_steps = 0
    for index in range(LENGTH):
        algorithm['oracle'][index] = []
        algorithm['oracle_prod'][index]= []
        state = env_train.reset(seed = 10*index)
        row, col = state
        done = False
        steps = 0.0
        prod = 1
        total_rewards = 0
        while (steps < 200):
            if not done:
                if np.random.uniform()<epsilon:
                    if env_train.state[0]  < env_train.rows-1:
                        action = 2#np.random.choice(np.arange(4))#1
                    else:
                        action = 1#np.random.choice(np.arange(4))#2
                else:
                    action = np.random.choice(np.arange(4))

                next_state, reward, done, info = env_train.step(action)
                row, col = next_state
                reward = -0.2 * (abs(row - (SIZE-1))) - 0.2 * (abs(col - (SIZE-1)))
                if info['next_state_index'] == action:
                    beta = 0.9/0.7#1.28571428571
                else:
                    beta = 0.1/0.3
                
                state = next_state
                total_rewards += (GAMMA**steps)*reward
                algorithm['oracle'][index].append((GAMMA**steps)*reward)#total_rewards)
                algorithm['oracle_prod'][index].append(prod)
                prod = prod * beta
                steps += 1 
            else:
                total_rewards += (GAMMA ** steps) * 0.0
                algorithm['oracle'][index].append(0.0)
                algorithm['oracle_prod'][index].append(prod)
                steps += 1 

                break
        avg_rewards += total_rewards
    print ("Algorithm Oracle Average Rewards", (1-GAMMA)*avg_rewards/LENGTH)
    
    print ("calculating estimated rewards now, this may take some time ...")
    for batch_size in Batches:
        algorithm['OEE'][batch_size] = {}
        avg_rewards = 0.0
        total_steps = 0
        for model_idx in range(10):
            print ("(batch_size, model index)", (batch_size, model_idx))
            algorithm['OEE'][batch_size][model_idx] = {}
            key = str(model_idx) + '_' + str(batch_size) + '_dot' + str(int(10*P_paramters)) + '_dot' + str(int(10*Q_parameters)) + '_' + str(SIZE)
            Net_sas, Net_sa = BetaEstimator(environment='GridWorld10x10', batch_size=batch_size, train=False, size=[SIZE, SIZE], key=key)
            for index in range(LENGTH):
                algorithm['OEE'][batch_size][model_idx][index] = []
                state = env_train.reset(seed = 10*index)
                row, col = state
                done = False
                steps = 0
                prod = 1
                total_rewards = 0
                while (steps < 200):
                    if not done:
                        if np.random.uniform()<epsilon:
                            if env_train.state[0]  < env_train.rows-1:
                                action = 2#np.random.choice(np.arange(4))#1
                            else:
                                action = 1#np.random.choice(np.arange(4))#2
                        else:
                            action = np.random.choice(np.arange(4))

                        next_state, reward, done, info = env_train.step(action)
                        reward = 1*(-0.2 * (abs(row - (SIZE-1))) - 0.2 * (abs(col - (SIZE-1))))
                        row, col = next_state
                        state = np.array(state)
                        state = np.expand_dims(state, axis=0)
                        next_state = np.array(next_state)
                        next_state = np.expand_dims(next_state, axis=0)
                        x = np.concatenate((state, [[action]], next_state), axis=1)
                        x = torch.from_numpy(x)
                        x = x.type(torch.FloatTensor)
                        beta1 = (Net_sas.predict(x))
                        beta1 = beta1.data.numpy()

                        y = np.concatenate((state, [[action]]), axis=1)
                        y = torch.from_numpy(y)
                        y = y.type(torch.FloatTensor)
                        beta2 = (Net_sa.predict(y))
                        beta2 = beta2.data.numpy()
                        beta = beta1[0][0]/beta2[0][0]
                        state = env_train.state
                        total_rewards += (GAMMA**steps)*reward*prod
                        algorithm['OEE'][batch_size][model_idx][index].append(total_rewards)
                        prod = prod * beta
                        steps += 1 
                    else:
                        total_rewards += (GAMMA ** steps) * 0.0 * prod
                        algorithm['OEE'][batch_size][model_idx][index].append(total_rewards)
                        steps += 1
                        break
                avg_rewards += total_rewards
                total_steps += steps
        print ("OEE Estimated Rewards", (1-GAMMA)*avg_rewards/(LENGTH*10))

    return algorithm

def eval_data(algorithm, key, batch_size=int(10**(5.0))):
    if key != 'OEE':
        mean_returns = np.zeros(200)
        std_returns = np.zeros(200)
        for i in range(200):
            return_per_timestep = np.zeros(len(list(algorithm[key].keys())))
            prod_per_timestep = np.zeros(len(list(algorithm[key].keys())))
            if key == 'oracle':
                for index in list((algorithm[key].keys())):
                    if i < len(algorithm[key][index]):
                        prod_per_timestep[index] = algorithm['oracle_prod'][index][i]
                    else:
                        prod_per_timestep[index] = algorithm['oracle_prod'][index][len(algorithm['oracle_prod'][index])-1]

            for index in list((algorithm[key].keys())):    
                if i < len(algorithm[key][index]):
                    if key == 'oracle':
                        return_per_timestep[index] = algorithm[key][index][i]*algorithm['oracle_prod'][index][i]/np.sum(prod_per_timestep)
                    else:
                        return_per_timestep[index] = algorithm[key][index][i]
                else: 
                    if key == 'oracle':
                        return_per_timestep[index] = algorithm[key][index][len(algorithm[key][index])-1]*algorithm['oracle_prod'][index][len(algorithm['oracle_prod'][index])-1]/np.sum(prod_per_timestep)
                    else:
                        return_per_timestep[index] = algorithm[key][index][len(algorithm[key][index])-1]#*algorithm['oracle_prod'][index][len(algorithm[key][index])-1]
            if key == 'oracle':
                if i > 0:
                    mean_returns[i] = np.sum(return_per_timestep) + mean_returns[i-1]
                else:
                    mean_returns[i] =  np.sum(return_per_timestep)
            else:
                mean_returns[i] = np.mean(return_per_timestep)
                std_returns[i] = np.std(return_per_timestep)
        return mean_returns, std_returns
    else: 
        mean_returns = np.zeros(200)
        std_returns = np.zeros(200)
        for i in range(200):
            model_returns = np.zeros(10)
            for model_idx in range(10):
                return_per_timestep = np.zeros(len(list(algorithm[key][batch_size][model_idx].keys())))
                for index in list((algorithm[key][batch_size][model_idx].keys())):    
                    if i < len(algorithm[key][batch_size][model_idx][index]):
                        return_per_timestep[index] += algorithm[key][batch_size][model_idx][index][i]
                    else: 
                        return_per_timestep[index] += algorithm[key][batch_size][model_idx][index][len(algorithm[key][batch_size][model_idx][index])-1]
                model_returns[model_idx] = np.mean(return_per_timestep)
            mean_returns[i] = np.mean(model_returns)
            std_returns[i] = np.std(model_returns)

        return mean_returns, std_returns


def plot_eval(mean_output, std_output, ax, color):
    x = np.arange(200) 
    mean = mean_output
    std = std_output
    ax.plot(x, mean, color, linewidth=2)
    ax.fill_between(x, mean - 1 * std, mean + 1 * std,color=color, alpha=0.1)
 
def eval_performance(config):

    fig, ax = plt.subplots()
    batch_size = config.eval_batch
    colors = ['#000000', '#00441b', '#7f0000', '#084081', '#4d004b']
    data_is = pickle.load(open('./gridworld/is_data.pkl','rb'))
    for index, epsilon in enumerate(config.epsilons):

        algorithm = main(epsilon, eval_batch=config.eval_batch)
        mean_output, std_output = eval_data(algorithm, key='true')
        mean_output = (1-GAMMA)*mean_output 
        std_output = (1-GAMMA)*std_output/np.sqrt(LENGTH)
        plot_eval(mean_output, std_output, ax, colors[0])

        mean_output, std_output = eval_data(algorithm, key='oracle')
        mean_output = (1-GAMMA)*mean_output 
        std_output = (1-GAMMA)*std_output/np.sqrt(LENGTH)

        plot_eval(mean_output, std_output, ax, colors[1])
        mean_output, std_output = eval_data(algorithm, key='OEE',batch_size=batch_size)
        mean_output = (1-GAMMA)*mean_output 
        std_output = (1-GAMMA)*std_output/np.sqrt(LENGTH)

        plot_eval(mean_output, std_output, ax, colors[2])
        if epsilon != 1.0:
            x = np.arange(200)
            for i in range(200):
                data_is[epsilon][i]= (1-GAMMA)*data_is[epsilon][i]
            ax.plot(x, data_is[epsilon], colors[3], linewidth=2)
            plt.legend(['True Rewards','Oracle',  'OEE (ours)', 'Importance Sampling'])
        else:
            plt.legend(['True Rewards','Oracle',  'OEE (ours)'])
        plt.grid(True)
        plt.xlabel('Time Steps')
        plt.ylabel('Average Returns')
        tikzplotlib.save('gridworld/assets/gridworld_evaluation' + str(int(10*epsilon)) + '_' + str(batch_size)+ '.tex')
        plt.savefig('gridworld/assets/gridworld_eval_trained_' + str(int(10*epsilon)) + '_' + str(batch_size)+'.png')
        fig, ax = plt.subplots()
plt.close()
