#!/usr/bin/env python3
import os
import numpy as np
import gym
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import _pickle as cPickle
from GridWorld.common.replaybuffer import ReplayBuffer
from GridWorld.model.betaforRL import BetaEstimator
import torch
import tikzplotlib
import statistics

from GridWorld.environments.GridWorld import GridWorld, save_model, collect_data

P_paramters = 0.9
Q_parameters = 0.7

def plot(input_data):
    batches = [1000,int(10**(3.5)) , int(10**(4)), int(10**(4.5)), int(10**(5)), int(10**(5.5))]
    sizes = [[10, 10], [20, 20], [40, 40]]
    colors = ['#662506', '#00441b', '#4d004b', '#084081']
    X = batches
    Y = np.arange(len(batches))
    plotting_data = {}
    mean = {}
    std = {}
    for size in sizes:
        mean[str(size)] = []
        std[str(size)] = []
        for batch_size in batches:
            data_keys = input_data[str(size)][batch_size][0].keys()
            error_list = []
            for key in data_keys:
                error_per_ensemble = 0.0
                for index in range(10):
                    error_per_ensemble += input_data[str(size)][batch_size][index][key]
                error_per_ensemble /= 10.0
                error_list.append(error_per_ensemble)
            mean[str(size)].append(statistics.mean(error_list))
            std[str(size)].append(statistics.stdev(error_list))
    
    fig, ax = plt.subplots()
    for index, size in enumerate(sizes):
        mean_size = np.asarray(mean[str(size)])
        print (mean_size)
        std_size = np.asarray(std[str(size)])

        ax.plot(Y, mean_size, marker='o', markerfacecolor='#000000', markersize=12, color=colors[index], linewidth=4)
        ax.fill_between(Y, mean_size - 1 * std_size, mean_size + 1 * std_size, color=colors[index], alpha=0.1)
    
    plt.grid(True)
    plt.xticks(Y, batches)
    plt.xlabel('Sample Size', fontsize=16)
    plt.ylabel('MSE Error', fontsize=16)
    plt.legend(sizes)
    tikzplotlib.save("gridworld_samplesize.tex")
    plt.savefig('gridworld.png')







def main(Collect_Data, estimate_beta, visualise_results, size):
    if Collect_Data:
        D_P, D_Q = collect_data(episodes=2000, p_parameters=P_paramters, q_parameters=Q_parameters, size=[size[0], size[1]])
        save_model(D_P, D_Q)
    if estimate_beta:
        for index in range(8, 10):
            for batch_size in [1000,int(10**(3.5)) , int(10**(4)), int(10**(4.5)), int(10**(5)), int(10**(5.5))]:
                key = str(index) + '_' + str(batch_size) + '_dot' + str(int(10*P_paramters)) + '_dot' + str(int(10*Q_parameters)) + '_' + str(size[0])
                Net_sas, Net_sa = BetaEstimator(environment='GridWorld10x10', batch_size=batch_size, train=True, size=[size[0], size[1]], key=key)
    if visualise_results:

        Beta_oracle = []
        Beta_oracle.append(np.array([1.285, 0.333, 0.333, 0.333]))
        Beta_oracle.append(np.array([0.333, 1.285, 0.333, 0.333]))
        Beta_oracle.append(np.array([0.333, 0.333, 1.285, 0.333]))
        Beta_oracle.append(np.array([0.333, 0.333, 0.333, 1.285]))
        Batches = [1000,int(10**(3.5)) , int(10**(4)), int(10**(4.5)), int(10**(5)), int(10**(5.5))]
        Data_Batches = {}
        for size in [[10, 10], [20, 20], [40, 40]]:
            env = GridWorld(size=(size[0], size[1]), action_probability=0.7)
            Data_Batches[str(size)] = {}
            for batch_size in Batches:
                Data_Batches[str(size)][batch_size] = {}
                for j in range(10):
                    print (batch_size, size, j)
                    Data_Batches[str(size)][batch_size][j] = {}
                    key = str(j) + '_' + str(batch_size) + '_dot' + str(int(10*P_paramters)) + '_dot' + str(int(10*Q_parameters)) + '_' + str(size[0])
                    Beta1, Beta2 = BetaEstimator(environment='GridWorld10x10', batch_size=batch_size, train=False, size=[size[0], size[1]], key=key)
                    sum = 0
                    total = 0
                    error_total = 0
                    for row in range(env.rows):
                        for col in range(env.columns):
                            state = [row, col]
                            env.state = [row, col]
                            state = np.array(state)
                            state = np.expand_dims(state, axis=0)
                            
                            for index, action in enumerate(range(4)):
                                _, reward, done, info = env.step(action)
                                error_actions = 0
                                for i in range(4):
                                    total += 1
                                    next_state = info['next_state'][i]
                                    next_state = np.array(next_state)
                                    next_state = np.expand_dims(next_state, axis=0)

                                
                                    x = np.concatenate((state, [[action]], next_state), axis=1)
                                    x = torch.from_numpy(x)
                                    x = x.type(torch.FloatTensor)
                                    output1 = Beta1.predict(x)
                                    output1 = output1.data.numpy()
                                    y = np.concatenate((state, [[action]]), axis=1)
                                    y = torch.from_numpy(y)
                                    y = y.type(torch.FloatTensor)
                                    output2 = Beta2.predict(y)
                                    output2 = output2.data.numpy()
                                    beta = output1[0][0]/output2[0][0]
                                    if info['p'][i] == 0.7:
                                        error = abs(beta - 1.285)
                                    else:
                                        error = abs(beta - 0.333)
                                    Data_Batches[str(size)][batch_size][j][total] = error
                                    error_actions += error
                                env.state = [row, col]
                                error_total += error_actions
        plot(Data_Batches)
        
        '''
        colors = ['#00441b', '#006d2c', '#238b45', '#41ae76', '#66c2a4', '#99d8c9', '#ccece6', '#e5f5f9', '#f7fcfd']
        X = list(Data_Batches.keys())
        Y = np.arange(len(Batches))
        fig, ax = plt.subplots()
        colors = ['#662506', '#00441b', '#4d004b', '#084081']
        for index, size in enumerate([[10, 10], [20, 20], [40, 40]]):
            mean = []
            std = []
            for batch_size in Batches:
                mean.append(np.mean(Data_Batches[str(size)][batch_size]))
                std.append(np.std(Data_Batches[str(size)][batch_size]))
            mean = np.asarray(mean)
            std = np.asarray(std)
            ax.plot(Y, mean, marker='o', markerfacecolor='#000000', markersize=12, color=colors[index], linewidth=4)
            ax.fill_between(Y, mean - 1 * std, mean + 1 * std, color=colors[index], alpha=0.1)

        plt.grid(True)

        plt.xticks(Y, Batches)
        plt.xlabel('Sample Size', fontsize=16)
        plt.ylabel('MSE Error', fontsize=16)
        plt.legend(X)
        tikzplotlib.save("gridworld_samplesize.tex")
        plt.savefig('gridworld.png')
        '''


if __name__ == '__main__':
    sizes = [5, 10, 20, 40]
    size = 10
    #for size in [10, 20, 40]:
    main(Collect_Data=False, estimate_beta=False, visualise_results=True, size=[size, size])
