#!/usr/bin/env python3
import os
import numpy as np
import gym
import math 
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import _pickle as cPickle
from DartEnv.environments.dartenv import DartEnv
from DartEnv.common.collect_data import collect_data
import torch
import tikzplotlib
from DartEnv.model.betaforRL import BetaEstimator
import statistics 

def reward_function(x):
    return -abs(x)

def save_model(D_P, D_Q):
    save_folder = 'DartEnv/environmentdata/DartEnv/'
    cPickle.dump(D_P, open(save_folder + 'DP.pkl', 'wb'))
    cPickle.dump(D_Q, open(save_folder + 'DQ.pkl', 'wb'))




def main(collect, estimate_beta, eval_results = False, identifier='', mean = 2.0):
    
    deployment_env = DartEnv(mean, 1.0)
    simulation_env = DartEnv(4.0, 2.0)
    if collect:
        D_P, D_Q = collect_data(simulation_env=simulation_env, deployment_env=deployment_env, episodes=1000000, buffer_size=1000000)
        save_model(D_P, D_Q)
    if estimate_beta:
        Net_sas, Net_sa = BetaEstimator(environment='DartEnv', batch_size=100000, train=True, identifier=identifier)
    if eval_results: 
        Net_sas, Net_sa = BetaEstimator(environment='DartEnv', batch_size=100000, train=False, identifier=identifier)
        true_average = {}
        estimated_average = {}
        simulation_average = {}
        for policy in np.linspace(-math.pi/3, math.pi/3, 20):
            true_average[policy] = []
            for index in range(1000):
                x = deployment_env.step(policy)
                true_average[policy].append(reward_function(x))
            print ("Policy", policy)
            print ("True Average", statistics.mean(true_average[policy]))
            estimated_average[policy] = []
            simulation_average[policy] = []
            for index in range(1000):
                x = simulation_env.step(policy)
                input_theta = torch.from_numpy(np.asarray([policy]))
                input_theta = input_theta.type(torch.FloatTensor)
                output1 = Net_sa.cpu().predict(input_theta)
                y = np.expand_dims(np.asarray([x]), axis=1)
                action = np.expand_dims(np.asarray([policy]), axis=1)
                y = np.concatenate((action, y), axis=1)
                y = torch.from_numpy(y)
                y = y.type(torch.FloatTensor)
                output2 = Net_sas.cpu().predict(y)
                beta = output2.data.numpy()[0]/output1.data.numpy()[0]
                #print (index, beta[0])
                estimated_average[policy].append(beta[0]*reward_function(x))
                simulation_average[policy].append(reward_function(x))
            print ("Estimated Average", statistics.mean(estimated_average[policy]))
            print ("Simulation Average", statistics.mean(simulation_average[policy]))



if __name__ == '__main__':
    train = False
    test = True
    if train:
        for i in range(10):
            main(collect=True, estimate_beta=True, eval_results=True, identifier=str(i) + '_2142_')
    if test:
        deployment_env = DartEnv(4.0, 1.0)
        simulation_env = DartEnv(4.0, 2.0)
        true_average = {}
        true = np.zeros(20)
        true_std = np.zeros(20)

        estimated_average = np.zeros((10, 20))
        estimated = np.zeros(20)
        estimated_std = np.zeros(20)

        simulation_average = {}
        simulation = np.zeros(20)
        simulation_std = np.zeros(20)

        for policy in np.linspace(-math.pi / 3, math.pi / 3, 20):
            print(int((policy + math.pi / 3) / (2 * math.pi / 58)))
            true_average[policy] = []
            simulation_average[policy] = []
            for index in range(100):
                x = deployment_env.step(policy)
                true_average[policy].append(reward_function(x))
                x = simulation_env.step(policy)
                simulation_average[policy].append(reward_function(x))
            for i in range(10):
                #print(i)
                Net_sas, Net_sa = BetaEstimator(environment='DartEnv', batch_size=100000, train=False,
                                                identifier=str(i) )
                temp_data = []
                for index in range(100):
                    x = simulation_env.step(policy)
                    input_theta = torch.from_numpy(np.asarray([policy]))
                    input_theta = input_theta.type(torch.FloatTensor)
                    output1 = Net_sa.cpu().predict(input_theta)
                    y = np.expand_dims(np.asarray([x]), axis=1)
                    action = np.expand_dims(np.asarray([policy]), axis=1)
                    y = np.concatenate((action, y), axis=1)
                    y = torch.from_numpy(y)
                    y = y.type(torch.FloatTensor)
                    output2 = Net_sas.cpu().predict(y)
                    beta = output2.data.numpy()[0] / output1.data.numpy()[0]
                    temp_data.append(beta[0] * reward_function(x))
                estimated_average[i, int((policy + math.pi / 3) / (2 * math.pi / 58))] = statistics.mean(temp_data)

        Policy = np.linspace(-math.pi / 3, math.pi / 3, 20)
        for policy in np.linspace(-math.pi / 3, math.pi / 3, 20):
            true[int((policy + math.pi / 3) / (2 * math.pi / 58))] = statistics.mean(true_average[policy])
            true_std[int((policy + math.pi / 3) / (2 * math.pi / 58))] = statistics.stdev(true_average[policy])

            estimated[int((policy + math.pi / 3) / (2 * math.pi / 58))] = np.mean(
                estimated_average[:, int((policy + math.pi / 3) / (2 * math.pi / 58))])
            estimated_std[int((policy + math.pi / 3) / (2 * math.pi / 58))] = np.std(
                estimated_average[:, int((policy + math.pi / 3) / (2 * math.pi / 58))])

            simulation[int((policy + math.pi / 3) / (2 * math.pi / 58))] = statistics.mean(simulation_average[policy])
            simulation_std[int((policy + math.pi / 3) / (2 * math.pi / 58))] = statistics.stdev(
                simulation_average[policy])


        
        colors = ['#000000', '#00441b', '#7f0000', '#084081', '#4d004b']
        
        plt.plot(Policy, true, '--', linewidth=2, color=colors[0])
        plt.fill_between(Policy, true - 1 * true_std, true + 1 * true_std, color=colors[0], alpha=0.1)
        plt.plot(Policy, estimated, linewidth=2, color=colors[1])
        plt.fill_between(Policy, estimated - 1 * estimated_std, estimated + 1 * estimated_std, color=colors[1],
                         alpha=0.1)
        plt.plot(Policy, simulation, linewidth=2, color=colors[2])
        plt.fill_between(Policy, simulation-1*simulation_std, simulation+1*simulation_std, color=colors[2], alpha=0.1)

        plt.xlabel('Theta')
        plt.ylabel('Average returns')

        plt.legend(['True Value', 'OEE (ours)', 'Simulated'])

        plt.grid(True)
        tikzplotlib.save("dart_evaluation.tex")
        plt.savefig('dart_evaluation.png')
