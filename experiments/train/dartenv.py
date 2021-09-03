#!/usr/bin/env python3
import os
import numpy as np
import gym
import math 
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import _pickle as cPickle
from dartenv.environments.dartenv import DartEnv
from dartenv.common.collect_data import collect_data
import torch
import tikzplotlib
from dartenv.model.betaforRL import BetaEstimator
import statistics 
from dartenv.config import dartconfigs 
def reward_function(x):
    return -abs(x)

def save_model(D_P, D_Q):
    save_folder = 'dartenv/environmentdata/DartEnv/'
    cPickle.dump(D_P, open(save_folder + 'DP.pkl', 'wb'))
    cPickle.dump(D_Q, open(save_folder + 'DQ.pkl', 'wb'))



if __name__ == '__main__':
    config = dartconfigs(index=0)
    if config.train:
        deployment_env = DartEnv(config.mean_p, config.std_p)
        simulation_env = DartEnv(config.mean_q, config.std_q)
        D_P, D_Q = collect_data(simulation_env=simulation_env, deployment_env=deployment_env, episodes=config.episodes, buffer_size=config.buffer)
        save_model(D_P, D_Q)
        for model_idx in range(10):
            key = '_' + str(int(config.mean_p)) + str(int(config.std_p)) + str(int(config.mean_q)) + str(int(config.std_q)) + '_' 
            identifier=str(model_idx) + key
            Net_sas, Net_sa = BetaEstimator(environment='DartEnv', batch_size=config.batch_size, train=True, identifier=identifier)

    if config.test:
        deployment_env = DartEnv(config.mean_p, config.std_p)
        simulation_env = DartEnv(config.mean_q, config.std_q)
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
            for model_idx in range(10):
                key = '_' + str(int(config.mean_p)) + str(int(config.std_p)) + str(int(config.mean_q)) + str(int(config.std_q)) + '_' 
                identifier=str(model_idx) + key
                Net_sas, Net_sa = BetaEstimator(environment='DartEnv', batch_size=config.batch_size, train=False,
                                                identifier=identifier)
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
                estimated_average[model_idx, int((policy + math.pi / 3) / (2 * math.pi / 58))] = statistics.mean(temp_data)

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
        tikzplotlib.save('./dartenv/assets/dart_evaluation' + key  + '.tex')
        plt.savefig('./dartenv/assets/dart_evaluation' + key + '.png')
