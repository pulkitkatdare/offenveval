#!/usr/bin/env python3
import os
import numpy as np
import gym
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import _pickle as cPickle
from gridworld.common.replaybuffer import ReplayBuffer
from gridworld.model.betaforRL import BetaEstimator
from gridworld.config import gridworldconfigs
from gridworld.environments.GridWorld import GridWorld, save_model, collect_data
from gridworld.eval_performance import eval_performance

import torch
import tikzplotlib
import statistics


P_paramters = 0.9
Q_parameters = 0.7

def main(config):
    if config.train:
        D_P, D_Q = collect_data(episodes=config.episodes, p_parameters=config.P_parameters, q_parameters=config.Q_parameters, size=[config.size[0], config.size[1]])
        save_model(D_P, D_Q)
        for index in range(10):
            for batch_size in config.batches:
                key = str(index) + '_' + str(batch_size) + '_dot' + str(int(10*config.P_parameters)) + '_dot' + str(int(10*config.Q_parameters)) + '_' + str(config.size[0])
                Net_sas, Net_sa = BetaEstimator(environment='GridWorld10x10', batch_size=batch_size, train=True, size=[config.size[0], config.size[1]], key=key)
    if config.test:
        eval_performance(config)

if __name__ == '__main__':
    config = gridworldconfigs()
    main(config)
