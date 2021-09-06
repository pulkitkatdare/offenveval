#!/usr/bin/env python3
import torch
from torch.autograd import Variable

import statistics
import numpy as np
import math
import pickle 
import tikzplotlib
import matplotlib.pyplot as plt

from cartpole.baselines.model import DynamicModel
from cartpole.common.collect_data import collect_data, save_model, load_model
from cartpole.model.betaforRL import BetaEstimator
from cartpole.config import cartpoleconfigs
from cartpole.environments.cartpole import CartPoleEnv
from cartpole.pytorchrl.rl.models.cem.actor_cartpole import ActorNetwork
from cartpole.eval import test

CUDA = torch.cuda.is_available()
GAMMA = 0.99
ENVIRONMENT = 'Cartpole-v0'


def main(config):
    if config.train:
        simulation_env = CartPoleEnv(gravity=config.simulation_parameter)
        deployment_env = CartPoleEnv(gravity=config.deployment_parameter)
        D_P, D_Q = collect_data(simulation_env, deployment_env, episodes=config.episodes, buffer_size=config.buffer)
        save_model(D_P, D_Q, environment=ENVIRONMENT)

        for model_idx in range(10):
            Net = BetaEstimator(environment=ENVIRONMENT, deployment_parameter=config.deployment_parameter, batch_size=config.batch_size, train=True, key=str(int(model_idx)))

    if config.test: 
        test(config)

if __name__ == '__main__':
    config = cartpoleconfigs(index=1)
    main(config)  

