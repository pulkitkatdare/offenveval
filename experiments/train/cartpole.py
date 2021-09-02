#!/usr/bin/env python3
from Cartpole.common.collect_data import collect_data, save_model, load_model
from Cartpole.model.betaforRL import BetaEstimator
import torch

from Cartpole.environments.cartpole import CartPoleEnv

CUDA = torch.cuda.is_available()
GAMMA = 0.99
ENVIRONMENT = 'Cartpole-v0'


def main(Collect_Data, simulation_parameter, deployment_parameter, key):
    if Collect_Data:
        simulation_parameter = simulation_parameter
        deployment_parameter = deployment_parameter

        simulation_env = CartPoleEnv(gravity=simulation_parameter)
        deployment_env = CartPoleEnv(gravity=deployment_parameter)

        D_P, D_Q = collect_data(simulation_env, deployment_env, episodes=3000)
        save_model(D_P, D_Q, environment=ENVIRONMENT)

    Net = BetaEstimator(environment=ENVIRONMENT, deployment_parameter=deployment_parameter, batch_size=80000, train=True, key=key)


if __name__ == '__main__':
    #i = 2
    #main(Collect_Data=True, simulation_parameter=9.8, deployment_parameter = 5 + 2.5*i)#0.0013 + (i+1)*0.0002)
    for i in range(5):
        for model_idx in range(10):
            main(Collect_Data=True, simulation_parameter=10.0, deployment_parameter=5 + 2.5*i, key=str(int(model_idx)))  # 0.0013 + (i+1)*0.0002)
