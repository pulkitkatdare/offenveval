from reacher.environment.gym_manipulator_envs import ReacherBulletEnv
from reacher.common.collect_data import collect_data, save_model, load_model
from reacher.model.betaforRL import BetaEstimator
from reacher.baselines.model import DynamicModel
from reacher.eval import eval_performance
from reacher.config import reacherconfigs

import torch
import pybullet
import gym

import os
import pickle

import numpy as np 
import matplotlib.pyplot as plt 
import tikzplotlib
import statistics
 

GAMMA = 0.99
def main(config):
    if config.train:
        deployment_env = ReacherBulletEnv(model_xml=os.getcwd() + '/reacher/environment/environment_parameters/' + config.xml_dir + '/reacher_p.xml', render=False)
        simulation_env = ReacherBulletEnv(model_xml=os.getcwd() + '/ReacherEnv/environment/environment_parameters/'+ config.xml_dir + '/reacher_q.xml', render=False)
        D_P, D_Q = collect_data(simulation_env=simulation_env, deployment_env=deployment_env, episodes=config.episodes, buffer_size=config.buffer)
        save_model(D_P, D_Q, environment='ReacherEnv')
        for model_idx in range(10):
            Net = BetaEstimator(environment='ReacherEnv', deployment_parameter=int(config.xml_dir), batch_size=config.batch_size,
                    train=True, key='_' + str(model_idx))
    if config.test:
        eval_performance(config)




if __name__ == '__main__':
    config = reacherconfigs(index=1)
    main(config)
 
