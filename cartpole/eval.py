#!/usr/bin/env python3
from cartpole.model.BetaNet import BetaNetwork
import torch
import statistics
from cartpole.environments.cartpole import CartPoleEnv
import numpy as np
from cartpole.baselines.model import DynamicModel
from cartpole.model.betaforRL import BetaEstimator
from cartpole.baselines.MLE import train_mle
from cartpole.baselines.common.collect_data import collect_data_for_IS, save_model_for_IS, load_model_for_IS

from cartpole.pytorchrl.rl.models.cem.actor_cartpole import ActorNetwork
from torch.autograd import Variable
import math
import pickle 
import tikzplotlib
import matplotlib.pyplot as plt

CUDA = torch.cuda.is_available()
GAMMA = 0.99
ENVIRONMENT = 'Cartpole-v0'
EPISODES = 100

def get_action(state_tuple, actor):
    state, action, next_state, reward, done = state_tuple 
    input_state = np.reshape(state, (1, actor.state_dim))
    input_state = torch.from_numpy(input_state)
    dtype = torch.FloatTensor
    input_state = Variable(input_state.type(dtype), requires_grad=False)
    a = actor(input_state)
    a = a.data.cpu().numpy()
    action = a[0][0]
    return action 


def get_mean(reward_per_timestep, product):
    out_mean = 0.0
    episodes, timesteps = np.shape(reward_per_timestep)
    for step in range(timesteps):
        sum_product = np.sum(product[:, step])
        out_mean += np.sum(np.multiply(reward_per_timestep[:, step], product[:, step]/sum_product))
    return out_mean

def get_true_baseline(episodes, timesteps, epsilon, env_parameters):
    actor = ActorNetwork(4, 2)
    actor.load_state_dict(torch.load('./cartpole/pytorchrl/saver/cem_cartpole.ptr'))

    env = CartPoleEnv(gravity=env_parameters)
    rewards_d = []
    avg_steps = 0

    for episode in range(episodes):
        env.seed(episode)
        np.random.seed(episode)
        state = env.reset()
        done = False
        cum_reward_sum = 0
        steps = 0
        for steps in range(timesteps):
            if not done:
                if np.random.uniform()<epsilon:
                    a = env.action_space.sample()
                else:
                    input_state = np.reshape(state, (1, actor.state_dim))
                    input_state = torch.from_numpy(input_state)
                    dtype = torch.FloatTensor
                    input_state = Variable(input_state.type(dtype), requires_grad=False)
                    a = actor(input_state)
                    a = a.data.cpu().numpy()
                    a = a[0][0]

                next_state, reward, done, info = env.step(a)
                state = next_state
                cum_reward_sum += (GAMMA**steps)*reward
        rewards_d.append(cum_reward_sum)
    print ("True Rewards", statistics.mean(rewards_d))

    return rewards_d


def get_oee_baseline(episodes, timesteps, epsilon, env_parameters, batch_size):
    env = CartPoleEnv(gravity=env_parameters)
    actor = ActorNetwork(4, 2)
    actor.load_state_dict(torch.load('./cartpole/pytorchrl/saver/cem_cartpole.ptr'))
    rewards_s = []
    avg_steps = 0
    approx_model_rewards = []
    for model_idx in range(10):
        print (model_idx)
        Beta1, Beta2 = BetaEstimator(environment=ENVIRONMENT, deployment_parameter=env_parameters, batch_size=batch_size, train=False, key=str(model_idx))
        rewards_s = np.zeros((episodes, timesteps))
        product = np.zeros((episodes, timesteps))
        for episode in range(episodes):
            print (episode)
            env.seed(episode)
            state = env.reset()
            done = False
            r = 0
            steps = 0
            prod = 1
            for steps in range(timesteps):
                if not done:
                    if np.random.uniform()<epsilon:
                        a = env.action_space.sample()
                    else:
                        input_state = np.reshape(state, (1, actor.state_dim))
                        input_state = torch.from_numpy(input_state)
                        dtype = torch.FloatTensor
                        input_state = Variable(input_state.type(dtype), requires_grad=False)
                        a = actor(input_state)
                        a = a.data.cpu().numpy()
                        a = a[0][0]
                    next_state, reward, done, info = env.step(a)
                    reward += reward

                    x = np.concatenate((state, [a], next_state), axis=0)
                    x = torch.from_numpy(x)
                    x = x.type(torch.FloatTensor)

                    y = np.concatenate((state, [a]), axis=0)
                    y = torch.from_numpy(y)
                    y = y.type(torch.FloatTensor)

                    beta1 = (Beta1.predict(x))
                    beta2 = (Beta2.predict(y))

                    beta1 = beta1.data.numpy()
                    beta2 = beta2.data.numpy()

                    state = next_state
                    r += (GAMMA**steps)*reward

                    rewards_s[episode, steps] = (GAMMA**steps)*reward
                    product[episode, steps] = prod
                    prod = prod*(beta1[0]/beta2[0])

        approx_model_rewards.append(get_mean(rewards_s, product))

    print("OEE Rewards:", statistics.mean(approx_model_rewards))

    return approx_model_rewards

def get_mle_baseline(episodes, timesteps, epsilon, env_parameters):
    actor = ActorNetwork(4, 2)
    actor.load_state_dict(torch.load('./cartpole/pytorchrl/saver/cem_cartpole.ptr'))
    env = CartPoleEnv(gravity=env_parameters)
    model = DynamicModel(state_dim=4, action_dim=1, learning_rate=0.0001, reg=0.01, seed=1234)
    model.load_state_dict(torch.load('./cartpole/baselines/' + 'model_' + str(int(env_parameters)) + '.ptr'))
    rewards_b = []
    for episode in range(episodes):
        state = env.reset()
        state_predicted = state
        done = False
        r = 0
        steps = 0
        prod = 1
        for steps in range(timesteps):
            if np.random.uniform()<epsilon:
                action = env.action_space.sample()
            else:
                input_state = np.reshape(state, (1, actor.state_dim))
                input_state = torch.from_numpy(input_state)
                dtype = torch.FloatTensor
                input_state = Variable(input_state.type(dtype), requires_grad=False)
                a = actor(input_state)
                a = a.data.cpu().numpy()
                action = a[0][0]

            input_state = np.reshape(state, (1, 4))

            input = np.concatenate((input_state, [[action]]),axis=1)
            input = torch.from_numpy(input)
            input = input.type(torch.FloatTensor)
            next_state = model(input)
            next_state =  next_state.data.numpy()
            next_state = 1e-1*np.random.randn(1, 4) + next_state

            done = bool(next_state[0, 0] < -4.8
                        or next_state[0, 0] > 4.8
                        or next_state[0, 2] < -0.418
                        or next_state[0, 2] > 0.418
                         )
            if done:
                reward = 0.0
                break
            else:
                reward = 1.0
            r += (GAMMA**steps)*reward
        rewards_b.append(r)
    print("MLE Rewards:", statistics.mean(rewards_b))
    return rewards_b

def get_IS_baseline(episodes, timesteps, epsilon, env_parameters):
    EPSILON_D = 0.3
    trajectory_length = timesteps
    deployment_env = CartPoleEnv(gravity=env_parameters)
    data = collect_data_for_IS(environment = deployment_env, episodes=1000)
    save_model_for_IS(data, environment=ENVIRONMENT, key='Q')

    actor = ActorNetwork(4, 2)
    actor.load_state_dict(torch.load('./cartpole/pytorchrl/saver/cem_cartpole.ptr'))
    data = load_model_for_IS(environment=ENVIRONMENT, key='Q')
    trajectories = len(data.keys())
    sum_total = 0.0
    prod_sum = 0.0
    for traj_idx in range(trajectories):
        print ('index:', traj_idx)
        prod = 1.0 
        sum_traj = 0.0 
        for time_step, state_tuple in enumerate(data[traj_idx]):
            if time_step > trajectory_length:
                break
            state, action, next_state, reward, done = state_tuple
            expert_action = get_action(state_tuple, actor)
            if expert_action == action[0]:
                out_ratio = (1 - epsilon/2)/(1 - EPSILON_D/2)
            else: 
                out_ratio = epsilon/EPSILON_D
            prod = prod*out_ratio
            sum_traj += (GAMMA**(time_step))*reward
        sum_total += prod*sum_traj 
        prod_sum += prod
    sum_total = sum_total/prod_sum
    print("IS Rewards:", sum_total)
    return sum_total

def test(config):
    if config.mle:
        if config.train_mle:
            train_mle(gravity=config.deployment_parameter, episodes=config.episodes)

    OEE = {}
    OEE['mean'] = []
    OEE['std'] = []

    Oracle = {}
    Oracle['mean'] = []
    Oracle['std'] = []

    MLE = {}
    MLE['mean'] = []
    MLE['std'] = []

    ISE = {}
    ISE['mean'] = []
    ISE['parameters'] = []



    for epsilon in config.epsilons:
        if config.true:
            rewards_t = get_true_baseline(episodes=config.eval_episodes, timesteps=config.timesteps, epsilon=epsilon, env_parameters=config.deployment_parameter)
            Oracle['mean'].append((1 - GAMMA) *statistics.mean(rewards_t))
            Oracle['std'].append((1 - GAMMA) * statistics.stdev(rewards_t)/np.sqrt(config.eval_episodes))

        if config.oee: 
            reward_per_model = get_oee_baseline(episodes=config.eval_episodes, timesteps=config.timesteps, epsilon=epsilon, env_parameters=config.deployment_parameter, batch_size=config.eval_batch_size)
            OEE['mean'].append((1 - GAMMA) * statistics.mean(reward_per_model))
            OEE['std'].append((1 - GAMMA) * statistics.stdev(reward_per_model)/np.sqrt(config.eval_episodes))

        if config.mle:
            reward_mle = get_mle_baseline(episodes=config.eval_episodes, timesteps=config.timesteps, epsilon=epsilon, env_parameters=config.deployment_parameter)
            MLE['mean'].append((1 - GAMMA) *statistics.mean(reward_mle))
            MLE['std'].append((1 - GAMMA) * statistics.stdev(reward_mle)/np.sqrt(config.eval_episodes))

        if config.IS:
            if epsilon != 0 :
                reward_IS = get_IS_baseline(episodes=config.eval_episodes, timesteps=config.timesteps, epsilon=epsilon, env_parameters=config.deployment_parameter) 
                ISE['mean'].append((1-GAMMA)*reward_IS)
                ISE['parameters'].append(epsilon)

    fig, ax = plt.subplots()
    colors = ['#000000', '#00441b', '#7f0000', '#084081', '#4d004b']
    legends = []
    if config.true:
        legends.append('True Value')
        ax.plot(config.epsilons, Oracle['mean'], color=colors[0],linewidth=2)
        X = np.asarray(config.epsilons)
        mean_algorithm = np.asarray(Oracle['mean'])
        std_algorithm =  np.asarray(Oracle['std'])
        ax.fill_between(X, mean_algorithm - 1 * std_algorithm, mean_algorithm + 1 * std_algorithm, color=colors[0], alpha=0.1)

    if config.oee:
        legends.append('OEE (ours)')
        ax.plot(config.epsilons, OEE['mean'], color=colors[1],linewidth=2)
        X = np.asarray(config.epsilons)
        mean_algorithm = np.asarray(OEE['mean'])
        std_algorithm =  np.asarray(OEE['std'])
        ax.fill_between(X, mean_algorithm - 1 * std_algorithm, mean_algorithm + 1 * std_algorithm, color=colors[1], alpha=0.1)
    
    if config.mle:
        legends.append('MLE')
        ax.plot(config.epsilons, MLE['mean'], color=colors[2], linewidth=2)
        X = np.asarray(config.epsilons)
        mean_algorithm = np.asarray(MLE['mean'])
        std_algorithm =  np.asarray(MLE['std'])
        ax.fill_between(X, mean_algorithm - 1 * std_algorithm, mean_algorithm + 1 * std_algorithm, color=colors[2], alpha=0.1)

    if config.IS:
        legends.append('IS')
        ax.plot(ISE['parameters'], ISE['mean'], color=colors[3], linewidth=2)

    ax.set_xlabel('Deployment Parameters', fontsize=16)
    ax.set_ylabel('Average Returns', fontsize=16)

    plt.legend(legends)
    plt.xticks(np.asarray(config.epsilons))
    plt.grid(True)
    if config.savepng:
        plt.savefig('./cartpole/assets/cartpole_' + str(config.deployment_parameter) + '.png')
    if config.savetex:
        tikzplotlib.save('./cartpole/assets/cartpole_eval_trained' + str(config.deployment_parameter) + '.tex')
    plt.close()
