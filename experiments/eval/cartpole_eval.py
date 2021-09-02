#!/usr/bin/env python3
from Cartpole.model.BetaNet import BetaNetwork
import torch
import statistics
from Cartpole.environments.cartpole import CartPoleEnv
import numpy as np
from Cartpole.Baselines.model import DynamicModel
from Cartpole.model.betaforRL import BetaEstimator


from pytorchrl.rl.models.cem.actor_cartpole import ActorNetwork
from torch.autograd import Variable
import math
import pickle 
import tikzplotlib
import matplotlib.pyplot as plt

CUDA = torch.cuda.is_available()
GAMMA = 0.99
ENVIRONMENT = 'Cartpole-v0'
EPISODES = 100

def get_mean(reward_per_timestep, product):
    out_mean = 0.0
    episodes, timesteps = np.shape(reward_per_timestep)
    for step in range(timesteps):
        sum_product = np.sum(product[:, step])
        out_mean += np.sum(np.multiply(reward_per_timestep[:, step], product[:, step]/sum_product))
    return out_mean

def main(simulation_parameter, deployment_parameter, episodes, timesteps=100, epsilon = 0.0, batch_size=20000):
    save_folder = 'Cartpole/environmentdata/' + ENVIRONMENT + '/'
    print (save_folder + 'net_sa' + str(100000) + '_' + str(deployment_parameter) + '.ptr')
    torch.manual_seed(1000)


    actor = ActorNetwork(4, 2)
    actor.load_state_dict(torch.load('pytorchrl/saver/cem_cartpole.ptr'))



    env = CartPoleEnv(gravity=deployment_parameter)
    rewards_d = []
    avg_steps = 0

    for episode in range(episodes):
        env.seed(episode)
        np.random.seed(episode)
        state = env.reset()
        done = False
        r = 0
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

                inter_reward = 0

                next_state, reward, done, info = env.step(a)
                #reward = (1 - (state[0]**2)/11.52 - (state[2]**2)/288)
                inter_reward += reward

                state = next_state

                if done:
                    avg_steps += 1#steps
            else:
                inter_reward = 0.0

            r += (GAMMA**steps)*inter_reward
        rewards_d.append(r)

    print ("Mean Rewards", statistics.mean(rewards_d))


    env = CartPoleEnv(gravity=simulation_parameter)
    rewards_s = []
    avg_steps = 0
    approx_model_rewards = []
    for model_idx in range(10):
        Beta1, Beta2 = BetaEstimator(environment=ENVIRONMENT, deployment_parameter=deployment_parameter, batch_size=80000, train=False, key=str(model_idx))
        rewards_s = np.zeros((episodes, timesteps))
        product = np.zeros((episodes, timesteps))
        for episode in range(episodes):
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
                    inter_reward = 0

                    next_state, reward, done, info = env.step(a)
                    #reward = (1 - (state[0] ** 2) / 11.52 - (state[2] ** 2) / 288)
                    inter_reward += reward

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
                    r += (GAMMA**steps)*inter_reward

                    rewards_s[episode, steps] = (GAMMA**steps)*inter_reward
                    product[episode, steps] = prod

                    prod = prod*(beta1[0]/beta2[0])

                    if done:
                        avg_steps += 1#steps
                else:
                    r += (GAMMA**steps)*0.0

                    rewards_s[episode, steps] = (GAMMA**steps)*0.0
                    product[episode, steps] = prod
        approx_model_rewards.append(get_mean(rewards_s, product))

    print("Mean Rewards:", statistics.mean(approx_model_rewards))

    model = DynamicModel(state_dim=4, action_dim=1, learning_rate=0.0001, reg=0.01, seed=1234)
    model.load_state_dict(torch.load('Cartpole/Baselines/' + 'model_' + str(int(deployment_parameter)) + '.ptr'))
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
          
            #input = np.concatenate((state_predicted, [action]))
            #input = torch.from_numpy(input)
            #input = input.type(torch.FloatTensor)
            #output = model(input)
            #x = state_predicted[0]
            #theta = state_predicted[1]
            #state_predicted = output.data.numpy()
            #theta_threshold = 12 * 2 * math.pi / 360
            #done = bool(
            #    x < -2.4
            #    or x > 2.4
            #    or theta < -theta_threshold
            #    or theta > theta_threshold
            #)

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

            #reward = (1 - (x ** 2) / 11.52 - (theta ** 2) / 288)
            r += (GAMMA**steps)*reward
        rewards_b.append(r)
    print("Mean Rewards:", statistics.mean(rewards_b))



    return approx_model_rewards, rewards_d, rewards_b

if __name__ == '__main__':
    for G in [5.0, 7.5, 10.0, 12.5, 15.0]:
        OEE = {}
        OEE['mean'] = []
        OEE['std'] = []
        OEE['parameters'] = []

        Oracle = {}


        Oracle['mean'] = []
        Oracle['std'] = []
        Oracle['parameters'] = []


        Baseline = {}

        Baseline['mean'] = []
        Baseline['std'] = []
        Baseline['parameters'] = []
        batches = [80000, 80000]#, 100000, 100000, 2000
        for i in range(1):
            for epsilon in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
                s, d, b = main(simulation_parameter=10.0, deployment_parameter=G, episodes=30, timesteps=100, epsilon=epsilon, batch_size=80000)
                OEE['mean'].append((1 - GAMMA) * statistics.mean(s))
                OEE['std'].append((1 - GAMMA) * statistics.stdev(s)/np.sqrt(100))
                OEE['parameters'].append(epsilon)

                Oracle['mean'].append((1 - GAMMA) *statistics.mean(d))
                Oracle['std'].append((1 - GAMMA) * statistics.stdev(d)/np.sqrt(100))
                Oracle['parameters'].append(epsilon)

                Baseline['mean'].append((1 - GAMMA) *statistics.mean(b))
                Baseline['std'].append((1 - GAMMA) * statistics.stdev(b)/np.sqrt(100))
                Baseline['parameters'].append(epsilon)





        fig, ax = plt.subplots()
        
        
        
        colors = ['#000000', '#00441b', '#7f0000', '#084081', '#4d004b']



        ax.plot(Oracle['parameters'], Oracle['mean'], color=colors[0],linewidth=2)
        ax.plot(OEE['parameters'], OEE['mean'], color=colors[1],linewidth=2)
        ax.plot(Baseline['parameters'], Baseline['mean'], color=colors[2], linewidth=2)


        data_is = pickle.load(open('./Cartpole/Baselines/IS_baseline.pkl','rb'))
        IS_x = []
        IS_y = []
        for epsilon in [0.2, 0.4, 0.6, 0.8, 1.0]:
            IS_x.append(epsilon)
            IS_y.append((1-GAMMA)*data_is['OPE'][(G,epsilon, 100)])

        ax.plot(IS_x, IS_y, color=colors[3], linewidth=2)

        X = np.asarray(Oracle['parameters'])
        mean_algorithm = np.asarray(OEE['mean'])
        std_algorithm =  np.asarray(OEE['std'])


        mean_oracle = np.asarray(Oracle['mean'])
        std_oracle  = np.asarray(Oracle['std'])

        mean_baseline = np.asarray(Baseline['mean'])
        std_baseline = np.asarray(Baseline['std'])

        ax.fill_between(X, mean_oracle - 1 * std_oracle, mean_oracle + 1 * std_oracle, color=colors[0], alpha=0.1)
        ax.fill_between(X, mean_algorithm - 1 * std_algorithm, mean_algorithm + 1 * std_algorithm, color=colors[1], alpha=0.1)
        ax.fill_between(X, mean_baseline - 1 * std_baseline, mean_baseline + 1 * std_baseline, color=colors[2], alpha=0.1)


        ax.set_xlabel('Deployment Parameters', fontsize=16)
        ax.set_ylabel('Average Returns', fontsize=16)

        plt.legend(['True Value', 'OEE (ours)', 'MLE', 'IS'])

        plt.xticks(X)
        #plt.ylim([0.6, 1.2])

        plt.grid(True)
        plt.savefig('cartpole_' + str(G) + '.png')
        tikzplotlib.save('cartpole_eval_trained' + str(G) + '.tex')
        #plt.show()
plt.close()
