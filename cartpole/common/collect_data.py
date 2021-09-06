from cartpole.common.replaybuffer import ReplayBuffer
import numpy as np
import _pickle as cPickle
from cartpole.pytorchrl.rl.models.cem.actor_cartpole import ActorNetwork
import torch
from torch.autograd import Variable


def collect_data(simulation_env, deployment_env, episodes=1000, buffer_size=1000000):
    D_P = ReplayBuffer(buffer_size=buffer_size)
    steps = 0
    actor = ActorNetwork(4, 2)
    actor.load_state_dict(torch.load('./cartpole/pytorchrl/saver/cem_cartpole.ptr'))
    for index in range(episodes):
        print ("Episodes", index)
        done = False
        state = deployment_env.reset()
        steps = 0
        while (steps < 200):
            if np.random.uniform() < 0.3:
                action = np.random.choice([0, 1], p = [0.5, 0.5])
            else:
                input_state = np.reshape(state, (1, actor.state_dim))
                input_state = torch.from_numpy(input_state)
                dtype = torch.FloatTensor
                input_state = Variable(input_state.type(dtype), requires_grad=False)
                a = actor(input_state)
                a = a.data.cpu().numpy()
                action = a[0][0]
            steps += 1
            reward = 0
            next_state, r, done, info = deployment_env.step(action)
            if done:
                break

            reward += r
            D_P.add(state, [action], next_state, reward, done)
            state = next_state
            
    D_Q = ReplayBuffer(buffer_size=buffer_size)
    steps = 0
    for index in range(episodes):
        print ("Episodes", index)
        done = False
        state = simulation_env.reset()
        steps = 0
        while (steps<200):
            if np.random.uniform() < 0.3:
                action = np.random.choice([0, 1], p=[0.5, 0.5])
            else:
                input_state = np.reshape(state, (1, actor.state_dim))
                input_state = torch.from_numpy(input_state)
                dtype = torch.FloatTensor
                input_state = Variable(input_state.type(dtype), requires_grad=False)
                a = actor(input_state)
                a = a.data.cpu().numpy()
                action = a[0][0]
            steps += 1
            reward = 0
            next_state, r, done, info = simulation_env.step(action)
            if done:
                break
            reward += r
            D_Q.add(state, [action], next_state, reward, done)
            state = next_state
    return D_P, D_Q

def save_model(D_P, D_Q, environment='GridWorld10x10'):
    save_folder = 'cartpole/environmentdata/' + environment + '/'
    cPickle.dump(D_P, open(save_folder + 'DP.pkl', 'wb'))
    cPickle.dump(D_Q, open(save_folder + 'DQ.pkl', 'wb'))

def load_model(environment='GridWorld10x10'):
    save_folder = 'cartpole/environmentdata/' + environment + '/'
    print (save_folder)
    D_P = cPickle.load(open(save_folder + 'DP.pkl', 'rb'))
    D_Q = cPickle.load(open(save_folder + 'DQ.pkl', 'rb'))
    
    return D_P, D_Q

    
    
    
