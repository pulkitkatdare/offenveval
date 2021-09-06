from Cartpole.common.replaybuffer import ReplayBuffer
import numpy as np
import _pickle as cPickle
from pytorchrl.rl.models.cem.actor_cartpole import ActorNetwork
import torch
from torch.autograd import Variable


def collect_data(environment, episodes=1000, buffer_size=1000000, epsilon=0.3):
    D_P = {}
    steps = 0
    actor = ActorNetwork(4, 2)
    actor.load_state_dict(torch.load('pytorchrl/saver/cem_cartpole.ptr'))
    for index in range(episodes):
        print ("Episodes", index)
        done = False
        D_P[index] = []
        state = environment.reset()
        steps = 0
        while (steps < 200):
            if np.random.uniform() < epsilon:
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
            next_state, r, done, info = environment.step(action)
            if done:
                break

            reward += r
            D_P[index].append((state, [action], next_state, reward, done))
            state = next_state
            
    return D_P

def save_model(data_buffer, environment='GridWorld10x10', key='p'):
    save_file = './Cartpole/Baselines/environmentdata/' + environment + '/D_' + key + '.pkl'
    cPickle.dump(data_buffer, open(save_file, 'wb'))

def load_model(environment='GridWorld10x10', key = ''):
    save_file = 'Cartpole/Baselines/environmentdata/' + environment + '/D_' + key + '.pkl'
    print (save_file)
    data_buffer = cPickle.load(open(save_file, 'rb'))
    
    return data_buffer

    
    
    
