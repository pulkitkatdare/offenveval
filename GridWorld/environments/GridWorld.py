import numpy as np
import gym
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from GridWorld.common.replaybuffer import ReplayBuffer
import _pickle as cPickle
from GridWorld.model.betaforRL import BetaEstimator
import torch


class GridWorld(object):
    def __init__(self, size = [5, 5], action_probability = 0.9):
        rows, columns = size
        self.rows = rows
        self.columns = columns
        self.action_probability = action_probability
        self.state = self.reset()
        self.action_space = [0, 1, 2, 3]
        self.goal = [self.rows-1, self.columns-1]
        self.probability = [(1-self.action_probability)/3, (1-self.action_probability)/3, (1-self.action_probability)/3, (1-self.action_probability)/3]
        
        
    def reset(self, seed = None):
        np.random.seed(seed)
        row = 0#np.random.randint(0, self.rows)
        col = 0#np.random.randint(0, self.columns)
        self.state = [row, col]
        self.data = {'rows': [], 'cols': []}
        self.data['rows'].append(self.state[0])
        self.data['cols'].append(self.state[1])
        return self.state
    
    def step(self, action):
        row, col = self.state
        p = self.probability.copy()
        p[action] = self.action_probability
        
        info = {'p': p}
        info['next_state'] = []
        info['next_state_index'] = []
        
        next_row = max(row-1, 0)
        next_col = col
        info['next_state'].append([next_row, next_col])
        
        next_col = min(col+1, self.columns-1)
        next_row = row
        info['next_state'].append([next_row, next_col])
        
        next_row = min(row+1, self.rows-1)
        next_col = col
        info['next_state'].append([next_row, next_col])
        
        next_col = max(col-1, 0)
        next_row = row
        info['next_state'].append([next_row, next_col])
            
        next_state_index = np.random.choice(self.action_space, p=info['p'])
        self.state = info['next_state'][next_state_index]
        info['next_state_index'].append(next_state_index)
        
        if (self.state[0] == self.goal[0]) and (self.state[1] == self.goal[1]):
            done = True
            reward = 0.0
        else:
            done = False
            reward = -1.0
        self.data['rows'].append(self.state[0])
        self.data['cols'].append(self.state[1])
        return self.state, reward, done, info
    
    def render(self):
        #plt.rcParams['axes.facecolor'] = '#bfd3e6'
        fig, ax = plt.subplots()
        ax.set_xlim([0, self.columns])
        ax.set_ylim([0, self.rows])
        ax.set_xticks(np.arange(0,self.columns))
        ax.set_yticks(np.arange(0, self.rows))
        #ax.axes.xaxis.set_ticklabels([])
        #ax.axes.yaxis.set_ticklabels([])
        plt.grid(True)
        for index, (row, col) in enumerate(zip(self.data['rows'][:-1], self.data['cols'][:-1])):
            next_col = self.data['cols'][index+1] - col
            next_row = self.data['rows'][index+1] - row
            ax.plot([col + 0.5, next_col+col+0.5], [row+0.5, next_row+row+0.5], '--', linewidth=2, color='#636363')
            ax.arrow(col + 0.5, row + 0.5, next_col/3, next_row/3,  length_includes_head=True, head_width=0.2, head_length=0.1)

        rect = patches.Rectangle((self.goal[1], self.goal[0]), 1, 1, linewidth=1, edgecolor='#000000', facecolor='#000000')
        ax.add_patch(rect)
        plt.show()

def collect_data(episodes, p_parameters, q_parameters, size):
    D_P = ReplayBuffer(buffer_size=10000000)
    env = GridWorld(size=(size[0], size[1]), action_probability=p_parameters)
    
    steps = 0
    
    for index in range(episodes):
        print ("Episodes", index)
        done = False
        state = env.reset()
        for i in range(1000):
            action = np.random.choice(4)
            steps += 1
            next_state, reward, done, info = env.step(action)
            D_P.add(state, [action], next_state, info['next_state_index'])
            state = next_state
            if done:
                break
            
    D_Q = ReplayBuffer(buffer_size=1000000)
    env = GridWorld(size=(size[0], size[1]), action_probability=q_parameters)
    steps = 0
    for index in range(episodes):
        done = False
        print("Episodes", index)
        state = env.reset()
        for i in range(1000):
            action = np.random.choice(4)
            steps += 1
            next_state, reward, done, info = env.step(action)
            D_Q.add(state, [action], next_state, info['next_state_index'])
            state = next_state
            if done:
                break
            
    return D_P, D_Q

def save_model(D_P, D_Q, environment='GridWorld10x10'):
    save_folder = 'GridWorld/environmentdata/' + environment + '/'
    cPickle.dump(D_P, open(save_folder + 'DP.pkl', 'wb'))
    cPickle.dump(D_Q, open(save_folder + 'DQ.pkl', 'wb'))
    
'''
D_P, D_Q = collect_data(episodes=1000, p_parameters=0.9, q_parameters=0.7)
save_model(D_P, D_Q)

Batches = [1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000]
Data_Batches = []
for batch_size in Batches:
    print (batch_size)
    Net = BetaEstimator(environment='GridWorld10x10', batch_size=batch_size, train=False)
    Beta_oracle = []
    Beta_oracle.append(np.array([1.285, 0.333, 0.333, 0.333]))
    Beta_oracle.append(np.array([0.333, 1.285, 0.333, 0.333]))
    Beta_oracle.append(np.array([0.333, 0.333, 1.285, 0.333]))
    Beta_oracle.append(np.array([0.333, 0.333, 0.333, 1.285]))
    
    
    env = GridWorld(size=(10, 10), action_probability=0.7)
    state = [np.random.randint(env.rows), np.random.randint(env.columns)]
    env.state = state
    state = np.array(state)
    state = np.expand_dims(state, axis=0)
    sum = 0
    total = 0
    state = np.array(state)
    for row in range(env.rows):
        for col in range(env.columns):
            state = [row, col]
            state = np.array(state)
            state = np.expand_dims(state, axis=0)
            for index, action in enumerate(range(4)):
                total += 1
                x = np.concatenate((state, [[action]]), axis=1)
                x = torch.from_numpy(x)
                x = x.type(torch.FloatTensor)
                output1 = (Net.predict(x))
                error = np.abs(output1.data.numpy() - Beta_oracle[index])**2
                sum += np.sqrt(np.sum(error))
    Data_Batches.append(sum/total)
    print (sum/total)

colors = ['#00441b', '#006d2c', '#238b45', '#41ae76', '#66c2a4', '#99d8c9', '#ccece6', '#e5f5f9', '#f7fcfd']
X = np.arange(len(Batches))
plt.plot(X, Data_Batches, marker='o', markerfacecolor='#238b45', markersize=12, color='#66c2a4', linewidth=4)
plt.grid(True)
plt.xticks(X, Batches)
plt.xlabel('Sample Size', fontsize=16)
plt.ylabel('L2 Error', fontsize=16)
plt.show()
'''
