from gridworld.common.replaybuffer import ReplayBuffer
import numpy as np
import _pickle as cPickle



def collect_data(simulation_env, deployment_env, episodes=1000, buffer_size=1000000):
    D_P = ReplayBuffer(buffer_size=buffer_size)
    steps = 0
    for index in range(episodes):
        print ("Episodes", index)
        done = False
        state = deployment_env.reset()
        steps = 0
        while (steps < 200):
            action = deployment_env.action_space.sample()
            steps += 1
            reward = 0

            for _ in range(5):
                next_state, r, done, info = deployment_env.step(action)
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
            action = simulation_env.action_space.sample()
            steps += 1
            reward = 0

            for _ in range(5):
                next_state, r, done, info = deployment_env.step(action)
                reward += r

            D_Q.add(state, [action], next_state, reward, done)
            state = next_state

    return D_P, D_Q

def save_model(D_P, D_Q, environment='GridWorld10x10'):
    save_folder = 'environmentdata/' + environment + '/'
    cPickle.dump(D_P, open(save_folder + 'DP.pkl', 'wb'))
    cPickle.dump(D_Q, open(save_folder + 'DQ.pkl', 'wb'))

def load_model(environment='GridWorld10x10'):
    save_folder = 'environmentdata/' + environment + '/'
    print (save_folder)
    D_P = cPickle.load(open(save_folder + 'DP.pkl', 'rb'))
    D_Q = cPickle.load(open(save_folder + 'DQ.pkl', 'rb'))
    
    return D_P, D_Q

    
    
    
