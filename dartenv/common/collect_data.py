from dartenv.common.replaybuffer import ReplayBuffer
import numpy as np
import _pickle as cPickle
import math 



def collect_data(simulation_env, deployment_env, episodes=1000, buffer_size=1000000):
    D_P = ReplayBuffer(buffer_size=buffer_size)
    steps = 0
    for index in range(episodes):
        print ("Episodes", index)
        action = np.random.uniform(-math.pi/3, math.pi/3)
        next_state = deployment_env.step(action)
        D_P.add(action, next_state)
            
    D_Q = ReplayBuffer(buffer_size=buffer_size)
    for index in range(episodes):
        print ("Episodes", index)
        action = np.random.uniform(-math.pi/3, math.pi/3)
        next_state = simulation_env.step(action)
        D_Q.add(action, next_state)

    return D_P, D_Q


def save_model(D_P, D_Q, environment='DartEnv'):
    save_folder = 'environmentdata/' + environment + '/'
    cPickle.dump(D_P, open(save_folder + 'DP.pkl', 'wb'))
    cPickle.dump(D_Q, open(save_folder + 'DQ.pkl', 'wb'))

def load_model(environment='DartEnv'):
    save_folder = 'environmentdata/' + environment + '/'
    print (save_folder)
    D_P = cPickle.load(open(save_folder + 'DP.pkl', 'rb'))
    D_Q = cPickle.load(open(save_folder + 'DQ.pkl', 'rb'))
    
    return D_P, D_Q

    
    
    
