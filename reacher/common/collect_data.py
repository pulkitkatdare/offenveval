from reacher.common.replaybuffer import ReplayBuffer
import numpy as np
import _pickle as cPickle
import torch
from torch.autograd import Variable


def collect_data(simulation_env, deployment_env, episodes=1000, buffer_size=1000000, epsilon=0.7):
    
    actor_critic, obs_rms = \
                torch.load('reacher/expert/pytorch-a2c-ppo-acktr-gail/trained_models/ppo/ReacherBulletEnv-v0.pt',
                                map_location='cpu')
    D_P = ReplayBuffer(buffer_size=buffer_size)
    for epsilon in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        for index in range(int(episodes/11.0)):
            print ("Episodes", index)
            done = False
            state = deployment_env.reset()
            steps = 0
            recurrent_hidden_states = torch.zeros(1,
                                                  actor_critic.recurrent_hidden_state_size)
            masks = torch.zeros(1, 1)
            while (steps < 150):
                if np.random.uniform() < epsilon:
                    action = [np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0)]
                else: 
                    with torch.no_grad():
                        obs = np.reshape(state, (1, 9))
                        obs = torch.from_numpy(obs).float().to('cpu')
                        value, action, _, recurrent_hidden_states = actor_critic.act(
                            obs, recurrent_hidden_states, masks, deterministic=True)
                    action = action.data.cpu().numpy()[0, :]

                #action = np.random.uniform(-1.0, 1.0, 2)
                #action += 0.0001*np.random.normal(0.0, 1.0, 2)
                next_state, r, done, info = deployment_env.step(action)
                if done:
                    break
                D_P.add(state, action, next_state, r, done)
                state = next_state
                steps += 1
            
    D_Q = ReplayBuffer(buffer_size=buffer_size)

    for epsilon in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        for index in range(int(episodes/11.0)):
            print ("Episodes", index)
            done = False
            state = simulation_env.reset()
            steps = 0
            recurrent_hidden_states = torch.zeros(1,
                                                  actor_critic.recurrent_hidden_state_size)
            masks = torch.zeros(1, 1)
            while (steps<150):
                if np.random.uniform() < epsilon:
                    action = [np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0)]
                else: 
                    with torch.no_grad():
                        obs = np.reshape(state, (1, 9))
                        obs = torch.from_numpy(obs).float().to('cpu')
                        value, action, _, recurrent_hidden_states = actor_critic.act(
                            obs, recurrent_hidden_states, masks, deterministic=True)
                    action = action.data.cpu().numpy()[0, :]
                #action = np.random.uniform(-1.0, 1.0, 2)
                #action += 0.0001*np.random.normal(0.0, 1.0, 2)
                next_state, r, done, info = simulation_env.step(action)
                if done:
                    break
                D_Q.add(state, action, next_state, r, done)
                state = next_state
                steps += 1
    return D_P, D_Q

def save_model(D_P, D_Q, environment='GridWorld10x10'):
    save_folder = 'reacher/environmentdata/' + environment + '/'
    cPickle.dump(D_P, open(save_folder + 'DP.pkl', 'wb'))
    cPickle.dump(D_Q, open(save_folder + 'DQ.pkl', 'wb'))

def load_model(environment='GridWorld10x10'):
    save_folder = 'reacher/environmentdata/' + environment + '/'
    print (save_folder)
    D_P = cPickle.load(open(save_folder + 'DP.pkl', 'rb'))
    D_Q = cPickle.load(open(save_folder + 'DQ.pkl', 'rb'))
    
    return D_P, D_Q

    
    
    
