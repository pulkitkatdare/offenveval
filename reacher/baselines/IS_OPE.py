import numpy as np 
import torch.nn as nn 
from Cartpole.Baselines.common.collect_data import collect_data, save_model, load_model
import torch
from Cartpole.environments.cartpole import CartPoleEnv
from pytorchrl.rl.models.cem.actor_cartpole import ActorNetwork
from torch.autograd import Variable
import pickle 

EPSILON_D = 0.3
CUDA = torch.cuda.is_available()
GAMMA = 0.99
ENVIRONMENT = 'Cartpole-v0'

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


def main(data=True, eval = False, calculate_true_values = True, EPSILON=0.2, trajectories=1000, trajectory_length=100, gravity=10.0):
    sum_total = None 
    true_value = None 
    if data is False:
        deployment_env = CartPoleEnv(gravity=gravity)
        Data = collect_data(environment = deployment_env, episodes=1000)
        save_model(Data, environment=ENVIRONMENT, key='Q')

    if eval is True:
        print ("True")
        actor = ActorNetwork(4, 2)
        actor.load_state_dict(torch.load('pytorchrl/saver/cem_cartpole.ptr'))
        data = load_model(environment=ENVIRONMENT, key='Q')
        print (len(data.keys()))
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
                    out_ratio = (1 - EPSILON/2)/(1 - EPSILON_D/2)
                else: 
                    out_ratio = EPSILON/EPSILON_D
                prod = prod*out_ratio
                sum_traj += (GAMMA**(time_step))*reward
            sum_total += prod*sum_traj 
            prod_sum += prod
        sum_total = sum_total/prod_sum
    if calculate_true_values is True:
        deployment_env = CartPoleEnv(gravity=gravity)
        data = collect_data(environment = deployment_env, episodes=trajectories, epsilon=EPSILON)
        true_value = 0.0 
        for traj_idx in range(trajectories):
            print ('index:', traj_idx)
            sum_traj = 0.0 
            for time_step, state_tuple in enumerate(data[traj_idx]):
                if time_step > trajectory_length:
                    break
                state, action, next_state, reward, done = state_tuple
                sum_traj += (GAMMA**(time_step))*reward
            true_value += sum_traj 
        true_value = true_value/trajectories

    return sum_total, true_value

if __name__ == '__main__':
    data = {}
    data['OPE'] = {}
    data['Oracle'] = {}
    for gravity in [5.0, 7.5, 10.0, 12.5, 15.0]:
        for EPSILON in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
                for length in range(4):
                    traj_length = (length+1)*50
                    sum_trajectories, true_value = main(data=False, eval = True, calculate_true_values = True, EPSILON=EPSILON, trajectories = 1000, trajectory_length=traj_length, gravity=gravity)
                    data['OPE'][(gravity, EPSILON, traj_length)] = sum_trajectories 
                    data['Oracle'][(gravity, EPSILON, traj_length)] = true_value

    with open('./Cartpole/Baselines/IS_baseline.pkl', 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

