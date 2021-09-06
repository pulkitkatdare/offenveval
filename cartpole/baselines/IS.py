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


def get_IS(data=False, eval = False, calculate_true_values = True, EPSILON=0.2, trajectories=1000, trajectory_length=100, gravity=10.0):
    deployment_env = CartPoleEnv(gravity=environment_parameter)
        Data = collect_data(environment = deployment_env, episodes=1000)
        save_model(Data, environment=ENVIRONMENT, key='Q')

    if eval is True:
        print ("True")
        actor = ActorNetwork(4, 2)
        actor.load_state_dict(torch.load('./cartpole/pytorchrl/saver/cem_cartpole.ptr'))
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


