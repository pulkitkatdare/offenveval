import numpy as np
import torch
from cartpole.environments.cartpole import CartPoleEnv
from cartpole.baselines.model import DynamicModel
from cartpole.pytorchrl.rl.models.cem.actor_cartpole import ActorNetwork
import matplotlib.pyplot as plt
import pickle
from cartpole.common.collect_data import load_model, collect_data, save_model
import statistics 
ENVIRONMENT = 'Cartpole-v0'
BATCH_SIZE = 64
GAMMA = 0.99

def get_action(state_tuple, actor):
    state = state_tuple 
    input_state = np.reshape(state, (1, actor.state_dim))
    input_state = torch.from_numpy(input_state)
    dtype = torch.FloatTensor
    input_state = Variable(input_state.type(dtype), requires_grad=False)
    a = actor(input_state)
    a = a.data.cpu().numpy()
    action = a[0][0]
    return action 

def train_mle(gravity, episodes):
    deployment_env = CartPoleEnv(gravity=gravity)
    DP, _ = collect_data(deployment_env, deployment_env, episodes=episodes, buffer_size=100000)
    test_data, _ = collect_data(deployment_env, deployment_env, episodes=10, buffer_size=2000)
    state_dim = 4
    action_dim = 1#env.action_space.n
    model = DynamicModel(state_dim=4, action_dim=1, learning_rate=0.00001, reg=0.01, seed=1234)
    for epoch in range(10):
        for i in range(int(DP.count/64)):
            s_pbatch, a_pbatch, next_state_pbatch, reward_p, terminal_p = DP.sample_batch(batch_size=BATCH_SIZE)

            input = np.concatenate((s_pbatch, a_pbatch), axis=1)
            input = torch.from_numpy(input)
            input = input.type(torch.FloatTensor)
            output = torch.from_numpy(next_state_pbatch)
            output = output.type(torch.FloatTensor)

            loss = model.train_step(input=input, output=output)
            if i % 500 == 0:
                y_hat = model.forward(input)
                loss = model.loss_fn(y_hat, output)
                print("Training Iterations", i, loss.data.numpy())
                with open('./cartpole/baselines/' + 'model_' + str(int(gravity)) + '.ptr', 'wb') as output:
                    torch.save(model.state_dict(), output) 

        s_pbatch, a_pbatch, next_state_pbatch, reward_p, terminal_p = test_data.sample_batch(batch_size=BATCH_SIZE)
        input = np.concatenate((s_pbatch, a_pbatch), axis=1)
        input = torch.from_numpy(input)
        input = input.type(torch.FloatTensor)
        output = torch.from_numpy(next_state_pbatch)
        output = output.type(torch.FloatTensor)

        y_hat = model.forward(input)
        loss = model.loss_fn(y_hat, output)
        print("Epoch Loss:", loss.data.numpy())