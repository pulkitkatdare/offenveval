from Cartpole.model.BetaNet import BetaNetwork
import torch 
import numpy as np
import _pickle as cPickle
from MountainCar.common.replaybuffer import ReplayBuffer

SAMPLE_SIZE = 128









'''
D_P = pickle.load(open("pkl/D_P.pkl", "rb"))
D_Q = pickle.load(open("pkl/D_Q.pkl", "rb"))

s_pbatch, a_pbatch, next_state_pbatch = D_P.sample_batch(batch_size = 1000)
s_qbatch, a_qbatch, next_state_qbatch = D_Q.sample_batch(batch_size = 1000)

p_state_action_state = np.concatenate((s_pbatch, np.expand_dims(a_pbatch, axis = 1), next_state_pbatch), axis = 1)
q_state_action_state = np.concatenate((s_qbatch, np.expand_dims(a_qbatch, axis = 1), next_state_qbatch), axis = 1)

p_state_action_state = np.expand_dims(p_state_action_state, axis = 2)
q_state_action_state = np.expand_dims(q_state_action_state, axis = 2)

p_state_action_state = torch.tensor(p_state_action_state)
q_state_action_state = torch.tensor(q_state_action_state)


Net = BetaNet(n_gaussians = N)
Net.train(p_state_action_state, q_state_action_state, epochs = 1000)

with open('pkl/net.pkl', 'wb') as output:
    pickle.dump(Net, output, pickle.HIGHEST_PROTOCOL)
'''

