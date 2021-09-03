from Cartpole.model.BetaNet import BetaNetwork
import torch 
import numpy as np
import _pickle as cPickle
from Cartpole.common.replaybuffer import ReplayBuffer

SAMPLE_SIZE = 64
CUDA = torch.cuda.is_available()
def BetaEstimator(environment, deployment_parameter, batch_size, train, key):
    save_folder = './Cartpole/environmentdata/' + environment

    Net_sas = BetaNetwork(state_dim=9, action_bound = 1000, learning_rate=5e-5, tau=1/batch_size, seed=1000, action_dim=1)#BetaNet(n_gaussians = batch_size)
    Net_sa = BetaNetwork(state_dim=5, action_bound = 1000, learning_rate=5e-5, tau=1/batch_size, seed=1000, action_dim=1)#BetaNet(n_gaussians = batch_size)
    
    if train:
        D_P = cPickle.load(open(save_folder + '/DP.pkl', 'rb'))
        D_Q = cPickle.load(open(save_folder + '/DQ.pkl', 'rb'))

        s_pbatch, a_pbatch, next_state_pbatch, reward_p, terminal_p = D_P.sample_batch(batch_size=batch_size)
        s_qbatch, a_qbatch, next_state_qbatch, reward_q, terminal_q = D_Q.sample_batch(batch_size=batch_size)

        p_state_action_state = np.concatenate((s_pbatch, a_pbatch, next_state_pbatch),
                                              axis=1)  # , next_state_pbatch), axis=1)#np.expand_dims(a_pbatch, axis=1)
        q_state_action_state = np.concatenate((s_qbatch, a_qbatch, next_state_qbatch),
                                              axis=1)  # , next_state_qbatch), axis=1)#np.expand_dims(a_qbatch, axis=1)

        p_state_action = np.concatenate((s_pbatch, a_pbatch), axis=1)
        q_state_action = np.concatenate((s_qbatch, a_qbatch), axis=1)

        batch_size, dimension_sas = np.shape(p_state_action_state)
        _, dimension_sa = np.shape(p_state_action);

        if CUDA: 
            Net_sas = Net_sas.cuda()
            Net_sa = Net_sa.cuda()
    
        for i in range(int(1.0*batch_size)):
            
            idx = np.random.randint(batch_size, size=SAMPLE_SIZE)

            states_p = p_state_action_state[idx, :]
            states_q = q_state_action_state[idx, :]

            states_p = torch.from_numpy(states_p)
            states_q = torch.from_numpy(states_q)
            
            states_p = states_p.type(torch.FloatTensor)
            states_q = states_q.type(torch.FloatTensor)

            if CUDA: 
                states_p = states_p.cuda()
                states_q = states_q.cuda()

            output_sas = Net_sas.train_step(states_p=states_p, states_q=states_q)

            states_p = p_state_action[idx, :]
            states_q = q_state_action[idx, :]

            states_p = torch.from_numpy(states_p)
            states_q = torch.from_numpy(states_q)

            states_p = states_p.type(torch.FloatTensor)
            states_q = states_q.type(torch.FloatTensor)

            if CUDA: 
                states_p = states_p.cuda()
                states_q = states_q.cuda()

            output_sa = Net_sa.train_step(states_p=states_p, states_q=states_q)
            
            if i % 100 == 0:
                print("Training Iterations", i, output_sas.cpu().data.numpy(), output_sa.cpu().data.numpy())
    
        
        with open(save_folder + '/net_sas_' + str(batch_size) + '_' + str(int(10*deployment_parameter))+ '_' + key + '.ptr', 'wb') as output:
            torch.save(Net_sas.cpu().state_dict(), output)
        with open(save_folder + '/net_sa' + str(batch_size) +'_' + str(int(10*deployment_parameter))+ '_' + key + '.ptr', 'wb') as output:
            torch.save(Net_sa.cpu().state_dict(), output)

    else:
        Net_sas.load_state_dict(torch.load(save_folder + '/net_sas_' + str(batch_size) + '_' + str(int(10*deployment_parameter)) + '_' + key + '.ptr'));
        Net_sa.load_state_dict(torch.load(save_folder + '/net_sa' + str(batch_size) + '_' + str(int(10*deployment_parameter))  + '_' + key + '.ptr'))
    return Net_sas, Net_sa








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

