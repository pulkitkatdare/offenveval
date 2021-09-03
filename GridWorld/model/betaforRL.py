#from GridWorld.model.BetaNet import BetaNetwork
from GridWorld.model.BetaNet import BetaNetwork
import torch 
import numpy as np
import _pickle as cPickle
from GridWorld.common.replaybuffer import ReplayBuffer

SAMPLE_SIZE = 128
def BetaEstimator(environment, batch_size, train, size, key):
    save_folder = 'GridWorld/environmentdata/' + environment
    D_P = cPickle.load(open(save_folder + '/DP.pkl', 'rb'))
    D_Q = cPickle.load(open(save_folder + '/DQ.pkl', 'rb'))

    s_pbatch, a_pbatch, next_state_pbatch, next_state_index_p = D_P.sample_batch(batch_size = batch_size)
    s_qbatch, a_qbatch, next_state_qbatch, next_state_index_q = D_Q.sample_batch(batch_size = batch_size)

    p_state_action_state = np.concatenate((s_pbatch, a_pbatch, next_state_pbatch),
                                          axis=1)  # , next_state_pbatch), axis=1)#np.expand_dims(a_pbatch, axis=1)
    q_state_action_state = np.concatenate((s_qbatch, a_qbatch, next_state_qbatch),
                                          axis=1)  # , next_state_qbatch), axis=1)#np.expand_dims(a_qbatch, axis=1)

    p_state_action = np.concatenate((s_pbatch, a_pbatch), axis=1)
    q_state_action = np.concatenate((s_qbatch, a_qbatch), axis=1)

    batch_size, dimension_sas = np.shape(p_state_action_state)
    _, dimension_sa = np.shape(p_state_action)

    Net_sas = BetaNetwork(state_dim=dimension_sas, action_bound=5, learning_rate=1e-5, tau=.1/batch_size, seed=1234,
                          action_dim=1)
    Net_sa = BetaNetwork(state_dim=dimension_sa, action_bound=5, learning_rate=1e-5, tau=.1/batch_size, seed=1234,
                         action_dim=1)

    if train:

        for i in range(int(batch_size)):

            idx = np.random.randint(batch_size, size=SAMPLE_SIZE)

            states_p = p_state_action_state[idx, :]
            states_q = q_state_action_state[idx, :]

            states_p = torch.from_numpy(states_p)
            states_q = torch.from_numpy(states_q)

            states_p = states_p.type(torch.FloatTensor)
            states_q = states_q.type(torch.FloatTensor)

            output_sas = Net_sas.train_step(states_p=states_p, states_q=states_q)

            states_p = p_state_action[idx, :]
            states_q = q_state_action[idx, :]

            states_p = torch.from_numpy(states_p)
            states_q = torch.from_numpy(states_q)

            states_p = states_p.type(torch.FloatTensor)
            states_q = states_q.type(torch.FloatTensor)

            output_sa = Net_sa.train_step(states_p=states_p, states_q=states_q)

            if i % 100 == 0:
                print("Training Iterations", i, output_sas.data.numpy(), output_sa.data.numpy())

        with open(save_folder + '/net_sas' + key + '.ptr',
                  'wb') as output:
            torch.save(Net_sas.state_dict(), output)
        with open(save_folder + '/net_sa' + key + '.ptr', 'wb') as output:
            torch.save(Net_sa.state_dict(), output)

    else:
        Net_sas.load_state_dict(
            torch.load(save_folder + '/net_sas' + key + '.ptr'))
        Net_sa.load_state_dict(
            torch.load(save_folder + '/net_sa' + key + '.ptr'))
    return Net_sas, Net_sa









