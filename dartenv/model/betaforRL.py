from dartenv.model.BetaNet import BetaNetwork
import torch 
import numpy as np
import _pickle as cPickle
from dartenv.common.replaybuffer import ReplayBuffer

SAMPLE_SIZE = 128
CUDA = torch.cuda.is_available()
def BetaEstimator(environment, batch_size, train, identifier=''):
    save_folder = 'dartenv/environmentdata/' + environment
    D_P = cPickle.load(open(save_folder + '/DP.pkl', 'rb'))
    D_Q = cPickle.load(open(save_folder + '/DQ.pkl', 'rb'))

    a_pbatch, next_state_pbatch = D_P.sample_batch(batch_size = batch_size)
    a_qbatch, next_state_qbatch = D_Q.sample_batch(batch_size = batch_size)

    a_pbatch = np.expand_dims(a_pbatch, axis=1)
    next_state_pbatch = np.expand_dims(next_state_pbatch, axis=1)

    a_qbatch = np.expand_dims(a_qbatch, axis=1)
    next_state_qbatch = np.expand_dims(next_state_qbatch, axis=1)


    p_state_action_state = np.concatenate((a_pbatch, next_state_pbatch),axis=1)  # , next_state_pbatch), axis=1)#np.expand_dims(a_pbatch, axis=1)
    q_state_action_state = np.concatenate((a_qbatch, next_state_qbatch),axis=1)  # , next_state_qbatch), axis=1)#np.expand_dims(a_qbatch, axis=1)

    p_state_action = a_pbatch
    q_state_action = a_qbatch

    batch_size, dimension_sas = np.shape(p_state_action_state)
    _, dimension_sa = np.shape(p_state_action)

    Net_sas = BetaNetwork(state_dim=dimension_sas, action_bound=1000, learning_rate=1e-5, tau=1/batch_size, seed=np.random.randint(1000),
                          action_dim=1)
    Net_sa = BetaNetwork(state_dim=dimension_sa, action_bound=1000, learning_rate=1e-5, tau=1/batch_size, seed=np.random.randint(1000),
                         action_dim=1)
    if CUDA: 
        Net_sas = Net_sas.cuda()
        Net_sa = Net_sa.cuda()

    if train:

        for i in range(int(batch_size)):

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

        with open(save_folder + '/net_sas' + str(batch_size) + identifier + '.ptr', 'wb') as output:
            torch.save(Net_sas.cpu().state_dict(), output)
        with open(save_folder + '/net_sa' + str(batch_size) + identifier + '.ptr', 'wb') as output:
            torch.save(Net_sa.cpu().state_dict(), output)

    else:
        Net_sas.load_state_dict(
            torch.load(save_folder + '/net_sas' + str(batch_size) + identifier + '.ptr'))
        Net_sa.load_state_dict(
            torch.load(save_folder + '/net_sa' + str(batch_size) + identifier + '.ptr'))
    return Net_sas, Net_sa









