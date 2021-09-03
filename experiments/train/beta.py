import numpy as np
import matplotlib.pyplot as plt
from beta.BetaNet import BetaNetwork
import torch
import tikzplotlib
from beta.config import Experiment 

colors = ['#000000', '#00441b', '#7f0000', '#084081', '#4d004b']


def calculate_normalpdf(x, mean, variance):
	first_term = 1/(np.sqrt(2*np.pi)*np.sqrt(variance))
	second_term = np.exp(-0.5*(((x-mean)**2)/variance))
	return first_term*second_term


def BetaEstimator(sample_size, data_p, data_q, config, train, key):

    batch_size, dimension = np.shape(data_p)
    Beta = BetaNetwork(state_dim=dimension, action_bound=config.action_bound, learning_rate=config.lr, tau=config.tau, seed=np.random.randint(0, 1000), action_dim=1) 

    if train:
        for i in range(int(sample_size)):
            idx = np.random.randint(batch_size, size=sample_size)

            states_p = data_p[idx, :]
            states_q = data_q[idx, :]

            states_p = torch.from_numpy(states_p)
            states_q = torch.from_numpy(states_q)

            states_p = states_p.type(torch.FloatTensor)
            states_q = states_q.type(torch.FloatTensor)

            output_beta = Beta.train_step(states_p=states_p, states_q=states_q)

            if i % 100 == 0:
                print("Training Iterations", i, output_beta.data.numpy())

        with open('./beta/data/beta_' + str(sample_size) + key + '.ptr', 'wb') as output:
                    torch.save(Beta.state_dict(), output)
                    
    else:
        Beta.load_state_dict(torch.load('./beta/data/beta_' + str(sample_size) + key + '.ptr'))
        
    return Beta



if __name__ == '__main__':

    config = Experiment(index=0)
    mean_p = config.mean_p
    variance_p = config.std_p
    mean_q = config.mean_q
    variance_q = config.std_q

    X = np.linspace(-4, 8, 100)
    sample_size = config.sample_sizes

    train = config.train 
    test = config.test

    if train:
        for sample in sample_size:
            data_p = np.random.normal(loc=mean_p, scale=np.sqrt(variance_p), size=(sample, 1))
            data_q = np.random.normal(loc=mean_q, scale=np.sqrt(variance_q), size=(sample, 1))
            for i in range(10):
                key = '_' + str(i) + '_' + str(int(mean_p)) + str(int(variance_p)) + str(int(mean_q)) + str(int(variance_q))
                Beta = BetaEstimator(sample_size=sample, data_p=data_p, data_q=data_q, train=True, key=key, config=config)
    if test:
        data = np.zeros((10, 100))
        baseline = []
        sample_size.append('Baselines')
        index = 0
        for sample in sample_size:
            print (sample)

            j = -1
            mean = []
            var = []
            for point in X:
                j += 1
                if sample == 'Baselines':
                    pdf_p = calculate_normalpdf(point, mean_p, variance_p)
                    pdf_q = calculate_normalpdf(point, mean_q, variance_q)
                    baseline.append(pdf_p/pdf_q)
                else:
                    data_p = np.random.normal(loc=mean_p, scale=np.sqrt(variance_p), size=(sample, 1))
                    data_q = np.random.normal(loc=mean_q, scale=np.sqrt(variance_q), size=(sample, 1))
                    for i in range(10):
                        key = '_' + str(i) + '_' + str(int(mean_p)) + str(int(variance_p)) + str(int(mean_q)) + str(int(variance_q))
                        Beta = BetaEstimator(sample_size=sample, data_p=data_p, data_q=data_q, train=False, key=key)
                        input = np.expand_dims([point], axis=1)
                        input = torch.from_numpy(input)
                        input = input.type(torch.FloatTensor)
                        output = Beta.predict(input)
                        output = output.data.numpy()
                        data[i, int(j)] = output[0, 0]
                    mean.append(np.mean(data[:, j]))
                    var.append(np.var(data[:, j]))
            if sample != 'Baselines':
                mean = np.array(mean)
                var = np.array(var)
                plt.plot(X, mean, linewidth=2, color=colors[index])
                plt.fill_between(X, mean-1*var, mean+1*var, color=colors[index], alpha=0.2)
            index += 1
        plt.plot(X, baseline, '--', linewidth=2, color='#4a1486')
        legends = []
        sample_sizes.remove('Baselines')
        for i in range(len(Sample_Sizes)):
                legends.append(str(sample_sizes[i]))
        legends.append('Oracle')
        plt.legend(legends)
        plt.grid(True)
        key = str(int(mean_p)) + str(int(variance_p)) + str(int(mean_q)) + str(int(variance_q))
        tikzplotlib.save('./beta/assets/beta_' + key +  '.tex')
        plt.savefig('./beta/assets/beta_' + key + '.png', dpi=300)

