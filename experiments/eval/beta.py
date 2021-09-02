import numpy as np
import matplotlib.pyplot as plt
from Beta.BetaNet import BetaNetwork
import torch
import tikzplotlib

BATCH_SIZE = 128

def BetaEstimator(sample_size, data_p, data_q, train=True, key=''):

    batch_size, dimension = np.shape(data_p)
    Beta = BetaNetwork(state_dim=dimension, action_bound=40.0, learning_rate=5e-6, tau=1e-4, seed=np.random.randint(0, 1000), action_dim=1) #10.0 # BetaNet(n_gaussians = batch_size)

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

        with open('./Beta/data/Baselines/beta_' + str(sample_size) + key + '.ptr', 'wb') as output:
                    torch.save(Beta.state_dict(), output)


    else:
        Beta.load_state_dict(torch.load('./Beta/data/Baselines/beta_' + str(sample_size) + key + '.ptr'))
    return Beta

def calculate_normalpdf(x, mean, variance):
	first_term = 1/(np.sqrt(2*np.pi)*np.sqrt(variance))
	second_term = np.exp(-0.5*(((x-mean)**2)/variance))
	return first_term*second_term

if __name__ == '__main__':

    train = False
    test = True
    mean_p = 2.0
    variance_p = 1.0
    mean_q = 4.0
    variance_q = 2.0

    colors = ['#000000', '#00441b', '#7f0000', '#084081', '#4d004b']#['#9e9ac8', '#807dba',

    X = np.linspace(-4, 8, 100)
    Sample_Sizes = [2000, 4000, 8000, 'Baselines']#[2000, 4000, 8000, 10000, 15000, 'Baselines']#, 1000, 20000]

    if train:
        for sample in Sample_Sizes:
            data_p = np.random.normal(loc=mean_p, scale=np.sqrt(variance_p), size=(sample, 1))
            data_q = np.random.normal(loc=mean_q, scale=np.sqrt(variance_q), size=(sample, 1))
            for i in range(10):
                key = '_' + str(i) + '_2142'
                Beta = BetaEstimator(sample_size=sample, data_p=data_p, data_q=data_q, train=True, key=key)
    if test:
        data = np.zeros((10, 100))
        baseline = []
        index = 0
        for sample in Sample_Sizes:
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
                        key = '_' + str(i) + '_2142'
                        Beta = BetaEstimator(sample_size=sample, data_p=data_p, data_q=data_q, train=False, key=key)
                        input = np.expand_dims([point], axis=1)
                        input = torch.from_numpy(input)
                        input = input.type(torch.FloatTensor)
                        output = Beta.predict(input)
                        output = output.data.numpy()
                        data[i, int(j)] = output[0, 0]
                    mean.append(np.mean(data[:, j]));
                    var.append(np.var(data[:, j]));
            if sample != 'Baselines':
                mean = np.array(mean);
                var = np.array(var);
                print (var)
                plt.plot(X, mean, linewidth=2, color=colors[index]);
                plt.fill_between(X, mean-1*var, mean+1*var, color=colors[index], alpha=0.2)
            index += 1
        plt.plot(X, baseline, '--', linewidth=2, color='#4a1486')
        #plt.plot(data_p, np.zeros(8000), 'x')
        #plt.plot(data_q, np.zeros(8000) + 130, 'o')
        legends = []
        Sample_Sizes.remove('Baselines')
        for i in range(len(Sample_Sizes)):
                legends.append(str(Sample_Sizes[i]))
        legends.append('Oracle')
        #legends.append('P-samples')
        #legends.append('Q-samples')
        plt.legend(legends)
        plt.grid(True)
        tikzplotlib.save("beta.tex")
        plt.savefig('beta.png', dpi=300)
#plt.show()
#
#
#	for index, sample_size in enumerate(Sample_Sizes):
#
#		data_p = np.random.normal(loc=mean_p, scale=np.sqrt(variance_p), size=(sample_size, 1))
#		data_q = np.random.normal(loc=mean_q, scale=np.sqrt(variance_q), size=(sample_size, 1))
#		for i in range(10):
#			key = '_' + str(i)
#			Beta = BetaEstimator(sample_size=sample_size, data_p=data_p, data_q=data_q, key=key)
#	'''







