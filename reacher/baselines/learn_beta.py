import numpy as np
import matplotlib.pyplot as plt
from Baselines.BetaNet import BetaNetwork
import torch
import tikzplotlib

BATCH_SIZE = 128

def BetaEstimator(sample_size, data_p, data_q, train=True):

	batch_size, dimension = np.shape(data_p)

	Beta = BetaNetwork(state_dim=dimension, action_bound=10.0, learning_rate=5e-6, tau=1e-4, seed=1234, action_dim=1)  # BetaNet(n_gaussians = batch_size)

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

		with open('Baselines/beta_' + str(sample_size) + '.ptr', 'wb') as output:
			torch.save(Beta.state_dict(), output)


	else:
		Beta.load_state_dict(torch.load('Baselines/beta_' + str(sample_size) + '.ptr'))
	return Beta

def calculate_normalpdf(x, mean, variance):
	first_term = 1/(np.sqrt(2*np.pi)*np.sqrt(variance))
	second_term = np.exp(-0.5*(((x-mean)**2)/variance))
	return first_term*second_term

if __name__ == '__main__':


	mean_p = 4.0
	variance_p = 1.0
	mean_q = 4.0
	variance_q = 2.0
	Sample_Sizes = [500, 1000, 2000, 4000, 8000]#, 1000, 20000]
	colors = ['#000000', '#00441b', '#7f0000', '#084081', '#4d004b']#['#9e9ac8', '#807dba',

	for index, sample_size in enumerate(Sample_Sizes):

		data_p = np.random.normal(loc=mean_p, scale=np.sqrt(variance_p), size=(sample_size, 1))
		data_q = np.random.normal(loc=mean_q, scale=np.sqrt(variance_q), size=(sample_size, 1))

		Beta = BetaEstimator(sample_size, data_p, data_q)
		X = np.linspace(-5, 8, 100)
		data = []
		baseline = []
		for point in X:
			pdf_p = calculate_normalpdf(point, mean_p, variance_p)
			pdf_q = calculate_normalpdf(point, mean_q, variance_q)
			baseline.append(pdf_p/pdf_q)

			input = np.expand_dims([point], axis=1)
			input = torch.from_numpy(input)
			input = input.type(torch.FloatTensor)
			output = Beta.predict(input)
			output = output.data.numpy()
			data.append(output[0, 0])

		plt.plot(X, data, linewidth=2, color=colors[index])
	plt.plot(X, baseline, '--', linewidth=2, color='#4a1486')
	legends = []
	for i in range(len(Sample_Sizes)):
		legends.append(str(Sample_Sizes[i]))
	legends.append('Oracle')
	plt.legend(legends)
	plt.grid(True)
	tikzplotlib.save("Beta_example.tex")
	plt.show()







