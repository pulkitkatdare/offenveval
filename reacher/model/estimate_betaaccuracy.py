import numpy as np
from BetaNet import BetaNetwork
import torch
import matplotlib.pyplot as plt
import tikzplotlib
import pickle


mean_p = -1.0
sigma_p = 1.0
mean_q = -2.0
sigma_q = 2.0
sigma = 1.0
size = 500
C = 1.0
dimension = 1

data_p = np.random.uniform(mean_p, sigma_p, (size, 1))
data_q = np.random.uniform(mean_q, sigma_q, (size, 1))

Beta = BetaNetwork(state_dim=dimension, action_bound = C, learning_rate=0.0005, tau=0.001, seed=1234)

for i in range(100):
	states_p = torch.from_numpy(data_p)
	states_q = torch.from_numpy(data_q)
	states_p = states_p.type(torch.FloatTensor)
	states_q = states_q.type(torch.FloatTensor)
	output = Beta.train(states_p=states_p, states_q=states_q)
	print ("Training Iterations", i,  output.data.numpy())
oracle = []
optimisation = []
input = np.zeros((1, 1))
for i in np.linspace(-2, 2, 100):
	input[0, 0] = i
	x = torch.from_numpy(input)
	x = x.type(torch.FloatTensor)
	out = Beta.predict(x).data.numpy()
	optimisation.append(out[0][0])
	if (i  > 1.0) or (i < -1.0):
		oracle.append(0.0)
	else:
		oracle.append(2)#((sigma_q/sigma_p)*np.exp((-((i-mean_p)**2/(2*sigma_p**2)) + ((i-mean_q)**2/(2*sigma_q**2))))))


colors = ['#00441b', '#006d2c', '#238b45', '#41ae76', '#66c2a4', '#99d8c9', '#ccece6', '#e5f5f9', '#f7fcfd']
X = np.linspace(-1, 1, 100)
plt.plot(X, optimisation)
plt.plot(X, oracle, linewidth=2)
plt.ylabel(' beta  (x)', fontsize=16)
plt.xlabel('x', fontsize=16)
plt.legend(['optimiser', 'Oracle'])
plt.grid(True)
pickle.dump(optimisation, open('optimisation' + str(size) + '.pkl', 'wb'))
#tikzplotlib.save('EstimatorAccuracy.tex')
plt.show()
'''

oracle = []
optimisation = []
input = np.zeros((1, 1))
for i in np.linspace(-4, 4, 100):
	input[0, 0] = i
	oracle.append(((sigma_q/sigma_p)*np.exp((-((i-mean_p)**2/(2*sigma_p**2)) + ((i-mean_q)**2/(2*sigma_q**2))))))
sizes = [50000, 10000, 5000]#, 2000, 5000, 10000]
colors = ['#00441b', '#006d2c', '#238b45']#, '#41ae76', '#66c2a4', '#99d8c9']#, '#ccece6', '#e5f5f9', '#f7fcfd']
X = np.linspace(-4, 4, 100)
for color, size in zip(colors, sizes):
	optimisation = pickle.load(open('optimisation' + str(size) + '.pkl', 'rb'))
	plt.plot(X, optimisation, color=color, linewidth=2)
sizes.append('Oracle')
plt.plot(X, oracle, linewidth=2)
plt.ylabel(' beta  (x)', fontsize=16)
plt.xlabel('x', fontsize=16)
plt.legend(sizes)
plt.grid(True)
#pickle.dump(optimisation, open('optimisation' + str(size) + '.pkl', 'wb'))
#tikzplotlib.save('EstimatorAccuracy.tex')
plt.show()

'''



'''
data_p = torch.tensor(data_p)
data_q = torch.tensor(data_q)

Beta_est = BetaNet(n_gaussians=size)
Beta_est.train(data_p, data_q, epochs=size)
Beta_est = ZetaNumpy(Beta_est)
sum = 0

for i in np.linspace(-4, 4, 100):
	input[0, 0] = i
	sum += abs(Beta_est.forward(input)[0][0]-np.exp((-(i-mean_p)**2 + (i-mean_q)**2)/2))
print (sum/100)


#pickle.dump(Beta_est, open('Beta_est' + str(size) + '.pkl', 'wb'))


import pickle
out_objects = []
out_objects.append(pickle.load(open('betaestimatorfiles/Beta_est5000.pkl', 'rb')))
out_objects.append(pickle.load(open('betaestimatorfiles/Beta_est2000.pkl', 'rb')))
out_objects.append(pickle.load(open('betaestimatorfiles/Beta_est1000.pkl', 'rb')))
out_objects.append(pickle.load(open('betaestimatorfiles/Beta_est100.pkl', 'rb')))

out = []
out.append([])
out.append([])
out.append([])
out.append([])

out_normal = []
input = np.zeros((1, 1))

for i in np.linspace(-4, 4, 100):
	input[0, 0] = i
	for j in range(4):
		out[j].append(np.log(out_objects[j].forward(input)[0][0]))
	out_normal.append(np.log(np.exp((-(i-mean_p)**2 + (i-mean_q)**2)/2)))
colors = ['#00441b', '#006d2c', '#238b45', '#41ae76', '#66c2a4', '#99d8c9', '#ccece6', '#e5f5f9', '#f7fcfd']
X = np.linspace(-4, 4, 100)
for j in range(4):
	plt.plot(X, out[j], color=colors[j], linewidth=2)

plt.plot(X, out_normal, linewidth=2)
plt.ylabel(' beta  (x)', fontsize=16)
plt.xlabel('x', fontsize=16)
plt.legend(['Estimator (k = 5000)', 'Estimator (k = 2000)', 'Estimator (k = 1000)', 'Estimator (k = 100)', 'Oracle'])
plt.grid(True)
#tikzplotlib.save('EstimatorAccuracy.tex')
plt.show()
'''

