import numpy as np
import torch
from Cartpole.environments.cartpole import CartPoleEnv
from MountainCar.environments.mountain_car import MountainCarEnv
from Acrobot.environments.acrobot import AcrobotEnv
from Baselines.model import DynamicModel
from pytorchrl.rl.models.cem.actor_cartpole import ActorNetwork
import matplotlib.pyplot as plt
import pickle
from Acrobot.acrobotv1.policy import Policy


ENVIRONMENT = 'Acrobot'
BATCH_SIZE = 64
def train(deployment_parameter):
	if ENVIRONMENT == 'Cartpole':
		from Cartpole.common.collect_data import load_model, collect_data, save_model
		simulation_parameter = 9.8
		simulation_env = CartPoleEnv(gravity=deployment_parameter)
		deployment_env = CartPoleEnv(gravity=deployment_parameter)
		DP, DQ = collect_data(simulation_env, deployment_env, episodes=1000)
		#save_model(DP, DQ, environment=ENVIRONMENT)
		env = CartPoleEnv(gravity=simulation_parameter)
		state_dim = env.observation_space.shape[0]
		action_dim = 1#env.action_space.n
		model = DynamicModel(state_dim=state_dim, action_dim=action_dim, learning_rate=0.00001, reg=0.01, seed=1234)
	elif ENVIRONMENT == 'Acrobot':
		from Acrobot.common.collect_data import load_model, collect_data, save_model
		simulation_parameters=[1.0, 1.0]
		simulation_env = AcrobotEnv(link_length_1=deployment_parameter[0], link_length_2=deployment_parameter[1])
		deployment_env = AcrobotEnv(link_length_1=deployment_parameter[0], link_length_2=deployment_parameter[1])
		DP, DQ = collect_data(simulation_env, deployment_env, episodes=100)
		#save_model(DP, DQ, environment=ENVIRONMENT)
		env = AcrobotEnv(simulation_parameters)
		state_dim = env.observation_space.shape[0]
		action_dim = 1#env.action_space.n
		model = DynamicModel(state_dim=state_dim, action_dim=action_dim, learning_rate=1e-3, reg=0.01, seed=1234)
	elif ENVIRONMENT == 'MountainCar':
		from MountainCar.common.collect_data import load_model, collect_data, save_model
		simulation_parameters=deployment_parameter
		simulation_env = MountainCarEnv(simulation_parameters)
		deployment_env = MountainCarEnv(deployment_parameter)
		DP, DQ = collect_data(simulation_env, deployment_env, episodes=100)
		#save_model(DP, DQ, environment=ENVIRONMENT)
		env = MountainCarEnv(simulation_parameters)
		state_dim = env.observation_space.shape[0]
		action_dim = 1#env.action_space.n
		model = DynamicModel(state_dim=state_dim, action_dim=action_dim, learning_rate=0.00001, reg = 0.01, seed = 1234)

	for i in range(1, int(157)):

		s_pbatch, a_pbatch, next_state_pbatch, reward_p, terminal_p = DP.sample_batch(batch_size=BATCH_SIZE)
		input = np.concatenate((s_pbatch, a_pbatch), axis=1)

		input = torch.from_numpy(input)
		input = input.type(torch.FloatTensor)
		output = torch.from_numpy(next_state_pbatch)
		output = output.type(torch.FloatTensor)

		loss = model.train_step(input=input, output=output)
		if i % 100 == 0:
			print("Training Iterations", i, loss.data.numpy())
			with open(ENVIRONMENT  + '/Baselines/' + 'model' + str(deployment_parameter) + '.ptr', 'wb') as output:
				torch.save(model.state_dict(), output)

def test(deployment_parameter=12.5):
	if ENVIRONMENT == 'Cartpole':

		env = CartPoleEnv(gravity=deployment_parameter)
		state_dim = env.observation_space.shape[0]
		action_dim = 1#env.action_space.n
		actor = ActorNetwork(4, 2)
		actor.load_state_dict(torch.load('./pytorchrl/saver/cem_cartpole'))
		model = DynamicModel(state_dim=state_dim, action_dim=action_dim, learning_rate=0.0001, reg=0.01, seed = 1234)
		model.load_state_dict(torch.load(ENVIRONMENT + '/Baselines/' + 'model' + str(deployment_parameter) +  '.ptr'))

	if ENVIRONMENT == 'Acrobot':

		env = AcrobotEnv(deployment_parameters)
		state_dim = env.observation_space.shape[0]
		action_dim = 1#env.action_space.n
		actor = Policy().to('cpu')
		actor.load_state_dict(torch.load('Acrobot/acrobotv1/checkpoint.pth'))
		model = DynamicModel(state_dim=state_dim, action_dim=action_dim, learning_rate=0.0001, reg=0.01, seed = 1234)
		model.load_state_dict(torch.load(ENVIRONMENT + '/Baselines/' + 'model' + str(sum(deployment_parameter)) + '.ptr'))

	state_roll_out = {}
	state_roll_out['model'] = {}
	state_roll_out['environment'] = {}
	for i in range(state_dim):
		state_roll_out['model'][i] = {}
		state_roll_out['environment'][i] = {}
	for index in range(100):
		print("Episodes", index)
		done = False
		state = env.reset()
		state_predicted = state
		for i in range(state_dim):
			state_roll_out['environment'][i][index] = []
			state_roll_out['model'][i][index] = []
			state_roll_out['environment'][i][index].append(state[i])
			state_roll_out['model'][i][index].append(state_predicted[i])
		steps = 0
		while not done:
			if ENVIRONMENT == 'Cartpole':
				input_state = np.reshape(state, (1, state_dim))
				input_state = torch.from_numpy(input_state)
				input_state = input_state.type(torch.FloatTensor)
				a = actor(input_state)
				a = a.data.cpu().numpy()
				action = a[0][0]
				steps += 1
				next_state, r, done, info = env.step(action)

				input_state = np.reshape(state_predicted, (1, state_dim))
				input_state = torch.from_numpy(input_state)
				input_state = input_state.type(torch.FloatTensor)
				action = actor(input_state)
				action = action.data.cpu().numpy()

				input = np.concatenate((state_predicted, [action]))
				input = torch.from_numpy(input)
				input = input.type(torch.FloatTensor)
				output = model(input)
				#state_predicted = output.data.numpy()
				#state = next_state
			if ENVIRONMENT == 'Acrobot':
				action, log_prob = actor.act(state_predicted)
				next_state, r, done, info = env.step(action)
				action, log_prob = actor.act(state_predicted)
				input = np.concatenate((state_predicted, [action]))
				input = torch.from_numpy(input)
				input = input.type(torch.FloatTensor)
				output = model(input)
				steps += 1
				if steps > 200:
					break

			state_predicted = output.data.numpy()
			state = next_state
			print (state_dim)
			for i in range(state_dim):
				state_roll_out['environment'][i][index].append(state[i])
				state_roll_out['model'][i][index].append(state_predicted[i])
	with open(ENVIRONMENT + '/Baselines/eval_' + str(sum(deployment_parameter)) + '.json', 'wb') as fp:
		pickle.dump(state_roll_out, fp)
def plot(deployment_parameter):
	state_roll_out = pickle.load(open(ENVIRONMENT + '/Baselines/eval_' + str(sum(deployment_parameter)) + '.json', 'rb'))

	fig, ax = plt.subplots(6)
	colors = ['#33a02c', '#33a02c', '#33a02c', '#33a02c', '#33a02c', '#33a02c'] #['#33a02c', '#b35806', '#1f78b4', '#5e3c99']
	legends = ['cos(theta1)', 'sin(theta1)', 'cos(theta2)',  'sin(theta2)', 'thetaDot1', 'thetaDot2']#['x-axis', 'x-dot', 'theta', 'theta_dot']
	for i in range(6):
		ax[i].plot(state_roll_out['model'][i][0], '--', color=colors[i], linewidth=2)#, label=legends[i])
		ax[i].plot(state_roll_out['environment'][i][0], color=colors[i], linewidth=2, label=legends[i])
		ax[i].legend(loc="upper right")
		ax[i].grid()

	plt.xlabel('Timesteps')
	plt.show()





if __name__ == '__main__':
	train_model = True
	test_model = False
	plot_model = False
	for j in range(5):#[0.0015, 0.0017, 0.0021, 0.0027, 0.0029, 0.0033]:
		i = 2*j
		if train_model:
			train(deployment_parameter = [1.0 + (i-5)*0.1, 1.0])
		if test_model:
			test(deployment_parameter = i)
		if plot_model:
			plot(deployment_parameter = i)
