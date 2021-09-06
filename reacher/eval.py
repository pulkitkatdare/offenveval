import pybullet
from reacher.environment.gym_manipulator_envs import ReacherBulletEnv
from reacher.common.collect_data import collect_data, save_model, load_model
from reacher.model.betaforRL import BetaEstimator
from reacher.baselines.model import DynamicModel
from reacher.baselines.MLE import train_mle

import torch
import gym
import os
import numpy as np 
import matplotlib.pyplot as plt 
import tikzplotlib
import statistics
import pickle 

GAMMA = 0.99


def get_mean(reward_per_timestep, product):
	out_mean = 0.0
	episodes, timesteps = np.shape(reward_per_timestep)
	for step in range(timesteps):
		sum_product = np.sum(product[:, step])
		#print (sum_product)
		#reward_per_timestep[:, step] = (np.multiply(reward_per_timestep[:, step], product[:, step]/sum_product))
		out_mean += np.sum(np.multiply(reward_per_timestep[:, step], product[:, step]/sum_product))
	return out_mean, reward_per_timestep




def validate(config, which_performance, epsilon, xml_dir, performance = {}):
	if which_performance == 'true'or which_performance =='simulated':
		if which_performance == 'true':
			env = ReacherBulletEnv(model_xml=os.getcwd() + '/reacher/environment/environment_parameters/' + xml_dir + '/reacher_p.xml', render=False)
		else:
			env = ReacherBulletEnv(model_xml=os.getcwd() + '/reacher/environment/environment_parameters/' + xml_dir + '/reacher_q.xml', render=False)
		actor_critic, obs_rms = \
					torch.load('./reacher/expert/pytorch-a2c-ppo-acktr-gail/trained_models/ppo/ReacherBulletEnv-v' + str(int(xml_dir)+1) +  '.pt',
								map_location='cpu')

		performance[epsilon] = {}
		performance[epsilon]['discounted'] = np.zeros(config.eval_episodes)
		performance[epsilon]['gamma'] = GAMMA 
		for episode in range(config.eval_episodes):
			state = env.reset()
			done  = False
			performance[epsilon][episode] = [] 
			recurrent_hidden_states = torch.zeros(1,
												  actor_critic.recurrent_hidden_state_size)
			masks = torch.zeros(1, 1)
			for step in range(config.eval_timesteps):
				if done is False:
					if np.random.uniform() < epsilon:
						action = [np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0)]
					else: 
						with torch.no_grad():
							obs = np.reshape(state, (1, 9))
							obs = torch.from_numpy(obs).float().to('cpu')
							value, action, _, recurrent_hidden_states = actor_critic.act(
								obs, recurrent_hidden_states, masks, deterministic=True)

						action = action.data.cpu().numpy()[0, :]
						#print (action)

					#action += 0.001*np.random.normal(0.0, 1.0, 2)
					next_state, reward, done, info = env.step(action)
					performance[epsilon][episode].append((GAMMA**step)*reward)
					state = next_state 
				else:
					break 

			#print (np.linalg.norm(next_state[2:4]))
			performance[epsilon]['discounted'][episode] = sum(performance[epsilon][episode])
		performance[epsilon]['summary'] = [np.mean(performance[epsilon]['discounted']), np.std(performance[epsilon]['discounted'])]
		return performance

	elif which_performance == 'oee':
		print ("entering", which_performance)
		env = ReacherBulletEnv(model_xml=os.getcwd() + '/reacher/environment/environment_parameters/' + xml_dir + '/reacher_q.xml', render=False)
		actor_critic, obs_rms = \
					torch.load('./reacher/expert/pytorch-a2c-ppo-acktr-gail/trained_models/ppo/ReacherBulletEnv-v' + str(int(xml_dir)+1)+ '.pt',
								map_location='cpu')
		performance[epsilon] = {}
		performance[epsilon]['model_summary'] = []
		performance[epsilon]['GAMMA'] = GAMMA 
		for model_idx in range(10):
			model_key = int(xml_dir)
			Beta1, Beta2 = BetaEstimator(environment='ReacherEnv', deployment_parameter=model_key, batch_size=config.batch_size,
														train=False, key='_' + str(model_idx))
			performance[epsilon][model_idx] = np.zeros((config.eval_episodes, config.eval_timesteps))
			product_information = np.zeros((config.eval_episodes, config.eval_timesteps))

			for episode in range(config.eval_episodes):
				env.seed(episode)
				state = env.reset()
				done  = False
				recurrent_hidden_states = torch.zeros(1,
													  actor_critic.recurrent_hidden_state_size)
				masks = torch.zeros(1, 1)
				prod = 1.0
				for step in range(config.eval_timesteps):
					if np.random.uniform() < epsilon:
						action = [np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0)]
					else: 
						with torch.no_grad():
							obs = np.reshape(state, (1, 9))
							obs = torch.from_numpy(obs).float().to('cpu')
							value, action, _, recurrent_hidden_states = actor_critic.act(
								obs, recurrent_hidden_states, masks, deterministic=True)
						action = action.data.cpu().numpy()[0, :] 
					
					#action += 0.0001*np.random.normal(0.0, 1.0, 2)
					next_state, reward, done, info = env.step(action)
					

					x = np.concatenate((state, action, next_state), axis=0)
					x = torch.from_numpy(x)
					x = x.type(torch.FloatTensor)

					y = np.concatenate((state, action), axis=0)
					y = torch.from_numpy(y)
					y = y.type(torch.FloatTensor)

					beta1 = (Beta1.predict(x))
					beta2 = (Beta2.predict(y))

					beta1 = beta1.data.numpy()
					beta2 = beta2.data.numpy()

					performance[epsilon][model_idx][episode, step] =  (GAMMA**step)*reward
					product_information[episode, step] = prod

					prod = prod*(beta1[0]/beta2[0])#np.clip((beta1[0]/beta2[0]), 1e-10, 1000.0)
					#print ("product", prod)
					state = next_state 
			mean_idx, performance[epsilon][model_idx] = get_mean(performance[epsilon][model_idx], product_information)
			performance[epsilon]['model_summary'].append(mean_idx)
		#print ("Here........", len( performance[epsilon]['model_summary']))
		performance[epsilon]['summary'] = [statistics.mean(performance[epsilon]['model_summary']), statistics.stdev(performance[epsilon]['model_summary'])]
		print (statistics.mean(performance[epsilon]['model_summary']))
		return performance
	elif which_performance=='model_based':

		env = ReacherBulletEnv(model_xml=os.getcwd() + '/reacher/environment/environment_parameters/' + config.xml_dir + '/reacher_p.xml', render=False)
		actor_critic, obs_rms = \
					torch.load('./reacher/expert/pytorch-a2c-ppo-acktr-gail/trained_models/ppo/ReacherBulletEnv-v'+ str(int(config.xml_dir) + 1)+ '.pt',
								map_location='cpu')
		performance[epsilon] = {}
		performance[epsilon]['discounted'] = np.zeros(config.eval_episodes)
		performance[epsilon]['gamma'] = GAMMA 

		model = DynamicModel(state_dim=9, action_dim=2, learning_rate=0.0001, reg=0.01, seed=1234)
		model.load_state_dict(torch.load('./reacher/baselines/' + xml_dir + '.ptr'))
		
		for episode in range(config.eval_episodes):
			state = env.reset()
			state = np.reshape(state, (1, 9))
			done  = False
			performance[epsilon][episode] = [] 
			recurrent_hidden_states = torch.zeros(1,
												  actor_critic.recurrent_hidden_state_size)
			masks = torch.zeros(1, 1)
			for steps in range(config.eval_timesteps):
				if np.random.uniform() < epsilon:
					action = [np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0)]
				else: 
					with torch.no_grad():
						obs = state#np.reshape(state, (1, 9))
						obs = torch.from_numpy(obs).float().to('cpu')
						value, action, _, recurrent_hidden_states = actor_critic.act(
							obs, recurrent_hidden_states, masks, deterministic=True)
					action = action.data.cpu().numpy()[0, :] 

				#action += 0.001*np.random.normal(0.0, 1.0, 2)
				input_state = np.reshape(state, (1, 9))
				action = np.reshape(action, (1, 2))
				input = np.concatenate((input_state, action),axis=1)
				input = torch.from_numpy(input)
				input = input.type(torch.FloatTensor)
				next_state = model(input)
				next_state =  next_state.data.numpy()
				next_state = 1e-1*np.random.randn(1, 9) + next_state
				reward_t1 = -100 * np.linalg.norm(next_state[0, 2:4])
				reward_t1 = reward_t1 + 100 * np.linalg.norm(state[0, 2:4])

				reward_t2 = (
					-0.10 * (np.abs(action[0, 0] * next_state[0, 6] + np.abs(action[0, 1] * next_state[0, 8])
							)  # work torque*angular_velocity
					- 0.01 * (np.abs(action[0, 0]) + np.abs(action[0, 1]))  # stall torque require some energy
				))

				if np.abs(np.abs(next_state[0, 8]) - 1) < 0.01:
					reward_t3 = -0.1 
				else:
					reward_t3 = 0.0

				reward = reward_t1 + reward_t2 + reward_t3

				performance[epsilon][episode].append((GAMMA**steps)*reward)
				state = next_state 

			performance[epsilon]['discounted'][episode] = sum(performance[epsilon][episode])
		performance[epsilon]['summary'] = [np.mean(performance[epsilon]['discounted']), np.std(performance[epsilon]['discounted'])]

		return performance
def plotter(config, ax, color, performance):
    x = np.array(list(performance.keys()))
    mean = np.zeros(len(performance.keys()))
    std = np.zeros(len(performance.keys()))

    for index, epsilon in enumerate(performance.keys()):
        mean[index] = performance[epsilon]['summary'][0]
        std[index] = performance[epsilon]['summary'][1]/np.sqrt(config.eval_episodes)

    ax.plot(x, mean, color, linewidth=2)
    ax.fill_between(x, mean - 1 * std, mean + 1 * std,color=color, alpha=0.1)

def eval_performance(config):
	if config.mle:
		if config.train_mle:
			train_mle(config)
	performance = {}
	oee = {}
	simulated = {}
	model = {}
	xml_dir = config.xml_dir

	for index, epsilon in enumerate(config.epsilons):
		if config.oee:
			oee = validate(config=config, which_performance='oee', performance=oee, epsilon=epsilon, xml_dir=xml_dir)
		if config.true:
			performance = validate(config=config, which_performance='true', epsilon=epsilon, performance=performance,xml_dir=xml_dir)
		if config.simulator:
			simulated = validate(config=config, which_performance='simulated', epsilon=epsilon, performance=simulated,xml_dir=xml_dir )
		if config.mle:
			model = validate(config=config, which_performance='model_based', performance=model, epsilon=epsilon, xml_dir=xml_dir)

	colors = ['#000000', '#00441b', '#7f0000', '#084081', '#4d004b']

	fig, ax = plt.subplots()
	legends = []
	if config.true:
		legends.append('True Value')
		plotter(config, ax, colors[0], performance)
	if config.oee:
		legends.append('OEE (ours)')
		plotter(config, ax, colors[1], oee)
	if config.simulator:
		legends.append('Simulator')
		plotter(config, ax, colors[2], simulated)
	if config.mle:
		legends.append('MLE')
		plotter(config, ax, colors[3], model)
	plt.grid(True)
	plt.xlabel('Epsilon')
	plt.ylabel('Average Returns')
	plt.legend(legends)
	if config.savetex:
		tikzplotlib.save('./reacher/assets/reacher_eval'+ config.xml_dir + '.tex')
	if config.savepng:
		plt.savefig('./reacher/assets/reacher_eval'+ config.xml_dir +  '.png')
	plt.close()