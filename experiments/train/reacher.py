import pybullet
from ReacherEnv.environment.gym_manipulator_envs import ReacherBulletEnv
from ReacherEnv.common.collect_data import collect_data, save_model, load_model
from ReacherEnv.model.betaforRL import BetaEstimator
import torch
from ReacherEnv.Baselines.model import DynamicModel
import gym
import os
import numpy as np 
import matplotlib.pyplot as plt 
import tikzplotlib
import statistics
import pickle 

EPISODES = 100
TIME_STEPS = 100
GAMMA = 0.99
BATCH_SIZE = 150000



def get_mean(reward_per_timestep, product):
    out_mean = 0.0
    episodes, timesteps = np.shape(reward_per_timestep)
    for step in range(timesteps):
        sum_product = np.sum(product[:, step])
        #print (sum_product)
        #reward_per_timestep[:, step] = (np.multiply(reward_per_timestep[:, step], product[:, step]/sum_product))
        out_mean += np.sum(np.multiply(reward_per_timestep[:, step], product[:, step]/sum_product))
    return out_mean, reward_per_timestep

def test_env():
	deployment_env = ReacherBulletEnv(model_xml=os.getcwd() + '/ReacherEnv/environment/environment_parameters/point_3/reacher_p.xml', render=True)
	deployment_env.reset()
	while True:
            deployment_env.step([0.0, 0.0])
def train(data = False, train = True, xml_dir=''):
    deployment_env = ReacherBulletEnv(model_xml=os.getcwd() + '/ReacherEnv/environment/environment_parameters/' + xml_dir + '/reacher_p.xml', render=False)
    simulation_env = ReacherBulletEnv(model_xml=os.getcwd() + '/ReacherEnv/environment/environment_parameters/'+ xml_dir + '/reacher_q.xml', render=False)
    if data is False:
        D_P, D_Q = collect_data(simulation_env=simulation_env, deployment_env=deployment_env, episodes=2000)
        save_model(D_P, D_Q, environment='ReacherEnv')
    if train is True:
        print ("training....")
        D_P, D_Q = load_model(environment='ReacherEnv')
        for model_idx in range(10):
            Net = BetaEstimator(environment='ReacherEnv', deployment_parameter=int(xml_dir[-1])*0.1 + 1.0, batch_size=BATCH_SIZE,
                    train=True, key='_' + str(model_idx))

def validate(which_performance, epsilon, xml_dir, performance = {}):
    if which_performance == 'true'or which_performance =='simulated':
        if which_performance == 'true':
            env = ReacherBulletEnv(model_xml=os.getcwd() + '/ReacherEnv/environment/environment_parameters/' + xml_dir + '/reacher_p.xml', render=False)
        else:
            env = ReacherBulletEnv(model_xml=os.getcwd() + '/ReacherEnv/environment/environment_parameters/' + xml_dir + '/reacher_q.xml', render=False)
        actor_critic, obs_rms = \
                    torch.load('ReacherEnv/expert/pytorch-a2c-ppo-acktr-gail/trained_models/ppo/ReacherBulletEnv-v' + str(xml_dir[-1]) +  '.pt',
                                map_location='cpu')

        performance[epsilon] = {}
        performance[epsilon]['discounted'] = np.zeros(EPISODES)
        performance[epsilon]['gamma'] = GAMMA 
        for episode in range(EPISODES):
            state = env.reset()
            done  = False
            performance[epsilon][episode] = [] 
            recurrent_hidden_states = torch.zeros(1,
                                                  actor_critic.recurrent_hidden_state_size)
            masks = torch.zeros(1, 1)
            for step in range(TIME_STEPS):
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
        env = ReacherBulletEnv(model_xml=os.getcwd() + '/ReacherEnv/environment/environment_parameters/' + xml_dir + '/reacher_q.xml', render=False)
        actor_critic, obs_rms = \
                    torch.load('ReacherEnv/expert/pytorch-a2c-ppo-acktr-gail/trained_models/ppo/ReacherBulletEnv-v' + xml_dir[-1]+ '.pt',
                                map_location='cpu')
        performance[epsilon] = {}
        performance[epsilon]['model_summary'] = []
        performance[epsilon]['GAMMA'] = GAMMA 
        for model_idx in range(10):
            model_key = int(xml_dir[-1])*0.1 + 1.0
            Beta1, Beta2 = BetaEstimator(environment='ReacherEnv', deployment_parameter=model_key, batch_size=BATCH_SIZE,
                                                        train=False, key='_' + str(model_idx))
            performance[epsilon][model_idx] = np.zeros((EPISODES, TIME_STEPS))
            product_information = np.zeros((EPISODES, TIME_STEPS))

            for episode in range(EPISODES):
                env.seed(episode)
                state = env.reset()
                done  = False
                recurrent_hidden_states = torch.zeros(1,
                                                      actor_critic.recurrent_hidden_state_size)
                masks = torch.zeros(1, 1)
                prod = 1.0
                for step in range(TIME_STEPS):
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

        env = ReacherBulletEnv(model_xml=os.getcwd() + '/ReacherEnv/environment/environment_parameters/' + xml_dir + '/reacher_p.xml', render=False)
        actor_critic, obs_rms = \
                    torch.load('ReacherEnv/expert/pytorch-a2c-ppo-acktr-gail/trained_models/ppo/ReacherBulletEnv-v'+ xml_dir[-1]+ '.pt',
                                map_location='cpu')
        performance[epsilon] = {}
        performance[epsilon]['discounted'] = np.zeros(EPISODES)
        performance[epsilon]['gamma'] = GAMMA 

        model = DynamicModel(state_dim=9, action_dim=2, learning_rate=0.0001, reg=0.01, seed=1234)
        model.load_state_dict(torch.load('./ReacherEnv/Baselines/' + xml_dir + '.ptr'))
        
        for episode in range(EPISODES):
            state = env.reset()
            state = np.reshape(state, (1, 9))
            done  = False
            performance[epsilon][episode] = [] 
            recurrent_hidden_states = torch.zeros(1,
                                                  actor_critic.recurrent_hidden_state_size)
            masks = torch.zeros(1, 1)
            for steps in range(TIME_STEPS):
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


def plotter(ax, color, performance):
    x = np.array(list(performance.keys()))
    mean = np.zeros(len(performance.keys()))
    std = np.zeros(len(performance.keys()))

    for index, epsilon in enumerate(performance.keys()):
        mean[index] = performance[epsilon]['summary'][0]
        std[index] = performance[epsilon]['summary'][1]/np.sqrt(EPISODES)

    ax.plot(x, mean, color, linewidth=2)
    ax.fill_between(x, mean - 1 * std, mean + 1 * std,color=color, alpha=0.1)

if __name__ == '__main__':
    for params in [2, 3]:#, 3, 4, 5]:
        performance = {}
        oee = {}
        simulated = {}
        model = {}
        xml_dir = 'point_' + str(params)

        #train(data=False, train=True, xml_dir=xml_dir)
        for index, epsilon in enumerate([0.1, 0.3, 0.5, 0.7, 0.9]):
            oee = validate(which_performance='oee', performance=oee, epsilon=epsilon, xml_dir=xml_dir)
            performance = validate(which_performance='true', epsilon=epsilon, performance=performance,xml_dir=xml_dir )
            simulated = validate(which_performance='simulated', epsilon=epsilon, performance=simulated,xml_dir=xml_dir )

            model = validate(which_performance='model_based', performance=model, epsilon=epsilon, xml_dir=xml_dir)

        data = {}
        data['true'] = performance 
        data['simulated'] = simulated
        data['oee'] = oee
        data['model_based'] = model


        #with open('./ReacherEnv/eval_data_' + str(params)+ '.pkl', 'wb') as handle:
        #    pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        data = pickle.load(open('./ReacherEnv/eval_data_' + str(params)+ '.pkl', 'rb'))
        performance = data['true']
        simulated = data['simulated']
        oee = data['oee']
        #model = data['model_based']
        colors = ['#000000', '#00441b', '#7f0000', '#084081', '#4d004b']

        fig, ax = plt.subplots()
        plotter(ax, colors[0], performance)
        plotter(ax, colors[1], oee)
        plotter(ax, colors[2], simulated)
        plotter(ax, colors[3], model)
        plt.grid(True)
        plt.xlabel('Epsilon')
        plt.ylabel('Average Returns')
        plt.legend(['True Value', 'OEE (ours)', 'Simulated', 'MLE'])
        tikzplotlib.save('reacher_eval'+ str(params) + '.tex')
        plt.savefig('reacher_eval'+ str(params) +  '.png')
	#train(data=False, train=True)
plt.close()
