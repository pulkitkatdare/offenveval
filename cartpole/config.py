import numpy as np 

class cartpoleconfigs(object):
    def __init__(self, index=0):
        self.simulation_parameter = 10.0
        if index == 0: 
            self.deployment_parameter = 5.0
        elif index == 1:
            self.deployment_parameter = 7.5 
        elif index == 2:
            self.deployment_parameter = 10.0
        elif index == 3:
            self.deployment_parameter = 12.5
        elif index == 4:
            self.deployment_parameter = 15.0
        self.episodes = 3000
        self.buffer = 100000
        self.batch_size = 80000
        self.train = False 
        self.test = True

        if self.test: 
            self.train_mle = False
            self.true = True 
            self.oee = True 
            self.eval_batch_size = 80000
            self.mle = True 
            self.IS = True

            self.eval_episodes = 100
            self.timesteps = 100
            self.epsilons = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

            self.savepng = True
            self.savetex = True 



