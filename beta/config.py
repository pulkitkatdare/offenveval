import numpy as np 

class Experiment(object):
    def __init__(self, index = 0):
        if index == 0:
            self.mean_p = 4.0
            self.mean_q = 4.0 
            self.std_p = 1.0 
            self.std_q = 2.0
            self.action_bound = 40
        elif index == 1:
            self.mean_p = 3.0
            self.mean_q = 4.0 
            self.std_p = 1.0 
            self.std_q = 2.0
            self.action_bound = 20 
        else:
            self.mean_p = 3.0
            self.mean_q = 4.0 
            self.std_p = 1.0 
            self.std_q = 2.0
            self.action_bound = 20

        self.sample_sizes = [500, 1000, 2000, 4000, 8000]
        self.train = True
        self.test = True
        self.lr = 5e-6 
        self.tau = 1e-4

