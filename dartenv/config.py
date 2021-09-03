import numpy as np 

class dartconfigs(object):
    def __init__(self, index=0):
        if index == 0: 
            self.mean_p = 4.0 
            self.mean_q = 4.0
            self.std_p = 1.0 
            self.std_q = 2.0
        elif index == 1:
            self.mean_p = 3.0 
            self.mean_q = 4.0
            self.std_p = 1.0 
            self.std_q = 2.0
        else:
            self.mean_p = 2.0 
            self.mean_q = 4.0
            self.std_p = 1.0 
            self.std_q = 2.0
        self.episodes = 100000
        self.buffer = 100000
        self.train = True 
        self.test = True
        self.batch_size = 1000

