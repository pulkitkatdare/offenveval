import gym 
import numpy as np 
import math 





class DartEnv(object):
    def __init__(self, mean_wind_speed=0.0, std_wind_speed=1.0):
        self.length = 100 
        self.mean = mean_wind_speed 
        self.variance = std_wind_speed
        self.velocity = 10

    def step(self, theta=0.0):
        time = self.length/self.velocity*math.cos(theta)
        x = time*(np.random.normal(self.mean, self.variance) +  self.velocity*math.sin(theta))

        return x 




