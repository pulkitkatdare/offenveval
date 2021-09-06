import numpy as np 

class gridworldconfigs(object):
	def __init__(self):
		self.P_parameters = 0.9
		self.Q_parameters = 0.7
		self.size = [10, 10]
		self.episodes = 2000
		self.buffer = 200000
		self.time_steps = 200
		self.batches = [1000, 10**(3.5), 10**(4.0), 10**(4.5), 10**(5), 10**(5.5)]
		self.train = True
		self.test = True
		if self.test:
			self.epsilons = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9]
			self.eval_batch = int(10**(5))