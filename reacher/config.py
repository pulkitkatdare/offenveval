import numpy as np 

class reacherconfigs(object):
	def __init__(self, index = 1):
		assert (index == 1  or index ==2)
		self.xml_dir = str(index)
		self.episodes = 3000
		self.buffer = 200000
		self.time_steps = 100
		self.batch_size = 150000
		self.train = True
		self.test = True

		if self.test:
			self.eval_episodes = 100
			self.eval_timesteps = 100
			self.mle = True
			if self.mle:
				self.train_mle = True
			self.simulator = True
			self.oee = True
			self.true = True
			self.epsilons = [0.0, 0.2, 0.4, 0.6, 1.0]
			self.savetex = True 
			self.savepng = True
