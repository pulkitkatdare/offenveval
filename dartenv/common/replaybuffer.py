import numpy as np
import random
from collections import deque

class ReplayBuffer(object):
    def __init__(self, buffer_size, random_seed=None):
        """
        The right side of the deque contains the most recent experiences
        The buffer stores a number of past experiences to stochastically sample from
        """
        self.buffer_size = buffer_size
        self.count = 0
        self.buffer = deque(maxlen=self.buffer_size)
        self.seed = random_seed
        if self.seed is not None:
            random.seed(self.seed)
        self.terminal_included = False

    def add(self, action, s2):
        experience = (action, s2)
        self.buffer.append(experience)
        self.count += 1

    def size(self):
        return self.count

    def sample_batch(self, batch_size):
        if self.count < batch_size:
            batch = random.sample(self.buffer, self.count)
        else:
            batch = random.sample(self.buffer, batch_size)

        a_batch = np.array([_[0] for _ in batch])
        s_batch = np.array([_[1] for _ in batch])

        return a_batch, s_batch

    def clear(self):
        self.buffer.clear()
        self.count = 0
