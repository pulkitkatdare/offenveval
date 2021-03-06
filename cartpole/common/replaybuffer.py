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

    def add(self, state, action, s2, reward, done=None):
        if done is not None:
            self.terminal_included = True
            experience = (state, action, s2, reward, done)
        else:
            experience = (state, action, s2, reward)
        self.buffer.append(experience)
        self.count += 1

    def size(self):
        return self.count

    def sample_batch(self, batch_size):
        if self.count < batch_size:
            batch = random.sample(self.buffer, self.count)
        else:
            batch = random.sample(self.buffer, batch_size)

        s_batch = np.array([_[0] for _ in batch])
        a_batch = np.array([_[1] for _ in batch])
        s2_batch = np.array([_[2] for _ in batch])
        reward = np.array([_[3] for _ in batch])
        if self.terminal_included:
            done = np.array([_[4] for _ in batch])
            return s_batch, a_batch, s2_batch, reward, done
        else:
            return s_batch, a_batch, s2_batch, reward

    def clear(self):
        self.buffer.clear()
        self.count = 0
