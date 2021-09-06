import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as weight_init


class CriticNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, learning_rate, seed, batch_size):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.seed = seed
        self.batch_size = batch_size
        super(CriticNetwork, self).__init__()
        
        self.layer1 = nn.Linear(self.state_dim, 24)
        n = weight_init._calculate_fan_in_and_fan_out(self.layer1.weight)[0]
        torch.manual_seed(self.seed)
        self.layer1.weight.data.uniform_(-math.sqrt(6. / n), math.sqrt(6. / n))
        self.layer2 = nn.Linear(24, 24)
        n = weight_init._calculate_fan_in_and_fan_out(self.layer2.weight)[0]
        torch.manual_seed(self.seed)
        self.layer2.weight.data.uniform_(-math.sqrt(6. / n), math.sqrt(6. / n))
        self.layer3 = nn.Linear(24, action_dim)
        n = weight_init._calculate_fan_in_and_fan_out(self.layer3.weight)[0]
        torch.manual_seed(self.seed)
        self.layer3.weight.data.uniform_(-math.sqrt(6. / n), math.sqrt(6. / n))
        
        self.loss_fn = torch.nn.MSELoss(size_average=True)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=0.00)
    
    def forward(self, x):
        y = F.relu(self.layer1(x))
        y = F.relu(self.layer2(y))
        y = self.layer3(y)
        return y
    
    def train(self, states, actions, y):
        self.optimizer.zero_grad()
        q_value = self.forward(states)
        actions = actions.data.numpy().astype(int)
        range_array = np.array(range(self.batch_size))
        index_range = np.arange(self.batch_size)
        index_range = np.reshape(index_range, (1, self.batch_size))
        q_value = q_value[index_range, actions]
        loss = self.loss_fn(q_value, y)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss
