import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as weight_init
from reacher.common.replaybuffer import ReplayBuffer


class DynamicModel(nn.Module):
    def __init__(self, state_dim, action_dim, learning_rate, reg = 0.0, seed = 1234):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.reg = reg
        self.seed = seed

        super(DynamicModel, self).__init__()
        torch.manual_seed(self.seed)
        self.layer1 = nn.Linear(self.state_dim + self.action_dim, 256)
        self.layer2 = nn.Linear(256, 512)
        self.layer3 = nn.Linear(512, 256)
        self.layer4 = nn.Linear(256, state_dim)
        torch.manual_seed(self.seed)

        self.loss_fn = torch.nn.MSELoss(size_average=True)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.reg)

    def forward(self, x):
        y = F.relu(self.layer1(x))
        y = F.relu(self.layer2(y))
        y = F.relu(self.layer3(y))
        y = self.layer4(y)
        y1 = 0.6*torch.tanh(y[:,0:1])
        y2 = 0.6*torch.tanh(y[:,1:2])
        y3 = 0.6*torch.tanh(y[:,2:3])
        y4 = 0.6*torch.tanh(y[:,3:4])
        y5 = torch.tanh(y[:, 4:5])
        y6 = torch.tanh(y[:, 5:6])
        y7 = 10*torch.tanh(y[:, 6:7])
        y8 = torch.tanh(y[:, 7:8])
        y9 = 10*torch.tanh(y[:, 8:9])

        activated = (y1, y2, y3, y4, y5, y6, y7, y8, y9)
        y = torch.cat(activated, dim=1)

        return y

    def train_step(self, input, output):
        self.optimizer.zero_grad()
        y_hat = self.forward(input)
        loss = self.loss_fn(y_hat, output)
        for params in self.parameters():
                loss += 0.5*self.reg*torch.norm(params)**2
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        return loss
