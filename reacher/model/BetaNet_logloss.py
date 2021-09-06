import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as weight_init
from MountainCar.common.replaybuffer import ReplayBuffer


class BetaNetwork(nn.Module):
    def __init__(self, state_dim, action_bound, learning_rate, tau, seed, action_dim=1):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bound = action_bound
        self.learning_rate = learning_rate
        self.tau = tau
        self.seed = seed

        super(BetaNetwork, self).__init__()
        torch.manual_seed(self.seed)
        self.layer1 = nn.Linear(self.state_dim, 256)
        torch.manual_seed(self.seed)
        self.layer2 = nn.Linear(256, 512)
        torch.manual_seed(self.seed)
        self.layer3 = nn.Linear(512, 256)
        torch.manual_seed(self.seed)
        self.layer4 = nn.Linear(256, 1)
        torch.manual_seed(self.seed)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=0.00)

    def forward(self, x):
        y = F.relu(self.layer1(x))
        y = F.relu(self.layer2(y))
        y = F.relu(self.layer3(y))
        y = torch.sigmoid(self.layer4(y))

        scaled_y = y

        return x, y, scaled_y

    def predict(self, x):
        x, y, scaled_y = self.forward(x)
        return scaled_y

    def train_step(self, states_p, states_q):
        self.optimizer.zero_grad()
        output1 = self.predict(states_p)
        output1 = torch.mean(torch.log(output1))
        output2 = torch.log(1 - self.predict(states_q))
        output2 = torch.mean((output2))
        output = output1 - output2
        for param in self.parameters():
            output += 0.5 * self.tau * torch.norm(param) ** 2
        output.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        return output
