# 7-16-2019 Added documentation while learning the code

import torch
import torch.nn as nn
from torch.distributions import Normal

class ActorCritic(nn.Module):
    def __init__(self, num_inputs, num_outputs, hidden_size, std=0.0):
        super(ActorCritic, self).__init__()
        
        # Seperate networks for actor and for critic
        # Both are simple 1-layer FC networks

        self.critic = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        
        self.actor = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_outputs),
        )

        self.log_std = nn.Parameter(torch.ones(1, num_outputs) * std)
        
    def forward(self, x):
        value = self.critic(x)
        mu    = self.actor(x)

        # Something like softmax
        std   = self.log_std.exp().expand_as(mu)
        dist  = Normal(mu, std)
        
        return dist, value
