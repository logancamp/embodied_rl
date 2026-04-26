import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal
import numpy as np

def layer_init(layer, std=np.sqrt(2), bias=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias)
    return layer

class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=64, continuous=False):
        super().__init__()
        self.continuous = continuous

        # Shared feature extractor
        self.shared = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden, hidden)),
            nn.Tanh(),
        )

        # Critic head — lower std init for value stability
        self.critic = layer_init(nn.Linear(hidden, 1), std=1.0)

        # Actor head
        if continuous:
            self.actor_mean = layer_init(nn.Linear(hidden, act_dim), std=0.01)
            self.actor_logstd = nn.Parameter(torch.zeros(act_dim))
        else:
            self.actor = layer_init(nn.Linear(hidden, act_dim), std=0.01)

    def get_value(self, x):
        return self.critic(self.shared(x))

    def get_action_and_value(self, x, action=None):
        features = self.shared(x)
        value = self.critic(features)

        if self.continuous:
            mean = self.actor_mean(features)
            std  = self.actor_logstd.exp().expand_as(mean)
            dist = Normal(mean, std)
            if action is None:
                action = dist.sample()
            if action.dim() == 1 and mean.dim() == 2:
                action = action.unsqueeze(-1)
            log_prob = dist.log_prob(action).sum(-1)
            entropy  = dist.entropy().sum(-1)
        else:
            logits = self.actor(features)
            dist = Categorical(logits=logits)
            if action is None:
                action = dist.sample()
            log_prob = dist.log_prob(action)
            entropy = dist.entropy()

        return action, log_prob, entropy, value