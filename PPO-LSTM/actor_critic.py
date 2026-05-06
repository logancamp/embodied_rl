import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal
import numpy as np

def layer_init(layer, std=np.sqrt(2), bias=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias)
    return layer


class CNNExtractor(nn.Module):
    def __init__(self, obs_shape, out_dim=512):
        super().__init__()
        c, h, w = obs_shape
        self.c = c
        self.cnn = nn.Sequential(
            layer_init(nn.Conv2d(c,  32, kernel_size=8, stride=4)), nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2)), nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1)), nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, c, h, w)
            flat  = self.cnn(dummy).shape[1]
        self.fc = nn.Sequential(
            layer_init(nn.Linear(flat, out_dim)), nn.ReLU()
        )
        self.out_dim = out_dim

    def forward(self, x):
        if x.max() > 1.0:
            x = x / 255.0
        if x.dim() == 3:
            x = x.unsqueeze(0)
        if x.shape[1] != self.c and x.shape[-1] == self.c:
            x = x.permute(0, 3, 1, 2)
        return self.fc(self.cnn(x))


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=64, continuous=False,
                 obs_shape=None, use_lstm=False):
        """
        obs_shape:  (C, H, W) for image obs, None for vector obs.
        use_lstm:   replace shared MLP trunk with LSTM for memory across timesteps.
        """
        super().__init__()
        self.continuous = continuous
        self.is_visual  = obs_shape is not None
        self.use_lstm   = use_lstm
        self.hidden     = hidden

        if self.is_visual:
            self.extractor = CNNExtractor(obs_shape, out_dim=512)
            feat_dim = 512
        else:
            self.extractor = None
            feat_dim = obs_dim

        if use_lstm:
            self.lstm = nn.LSTM(feat_dim, hidden, batch_first=True)
            for name, param in self.lstm.named_parameters():
                if 'bias' in name:
                    nn.init.constant_(param, 0)
                elif 'weight' in name:
                    nn.init.orthogonal_(param)
        else:
            # original shared MLP trunk — unchanged
            self.shared = nn.Sequential(
                layer_init(nn.Linear(feat_dim, hidden)),
                nn.Tanh(),
                layer_init(nn.Linear(hidden, hidden)),
                nn.Tanh(),
            )

        # Critic
        self.critic = layer_init(nn.Linear(hidden, 1), std=1.0)

        # Actor
        if continuous:
            self.actor_mean   = layer_init(nn.Linear(hidden, act_dim), std=0.01)
            self.actor_logstd = nn.Parameter(torch.zeros(act_dim))
        else:
            self.actor = layer_init(nn.Linear(hidden, act_dim), std=0.01)

    def _extract(self, x):
        if self.extractor is not None:
            return self.extractor(x)
        return x

    def get_initial_state(self, device):
        h = torch.zeros(1, 1, self.hidden).to(device)
        c = torch.zeros(1, 1, self.hidden).to(device)
        return h, c

    def _trunk(self, features, hx=None, cx=None):
        if self.use_lstm:
            # ensure features is batched: (feat,) -> (1, feat)
            if features.dim() == 1:
                features = features.unsqueeze(0)
            # (batch, feat) -> (batch, 1, feat) for seq_len=1
            out, (hx, cx) = self.lstm(features.unsqueeze(1), (hx, cx))
            out = out.squeeze(1)  # (batch, hidden)
            return out, hx, cx
        else:
            return self.shared(features), None, None

    def get_value(self, x, hx=None, cx=None):
        features = self._extract(x)
        out, _, _ = self._trunk(features, hx, cx)
        return self.critic(out)

    def get_action_and_value(self, x, action=None, hx=None, cx=None):
        features = self._extract(x)
        out, hx, cx = self._trunk(features, hx, cx)
        value = self.critic(out)

        if self.continuous:
            mean = self.actor_mean(out)
            std  = self.actor_logstd.exp().expand_as(mean)
            dist = Normal(mean, std)
            if action is None:
                action = dist.sample()
            if action.dim() == 1 and mean.dim() == 2:
                action = action.unsqueeze(-1)
            log_prob = dist.log_prob(action).sum(-1)
            entropy  = dist.entropy().sum(-1)
        else:
            logits = self.actor(out)
            dist   = Categorical(logits=logits)
            if action is None:
                action = dist.sample()
            log_prob = dist.log_prob(action)
            entropy  = dist.entropy()

        return action, log_prob, entropy, value, hx, cx