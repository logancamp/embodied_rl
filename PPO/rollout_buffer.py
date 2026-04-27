import torch

class RolloutBuffer:
    def __init__(self, steps, obs_dim, act_dim, device, continuous=False, obs_shape=None):
        if obs_shape is not None:
            self.obs = torch.zeros(steps, *obs_shape).to(device)
        else:
            self.obs = torch.zeros(steps, obs_dim).to(device)
        self.actions = torch.zeros(steps, act_dim if continuous else 1).to(device)
        self.logprobs = torch.zeros(steps).to(device)
        self.rewards = torch.zeros(steps).to(device)
        self.dones = torch.zeros(steps).to(device)
        self.values = torch.zeros(steps).to(device)
        self.ptr = 0

    def push(self, obs, action, logprob, reward, done, value):
        self.obs[self.ptr] = obs
        self.actions[self.ptr] = action
        self.logprobs[self.ptr] = logprob
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done
        self.values[self.ptr] = value.squeeze()
        self.ptr += 1

    def reset(self): self.ptr = 0