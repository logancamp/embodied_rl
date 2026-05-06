import torch

class RolloutBuffer:
    def __init__(self, steps, obs_dim, act_dim, device, continuous=False, obs_shape=None, hidden_size=None):
        if obs_shape is not None:
            self.obs = torch.zeros(steps, *obs_shape).to(device)
        else:
            self.obs = torch.zeros(steps, obs_dim).to(device)
        self.actions  = torch.zeros(steps, act_dim if continuous else 1).to(device)
        self.logprobs = torch.zeros(steps).to(device)
        self.rewards  = torch.zeros(steps).to(device)
        self.dones    = torch.zeros(steps).to(device)
        self.values   = torch.zeros(steps).to(device)

        # LSTM hidden states — None if not using LSTM
        if hidden_size is not None:
            self.hx: torch.Tensor | None = torch.zeros(steps, hidden_size).to(device)
            self.cx: torch.Tensor | None = torch.zeros(steps, hidden_size).to(device)
        else:
            self.hx: torch.Tensor | None = None
            self.cx: torch.Tensor | None = None

        self.ptr = 0

    def push(self, obs, action, logprob, reward, done, value, hx=None, cx=None):
        self.obs[self.ptr]      = obs
        self.actions[self.ptr]  = action
        self.logprobs[self.ptr] = logprob
        self.rewards[self.ptr]  = reward
        self.dones[self.ptr]    = done
        self.values[self.ptr]   = value.squeeze()
        if self.hx is not None and hx is not None:
            self.hx[self.ptr] = hx.squeeze()
            self.cx[self.ptr] = cx.squeeze()  # type: ignore
        self.ptr += 1

    def reset(self): self.ptr = 0