import gymnasium as gym # type: ignore
import numpy as np
from gymnasium.wrappers import TimeLimit # type: ignore


def strip_timelimit(env):
    """Remove the TimeLimit wrapper if present."""
    if isinstance(env, TimeLimit):
        return env.env
    return env


class MetricsPrinter(gym.Wrapper):
    """Prints custom metrics on episode end. Suppresses ppo.py's generic print."""
    def __init__(self, env, fmt_fn):
        super().__init__(env)
        self._fmt_fn    = fmt_fn
        self._ep_reward = 0.0
        self._ep_len    = 0
        self._ep_count  = 0

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._ep_reward += float(reward)
        self._ep_len    += 1

        if terminated or truncated:
            self._ep_count += 1
            print(self._fmt_fn(self._ep_count, self._ep_len, self._ep_reward))
            self._ep_reward = 0.0
            self._ep_len    = 0

        return obs, reward, terminated, truncated, info


class PhaseSwitch(gym.Wrapper):
    """
    Runs with TimeLimit until avg episode reward exceeds threshold,
    then strips it and runs forever.
    """
    def __init__(self, env, threshold, window=10):
        super().__init__(env)
        self.threshold  = threshold
        self.window     = window
        self._rewards   = []
        self._ep_reward = 0.0
        self._switched  = False

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._ep_reward += reward

        if terminated or truncated:
            self._rewards.append(self._ep_reward)
            self._ep_reward = 0.0
            if len(self._rewards) > self.window:
                self._rewards.pop(0)

            if (not self._switched
                    and len(self._rewards) >= self.window
                    and np.mean(self._rewards) > self.threshold):
                self.env = strip_timelimit(self.env)
                self._switched = True
                print('  Phase switch — TimeLimit removed, running forever')

        return obs, reward, terminated, truncated, info
    
    
class ShapedMountainCar(gym.Wrapper):
    def step(self, action):
        obs, reward, term, trunc, info = self.env.step(action)
        pos, vel = obs
        reward += 0.01 * np.sin(3 * pos)  # height-based shaping
        return obs, reward, term, trunc, info