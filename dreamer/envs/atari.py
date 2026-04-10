"""
envs/atari.py — Atari environment wrapper for DreamerV3.

Matches the preprocessing described in the DreamerV3 paper exactly:
  - 64x64 RGB observations (not grayscale, no frame stacking)
  - 4-frame action repeat
  - Terminal on life loss during training
  - No-op starts (up to 30 frames)
  - Observations normalised to [0, 1]

DreamerV3 does NOT stack frames. The RSSM's recurrent deterministic state
handles temporal context instead. One clean frame per step.
"""

import numpy as np
import gymnasium as gym
import ale_py
from PIL import Image
from gymnasium import spaces
from typing import Tuple, Dict, Optional

# Register ALE environments with gymnasium.
# Must be called before gym.make('ALE/...') or the ALE namespace won't be found.
gym.register_envs(ale_py)


class AtariEnv:
    """Atari environment with DreamerV3 preprocessing."""

    def __init__(
        self,
        name: str,
        action_repeat: int = 4,
        obs_size: int = 64,
        terminal_on_life_loss: bool = True,
        noop_max: int = 30,
        seed: Optional[int] = None,
    ):
        self._action_repeat = action_repeat
        self._obs_size = obs_size
        self._terminal_on_life_loss = terminal_on_life_loss
        self._noop_max = noop_max
        self._seed = seed

        self._env = gym.make(
            name,
            obs_type="rgb",
            frameskip=1,
            repeat_action_probability=0.0,
            full_action_space=False,
            render_mode=None,
        )

        self._lives: int = 0
        self._episode_return: float = 0.0
        self._episode_length: int = 0

    @property
    def observation_space(self) -> spaces.Box:
        return spaces.Box(0.0, 1.0, (3, self._obs_size, self._obs_size), dtype=np.float32)

    @property
    def action_space(self) -> spaces.Discrete:
        return self._env.action_space  # type: ignore[return-value]

    @property
    def num_actions(self) -> int:
        return int(self._env.action_space.n)  # type: ignore[attr-defined]

    def reset(self) -> Tuple[np.ndarray, Dict]:
        obs, info = self._env.reset(seed=self._seed)
        self._seed = None

        noop_count = np.random.randint(1, self._noop_max + 1)
        for _ in range(noop_count):
            obs, _, terminated, truncated, info = self._env.step(0)
            if terminated or truncated:
                obs, info = self._env.reset()

        self._lives = int(info.get('lives', 0))
        self._episode_return = 0.0
        self._episode_length = 0
        return self._process_obs(np.asarray(obs)), dict(info)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        total_reward = 0.0
        last_obs: np.ndarray = np.zeros((210, 160, 3), dtype=np.uint8)
        terminated = False
        truncated = False
        info: Dict = {}

        for _ in range(self._action_repeat):
            raw_obs, reward, terminated, truncated, info = self._env.step(action)
            last_obs = np.asarray(raw_obs)
            total_reward += float(reward)
            if terminated or truncated:
                break

        self._episode_return += total_reward
        self._episode_length += 1

        done = terminated or truncated
        if self._terminal_on_life_loss:
            new_lives = int(info.get('lives', 0))
            if new_lives < self._lives:
                done = True
            self._lives = new_lives

        if terminated or truncated:
            info['episode_return'] = self._episode_return
            info['episode_length'] = self._episode_length

        return self._process_obs(last_obs), total_reward, done, terminated or truncated, info

    def close(self) -> None:
        self._env.close()

    def _process_obs(self, obs: np.ndarray) -> np.ndarray:
        img = Image.fromarray(obs)
        img = img.resize((self._obs_size, self._obs_size), Image.Resampling.BILINEAR)
        arr = np.array(img, dtype=np.float32) / 255.0
        return np.transpose(arr, (2, 0, 1))


def make_eval_env(cfg) -> AtariEnv:
    return AtariEnv(
        name=cfg.env.name,
        action_repeat=cfg.env.action_repeat,
        obs_size=cfg.env.obs_size,
        terminal_on_life_loss=False,
        noop_max=1,
        seed=42,
    )


def make_train_env(cfg, seed: Optional[int] = None) -> AtariEnv:
    return AtariEnv(
        name=cfg.env.name,
        action_repeat=cfg.env.action_repeat,
        obs_size=cfg.env.obs_size,
        terminal_on_life_loss=cfg.env.terminal_on_life_loss,
        noop_max=cfg.env.noop_max,
        seed=seed,
    )