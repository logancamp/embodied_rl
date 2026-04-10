"""
training/replay_buffer.py — Replay buffer for DreamerV3.

DreamerV3 stores entire episode trajectories and samples contiguous
subsequences of length batch_length for training. This is important:
  - Sequences preserve temporal structure the RSSM needs to learn from
  - Random individual transitions (like DQN's buffer) would break this

Storage format:
  Each episode is stored as a dict of numpy arrays.
  We maintain a ring buffer of episodes (oldest evicted when full).

Sampling:
  1. Sample batch_size episodes (uniform, weighted by length)
  2. Sample a random contiguous subsequence of length batch_length from each
  3. Return stacked batch tensors

Memory:
  Each transition stores: obs (3*64*64*4 bytes ≈ 49KB), action (4B), 
  reward (4B), done (1B). At 1M steps: ~49GB naively.
  We store uint8 observations (not float) to reduce by 4x → ~12GB.
  Still large — reduce replay_size in debug config.
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple


class ReplayBuffer:
    """Episode-based replay buffer with subsequence sampling.
    
    Args:
        capacity:     maximum number of transitions stored
        obs_shape:    observation shape e.g. (3, 64, 64)
        num_actions:  number of discrete actions
        batch_size:   sequences per training batch
        batch_length: timesteps per sequence
        device:       device to place sampled tensors on
    """

    def __init__(
        self,
        capacity: int,
        obs_shape: Tuple[int, ...],
        num_actions: int,
        batch_size: int,
        batch_length: int,
        device: torch.device,
    ):
        self.capacity = capacity
        self.obs_shape = obs_shape
        self.num_actions = num_actions
        self.batch_size = batch_size
        self.batch_length = batch_length
        self.device = device

        self._episodes: List[Dict[str, np.ndarray]] = []
        self._total_steps = 0

        # Temporary storage for the current in-progress episode
        self._current_obs: List[np.ndarray] = []
        self._current_actions: List[int] = []
        self._current_rewards: List[float] = []
        self._current_dones: List[bool] = []
        self._current_is_first: List[bool] = []

    # ── Episode building ───────────────────────────────────────────────────────

    def start_episode(self, obs: np.ndarray) -> None:
        """Call at the start of each episode with the initial observation."""
        self._current_obs = [obs]
        self._current_actions = []
        self._current_rewards = []
        self._current_dones = []
        self._current_is_first = [True]

    def add_step(
        self,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        """Add a single transition to the current episode."""
        self._current_actions.append(action)
        self._current_rewards.append(reward)
        self._current_obs.append(next_obs)
        self._current_dones.append(done)
        self._current_is_first.append(False)  # only first step is is_first=True

    def end_episode(self) -> int:
        """Finalise and store the current episode. Returns its length."""
        T = len(self._current_actions)
        if T < 2:
            # Episode too short to be useful at all — discard
            self._current_obs.clear()
            return 0

        episode: Dict[str, np.ndarray] = {
            # obs: [T+1, *obs_shape] uint8 (saves 4x vs float32)
            'obs':      np.array(
                [(o * 255).astype(np.uint8) for o in self._current_obs],
                dtype=np.uint8,
            ),
            'action':   np.array(self._current_actions, dtype=np.int32),     # [T]
            'reward':   np.array(self._current_rewards, dtype=np.float32),   # [T]
            'done':     np.array(self._current_dones, dtype=np.bool_),       # [T]
            'is_first': np.array(self._current_is_first[:-1], dtype=np.bool_),  # [T]
        }

        self._episodes.append(episode)
        self._total_steps += T

        # Evict oldest episodes when over capacity
        while self._total_steps > self.capacity and len(self._episodes) > 1:
            removed = self._episodes.pop(0)
            self._total_steps -= len(removed['action'])

        return T

    # ── Sampling ───────────────────────────────────────────────────────────────

    def sample(self) -> Dict[str, torch.Tensor]:
        """Sample a batch of contiguous subsequences.
        
        Returns dict with keys:
            obs:      [B, T, C, H, W]   float32 in [0, 1]
            action:   [B, T, num_actions] float32 one-hot
            reward:   [B, T]             float32
            done:     [B, T]             float32 (0 or 1)
            is_first: [B, T]             float32 (0 or 1)
        where B = batch_size, T = batch_length
        """
        assert self.ready(), (
            f"Buffer not ready: need episodes with length >= {self.batch_length}, "
            f"have {len(self._episodes)} episodes."
        )

        # Sample episodes, weighted by length (longer episodes contribute more starts)
        lengths = np.array([len(ep['action']) for ep in self._episodes], dtype=np.float64)
        lengths = np.maximum(lengths - self.batch_length + 1, 0)  # valid start positions
        probs = lengths / lengths.sum()

        obs_list, action_list, reward_list, done_list, isfirst_list = [], [], [], [], []

        for _ in range(self.batch_size):
            ep_idx = np.random.choice(len(self._episodes), p=probs)
            ep = self._episodes[ep_idx]
            T_ep = len(ep['action'])
            # Random start position in valid range
            start = np.random.randint(0, T_ep - self.batch_length + 1)
            end = start + self.batch_length

            obs_slice = ep['obs'][start:end + 1]      # T+1 for shifted obs
            obs_list.append(obs_slice[:-1])            # [T, C, H, W]
            action_list.append(ep['action'][start:end])
            reward_list.append(ep['reward'][start:end])
            done_list.append(ep['done'][start:end])
            isfirst_list.append(ep['is_first'][start:end])

        def to_tensor(arr: list, dtype: torch.dtype) -> torch.Tensor:
            return torch.tensor(np.array(arr), dtype=dtype, device=self.device)

        obs = to_tensor(obs_list, torch.uint8).float() / 255.0   # [B, T, C, H, W]
        actions_idx = to_tensor(action_list, torch.long)          # [B, T]
        # One-hot encode actions
        actions_oh = torch.zeros(
            *actions_idx.shape, self.num_actions,
            device=self.device, dtype=torch.float32
        )
        actions_oh.scatter_(-1, actions_idx.unsqueeze(-1), 1.0)   # [B, T, num_actions]

        return {
            'obs':      obs,                                         # [B, T, C, H, W]
            'action':   actions_oh,                                  # [B, T, num_actions]
            'reward':   to_tensor(reward_list, torch.float32),       # [B, T]
            'done':     to_tensor(done_list, torch.float32),         # [B, T]
            'is_first': to_tensor(isfirst_list, torch.float32),      # [B, T]
        }

    def ready(self) -> bool:
        """True if buffer has enough data to yield a full batch."""
        valid = [ep for ep in self._episodes if len(ep['action']) >= self.batch_length]
        return len(valid) >= 1  # we can repeat-sample if needed

    @property
    def num_steps(self) -> int:
        return self._total_steps

    @property
    def num_episodes(self) -> int:
        return len(self._episodes)