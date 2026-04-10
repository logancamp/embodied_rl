"""
utils/logging.py — Metrics tracking, checkpointing, and episode logging.

Keeps training clean by centralising all I/O. The Logger owns the log
directory, writes JSON-lines for easy post-hoc analysis, and saves/loads
checkpoints as plain PyTorch state dicts.
"""

import os
import json
import time
import torch
import numpy as np
from collections import defaultdict
from typing import Any, Dict, Optional


class MetricsLogger:
    """Accumulates scalar metrics and flushes them periodically.
    
    Metrics are written as JSON-lines to {logdir}/metrics.jsonl.
    Each line is a flat dict with a 'step' key and all accumulated metrics.
    
    Usage:
        logger = MetricsLogger(logdir='logs/run')
        logger.log('train/reward_loss', 0.42, step=1000)
        logger.flush(step=1000)   # writes accumulated metrics to disk
    """

    def __init__(self, logdir: str):
        os.makedirs(logdir, exist_ok=True)
        self.logdir = logdir
        self._buffer: Dict[str, list] = defaultdict(list)
        self._metrics_path = os.path.join(logdir, 'metrics.jsonl')
        self._start_time = time.time()
        self._last_flush_step = 0

    def log(self, key: str, value: Any, step: Optional[int] = None) -> None:
        """Buffer a scalar metric. Values are averaged when flushed."""
        if isinstance(value, torch.Tensor):
            value = value.detach().float().mean().item()
        elif isinstance(value, np.ndarray):
            value = float(value.mean())
        self._buffer[key].append(float(value))

    def flush(self, step: int) -> Dict[str, float]:
        """Average buffered metrics, write to disk, print to console, return dict."""
        if not self._buffer:
            return {}

        row: Dict[str, float] = {'step': step}
        for key, values in self._buffer.items():
            row[key] = float(np.mean(values))
        row['fps'] = (step - self._last_flush_step) / max(
            time.time() - self._start_time + 1e-8, 1e-8
        )

        # Write to disk
        with open(self._metrics_path, 'a') as f:
            f.write(json.dumps(row) + '\n')

        self._buffer.clear()
        self._last_flush_step = step
        self._start_time = time.time()
        return row

    @staticmethod
    def _format(row: Dict[str, float]) -> str:
        step = int(row['step'])
        parts = [f"Step {step:>8d}"]
        priority_keys = [
            'eval/mean_return', 'eval/mean_episode_length',
            'wm/loss', 'wm/kl', 'wm/reward_loss', 'wm/recon_loss',
            'ac/actor_loss', 'ac/critic_loss', 'ac/entropy',
        ]
        shown = set()
        for k in priority_keys:
            if k in row:
                parts.append(f"{k.split('/')[-1]}: {row[k]:.4f}")
                shown.add(k)
        # Show remaining metrics not already shown (except step/fps)
        for k, v in row.items():
            if k not in shown and k not in ('step', 'fps'):
                parts.append(f"{k}: {v:.4f}")
        parts.append(f"fps: {row.get('fps', 0):.0f}")
        return "  |  ".join(parts)


class EpisodeLogger:
    """Tracks per-episode statistics during rollouts."""

    def __init__(self):
        self._episode_returns: list[float] = []
        self._episode_lengths: list[int] = []
        self._current_return = 0.0
        self._current_length = 0

    def step(self, reward: float, done: bool) -> None:
        self._current_return += reward
        self._current_length += 1
        if done:
            self._episode_returns.append(self._current_return)
            self._episode_lengths.append(self._current_length)
            self._current_return = 0.0
            self._current_length = 0

    def pop(self) -> Dict[str, float]:
        """Return and clear accumulated episode stats."""
        if not self._episode_returns:
            return {}
        stats = {
            'train/mean_episode_return': float(np.mean(self._episode_returns)),
            'train/mean_episode_length': float(np.mean(self._episode_lengths)),
            'train/num_episodes': len(self._episode_returns),
        }
        self._episode_returns.clear()
        self._episode_lengths.clear()
        return stats


# ── Checkpointing ──────────────────────────────────────────────────────────────

def save_checkpoint(
    logdir: str,
    step: int,
    world_model: torch.nn.Module,
    actor: torch.nn.Module,
    critic: torch.nn.Module,
    wm_opt: torch.optim.Optimizer,
    actor_opt: torch.optim.Optimizer,
    critic_opt: torch.optim.Optimizer,
) -> None:
    """Save all model and optimizer states to a single checkpoint file."""
    path = os.path.join(logdir, f'checkpoint_{step:08d}.pt')
    torch.save({
        'step': step,
        'world_model': world_model.state_dict(),
        'actor': actor.state_dict(),
        'critic': critic.state_dict(),
        'wm_opt': wm_opt.state_dict(),
        'actor_opt': actor_opt.state_dict(),
        'critic_opt': critic_opt.state_dict(),
    }, path)
    print(f"Checkpoint saved: {path}")

    # Keep only the last 3 checkpoints to save disk space
    _prune_checkpoints(logdir, keep=3)


def load_checkpoint(
    path: str,
    world_model: torch.nn.Module,
    actor: torch.nn.Module,
    critic: torch.nn.Module,
    wm_opt: torch.optim.Optimizer,
    actor_opt: torch.optim.Optimizer,
    critic_opt: torch.optim.Optimizer,
    device: torch.device,
) -> int:
    """Load checkpoint. Returns the step number."""
    ckpt = torch.load(path, map_location=device)
    world_model.load_state_dict(ckpt['world_model'])
    actor.load_state_dict(ckpt['actor'])
    critic.load_state_dict(ckpt['critic'])
    wm_opt.load_state_dict(ckpt['wm_opt'])
    actor_opt.load_state_dict(ckpt['actor_opt'])
    critic_opt.load_state_dict(ckpt['critic_opt'])
    print(f"Loaded checkpoint from step {ckpt['step']}")
    return int(ckpt['step'])


def latest_checkpoint(logdir: str) -> Optional[str]:
    """Return path to the most recent checkpoint in logdir, or None."""
    if not os.path.exists(logdir):
        return None
    checkpoints = sorted([
        f for f in os.listdir(logdir) if f.startswith('checkpoint_') and f.endswith('.pt')
    ])
    return os.path.join(logdir, checkpoints[-1]) if checkpoints else None


def _prune_checkpoints(logdir: str, keep: int) -> None:
    checkpoints = sorted([
        f for f in os.listdir(logdir) if f.startswith('checkpoint_') and f.endswith('.pt')
    ])
    for old in checkpoints[:-keep]:
        os.remove(os.path.join(logdir, old))