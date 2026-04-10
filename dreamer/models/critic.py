"""
models/critic.py — Critic (value function) for DreamerV3.

The critic estimates the expected lambda-return from a latent state.
Like the reward head, it uses a two-hot distributional output — this gives
it calibrated uncertainty estimates and handles value scale variation.

Two networks:
  critic:        trained with gradient descent
  target_critic: EMA copy of critic, used for stable bootstrap targets

The target network is NOT a separate module with its own optimizer.
It is updated manually every step via EMA:
  target = ema_decay * target + (1 - ema_decay) * critic

Architecture: identical to RewardHead — MLP + two-hot output.
"""

import copy
import torch
import torch.nn as nn
from utils.misc import MLP, TwoHotDist


class Critic(nn.Module):
    """Distributional value function critic.
    
    Args:
        latent_dim: flat RSSM state dimension
        units:      MLP hidden width
        layers:     number of MLP hidden layers
        bins:       two-hot bin count
        low, high:  bin range in symlog space
        ema_decay:  EMA decay for target network
    """

    def __init__(
        self,
        latent_dim: int,
        units: int = 512,
        layers: int = 2,
        bins: int = 255,
        low: float = -20.0,
        high: float = 20.0,
        ema_decay: float = 0.98,
    ):
        super().__init__()
        self.dist = TwoHotDist(low=low, high=high, num_bins=bins)
        self.ema_decay = ema_decay

        self.mlp = MLP(latent_dim, bins, units, layers)

        # Target critic: copy of mlp weights, updated by EMA not gradients
        # We keep it as a separate module so its parameters are excluded
        # from the optimizer.
        self._target_mlp = copy.deepcopy(self.mlp)
        for p in self._target_mlp.parameters():
            p.requires_grad_(False)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """Return value distribution logits from online critic.
        
        Args:
            latent: [..., latent_dim]
        Returns:
            [..., bins] logits
        """
        return self.mlp(latent)

    def predict(self, latent: torch.Tensor) -> torch.Tensor:
        """Return scalar value estimate (online critic).
        
        Args:
            latent: [..., latent_dim]
        Returns:
            [...] real-valued value estimates
        """
        return self.dist.decode(self.forward(latent))

    def target_predict(self, latent: torch.Tensor) -> torch.Tensor:
        """Return scalar value estimate from target critic (no grad).
        
        Args:
            latent: [..., latent_dim]
        Returns:
            [...] real-valued value estimates
        """
        with torch.no_grad():
            logits = self._target_mlp(latent)
        return self.dist.decode(logits)

    def loss(self, latent: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Two-hot cross-entropy loss against lambda-return targets.
        
        We use the ONLINE critic logits against the STOP-GRADIENT of targets.
        This is the standard distributional RL update.
        
        Args:
            latent:  [..., latent_dim]
            targets: [...] real-valued lambda-return targets (stop-grad applied here)
        Returns:
            scalar mean loss
        """
        logits = self.forward(latent)
        return self.dist.loss(logits, targets.detach()).mean()

    @torch.no_grad()
    def update_target(self) -> None:
        """EMA update of target critic from online critic.
        
        Call once per training step AFTER the optimizer step.
        """
        for online, target in zip(self.mlp.parameters(), self._target_mlp.parameters()):
            target.data.lerp_(online.data, 1.0 - self.ema_decay)
            # lerp_(end, weight) = (1-weight)*self + weight*end
            # So: target = ema_decay * target + (1-ema_decay) * online ✓
