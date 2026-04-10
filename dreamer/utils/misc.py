"""
utils/misc.py — Core mathematical utilities for DreamerV3.

All functions here appear throughout the model. Getting these right is
critical — a bug here silently corrupts training everywhere.

Key paper references:
  - symlog/symexp: Section 2 "Symlog Predictions"
  - TwoHotDist:    Section 2 "Distributional Outputs"
  - MLP:           Appendix B architecture details
"""

import os
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Optional
from dataclasses import dataclass, field


# ── Symlog / Symexp ────────────────────────────────────────────────────────────

def symlog(x: torch.Tensor) -> torch.Tensor:
    """Symmetric log: sign(x) * log(|x| + 1).
    
    Compresses large values while preserving sign and small value identity.
    Used to normalize rewards and value targets across games with very
    different reward scales — a key reason DreamerV3 uses fixed hyperparameters.
    """
    return torch.sign(x) * torch.log(torch.abs(x) + 1.0)


def symexp(x: torch.Tensor) -> torch.Tensor:
    """Inverse of symlog: sign(x) * (exp(|x|) - 1).
    
    Used to decode symlog-space predictions back to real values.
    """
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1.0)


# ── Two-Hot Distributional Encoding ───────────────────────────────────────────

class TwoHotDist:
    """Distributional output using two-hot encoding over fixed bins.
    
    Instead of predicting a scalar reward or value, the network outputs
    a distribution over a fixed set of bins. The target is encoded as a
    "two-hot" vector — nonzero at only the two bins bracketing the target,
    with weights proportional to distance. Enables distributional RL.
    
    All bin operations happen in symlog space. Targets are symlog-transformed
    before encoding; predictions are symexp-transformed after decoding.
    
    Paper: bins span [-20, 20] in symlog space (255 bins).
    In real space this covers roughly [-4.85e8, 4.85e8] — any Atari reward.
    """

    def __init__(self, low: float = -20.0, high: float = 20.0, num_bins: int = 255):
        self.low = low
        self.high = high
        self.num_bins = num_bins
        # Fixed bin centers in symlog space. Registered as buffer in models.
        self._bins = torch.linspace(low, high, num_bins)

    def bins(self, device: torch.device) -> torch.Tensor:
        return self._bins.to(device)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode real-valued scalars to two-hot vectors in symlog space.
        
        Args:
            x: [...] real-valued targets
        Returns:
            [..., num_bins] two-hot distribution
        """
        bins = self.bins(x.device)
        x_sl = symlog(x).clamp(self.low, self.high)

        # How many bins are <= x_sl: gives lower bin index
        below = (bins.unsqueeze(0) <= x_sl.unsqueeze(-1)).sum(-1) - 1
        below = below.clamp(0, self.num_bins - 2)
        above = below + 1

        lower_val = bins[below]
        upper_val = bins[above]

        # Linear interpolation weights
        span = (upper_val - lower_val).clamp(min=1e-8)
        upper_weight = (x_sl - lower_val) / span
        lower_weight = 1.0 - upper_weight

        encoding = torch.zeros(*x_sl.shape, self.num_bins, device=x.device)
        encoding.scatter_(-1, below.unsqueeze(-1), lower_weight.unsqueeze(-1))
        encoding.scatter_(-1, above.unsqueeze(-1), upper_weight.unsqueeze(-1))
        return encoding

    def decode(self, logits: torch.Tensor) -> torch.Tensor:
        """Decode logits to real-valued scalar prediction.
        
        Args:
            logits: [..., num_bins] unnormalized
        Returns:
            [...] real-valued prediction (symexp applied)
        """
        bins = self.bins(logits.device)
        probs = F.softmax(logits, dim=-1)
        return symexp((probs * bins).sum(-1))

    def loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Cross-entropy loss between predicted logits and encoded targets.
        
        Args:
            logits:  [..., num_bins]
            targets: [...] real-valued scalar targets
        Returns:
            [...] per-element loss values (not yet reduced)
        """
        encoded = self.encode(targets)                        # [..., num_bins]
        log_probs = F.log_softmax(logits, dim=-1)             # [..., num_bins]
        return -(encoded * log_probs).sum(-1)                 # [...]


# ── MLP building block ─────────────────────────────────────────────────────────

class MLP(nn.Module):
    """Feed-forward MLP with SiLU activations.
    
    Used throughout DreamerV3 for prediction heads, actor, and critic.
    All hidden layers have `units` dimensions. No normalization (removed in V3).
    
    Args:
        in_dim:   input dimension
        out_dim:  output dimension (None = no output projection, returns hidden)
        units:    hidden layer width
        layers:   number of hidden layers
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: Optional[int],
        units: int,
        layers: int,
    ):
        super().__init__()
        hidden_layers: list[nn.Module] = []
        dim = in_dim
        for _ in range(layers):
            hidden_layers.append(nn.Linear(dim, units))
            hidden_layers.append(nn.SiLU())
            dim = units
        self.hidden = nn.Sequential(*hidden_layers)
        self.out = nn.Linear(dim, out_dim) if out_dim is not None else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.hidden(x)
        return self.out(h) if self.out is not None else h


# ── Return normalization ───────────────────────────────────────────────────────

class ReturnNormalizer:
    """EMA-based normalizer for imagined returns.
    
    Tracks the 5th and 95th percentile of returns using exponential moving
    averages. Divides returns by (P95 - P5), clamped to a minimum scale.
    This keeps the actor loss scale stable across games.
    
    Paper: decay=0.99, limit=1.0
    """

    def __init__(self, decay: float = 0.99, limit: float = 1.0):
        self.decay = decay
        self.limit = limit
        self._low: Optional[float] = None
        self._high: Optional[float] = None

    def update(self, returns: torch.Tensor) -> None:
        low = float(torch.quantile(returns.detach().float(), 0.05).item())
        high = float(torch.quantile(returns.detach().float(), 0.95).item())
        if self._low is None or self._high is None:
            self._low, self._high = low, high
        else:
            self._low = self.decay * self._low + (1 - self.decay) * low
            self._high = self.decay * self._high + (1 - self.decay) * high

    @property
    def scale(self) -> float:
        if self._low is None or self._high is None:
            return 1.0
        return max(self._high - self._low, self.limit)

    def normalize(self, returns: torch.Tensor) -> torch.Tensor:
        return returns / self.scale


# ── Config loading ─────────────────────────────────────────────────────────────

class Config:
    """Hierarchical config loaded from yaml files.
    
    Supports nested attribute access (cfg.model.deter) and merging
    multiple yaml files (base + size override).
    
    Uses __getattr__ returning Any so type checkers treat all attribute
    access as valid — appropriate for a dynamic config object.
    """

    def __init__(self, d: dict) -> None:
        for k, v in d.items():
            object.__setattr__(self, k, Config(v) if isinstance(v, dict) else v)

    def __getattr__(self, name: str) -> Any:
        # This is only called when normal attribute lookup fails.
        # Returning Any here tells Pylance that any attribute access is valid.
        raise AttributeError(f"Config has no attribute '{name}'")

    def __repr__(self) -> str:
        return f"Config({self.__dict__})"


def load_config(*yaml_paths: str) -> Config:
    """Load and merge yaml config files left to right (later overrides earlier).
    
    Usage:
        cfg = load_config('configs/base.yaml', 'configs/s_size.yaml')
    """
    merged: dict = {}
    for path in yaml_paths:
        with open(path) as f:
            data = yaml.safe_load(f)
        _deep_update(merged, data)
    return Config(merged)


def _deep_update(base: dict, override: dict) -> dict:
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def get_device(cfg: Config) -> torch.device:
    """Resolve device string from config, falling back gracefully."""
    requested = getattr(cfg, 'device', 'cpu')
    if requested == 'cuda':
        if torch.cuda.is_available():
            return torch.device('cuda')
        print("Warning: CUDA not available, falling back to CPU.")
        return torch.device('cpu')
    elif requested == 'mps':
        if torch.backends.mps.is_available():
            return torch.device('mps')
        print("Warning: MPS not available, falling back to CPU.")
        return torch.device('cpu')
    return torch.device('cpu')