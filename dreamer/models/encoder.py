"""
models/encoder.py — CNN image encoder for DreamerV3.

Architecture (from paper Appendix B):
  4 convolutional layers
  Kernel size 4, stride 2 at each layer
  Channel multipliers: [1, 2, 3, 4] * cnn_depth
  SiLU activations
  No normalisation (removed in V3 vs V2)
  Output: flattened feature vector

For 64x64 input and cnn_depth=32:
  64x64x3  → Conv(3,  32, 4, 2) → 31x31x32
  31x31x32 → Conv(32, 64, 4, 2) → 14x14x64
  14x14x64 → Conv(64, 96, 4, 2) → 6x6x96
  6x6x96   → Conv(96,128, 4, 2) → 2x2x128
  flatten  → 512-dim vector

The embed_dim fed into the RSSM posterior = 4 * 4 * 4 * cnn_depth
  cnn_depth=24 (XS): 4*4*96 = 1536
  cnn_depth=32 (S):  4*4*128 = 2048
"""

import torch
import torch.nn as nn
from typing import Tuple


class Encoder(nn.Module):
    """CNN image encoder.
    
    Args:
        obs_channels: number of input channels (3 for RGB)
        cnn_depth:    base channel multiplier
        obs_size:     spatial size of input (assumed square)
    """

    CHANNEL_MULTS = (1, 2, 3, 4)  # matches paper exactly

    def __init__(self, obs_channels: int = 3, cnn_depth: int = 32, obs_size: int = 64):
        super().__init__()
        self.cnn_depth = cnn_depth
        self.obs_size = obs_size

        channels = [obs_channels] + [m * cnn_depth for m in self.CHANNEL_MULTS]
        layers: list[nn.Module] = []
        for in_c, out_c in zip(channels[:-1], channels[1:]):
            layers.append(nn.Conv2d(in_c, out_c, kernel_size=4, stride=2))
            layers.append(nn.SiLU())

        self.cnn = nn.Sequential(*layers)

        # Compute flat output dimension by doing a dry run
        with torch.no_grad():
            dummy = torch.zeros(1, obs_channels, obs_size, obs_size)
            out = self.cnn(dummy)
            self.embed_dim = int(out.numel() / out.shape[0])

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Encode observations to flat embedding vectors.
        
        Args:
            obs: [..., C, H, W] float32 in [0, 1]
                 Leading dims are arbitrary (B, or B*T, etc.)
        Returns:
            [..., embed_dim] float32
        """
        # Flatten all leading dims except spatial so we can run a single CNN forward
        lead_shape = obs.shape[:-3]
        x = obs.reshape(-1, *obs.shape[-3:])          # (N, C, H, W)
        x = self.cnn(x)                               # (N, C', H', W')
        x = x.reshape(-1, self.embed_dim)             # (N, embed_dim)
        return x.reshape(*lead_shape, self.embed_dim)  # (..., embed_dim)
