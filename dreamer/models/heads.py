"""
models/heads.py — World model prediction heads for DreamerV3.

Three heads, all taking the flat RSSM latent state as input:

  Decoder:       latent → predicted observation (MSE reconstruction loss)
  RewardHead:    latent → predicted reward       (two-hot distributional loss)
  ContinueHead:  latent → predicted continue     (binary cross-entropy loss)

The decoder uses a transposed CNN that mirrors the encoder.
The reward and continue heads are small MLPs.

All training losses for the world model flow through these three heads
plus the KL loss from the RSSM. No RL gradients touch the world model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.misc import MLP, TwoHotDist, symlog


class Decoder(nn.Module):
    """Transposed CNN that reconstructs observations from latent states.
    
    Mirrors the encoder exactly: 4 transposed conv layers,
    channel multipliers [4, 3, 2, 1] * cnn_depth (reversed encoder),
    kernel 5, stride 2.
    
    The reconstruction target is the raw observation in [0,1].
    Loss is MSE (treating each pixel as a Gaussian with unit variance).
    
    Args:
        latent_dim:   dimension of flat RSSM state (deter + stoch*classes)
        cnn_depth:    base channel multiplier (must match encoder)
        obs_channels: output channels (3 for RGB)
        obs_size:     output spatial size (must match encoder input)
    """

    CHANNEL_MULTS = (4, 3, 2, 1)

    def __init__(
        self,
        latent_dim: int,
        cnn_depth: int = 32,
        obs_channels: int = 3,
        obs_size: int = 64,
    ):
        super().__init__()
        self.cnn_depth = cnn_depth
        self.obs_size = obs_size

        # Project latent → spatial feature map matching encoder's output shape
        # Encoder output: (4*cnn_depth) channels at spatial 2x2
        first_channels = 4 * cnn_depth
        self.linear = nn.Linear(latent_dim, first_channels * 2 * 2)
        self.first_channels = first_channels

        channels = [m * cnn_depth for m in self.CHANNEL_MULTS]
        # Add output channel count
        out_channels = channels[1:] + [obs_channels]

        layers: list[nn.Module] = []
        for i, (in_c, out_c) in enumerate(zip(channels, out_channels)):
            layers.append(nn.ConvTranspose2d(in_c, out_c, kernel_size=5, stride=2))
            if i < len(channels) - 1:  # no activation on final layer
                layers.append(nn.SiLU())

        self.cnn = nn.Sequential(*layers)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode latent state to observation.
        
        Args:
            latent: [..., latent_dim]
        Returns:
            [..., C, H, W] predicted observation in [0, 1] range
        """
        lead_shape = latent.shape[:-1]
        x = latent.reshape(-1, latent.shape[-1])

        x = self.linear(x)
        x = x.reshape(-1, self.first_channels, 2, 2)
        x = self.cnn(x)

        # Crop to exact target size (transposed conv output may be slightly larger)
        x = x[..., :self.obs_size, :self.obs_size]
        x = torch.sigmoid(x)   # pixels in [0, 1]

        return x.reshape(*lead_shape, *x.shape[1:])

    def loss(self, latent: torch.Tensor, obs_target: torch.Tensor) -> torch.Tensor:
        """MSE reconstruction loss.
        
        Args:
            latent:     [..., latent_dim]
            obs_target: [..., C, H, W] in [0, 1]
        Returns:
            scalar mean loss
        """
        pred = self.forward(latent)
        return F.mse_loss(pred, obs_target)


class RewardHead(nn.Module):
    """Predicts reward from latent state using two-hot distributional output.
    
    The two-hot encoding over symlog-transformed rewards lets the model
    represent uncertainty over reward values and handles the large dynamic
    range of rewards across games without manual scaling.
    
    Args:
        latent_dim: flat RSSM state dimension
        units:      MLP hidden width
        layers:     number of MLP hidden layers
        bins:       number of two-hot bins
        low, high:  bin range in symlog space
    """

    def __init__(
        self,
        latent_dim: int,
        units: int = 512,
        layers: int = 2,
        bins: int = 255,
        low: float = -20.0,
        high: float = 20.0,
    ):
        super().__init__()
        self.dist = TwoHotDist(low=low, high=high, num_bins=bins)
        self.mlp = MLP(latent_dim, bins, units, layers)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """Return logits over reward bins.
        
        Args:
            latent: [..., latent_dim]
        Returns:
            [..., bins] unnormalized logits
        """
        return self.mlp(latent)

    def predict(self, latent: torch.Tensor) -> torch.Tensor:
        """Return predicted reward (real-valued, symexp decoded).
        
        Args:
            latent: [..., latent_dim]
        Returns:
            [...] scalar reward predictions
        """
        return self.dist.decode(self.forward(latent))

    def loss(self, latent: torch.Tensor, reward_target: torch.Tensor) -> torch.Tensor:
        """Two-hot cross-entropy loss.
        
        Args:
            latent:        [..., latent_dim]
            reward_target: [...] real-valued reward targets
        Returns:
            scalar mean loss
        """
        logits = self.forward(latent)
        return self.dist.loss(logits, reward_target).mean()


class ContinueHead(nn.Module):
    """Predicts episode continuation (1 - done) from latent state.
    
    Binary output: 1 = episode continues, 0 = episode ended.
    Used both as a training target and during imagination to weight returns.
    
    Args:
        latent_dim: flat RSSM state dimension
        units:      MLP hidden width
        layers:     number of MLP hidden layers
    """

    def __init__(self, latent_dim: int, units: int = 512, layers: int = 2):
        super().__init__()
        self.mlp = MLP(latent_dim, 1, units, layers)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """Return continuation logits.
        
        Args:
            latent: [..., latent_dim]
        Returns:
            [..., 1] continuation logits (before sigmoid)
        """
        return self.mlp(latent)

    def predict(self, latent: torch.Tensor) -> torch.Tensor:
        """Return continuation probability in [0, 1].
        
        Args:
            latent: [..., latent_dim]
        Returns:
            [...] continuation probabilities
        """
        return torch.sigmoid(self.forward(latent)).squeeze(-1)

    def loss(self, latent: torch.Tensor, continue_target: torch.Tensor) -> torch.Tensor:
        """Binary cross-entropy loss.
        
        Args:
            latent:          [..., latent_dim]
            continue_target: [...] float targets in {0.0, 1.0}
        Returns:
            scalar mean loss
        """
        logits = self.forward(latent).squeeze(-1)
        return F.binary_cross_entropy_with_logits(logits, continue_target)
