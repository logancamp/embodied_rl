"""
models/actor.py — Actor (policy) network for DreamerV3.

The actor never sees raw pixels. It takes the flat RSSM latent state
(deterministic + stochastic components concatenated) and outputs a
categorical distribution over discrete actions.

Key design points:
  - Operates entirely in latent space, trained on imagined trajectories
  - Uses a uniform mixture (unimix=0.01) for action distribution to prevent
    entropy collapse: final_prob = (1 - unimix) * softmax + unimix * uniform
  - Entropy regularisation during training keeps exploration from collapsing
  - Gradients flow through the discrete latent via straight-through estimator

Architecture (paper Appendix B):
  MLP: [units]*layers hidden layers with SiLU activations
  Output: Linear → num_actions logits
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.misc import MLP


class Actor(nn.Module):
    """Discrete action policy over latent RSSM states.
    
    Args:
        latent_dim:  flat RSSM state dimension (deter + stoch*classes)
        num_actions: number of discrete actions
        units:       MLP hidden width
        layers:      number of MLP hidden layers
        unimix:      uniform mixture coefficient for action distribution
    """

    def __init__(
        self,
        latent_dim: int,
        num_actions: int,
        units: int = 512,
        layers: int = 2,
        unimix: float = 0.01,
    ):
        super().__init__()
        self.num_actions = num_actions
        self.unimix = unimix
        self.mlp = MLP(latent_dim, num_actions, units, layers)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """Return action logits.
        
        Args:
            latent: [..., latent_dim]
        Returns:
            [..., num_actions] logits
        """
        return self.mlp(latent)

    def distribution(self, latent: torch.Tensor) -> torch.distributions.Categorical:
        """Return action distribution with uniform mixture.
        
        The uniform mixture prevents the distribution from collapsing to
        near-zero entropy, which destabilises training.
        
        Args:
            latent: [..., latent_dim]
        Returns:
            Categorical distribution over actions
        """
        logits = self.forward(latent)
        probs = F.softmax(logits, dim=-1)
        # Mix with uniform distribution
        uniform = torch.ones_like(probs) / self.num_actions
        probs = (1.0 - self.unimix) * probs + self.unimix * uniform
        return torch.distributions.Categorical(probs=probs)

    def act(self, latent: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """Sample or take the argmax action.
        
        Args:
            latent:        [..., latent_dim]
            deterministic: if True, return argmax (for evaluation)
        Returns:
            [...] integer action indices
        """
        dist = self.distribution(latent)
        if deterministic:
            return dist.probs.argmax(dim=-1)
        return dist.sample()

    def act_one_hot(self, latent: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """Return action as one-hot vector for use as RSSM input.
        
        Args:
            latent: [..., latent_dim]
        Returns:
            [..., num_actions] one-hot
        """
        action = self.act(latent, deterministic)
        return F.one_hot(action, self.num_actions).float()

    def entropy(self, latent: torch.Tensor) -> torch.Tensor:
        """Return entropy of the action distribution.
        
        Args:
            latent: [..., latent_dim]
        Returns:
            [...] entropy values
        """
        return self.distribution(latent).entropy()

    def log_prob(self, latent: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Return log probability of given actions.
        
        Args:
            latent: [..., latent_dim]
            action: [...] integer action indices
        Returns:
            [...] log probabilities
        """
        return self.distribution(latent).log_prob(action)
