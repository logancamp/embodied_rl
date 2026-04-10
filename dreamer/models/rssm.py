"""
models/rssm.py — Recurrent State Space Model (RSSM) for DreamerV3.

The RSSM maintains a latent state with two components:
  h (deterministic): GRU hidden state — carries memory across timesteps
  z (stochastic):    categorical sample — represents current uncertainty

This split is deliberate:
  - h gives smooth temporal continuity for planning
  - z gives the model the ability to represent genuine uncertainty
    (two futures can diverge from the same h with different z)

The RSSM has two modes:
  observe (posterior): given real observation embed, infer z
  imagine (prior):     without observation, predict z from h alone

During training, sequences are processed in observe mode.
During imagination, the actor interacts only with the prior.

Architecture (paper Appendix B):
  GRU input MLP: Linear(z_flat + action) → SiLU → Linear → GRU hidden
  Deterministic:  h_t = GRUCell(input, h_{t-1})
  Prior logits:   MLP(h_t)   → (stoch, classes)
  Posterior logits: MLP(h_t, embed_t) → (stoch, classes)
  
  Straight-through gradients through the categorical sample.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, NamedTuple, Optional


class RSSMState(NamedTuple):
    """Complete RSSM latent state at a single timestep.
    
    deter: [B, deter]           deterministic GRU hidden state
    stoch: [B, stoch, classes]  categorical one-hot sample (straight-through)
    logit: [B, stoch, classes]  logits that produced stoch (for KL computation)
    """
    deter: torch.Tensor
    stoch: torch.Tensor
    logit: torch.Tensor

    @property
    def flat(self) -> torch.Tensor:
        """Concatenated [deter, stoch_flat] used as input to actor/critic/heads."""
        B = self.deter.shape[0]
        return torch.cat([self.deter, self.stoch.reshape(B, -1)], dim=-1)

    @property
    def dim(self) -> int:
        return self.deter.shape[-1] + self.stoch.shape[-2] * self.stoch.shape[-1]


class RSSM(nn.Module):
    """Recurrent State Space Model.
    
    Args:
        deter:       deterministic state size
        stoch:       number of categorical variables
        classes:     classes per categorical variable
        embed_dim:   size of encoder output
        action_size: number of discrete actions
        units:       MLP hidden size
    """

    def __init__(
        self,
        deter: int,
        stoch: int,
        classes: int,
        embed_dim: int,
        action_size: int,
        units: int,
    ):
        super().__init__()
        self.deter = deter
        self.stoch = stoch
        self.classes = classes

        stoch_flat = stoch * classes

        # ── Sequence model ─────────────────────────────────────────────────────
        # Input transform before GRU: project [z_flat, one_hot_action] → GRU input
        self.img_in = nn.Sequential(
            nn.Linear(stoch_flat + action_size, units),
            nn.SiLU(),
            nn.Linear(units, deter),
        )
        # Note: GRUCell falls back to CPU on MPS — this is expected and handled
        # by PYTORCH_ENABLE_MPS_FALLBACK=1. It's slower than CUDA but functional.
        self.gru = nn.GRUCell(deter, deter)

        # ── Prior (imagination) ────────────────────────────────────────────────
        # Predict stochastic state from deterministic state alone
        self.prior_mlp = nn.Sequential(
            nn.Linear(deter, units),
            nn.SiLU(),
            nn.Linear(units, stoch * classes),
        )

        # ── Posterior (observe) ────────────────────────────────────────────────
        # Predict stochastic state from deterministic state + encoder embedding
        self.post_mlp = nn.Sequential(
            nn.Linear(deter + embed_dim, units),
            nn.SiLU(),
            nn.Linear(units, stoch * classes),
        )

        self._action_size = action_size

    # ── State initialization ───────────────────────────────────────────────────

    def initial_state(self, batch_size: int, device: torch.device) -> RSSMState:
        """Return a zero state for the start of an episode."""
        deter = torch.zeros(batch_size, self.deter, device=device)
        stoch = torch.zeros(batch_size, self.stoch, self.classes, device=device)
        logit = torch.zeros(batch_size, self.stoch, self.classes, device=device)
        return RSSMState(deter, stoch, logit)

    # ── Single-step operations ─────────────────────────────────────────────────

    def observe_step(
        self,
        prev_state: RSSMState,
        prev_action: torch.Tensor,
        embed: torch.Tensor,
        is_first: torch.Tensor,
    ) -> Tuple[RSSMState, RSSMState]:
        """Single posterior step given a real observation embedding.
        
        Args:
            prev_state:  RSSMState at t-1
            prev_action: [B, action_size] one-hot action taken at t-1
            embed:       [B, embed_dim] encoder output at t
            is_first:    [B] bool — True if this is the first step of an episode
        Returns:
            (prior_state, posterior_state) both at time t
        """
        # Reset state at episode boundaries
        prev_state = self._mask_state(prev_state, is_first)
        prev_action = prev_action * (1.0 - is_first.float().unsqueeze(-1))

        # Advance deterministic state
        h = self._gru_step(prev_state, prev_action)

        # Prior: predict z from h only
        prior_logit = self.prior_mlp(h).reshape(-1, self.stoch, self.classes)
        prior_stoch = self._straight_through(prior_logit)
        prior = RSSMState(h, prior_stoch, prior_logit)

        # Posterior: refine z using real observation
        post_input = torch.cat([h, embed], dim=-1)
        post_logit = self.post_mlp(post_input).reshape(-1, self.stoch, self.classes)
        post_stoch = self._straight_through(post_logit)
        posterior = RSSMState(h, post_stoch, post_logit)

        return prior, posterior

    def imagine_step(
        self,
        prev_state: RSSMState,
        action: torch.Tensor,
    ) -> RSSMState:
        """Single prior step without observation (for imagination rollouts).
        
        Args:
            prev_state: RSSMState at t-1
            action:     [B, action_size] one-hot action to take
        Returns:
            RSSMState at t (prior only)
        """
        h = self._gru_step(prev_state, action)
        logit = self.prior_mlp(h).reshape(-1, self.stoch, self.classes)
        stoch = self._straight_through(logit)
        return RSSMState(h, stoch, logit)

    # ── Sequence processing ────────────────────────────────────────────────────

    def observe_sequence(
        self,
        embeds: torch.Tensor,
        actions: torch.Tensor,
        is_first: torch.Tensor,
        initial_state: Optional[RSSMState] = None,
    ) -> Tuple['SequenceOutput', 'SequenceOutput']:
        """Process a full sequence in posterior mode.
        
        Args:
            embeds:        [T, B, embed_dim]
            actions:       [T, B, action_size] one-hot, action AT t used to
                           transition FROM t to t+1 (so we use prev action)
            is_first:      [T, B] bool
            initial_state: optional starting state (zeros if None)
        Returns:
            (priors, posteriors): SequenceOutput with [T, B, ...] tensors
        """
        T, B = embeds.shape[:2]
        device = embeds.device

        state = initial_state or self.initial_state(B, device)

        priors_deter, priors_stoch, priors_logit = [], [], []
        posts_deter, posts_stoch, posts_logit = [], [], []

        # The action used to transition INTO step t is the action taken AT t-1
        # We prepend a zero action for t=0 (beginning of sequence)
        prev_action = torch.zeros(B, self._action_size, device=device)

        for t in range(T):
            prior, posterior = self.observe_step(
                state, prev_action, embeds[t], is_first[t]
            )
            priors_deter.append(prior.deter)
            priors_stoch.append(prior.stoch)
            priors_logit.append(prior.logit)
            posts_deter.append(posterior.deter)
            posts_stoch.append(posterior.stoch)
            posts_logit.append(posterior.logit)

            state = posterior
            prev_action = actions[t]

        def _stack(lst: list) -> torch.Tensor:
            return torch.stack(lst, dim=0)

        priors = SequenceOutput(
            deter=_stack(priors_deter),
            stoch=_stack(priors_stoch),
            logit=_stack(priors_logit),
        )
        posteriors = SequenceOutput(
            deter=_stack(posts_deter),
            stoch=_stack(posts_stoch),
            logit=_stack(posts_logit),
        )
        return priors, posteriors

    # ── KL divergence ──────────────────────────────────────────────────────────

    def kl_loss(
        self,
        prior: 'SequenceOutput',
        posterior: 'SequenceOutput',
        balance: float = 0.8,
        free_bits: float = 1.0,
    ) -> torch.Tensor:
        """KL divergence with balancing and free bits.
        
        DreamerV3 uses two modifications to the standard KL:

        1. Balancing: split the KL into two terms and weight them.
           This prevents either the prior or posterior from being ignored.
             loss = balance * KL(sg(post) || prior)      [train prior]
                  + (1-balance) * KL(post || sg(prior))  [train posterior]

        2. Free bits: clamp each categorical's KL from below at `free_bits`.
           This prevents the model from over-regularising early in training
           when the prior can't yet match the posterior.

        Args:
            prior:     SequenceOutput with logit [T, B, stoch, classes]
            posterior: SequenceOutput with logit [T, B, stoch, classes]
            balance:   weight for the prior-training term (paper: 0.8)
            free_bits: minimum KL per latent variable in nats (paper: 1.0)
        Returns:
            scalar KL loss
        """
        # [T, B, stoch] KL per categorical variable
        kl_prior = _categorical_kl(posterior.logit.detach(), prior.logit)
        kl_post  = _categorical_kl(posterior.logit, prior.logit.detach())

        # Free bits: clamp each variable's KL from below
        kl_prior = kl_prior.clamp(min=free_bits)
        kl_post  = kl_post.clamp(min=free_bits)

        # Sum over stoch variables, mean over T and B
        loss = balance * kl_prior.sum(-1).mean() + (1 - balance) * kl_post.sum(-1).mean()
        return loss

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _gru_step(self, state: RSSMState, action: torch.Tensor) -> torch.Tensor:
        """Compute next deterministic state h given previous state and action."""
        B = state.deter.shape[0]
        z_flat = state.stoch.reshape(B, -1)
        x = torch.cat([z_flat, action], dim=-1)
        x = self.img_in(x)
        return self.gru(x, state.deter)

    def _straight_through(self, logit: torch.Tensor) -> torch.Tensor:
        """Sample from categorical and apply straight-through estimator.
        
        Forward pass: one-hot of argmax sample (discrete, not differentiable)
        Backward pass: gradients flow through soft probabilities
        This allows the discrete bottleneck to be trained end-to-end.
        """
        # logit: [B, stoch, classes]
        probs = F.softmax(logit, dim=-1)                          # [B, stoch, classes]
        indices = torch.distributions.Categorical(probs=probs).sample()  # [B, stoch]
        one_hot = F.one_hot(indices, self.classes).float()        # [B, stoch, classes]
        # Straight-through: use one_hot in forward, probs gradient in backward
        return one_hot + probs - probs.detach()

    def _mask_state(self, state: RSSMState, is_first: torch.Tensor) -> RSSMState:
        """Zero out state components where is_first is True."""
        mask = (1.0 - is_first.float()).unsqueeze(-1)             # [B, 1]
        deter = state.deter * mask
        mask3d = mask.unsqueeze(-1)                               # [B, 1, 1]
        stoch = state.stoch * mask3d
        logit = state.logit * mask3d
        return RSSMState(deter, stoch, logit)


class SequenceOutput(NamedTuple):
    """Sequence of RSSM states: all tensors have shape [T, B, ...]."""
    deter: torch.Tensor   # [T, B, deter]
    stoch: torch.Tensor   # [T, B, stoch, classes]
    logit: torch.Tensor   # [T, B, stoch, classes]

    @property
    def flat(self) -> torch.Tensor:
        """[T, B, deter + stoch*classes] concatenated representation."""
        T, B = self.deter.shape[:2]
        return torch.cat([self.deter, self.stoch.reshape(T, B, -1)], dim=-1)

    def as_state(self, t: int) -> RSSMState:
        """Extract a single-timestep RSSMState at index t."""
        return RSSMState(self.deter[t], self.stoch[t], self.logit[t])


# ── KL helper (module-level, not a method) ────────────────────────────────────

def _categorical_kl(logit_p: torch.Tensor, logit_q: torch.Tensor) -> torch.Tensor:
    """KL(P || Q) for independent categorical distributions.
    
    Args:
        logit_p: [..., stoch, classes] logits for distribution P
        logit_q: [..., stoch, classes] logits for distribution Q
    Returns:
        [..., stoch] KL per categorical variable in nats
    """
    p = F.softmax(logit_p, dim=-1).clamp(min=1e-8)
    q = F.softmax(logit_q, dim=-1).clamp(min=1e-8)
    return (p * (p.log() - q.log())).sum(-1)