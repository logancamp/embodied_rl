"""
training/world_model.py — World model training for DreamerV3.

The world model groups:
  - Encoder (CNN)
  - RSSM
  - Decoder (reconstruction head)
  - RewardHead
  - ContinueHead

All trained jointly on a single combined loss:
  L = reconstruction + kl_scale * kl + reward_scale * reward + continue_scale * continue

No RL signal touches the world model. It learns purely by predicting
the future from compressed latent representations.

The WorldModel class wraps all five components and exposes:
  - train_step(batch) → loss dict + posterior sequence for imagination
  - encode_sequence(batch) → posterior sequence (no gradient update)
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple

from models.encoder import Encoder
from models.rssm import RSSM, SequenceOutput
from models.heads import Decoder, RewardHead, ContinueHead


class WorldModel(nn.Module):
    """Complete world model: encoder + RSSM + prediction heads.
    
    Args:
        cfg: Config object with model and world_model sub-configs
        num_actions: number of discrete actions
        obs_shape:   observation shape (C, H, W)
        device:      compute device
    """

    def __init__(self, cfg, num_actions: int, obs_shape: Tuple[int, ...], device: torch.device):
        super().__init__()
        self.cfg = cfg
        self.device = device

        m = cfg.model
        wm = cfg.world_model
        tw = cfg.twohot

        # ── Sub-modules ────────────────────────────────────────────────────────
        self.encoder = Encoder(
            obs_channels=obs_shape[0],
            cnn_depth=m.cnn_depth,
            obs_size=obs_shape[1],
        )

        self.rssm = RSSM(
            deter=m.deter,
            stoch=m.stoch,
            classes=m.classes,
            embed_dim=self.encoder.embed_dim,
            action_size=num_actions,
            units=m.units,
        )

        latent_dim = m.deter + m.stoch * m.classes

        self.decoder = Decoder(
            latent_dim=latent_dim,
            cnn_depth=m.cnn_depth,
            obs_channels=obs_shape[0],
            obs_size=obs_shape[1],
        )

        self.reward_head = RewardHead(
            latent_dim=latent_dim,
            units=m.units,
            layers=m.mlp_layers,
            bins=tw.bins,
            low=tw.low,
            high=tw.high,
        )

        self.continue_head = ContinueHead(
            latent_dim=latent_dim,
            units=m.units,
            layers=m.mlp_layers,
        )

        self.to(device)

    # ── Training ───────────────────────────────────────────────────────────────

    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
        optimizer: torch.optim.Optimizer,
    ) -> Tuple[Dict[str, float], SequenceOutput]:
        """Single world model training step.
        
        Args:
            batch:     dict from ReplayBuffer.sample() — all [B, T, ...]
            optimizer: world model Adam optimizer
        Returns:
            (loss_dict, posteriors) where posteriors has shape [T, B, ...]
        """
        optimizer.zero_grad()

        obs      = batch['obs']       # [B, T, C, H, W]
        actions  = batch['action']    # [B, T, num_actions]
        rewards  = batch['reward']    # [B, T]
        dones    = batch['done']      # [B, T]
        is_first = batch['is_first']  # [B, T]

        B, T = obs.shape[:2]

        # ── Encode observations ────────────────────────────────────────────────
        # Reshape to (B*T, ...) for CNN, then reshape back
        obs_flat = obs.reshape(B * T, *obs.shape[2:])
        embeds_flat = self.encoder(obs_flat)                      # (B*T, embed_dim)
        embeds = embeds_flat.reshape(B, T, -1)                    # (B, T, embed_dim)

        # Transpose to [T, B, ...] for sequence processing
        embeds   = embeds.permute(1, 0, 2)                        # [T, B, embed_dim]
        actions  = actions.permute(1, 0, 2)                       # [T, B, num_actions]
        is_first = is_first.permute(1, 0)                         # [T, B]

        # ── Run RSSM in observe mode ───────────────────────────────────────────
        priors, posteriors = self.rssm.observe_sequence(embeds, actions, is_first)

        # Flat latent for heads: [T, B, latent_dim]
        latent_flat = posteriors.flat

        # ── Prediction losses ──────────────────────────────────────────────────

        # Reconstruction loss: predict obs from latent
        # Reshape for decoder: [T*B, latent_dim] → [T*B, C, H, W]
        latent_2d = latent_flat.reshape(T * B, -1)
        obs_target = obs.permute(1, 0, 2, 3, 4).reshape(T * B, *obs.shape[2:])
        recon_loss = self.decoder.loss(latent_2d, obs_target)

        # Reward loss: [T, B] targets
        reward_t = batch['reward'].permute(1, 0)                  # [T, B]
        reward_loss = self.reward_head.loss(latent_flat, reward_t)

        # Continue loss: [T, B] targets (1 - done)
        done_t = batch['done'].permute(1, 0)                      # [T, B]
        continue_target = 1.0 - done_t
        continue_loss = self.continue_head.loss(latent_flat, continue_target)

        # ── KL loss ────────────────────────────────────────────────────────────
        wm = self.cfg.world_model
        kl_loss = self.rssm.kl_loss(
            priors, posteriors,
            balance=wm.kl_balance,
            free_bits=wm.kl_free,
        )

        # ── Combined loss ──────────────────────────────────────────────────────
        loss = (
            recon_loss
            + wm.kl_scale * kl_loss
            + wm.reward_scale * reward_loss
            + wm.continue_scale * continue_loss
        )

        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), wm.grad_clip)
        optimizer.step()

        loss_dict = {
            'wm/loss':          loss.item(),
            'wm/recon_loss':    recon_loss.item(),
            'wm/kl':            kl_loss.item(),
            'wm/reward_loss':   reward_loss.item(),
            'wm/continue_loss': continue_loss.item(),
        }
        return loss_dict, posteriors

    # ── Inference (no grad update) ─────────────────────────────────────────────

    @torch.no_grad()
    def encode_sequence(self, batch: Dict[str, torch.Tensor]) -> SequenceOutput:
        """Encode a batch to posterior states without updating weights."""
        obs      = batch['obs']
        actions  = batch['action']
        is_first = batch['is_first']
        B, T = obs.shape[:2]

        obs_flat = obs.reshape(B * T, *obs.shape[2:])
        embeds = self.encoder(obs_flat).reshape(B, T, -1).permute(1, 0, 2)
        actions = actions.permute(1, 0, 2)
        is_first = is_first.permute(1, 0)

        _, posteriors = self.rssm.observe_sequence(embeds, actions, is_first)
        return posteriors

    @property
    def latent_dim(self) -> int:
        m = self.cfg.model
        return m.deter + m.stoch * m.classes
