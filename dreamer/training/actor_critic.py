"""
training/actor_critic.py — Actor-critic training via imagination for DreamerV3.

The key insight: after the world model is trained, we can run the RSSM
in imagination mode (prior only, no real observations) to generate rollouts
entirely inside the model's learned latent space. The actor and critic are
then trained on these imagined trajectories.

Training flow:
  1. Sample starting states from recent world model posteriors
  2. Roll out H=15 steps using actor + RSSM prior
  3. Predict rewards and continue signals at each imagined step
  4. Compute lambda-return targets
  5. Update critic toward those targets (two-hot loss)
  6. Update actor to maximise normalised returns + entropy

Why this works:
  Actor/critic never see real environment rewards during training.
  They learn purely from the world model's predictions — which were
  themselves learned from real experience.
  This separation allows many gradient steps per real step (train_ratio=512).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple

from models.rssm import RSSM, RSSMState
from models.actor import Actor
from models.critic import Critic
from models.heads import RewardHead, ContinueHead
from utils.misc import ReturnNormalizer


class ActorCritic:
    """Manages actor and critic training via imagined rollouts.
    
    NOT a nn.Module because we manage two separate optimizers.
    
    Args:
        actor:          Actor module
        critic:         Critic module
        rssm:           RSSM (used for imagination, weights frozen during AC training)
        reward_head:    RewardHead (frozen, used to score imagined states)
        continue_head:  ContinueHead (frozen, used to predict episode continuation)
        cfg:            Config object
    """

    def __init__(
        self,
        actor: Actor,
        critic: Critic,
        rssm: RSSM,
        reward_head: RewardHead,
        continue_head: ContinueHead,
        cfg,
    ):
        self.actor = actor
        self.critic = critic
        self.rssm = rssm
        self.reward_head = reward_head
        self.continue_head = continue_head
        self.cfg = cfg

        ac = cfg.actor
        cr = cfg.critic
        self.actor_opt = torch.optim.Adam(actor.parameters(), lr=ac.lr, eps=1e-8)
        self.critic_opt = torch.optim.Adam(critic.parameters(), lr=cr.lr, eps=1e-8)

        self._return_norm = ReturnNormalizer(
            decay=cfg.returns.return_norm_decay,
            limit=cfg.returns.return_norm_limit,
        )

    def train_step(
        self,
        start_states: 'SequenceOutput',  # noqa: F821
    ) -> Dict[str, float]:
        """Single actor-critic training step.
        
        Args:
            start_states: posterior sequence from world model [T, B, ...]
                         We sample from these as starting points for imagination.
        Returns:
            dict of scalar loss metrics
        """
        r = self.cfg.returns
        ac = self.cfg.actor
        cr = self.cfg.critic

        # ── Sample starting states ─────────────────────────────────────────────
        # Flatten [T, B] → [T*B] and randomly select batch_size starting points
        T, B = start_states.deter.shape[:2]
        flat_deter = start_states.deter.reshape(T * B, -1).detach()
        flat_stoch = start_states.stoch.reshape(T * B, *start_states.stoch.shape[2:]).detach()
        flat_logit = start_states.logit.reshape(T * B, *start_states.logit.shape[2:]).detach()

        # Use all start states (same as paper — no sub-sampling)
        start = RSSMState(flat_deter, flat_stoch, flat_logit)

        # ── Imagination rollout ────────────────────────────────────────────────
        states, actions, rewards, continues = self._imagine(start, r.horizon)
        # states:   list of H+1 RSSMState (including start)
        # actions:  [H, N, num_actions] one-hot
        # rewards:  [H, N] predicted rewards (real space)
        # continues:[H, N] predicted continue prob

        # Stack states for critic computation
        # We need states[0:H] for actor updates, states[1:H+1] for value bootstrap
        H = r.horizon
        N = flat_deter.shape[0]

        latents = torch.stack([s.flat for s in states], dim=0)  # [H+1, N, latent_dim]

        # ── Compute lambda returns ─────────────────────────────────────────────
        with torch.no_grad():
            values = self.critic.target_predict(latents)       # [H+1, N]

        returns = _lambda_return(
            rewards=rewards,                                    # [H, N]
            values=values,                                      # [H+1, N]
            continues=continues,                                # [H, N]
            gamma=r.gamma,
            lam=r.lam,
        )                                                       # [H, N]

        # ── Update return normalizer ───────────────────────────────────────────
        self._return_norm.update(returns)
        norm_returns = self._return_norm.normalize(returns)    # [H, N]

        # ── Critic loss ────────────────────────────────────────────────────────
        self.critic_opt.zero_grad()
        critic_loss = self.critic.loss(latents[:-1].detach(), returns)
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.cfg.critic.grad_clip)
        self.critic_opt.step()
        self.critic.update_target()

        # ── Actor loss ─────────────────────────────────────────────────────────
        # Re-compute actor distribution at each imagined state to get entropy
        # We use stop-gradient on the states — actor gradients don't flow
        # back through the world model weights (only through the actions taken)
        self.actor_opt.zero_grad()

        latents_sg = latents[:-1].detach()                     # [H, N, latent_dim]
        entropy = self.actor.entropy(
            latents_sg.reshape(H * N, -1)
        ).reshape(H, N)                                         # [H, N]

        # Actor loss: maximise normalised returns + entropy bonus
        actor_loss = -(norm_returns.detach() + ac.entropy_coeff * entropy).mean()

        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), ac.grad_clip)
        self.actor_opt.step()

        return {
            'ac/actor_loss':   actor_loss.item(),
            'ac/critic_loss':  critic_loss.item(),
            'ac/entropy':      entropy.mean().item(),
            'ac/mean_return':  returns.mean().item(),
            'ac/return_scale': self._return_norm.scale,
        }

    # ── Imagination ────────────────────────────────────────────────────────────

    def _imagine(
        self,
        start: RSSMState,
        horizon: int,
    ):
        """Roll out imagined trajectories for `horizon` steps.
        
        No real environment interaction. Uses:
          actor   → choose action from current latent
          rssm    → predict next latent state (prior step)
          reward_head  → predict reward at each state
          continue_head → predict episode continuation
          
        Returns:
            states:   list of H+1 RSSMState
            actions:  [H, N, num_actions] float tensor
            rewards:  [H, N] float tensor
            continues:[H, N] float tensor
        """
        states = [start]
        all_actions = []
        all_rewards = []
        all_continues = []

        state = start
        with torch.no_grad():
            # Reward and continue heads don't need gradients for imagination
            # Actor needs gradients for its own update (but not through world model)
            pass

        for _ in range(horizon):
            # Actor selects action (with gradients for actor update)
            action_oh = self.actor.act_one_hot(state.flat.detach())  # [N, num_actions]

            # World model predicts next state (no grad through world model)
            with torch.no_grad():
                next_state = self.rssm.imagine_step(state, action_oh)
                reward = self.reward_head.predict(next_state.flat)    # [N]
                cont   = self.continue_head.predict(next_state.flat)  # [N]

            states.append(next_state)
            all_actions.append(action_oh)
            all_rewards.append(reward)
            all_continues.append(cont)
            state = next_state

        actions   = torch.stack(all_actions,   dim=0)   # [H, N, num_actions]
        rewards   = torch.stack(all_rewards,   dim=0)   # [H, N]
        continues = torch.stack(all_continues, dim=0)   # [H, N]

        return states, actions, rewards, continues


# ── Lambda return computation ─────────────────────────────────────────────────

def _lambda_return(
    rewards: torch.Tensor,
    values: torch.Tensor,
    continues: torch.Tensor,
    gamma: float,
    lam: float,
) -> torch.Tensor:
    """Compute TD(lambda) returns for imagined trajectories.
    
    V_lambda_t = r_t + gamma * c_t * [(1-lam)*V_t+1 + lam*V_lambda_t+1]
    
    This interpolates between TD(1) (Monte Carlo, lam=1) and TD(0) (lam=0).
    DreamerV3 uses lam=0.95 — mostly Monte Carlo but with some bootstrapping.
    
    Args:
        rewards:   [H, N]   predicted rewards at each imagined step
        values:    [H+1, N] predicted values including bootstrap at H
        continues: [H, N]   episode continuation probabilities
        gamma:     discount factor (0.997 in paper)
        lam:       lambda for TD(lambda) (0.95 in paper)
    Returns:
        returns: [H, N] lambda-return targets
    """
    H = rewards.shape[0]
    returns = []

    # Bootstrap from the last value
    last = values[H]

    for t in reversed(range(H)):
        # Single-step return + discounted lambda-return
        td_target = values[t + 1]
        last = rewards[t] + gamma * continues[t] * (
            (1.0 - lam) * td_target + lam * last
        )
        returns.insert(0, last)

    return torch.stack(returns, dim=0)  # [H, N]
