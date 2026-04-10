"""
train.py — Main DreamerV3 training loop.

Usage:
    # Debug run on Mac (XS model, MPS):
    python train.py --configs configs/base.yaml configs/debug.yaml

    # Full training on RTX 3070 (S model, CUDA):
    python train.py --configs configs/base.yaml configs/s_size.yaml

    # Resume from checkpoint:
    python train.py --configs configs/base.yaml configs/s_size.yaml --resume

Training flow:
  1. Collect seed_steps of random experience
  2. Loop:
       a. Collect 1 real environment step (using actor)
       b. Every step: run train_ratio world model + actor-critic updates
       c. Log, evaluate, checkpoint on schedule
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from tqdm import tqdm

# Allow imports from project root
sys.path.insert(0, os.path.dirname(__file__))

from utils.misc import load_config, get_device
from utils.logging import MetricsLogger, EpisodeLogger, save_checkpoint, load_checkpoint, latest_checkpoint
from envs.atari import make_train_env, make_eval_env
from models.actor import Actor
from models.critic import Critic
from training.replay_buffer import ReplayBuffer
from training.world_model import WorldModel
from training.actor_critic import ActorCritic


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--configs', nargs='+',
        default=['configs/base.yaml', 'configs/s_size.yaml'],
        help='Config yaml files (merged left to right).'
    )
    parser.add_argument('--resume', action='store_true', help='Resume from latest checkpoint.')
    parser.add_argument('--seed', type=int, default=0)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate(actor, world_model, env, num_episodes: int, device: torch.device) -> dict:
    """Run evaluation episodes using the current policy (deterministic actions)."""
    returns = []
    lengths = []

    for _ in range(num_episodes):
        obs, _ = env.reset()
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        state = world_model.rssm.initial_state(1, device)
        prev_action = torch.zeros(1, env.num_actions, device=device)

        ep_return = 0.0
        ep_length = 0
        done = False

        while not done:
            with torch.no_grad():
                # Encode observation
                embed = world_model.encoder(obs_tensor)

                # Posterior step
                is_first = torch.zeros(1, device=device)
                _, state = world_model.rssm.observe_step(state, prev_action, embed, is_first)

                # Actor selects action (deterministic for evaluation)
                action = actor.act(state.flat, deterministic=True)

            action_int = int(action.item())
            obs, reward, done, truncated, info = env.step(action_int)
            done = done or truncated
            ep_return += reward
            ep_length += 1

            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            prev_action = F.one_hot(action, env.num_actions).float()
            if done:
                break

        returns.append(ep_return)
        lengths.append(ep_length)

    return {
        'eval/mean_return':         float(np.mean(returns)),
        'eval/max_return':          float(np.max(returns)),
        'eval/mean_episode_length': float(np.mean(lengths)),
    }


def main():
    args = parse_args()
    cfg = load_config(*args.configs)
    set_seed(args.seed)
    device = get_device(cfg)
    print(f"Device: {device}")

    # MPS (Apple Silicon) — force float32 throughout, disable any float16 paths
    if device.type == 'mps':
        torch.set_default_dtype(torch.float32)

    # ── Environments ───────────────────────────────────────────────────────────
    train_env = make_train_env(cfg, seed=args.seed)
    eval_env  = make_eval_env(cfg)
    num_actions = train_env.num_actions
    obs_shape   = train_env.observation_space.shape       # (C, H, W)
    print(f"Env: {cfg.env.name} | Actions: {num_actions} | Obs: {obs_shape}")

    # ── Models ─────────────────────────────────────────────────────────────────
    world_model = WorldModel(cfg, num_actions, obs_shape, device)
    latent_dim  = world_model.latent_dim

    actor = Actor(
        latent_dim=latent_dim,
        num_actions=num_actions,
        units=cfg.model.units,
        layers=cfg.model.mlp_layers,
        unimix=cfg.actor.unimix,
    ).to(device)

    critic = Critic(
        latent_dim=latent_dim,
        units=cfg.model.units,
        layers=cfg.model.mlp_layers,
        bins=cfg.twohot.bins,
        low=cfg.twohot.low,
        high=cfg.twohot.high,
        ema_decay=cfg.critic.ema_decay,
    ).to(device)

    wm_opt = torch.optim.Adam(world_model.parameters(), lr=cfg.world_model.lr, eps=1e-8)

    ac_trainer = ActorCritic(
        actor=actor,
        critic=critic,
        rssm=world_model.rssm,
        reward_head=world_model.reward_head,
        continue_head=world_model.continue_head,
        cfg=cfg,
    )

    # Count and report parameters
    wm_params = sum(p.numel() for p in world_model.parameters())
    ac_params  = sum(p.numel() for p in actor.parameters()) + sum(p.numel() for p in critic.parameters())
    print(f"World model params: {wm_params:,}")
    print(f"Actor+Critic params: {ac_params:,}")

    # ── Replay buffer ──────────────────────────────────────────────────────────
    buffer = ReplayBuffer(
        capacity=cfg.training.replay_size,
        obs_shape=obs_shape,
        num_actions=num_actions,
        batch_size=cfg.training.batch_size,
        batch_length=cfg.training.batch_length,
        device=device,
    )

    # ── Logging ────────────────────────────────────────────────────────────────
    logger = MetricsLogger(cfg.logging.logdir)
    ep_logger = EpisodeLogger()

    # ── Resume from checkpoint ─────────────────────────────────────────────────
    start_step = 0
    if args.resume:
        ckpt = latest_checkpoint(cfg.logging.logdir)
        if ckpt:
            start_step = load_checkpoint(
                ckpt, world_model, actor, critic,
                wm_opt, ac_trainer.actor_opt, ac_trainer.critic_opt,
                device,
            )

    # ── Seed the buffer with random actions ────────────────────────────────────
    print(f"Collecting {cfg.training.seed_steps} seed steps with random policy...")
    obs, _ = train_env.reset()
    buffer.start_episode(obs)

    for _ in range(cfg.training.seed_steps):
        action = int(train_env.action_space.sample())
        next_obs, reward, done, truncated, info = train_env.step(action)
        buffer.add_step(action, reward, next_obs, done or truncated)
        ep_logger.step(reward, done or truncated)

        if done or truncated:
            buffer.end_episode()
            obs, _ = train_env.reset()
            buffer.start_episode(obs)
        else:
            obs = next_obs

    # Force-save the in-progress episode even if it didn't naturally terminate
    buffer.end_episode()
    print(f"Buffer has {buffer.num_steps} steps from {buffer.num_episodes} episodes.")

    # ── Main training loop ─────────────────────────────────────────────────────
    print("Starting training...")

    obs, _ = train_env.reset()
    buffer.start_episode(obs)

    rssm_state = world_model.rssm.initial_state(1, device)
    prev_action = torch.zeros(1, num_actions, device=device)
    is_episode_start = True

    last_return  = float('nan')
    last_eval    = float('nan')
    last_wm_loss = float('nan')
    last_kl      = float('nan')

    step = start_step
    while step < cfg.training.total_steps:
        # Each iteration of this outer loop = one log_every-sized chunk
        chunk_end = min(step + cfg.logging.log_every, cfg.training.total_steps)
        pct = 100 * step / cfg.training.total_steps

        desc = (
            f"[{pct:3.0f}%] "
            f"ret={last_return:+.1f} "
            f"loss={last_wm_loss:.3f} "
            f"kl={last_kl:.2f} "
            f"eval={last_eval:+.1f}"
        )

        with tqdm(
            total=chunk_end - step,
            desc=desc,
            unit="step",
            dynamic_ncols=True,
            leave=True,
        ) as pbar:
            while step < chunk_end:

                # ── Collect one environment step ───────────────────────────────
                with torch.no_grad():
                    obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                    embed      = world_model.encoder(obs_tensor)
                    is_first   = torch.tensor([is_episode_start], dtype=torch.float32, device=device)
                    _, rssm_state = world_model.rssm.observe_step(
                        rssm_state, prev_action, embed, is_first
                    )
                    action_tensor = actor.act(rssm_state.flat, deterministic=False)
                    action_int    = int(action_tensor.item())

                action_oh = F.one_hot(action_tensor, num_actions).float()
                next_obs, reward, done, truncated, info = train_env.step(action_int)
                episode_done = done or truncated

                buffer.add_step(action_int, reward, next_obs, episode_done)
                ep_logger.step(reward, episode_done)

                prev_action      = action_oh
                is_episode_start = False

                if episode_done:
                    buffer.end_episode()
                    obs, _ = train_env.reset()
                    buffer.start_episode(obs)
                    rssm_state   = world_model.rssm.initial_state(1, device)
                    prev_action  = torch.zeros(1, num_actions, device=device)
                    is_episode_start = True
                else:
                    obs = next_obs

                # ── Training updates ───────────────────────────────────────────
                if buffer.ready():
                    batch = buffer.sample()
                    wm_metrics, posteriors = world_model.train_step(batch, wm_opt)
                    for k, v in wm_metrics.items():
                        logger.log(k, v)
                    ac_metrics = ac_trainer.train_step(posteriors)
                    for k, v in ac_metrics.items():
                        logger.log(k, v)

                ep_stats = ep_logger.pop()
                for k, v in ep_stats.items():
                    logger.log(k, v)

                step += 1
                pbar.update(1)

        # ── End of chunk: flush metrics ────────────────────────────────────────
        metrics = logger.flush(step)
        if metrics:
            last_return  = metrics.get('train/mean_episode_return', last_return)
            last_wm_loss = metrics.get('wm/loss', last_wm_loss)
            last_kl      = metrics.get('wm/kl', last_kl)

        # ── Evaluation ─────────────────────────────────────────────────────────
        if step % cfg.logging.eval_every == 0:
            eval_metrics = evaluate(actor, world_model, eval_env, cfg.logging.eval_episodes, device)
            for k, v in eval_metrics.items():
                logger.log(k, v)
            last_eval = eval_metrics['eval/mean_return']
            print(f"  ╔{'═'*50}╗")
            print(f"  ║  EVAL @ step {step:,}  |  mean={last_eval:.1f}  max={eval_metrics['eval/max_return']:.1f}{'':>12}║")
            print(f"  ╚{'═'*50}╝")

        # ── Checkpoint ─────────────────────────────────────────────────────────
        if step % cfg.logging.checkpoint_every == 0:
            save_checkpoint(
                cfg.logging.logdir, step,
                world_model, actor, critic,
                wm_opt, ac_trainer.actor_opt, ac_trainer.critic_opt,
            )

        # ── Collect one environment step ───────────────────────────────────────
        with torch.no_grad():
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            embed      = world_model.encoder(obs_tensor)
            is_first   = torch.tensor([is_episode_start], dtype=torch.float32, device=device)

            _, rssm_state = world_model.rssm.observe_step(
                rssm_state, prev_action, embed, is_first
            )
            action_tensor = actor.act(rssm_state.flat, deterministic=False)
            action_int    = int(action_tensor.item())

        action_oh = F.one_hot(action_tensor, num_actions).float()
        next_obs, reward, done, truncated, info = train_env.step(action_int)
        episode_done = done or truncated

        buffer.add_step(action_int, reward, next_obs, episode_done)
        ep_logger.step(reward, episode_done)

        prev_action      = action_oh
        is_episode_start = False

        if episode_done:
            buffer.end_episode()
            obs, _ = train_env.reset()
            buffer.start_episode(obs)
            rssm_state   = world_model.rssm.initial_state(1, device)
            prev_action  = torch.zeros(1, num_actions, device=device)
            is_episode_start = True
        else:
            obs = next_obs

    # Final evaluation
    print("\nFinal evaluation...")
    eval_metrics = evaluate(actor, world_model, eval_env, cfg.logging.eval_episodes * 2, device)
    print(f"Final mean return: {eval_metrics['eval/mean_return']:.2f}")
    print(f"Final max return:  {eval_metrics['eval/max_return']:.2f}")

    train_env.close()
    eval_env.close()


if __name__ == '__main__':
    main()