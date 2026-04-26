import signal
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import json
from torch.utils.tensorboard import SummaryWriter

import gymnasium as gym  # type: ignore
import optuna # type: ignore

from rollout_buffer import RolloutBuffer
from actor_critic import ActorCritic

def compute_gae(rewards, values, dones, next_value, gamma=0.99, lam=0.95):
    T = len(rewards)
    advantages = torch.zeros(T)
    last_gae = 0.0

    for t in reversed(range(T)):
        next_val = next_value if t == T - 1 else values[t + 1]
        next_nterm = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_val * next_nterm - values[t]
        last_gae = delta + gamma * lam * next_nterm * last_gae
        advantages[t] = last_gae

    returns = advantages + values
    return advantages, returns


def ppo_update(model, optimizer, buffer, returns, advantages,
               clip_eps=0.2, vf_coef=0.5, continuous=False, ent_coef=0.01,
               n_epochs=10, batch_size=64, max_grad_norm=0.5):

    obs = buffer.obs[:buffer.ptr]
    
    actions = buffer.actions[:buffer.ptr].squeeze(-1)
    if not continuous:
        actions = actions.long()
        
    logprobs = buffer.logprobs[:buffer.ptr]
    advs = advantages[:buffer.ptr]
    rets = returns[:buffer.ptr]

    advs = (advs - advs.mean()) / (advs.std() + 1e-8)
    T = obs.shape[0]
    metrics = {'pg_loss': [], 'vf_loss': [], 'entropy': [], 'clip_frac': []}

    for _ in range(n_epochs):
        idxs = torch.randperm(T)
        for start in range(0, T, batch_size):
            mb = idxs[start:start + batch_size]

            _, new_lp, entropy, new_val = model.get_action_and_value(obs[mb], actions[mb])

            ratio = (new_lp - logprobs[mb]).exp()
            pg_loss1 = -advs[mb] * ratio
            pg_loss2 = -advs[mb] * ratio.clamp(1 - clip_eps, 1 + clip_eps)
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()

            vf_loss = F.mse_loss(new_val.squeeze(), rets[mb])
            ent_loss = entropy.mean()
            loss = pg_loss + vf_coef * vf_loss - ent_coef * ent_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

            clip_frac = ((ratio - 1).abs() > clip_eps).float().mean()
            metrics['pg_loss'].append(pg_loss.item())
            metrics['vf_loss'].append(vf_loss.item())
            metrics['entropy'].append(ent_loss.item())
            metrics['clip_frac'].append(clip_frac.item())

    return {k: np.mean(v) for k, v in metrics.items()}


def space_dim(space):
    if hasattr(space, 'n'):
        return int(space.n)
    if getattr(space, 'shape', None) is not None:
        return int(np.prod(space.shape))
    raise ValueError(f"Unable to infer dimension from space {space}")


def train(env_id, total_steps=float('inf'), gamma=0.99, gae_lambda=0.95, rollout_steps=2048,
          hidden=64, lr=3e-4, render=False,
          return_score=False, verbose=True, silent_episodes=False,
          checkpoint=None, env_factory=None, render_factory=None,
          **ppo_kwargs):

    writer = SummaryWriter(f'runs/{env_id}') if verbose else None
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # handle terminal close and kill signals
    def _handle_signal(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGHUP, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    env = env_factory() if env_factory else gym.make(env_id)
    continuous = hasattr(env.action_space, 'shape') and env.action_space.shape != ()

    render_env = None
    if render:
        render_env = render_factory() if render_factory else gym.make(env_id, render_mode="human")
            
    obs_dim = space_dim(env.observation_space)
    act_dim = space_dim(env.action_space)

    model = ActorCritic(obs_dim, act_dim, hidden, continuous).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, eps=1e-5)
    buffer = RolloutBuffer(rollout_steps, obs_dim, act_dim if continuous else 1, device, continuous)

    if checkpoint and verbose:
        model.load_state_dict(torch.load(checkpoint, map_location=device))
        print(f'  Resumed from {checkpoint}')

    obs, _ = env.reset()
    obs = torch.tensor(obs, dtype=torch.float32).flatten().to(device)

    render_obs = None
    if render_env:
        render_obs, _ = render_env.reset()
        render_obs = np.array(render_obs).flatten()

    step = 0
    ep_count = 0
    ep_len = 0
    ep_reward = 0.0
    recent_lengths = []
    recent_rewards = []
    best_avg = 0.0
    base_ent_coef = ppo_kwargs.pop('ent_coef', 0.01)
    interrupted = False

    while step < total_steps:
        try:
            model.eval()
            buffer.reset()

            optimizer.param_groups[0]['lr'] = lr * (1.0 - step / total_steps)

            for _ in range(rollout_steps):
                with torch.no_grad():
                    action, logprob, _, value = model.get_action_and_value(obs)

                a = action.cpu().numpy()
                if not continuous:
                    a = int(a)

                next_obs, reward, terminated, truncated, _ = env.step(a)
                done = terminated or truncated
                buffer.push(obs, action, logprob, torch.tensor(reward), torch.tensor(done), value)
                obs = torch.tensor(next_obs, dtype=torch.float32).flatten().to(device)
                step += 1
                ep_len += 1
                ep_reward += reward

                if render_env:
                    with torch.no_grad():
                        render_action, _, _, _ = model.get_action_and_value(
                            torch.tensor(render_obs, dtype=torch.float32).to(device))
                    render_obs, _, term, trunc, _ = render_env.step(
                        int(render_action.cpu().numpy()) if not continuous
                        else render_action.cpu().numpy())
                    render_obs = np.array(render_obs).flatten()
                    if term or trunc:
                        render_obs, _ = render_env.reset()
                        render_obs = np.array(render_obs).flatten()

                if done:
                    ep_count += 1
                    recent_lengths.append(ep_len)
                    recent_rewards.append(ep_reward)
                    
                    if len(recent_rewards) > 100:
                        recent_rewards.pop(0)
                    ep_reward = 0.0 

                    if len(recent_lengths) >= 10:
                        avg_len = np.mean(recent_lengths)
                        ent_coef = base_ent_coef * max(0.1, 1.0 - avg_len / 1000.0)
                        ppo_kwargs['ent_coef'] = ent_coef
                        if writer:
                            writer.add_scalar('charts/ent_coef', ent_coef, step)

                    if len(recent_lengths) > 100:
                        recent_lengths.pop(0)

                    if len(recent_lengths) >= 10 and np.mean(recent_lengths) > best_avg:
                        best_avg = np.mean(recent_lengths)
                        best_path = f'checkpoints/{env_id.replace("/", "_")}_best.pt'
                        
                        if verbose:
                            os.makedirs('checkpoints', exist_ok=True)
                            torch.save(model.state_dict(), best_path)
                            print(f'  New best avg: {best_avg:.1f} — saved best checkpoint')

                    if writer:
                        writer.add_scalar('charts/episode_length', ep_len, step)
                        writer.add_scalar('charts/episode_count',  ep_count, step)

                    if verbose and not silent_episodes:
                        print(f'  Episode {ep_count} | length {ep_len} | reward {ep_reward:.1f}')

                    ep_len = 0
                    obs, _ = env.reset()
                    obs = torch.tensor(obs, dtype=torch.float32).flatten().to(device)


            with torch.no_grad():
                next_val = model.get_value(obs).squeeze().cpu()

            advs, rets = compute_gae(
                buffer.rewards[:buffer.ptr].cpu(),
                buffer.values[:buffer.ptr].cpu(),
                buffer.dones[:buffer.ptr].cpu(),
                next_val,
                gamma=gamma,
                lam=gae_lambda,
            )

            model.train()
            metrics = ppo_update(model, optimizer, buffer, rets.to(device), advs.to(device),
                     continuous=continuous, **ppo_kwargs)

            if writer:
                writer.add_scalar('losses/policy_loss', metrics['pg_loss'],  step)
                writer.add_scalar('losses/value_loss', metrics['vf_loss'],  step)
                writer.add_scalar('losses/entropy', metrics['entropy'],  step)
                writer.add_scalar('losses/clip_frac', metrics['clip_frac'], step)

            if verbose:
                print(f'Step {step:>8} | ' +
                      ' | '.join(f'{k}={v:.4f}' for k, v in metrics.items()))

        except KeyboardInterrupt:
            interrupted = True

        if interrupted:
            print('\n  Interrupted — saving checkpoint...')
            break

    if verbose:
        os.makedirs('checkpoints', exist_ok=True)
        ckpt_path = f'checkpoints/{env_id.replace("/", "_")}_latest.pt'
        torch.save(model.state_dict(), ckpt_path)
        print(f'  Saved → {ckpt_path}')
        
    env.close()
    if render_env: render_env.close()
    if writer:     writer.close()

    if return_score:
        return float(np.mean(recent_rewards)) if recent_rewards else 0.0
    return model


def tune(env_id, n_trials=50, total_steps=50_000):
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    os.makedirs('configs', exist_ok=True)
    config_file = f'configs/{env_id.replace("/", "_")}.json'

    best = {'score': -float('inf'), 'params': {}}

    def objective(trial):
        lr = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
        ent_coef = trial.suggest_float('ent_coef', 1e-4, 0.1, log=True)
        vf_coef = trial.suggest_float('vf_coef', 0.25, 1.0)
        clip_eps = trial.suggest_float('clip_eps', 0.1, 0.4)
        gamma = trial.suggest_float('gamma', 0.95, 0.999)
        gae_lambda = trial.suggest_float('gae_lambda', 0.8, 0.99)
        max_grad_norm = trial.suggest_float('max_grad_norm', 0.3, 1.0)
        hidden = trial.suggest_categorical('hidden', [64, 128, 256])
        n_epochs = trial.suggest_int('n_epochs', 2, 10)
        batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
        rollout_steps = trial.suggest_categorical('rollout_steps', [512, 1024, 2048])

        score = train(
            env_id = env_id,
            total_steps = total_steps,
            rollout_steps = rollout_steps,
            hidden = hidden,
            lr = lr,
            render = False,
            return_score = True,
            verbose = False,
            clip_eps = clip_eps,
            vf_coef = vf_coef,
            ent_coef = ent_coef,
            n_epochs = n_epochs,
            batch_size = batch_size,
            gamma = gamma,
            gae_lambda = gae_lambda,
            max_grad_norm = max_grad_norm,
        )

        if score > best['score']:
            best['score'] = score
            best['params'] = trial.params

        bar_filled = int((trial.number + 1) / n_trials * 40)
        bar = '█' * bar_filled + '░' * (40 - bar_filled)
        print(f'\r  [{bar}] {trial.number+1}/{n_trials}  best={best["score"]:.1f}',
              end='', flush=True)
        return score

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    print()

    best_params = study.best_params
    best_score = study.best_value

    with open(config_file, 'w') as f:
        json.dump({'env_id': env_id, 'score': best_score, 'params': best_params}, f, indent=2)

    print(f'  Saved → {config_file}  (score: {best_score:.1f})')
    return best_params, best_score