import signal
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import json
import sys

import gymnasium as gym  # type: ignore
import optuna # type: ignore

from rollout_buffer import RolloutBuffer
from actor_critic import ActorCritic
from torch.utils.tensorboard import SummaryWriter

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


def ppo_update(model,
               optimizer,
               buffer,
               returns,
               advantages,
               clip_eps=0.2,
               vf_coef=0.5,
               continuous=False,
               ent_coef=0.01,
               n_epochs=10,
               batch_size=64,
               max_grad_norm=0.5):

    obs      = buffer.obs[:buffer.ptr]
    actions  = buffer.actions[:buffer.ptr].squeeze(-1)
    if not continuous:
        actions = actions.long()
    logprobs = buffer.logprobs[:buffer.ptr]
    advs     = advantages[:buffer.ptr]
    rets     = returns[:buffer.ptr]

    hx_buf = buffer.hx[:buffer.ptr] if buffer.hx is not None else None
    cx_buf = buffer.cx[:buffer.ptr] if buffer.cx is not None else None

    advs = (advs - advs.mean()) / (advs.std() + 1e-8)
    T = obs.shape[0]
    metrics = {'pg_loss': [], 'vf_loss': [], 'entropy': [], 'clip_frac': []}

    for _ in range(n_epochs):
        idxs = torch.randperm(T)
        for start in range(0, T, batch_size):
            mb = idxs[start:start + batch_size]

            if hx_buf is not None:
                hx_mb = hx_buf[mb].unsqueeze(0)
                cx_mb = cx_buf[mb].unsqueeze(0)  # type: ignore
            else:
                hx_mb = cx_mb = None

            _, new_lp, entropy, new_val, _, _ = model.get_action_and_value(
                obs[mb], actions[mb], hx=hx_mb, cx=cx_mb)

            ratio    = (new_lp - logprobs[mb]).exp()
            pg_loss1 = -advs[mb] * ratio
            pg_loss2 = -advs[mb] * ratio.clamp(1 - clip_eps, 1 + clip_eps)
            pg_loss  = torch.max(pg_loss1, pg_loss2).mean()

            vf_loss  = F.mse_loss(new_val.squeeze(), rets[mb])
            ent_loss = entropy.mean()
            loss     = pg_loss + vf_coef * vf_loss - ent_coef * ent_loss

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


def _get_device(obs_shape=None):
    if torch.cuda.is_available():
        return torch.device('cuda'), 'CUDA'
    elif torch.backends.mps.is_available() and obs_shape is not None:
        return torch.device('mps'), 'MPS'
    else:
        return torch.device('cpu'), 'CPU'


def _save_metrics(env_id, variant, step, ep_count, recent_rewards,
                  start_time, steps_to_threshold):
    """Save comparison metrics to metrics/<variant>/<env>.json"""
    if not recent_rewards:
        return
    metrics_dir = f'metrics/{variant}'
    os.makedirs(metrics_dir, exist_ok=True)
    safe_env = env_id.replace('/', '_')
    report = {
        'env_id':             env_id,
        'variant':            variant,
        'mean_reward':        float(np.mean(recent_rewards)),
        'std_reward':         float(np.std(recent_rewards)),
        'max_reward':         float(np.max(recent_rewards)),
        'total_steps':        step,
        'total_episodes':     ep_count,
        'wall_time_s':        round(time.time() - start_time, 1),
        'steps_to_threshold': steps_to_threshold,
    }
    with open(f'{metrics_dir}/{safe_env}.json', 'w') as f:
        json.dump(report, f, indent=2)


def train(env_id,
          total_steps=float('inf'),
          gamma=0.99,
          gae_lambda=0.95,
          rollout_steps=2048,
          hidden=64,
          lr=3e-4,
          render=False,
          return_score=False,
          verbose=True,
          silent_episodes=False,
          checkpoint=None,
          env_factory=None,
          render_factory=None,
          use_lstm=False,
          variant=None,
          **ppo_kwargs):

    writer = SummaryWriter(f'runs/{env_id}') if verbose else None

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

    obs_raw, _ = env.reset()
    obs_arr = np.array(obs_raw)
    obs_space_shape = obs_arr.shape
    if len(obs_space_shape) == 3:
        if obs_space_shape[2] in (1, 3, 4) and obs_space_shape[2] < obs_space_shape[0]:
            obs_shape     = (obs_space_shape[2], obs_space_shape[0], obs_space_shape[1])
            needs_permute = True
        else:
            obs_shape     = obs_space_shape
            needs_permute = False
    else:
        obs_shape     = None
        needs_permute = False

    device, device_name = _get_device(obs_shape)
    if verbose:
        print(f'  Using {device_name}')

    model     = ActorCritic(obs_dim, act_dim, hidden, continuous,
                            obs_shape=obs_shape, use_lstm=use_lstm).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, eps=1e-5)
    buffer    = RolloutBuffer(rollout_steps, obs_dim, act_dim if continuous else 1,
                              device, continuous, obs_shape=obs_shape,
                              hidden_size=hidden if use_lstm else None)

    if checkpoint and verbose:
        model.load_state_dict(torch.load(checkpoint, map_location=device))
        print(f'  Resumed from {checkpoint}')

    def _to_tensor(o):
        if obs_shape:
            t = torch.tensor(np.array(o), dtype=torch.float32)
            if needs_permute:
                t = t.permute(2, 0, 1)
        else:
            t = torch.tensor(o, dtype=torch.float32).flatten()
        return t.to(device)

    obs = _to_tensor(obs_raw)

    hx, cx = model.get_initial_state(device) if use_lstm else (None, None)

    render_obs = None
    if render_env:
        render_obs, _ = render_env.reset()
        render_obs = np.array(render_obs)

    step               = 0
    ep_count           = 0
    ep_len             = 0
    ep_reward          = 0.0
    recent_lengths     = []
    recent_rewards     = []
    best_avg           = 0.0
    base_ent_coef      = ppo_kwargs.pop('ent_coef', 0.01)
    interrupted        = False
    start_time         = time.time()
    steps_to_threshold = None
    variant_name       = variant or ('ppo_lstm' if use_lstm else 'ppo')

    while step < total_steps:
        try:
            model.eval()
            buffer.reset()

            optimizer.param_groups[0]['lr'] = lr * (1.0 - step / total_steps)

            for _ in range(rollout_steps):
                with torch.no_grad():
                    action, logprob, _, value, hx, cx = model.get_action_and_value(
                        obs, hx=hx, cx=cx)

                a = action.cpu().numpy()
                if not continuous:
                    a = int(a)
                else:
                    a = a.flatten()

                next_obs, reward, terminated, truncated, _ = env.step(a)
                done = terminated or truncated

                hx_store = hx.squeeze() if hx is not None else None
                cx_store = cx.squeeze() if cx is not None else None
                buffer.push(obs, action, logprob, torch.tensor(reward),
                            torch.tensor(done), value, hx_store, cx_store)
                obs = _to_tensor(next_obs)

                step      += 1
                ep_len    += 1
                ep_reward += reward

                if render_env:
                    render_t = torch.tensor(np.array(render_obs), dtype=torch.float32)
                    if needs_permute:
                        render_t = render_t.permute(2, 0, 1)
                    elif not obs_shape:
                        render_t = render_t.flatten()
                    with torch.no_grad():
                        render_action, _, _, _, _, _ = model.get_action_and_value(
                            render_t.to(device), hx=hx, cx=cx)
                    render_obs, _, term, trunc, _ = render_env.step(
                        int(render_action.cpu().numpy()) if not continuous
                        else render_action.cpu().numpy().flatten())
                    render_obs = np.array(render_obs)
                    if term or trunc:
                        render_obs, _ = render_env.reset()
                        render_obs = np.array(render_obs)

                if done:
                    ep_count += 1
                    recent_lengths.append(ep_len)
                    recent_rewards.append(ep_reward)

                    if len(recent_rewards) > 100:
                        recent_rewards.pop(0)
                    ep_reward = 0.0

                    if len(recent_lengths) >= 10:
                        avg_len  = np.mean(recent_lengths)
                        ent_coef = base_ent_coef * max(0.1, 1.0 - avg_len / 1000.0)
                        ppo_kwargs['ent_coef'] = ent_coef
                        if writer:
                            writer.add_scalar('charts/ent_coef', ent_coef, step)

                    if len(recent_lengths) > 100:
                        recent_lengths.pop(0)

                    if steps_to_threshold is None and len(recent_rewards) >= 10:
                        if np.mean(recent_rewards) > 0:
                            steps_to_threshold = step

                    if len(recent_lengths) >= 10 and np.mean(recent_lengths) > best_avg:
                        best_avg  = np.mean(recent_lengths)
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
                    obs_raw, _ = env.reset()
                    obs = _to_tensor(obs_raw)
                    if use_lstm:
                        hx, cx = model.get_initial_state(device)

            with torch.no_grad():
                next_val = model.get_value(obs, hx=hx, cx=cx).squeeze().cpu()

            advs, rets = compute_gae(
                buffer.rewards[:buffer.ptr].cpu(),
                buffer.values[:buffer.ptr].cpu(),
                buffer.dones[:buffer.ptr].cpu(),
                next_val,
                gamma=gamma,
                lam=gae_lambda,
            )

            model.train()
            metrics = ppo_update(model, optimizer, buffer,
                                 rets.to(device), advs.to(device),
                                 continuous=continuous, **ppo_kwargs)

            if writer:
                writer.add_scalar('losses/policy_loss', metrics['pg_loss'],  step)
                writer.add_scalar('losses/value_loss',  metrics['vf_loss'],  step)
                writer.add_scalar('losses/entropy',     metrics['entropy'],  step)
                writer.add_scalar('losses/clip_frac',   metrics['clip_frac'], step)

            if verbose:
                print(f'Step {step:>8} | ' +
                      ' | '.join(f'{k}={v:.4f}' for k, v in metrics.items()))

        except KeyboardInterrupt:
            if return_score:
                raise
            interrupted = True

        if interrupted:
            if verbose:
                print('\n  Interrupted — saving checkpoint...')
            break

    if verbose:
        os.makedirs('checkpoints', exist_ok=True)
        ckpt_path = f'checkpoints/{env_id.replace("/", "_")}_latest.pt'
        torch.save(model.state_dict(), ckpt_path)
        print(f'  Saved → {ckpt_path}')
        _save_metrics(env_id, variant_name, step, ep_count, recent_rewards,
                      start_time, steps_to_threshold)

    env.close()
    if render_env: render_env.close()
    if writer:     writer.close()

    if return_score:
        return float(np.mean(recent_rewards)) if recent_rewards else 0.0
    return model


def tune(env_id, n_trials=50, total_steps=50_000, env_factory=None):
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    os.makedirs('../configs', exist_ok=True)
    config_file = f'../configs/{env_id.replace("/", "_")}.json'

    try:
        _tty = open('/dev/tty', 'w')
    except:
        _tty = sys.stdout

    _sample_env = env_factory() if env_factory else gym.make(env_id)
    _sample_obs, _ = _sample_env.reset()
    _sample_shape = np.array(_sample_obs).shape
    _sample_env.close()
    _obs_shape_hint = _sample_shape if len(_sample_shape) == 3 else None
    _, device_name = _get_device(_obs_shape_hint)
    _tty.write(f'  Using {device_name}\n')
    _tty.flush()

    best = {'score': -float('inf'), 'params': {}}

    def _bar(trial_num, best_score):
        bar_filled = int(trial_num / n_trials * 40)
        bar        = '█' * bar_filled + '░' * (40 - bar_filled)
        _tty.write(f'\r  [{bar}] {trial_num}/{n_trials}  best={best_score:.1f}')
        _tty.flush()

    study = optuna.create_study(direction='maximize')
    _interrupted = [False]

    def objective(trial):
        if _interrupted[0]:
            study.stop()
            raise optuna.exceptions.TrialPruned()

        try:
            lr            = trial.suggest_float('lr',            1e-5, 1e-3, log=True)
            ent_coef      = trial.suggest_float('ent_coef',      1e-4, 0.1,  log=True)
            vf_coef       = trial.suggest_float('vf_coef',       0.25, 1.0)
            clip_eps      = trial.suggest_float('clip_eps',      0.1,  0.4)
            gamma         = trial.suggest_float('gamma',         0.95, 0.999)
            gae_lambda    = trial.suggest_float('gae_lambda',    0.8,  0.99)
            max_grad_norm = trial.suggest_float('max_grad_norm', 0.3,  1.0)
            # expanded search space for LSTM
            hidden        = trial.suggest_categorical('hidden',        [128, 256, 512])
            n_epochs      = trial.suggest_int('n_epochs',              2, 6)
            batch_size    = trial.suggest_categorical('batch_size',    [32, 64, 128])
            rollout_steps = trial.suggest_categorical('rollout_steps', [512, 1024, 2048, 4096])

            score = train(
                env_id        = env_id,
                total_steps   = total_steps,
                rollout_steps = rollout_steps,
                hidden        = hidden,
                env_factory   = env_factory,
                lr            = lr,
                render        = False,
                return_score  = True,
                verbose       = False,
                use_lstm      = True,
                clip_eps      = clip_eps,
                vf_coef       = vf_coef,
                ent_coef      = ent_coef,
                n_epochs      = n_epochs,
                batch_size    = batch_size,
                gamma         = gamma,
                gae_lambda    = gae_lambda,
                max_grad_norm = max_grad_norm,
            )
        except KeyboardInterrupt:
            _interrupted[0] = True
            study.stop()
            raise optuna.exceptions.TrialPruned()

        if score > best['score']:
            best['score']  = score
            best['params'] = trial.params

        _bar(trial.number + 1, best['score'])
        return score

    _bar(0, float('-inf'))
    study.optimize(objective, n_trials=n_trials)

    _tty.write('\n')
    _tty.flush()
    if _tty != sys.stdout:
        _tty.close()

    if study.best_trial:
        best_params = study.best_params
        best_score  = study.best_value
        with open(config_file, 'w') as f:
            json.dump({'env_id': env_id, 'score': best_score, 'params': best_params}, f, indent=2)
        print(f'  Saved → {config_file}  (score: {best_score:.1f})')
        if _interrupted[0]:
            sys.exit(1)
        return study.best_params, study.best_value

    if _interrupted[0]:
        sys.exit(1)
    return {}, 0.0