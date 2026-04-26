import argparse
import json
import os
import gymnasium as gym # type: ignore
from gymnasium.wrappers import NormalizeObservation # type: ignore
from ppo import train, tune
from wrappers import PhaseSwitch, MetricsPrinter, strip_timelimit

ENV_ID = 'CartPole-v1'

def config_path(env_id):
    return f'configs/{env_id.replace("/", "_")}.json'


# ── Helpers ────────────────────────────────────────────────────────────────────
def _fmt_reward(c, l, r):
    return f'  Episode {c} | reward {r:.1f}'

def _fmt_length(c, l, r):
    return f'  Episode {c} | length {l}'

def _fmt_lunar(c, l, r):
    status = 'LANDED ✓' if r > 100 else 'CRASHED ✗' if r < -50 else 'ok'
    return f'  Episode {c} | reward {r:.1f} | {status}'

def _fmt_walk(c, l, r):
    status = 'FELL ✗' if r < -50 else 'walking' if r > 100 else 'struggling'
    return f'  Episode {c} | reward {r:.1f} | {status}'

def _fmt_mujoco(c, l, r):
    return f'  Episode {c} | reward {r:.1f} | length {l}'


def _make(env_id, render=False, normalize_obs=False,
          strip_tl=False, phase_threshold=None, fmt_fn=_fmt_reward):
    """Factory helper — builds the env with requested wrappers."""
    kw = {'render_mode': 'human'} if render else {}
    env = gym.make(env_id, **kw)
    if strip_tl:
        env = strip_timelimit(env)
    if phase_threshold is not None:
        env = PhaseSwitch(env, threshold=phase_threshold)
    if normalize_obs:
        env = NormalizeObservation(env)
    if not render:
        env = MetricsPrinter(env, fmt_fn)
    return env


def _factory(env_id, **kw):
    """Returns (train_factory, render_factory) pair."""
    return (
        lambda: _make(env_id, render=False, **kw),
        lambda: _make(env_id, render=True,  **{k: v for k, v in kw.items() if k != 'fmt_fn'}),
    )


# ── Env factories ──────────────────────────────────────────────────────────────
FACTORIES = {
    # ── Classic control ───────────────────────────────────────────────────────
    'CartPole-v1': _factory(
        'CartPole-v1', strip_tl=True, fmt_fn=_fmt_length),

    'Acrobot-v1': _factory(
        'Acrobot-v1', fmt_fn=_fmt_length),          # shorter = better

    'MountainCar-v0': _factory(
        'MountainCar-v0', fmt_fn=_fmt_reward),       # sparse, reward only at top

    'MountainCarContinuous-v0': _factory(
        'MountainCarContinuous-v0', fmt_fn=_fmt_reward),

    # ── Pendulum — phase switch to infinite once mastered ─────────────────────
    'Pendulum-v1': _factory(
        'Pendulum-v1', phase_threshold=-200, fmt_fn=_fmt_reward),

    # ── Box2D ─────────────────────────────────────────────────────────────────
    'LunarLander-v3': _factory(
        'LunarLander-v3', fmt_fn=_fmt_lunar),

    'LunarLanderContinuous-v3': _factory(
        'LunarLanderContinuous-v3', fmt_fn=_fmt_lunar),

    'BipedalWalker-v3': _factory(
        'BipedalWalker-v3', fmt_fn=_fmt_walk),

    'BipedalWalkerHardcore-v3': _factory(
        'BipedalWalkerHardcore-v3', fmt_fn=_fmt_walk),

    # ── MuJoCo locomotion — obs normalization essential ───────────────────────
    'HalfCheetah-v4': _factory(
        'HalfCheetah-v4', normalize_obs=True, fmt_fn=_fmt_mujoco),

    'Hopper-v4': _factory(
        'Hopper-v4', normalize_obs=True, fmt_fn=_fmt_mujoco),

    'Walker2d-v4': _factory(
        'Walker2d-v4', normalize_obs=True, fmt_fn=_fmt_mujoco),

    'Ant-v4': _factory(
        'Ant-v4', normalize_obs=True, fmt_fn=_fmt_mujoco),

    'Humanoid-v4': _factory(
        'Humanoid-v4', normalize_obs=True, fmt_fn=_fmt_mujoco),

    'HumanoidStandup-v4': _factory(
        'HumanoidStandup-v4', normalize_obs=True, fmt_fn=_fmt_mujoco),

    'Swimmer-v4': _factory(
        'Swimmer-v4', normalize_obs=True, fmt_fn=_fmt_mujoco),

    # ── MuJoCo manipulation ───────────────────────────────────────────────────
    'Reacher-v4': _factory(
        'Reacher-v4', normalize_obs=True, fmt_fn=_fmt_mujoco),

    'Pusher-v4': _factory(
        'Pusher-v4', normalize_obs=True, fmt_fn=_fmt_mujoco),

    # ── MuJoCo balance — strip timelimit, balance forever ────────────────────
    'InvertedPendulum-v4': _factory(
        'InvertedPendulum-v4', normalize_obs=True, strip_tl=True, fmt_fn=_fmt_length),

    'InvertedDoublePendulum-v4': _factory(
        'InvertedDoublePendulum-v4', normalize_obs=True, strip_tl=True, fmt_fn=_fmt_length),
}
# ──────────────────────────────────────────────────────────────────────────────


def run_train(env_id, checkpoint=None):
    path = config_path(env_id)

    if os.path.exists(path):
        with open(path) as f:
            cfg = json.load(f)
        params = cfg['params']
        print(f'  Loaded config for {env_id} (tuned score: {cfg["score"]:.1f})')
    else:
        print(f'  No config found for {env_id} — tuning first...')
        params, _ = tune(env_id)

    env_factory, render_factory = FACTORIES.get(
        env_id,
        (lambda: gym.make(env_id),
         lambda: gym.make(env_id, render_mode='human'))
    )

    train(
        env_id          = env_id,
        total_steps     = float('inf'),  # type: ignore
        render          = True,
        checkpoint      = checkpoint,
        env_factory     = env_factory,
        render_factory  = render_factory,
        silent_episodes = env_id in FACTORIES,
        gamma           = params.pop('gamma',         0.99),
        gae_lambda      = params.pop('gae_lambda',    0.95),
        rollout_steps   = params.pop('rollout_steps', 2048),
        hidden          = params.pop('hidden',        64),
        lr              = params.pop('lr',            3e-4),
        **params,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--tune',       action='store_true')
    parser.add_argument('--env',        type=str, default=ENV_ID)
    parser.add_argument('--trials',     type=int, default=50)
    parser.add_argument('--checkpoint', type=str, default=None)
    args = parser.parse_args()

    if args.tune:
        tune(args.env, args.trials)
    else:
        run_train(args.env, args.checkpoint)