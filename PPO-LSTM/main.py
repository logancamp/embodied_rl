import argparse
import json
import os
import gymnasium as gym # type: ignore
from gymnasium.wrappers import NormalizeObservation, GrayscaleObservation, ResizeObservation, FrameStackObservation # type: ignore
from ppo import train, tune
from wrappers import PhaseSwitch, MetricsPrinter, strip_timelimit
import ale_py  # type: ignore
import gymnasium # type: ignore

gymnasium.register_envs(ale_py)

ENV_ID = 'CartPole-v1'

def config_path(env_id):
    return f'../configs/{env_id.replace("/", "_")}.json'


# ── Format helpers ─────────────────────────────────────────────────────────────
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

def _fmt_game(c, l, r):
    return f'  Episode {c} | score {r:.0f} | length {l}'


# ── Factory helpers ────────────────────────────────────────────────────────────
def _make(env_id,
          render=False,
          normalize_obs=False,
          strip_tl=False,
          phase_threshold=None,
          extra_wrapper=None,
          fmt_fn=_fmt_reward,
          silent=False):

    kw = {'render_mode': 'human'} if render else {}
    env = gym.make(env_id, **kw)
    if strip_tl:
        env = strip_timelimit(env)
    if extra_wrapper is not None:
        env = extra_wrapper(env)
    if phase_threshold is not None:
        env = PhaseSwitch(env, threshold=phase_threshold)
    if normalize_obs:
        env = NormalizeObservation(env)
    if not render and not silent:
        env = MetricsPrinter(env, fmt_fn)
    return env


def _make_atari(env_id, render=False, fmt_fn=_fmt_game, silent=False):
    ale_py.ALEInterface.setLoggerMode(ale_py.LoggerMode.Error)
    kw = {'render_mode': 'human'} if render else {}
    env = gym.make(env_id, **kw)
    env = GrayscaleObservation(env)
    env = ResizeObservation(env, (84, 84))
    env = FrameStackObservation(env, 4)
    if not render and not silent:
        env = MetricsPrinter(env, fmt_fn)
    return env


def _factory(env_id, **kw):
    """Returns (train_factory, render_factory, tune_factory) triple."""
    return (
        lambda: _make(env_id, render=False, **kw),
        lambda: _make(env_id, render=True,  **{k: v for k, v in kw.items() if k != 'fmt_fn'}),
        lambda: _make(env_id, render=False, silent=True, **kw),
    )


# ── Env factories ──────────────────────────────────────────────────────────────
FACTORIES = {
    # ── Atari ─────────────────────────────────────────────────────────────────
    'ALE/MsPacman-v5': (
        lambda: _make_atari('ALE/MsPacman-v5'),
        lambda: _make_atari('ALE/MsPacman-v5', render=True),
        lambda: _make_atari('ALE/MsPacman-v5', silent=True),
    ),
    'ALE/Pong-v5': (
        lambda: _make_atari('ALE/Pong-v5'),
        lambda: _make_atari('ALE/Pong-v5', render=True),
        lambda: _make_atari('ALE/Pong-v5', silent=True),
    ),
    'ALE/Breakout-v5': (
        lambda: _make_atari('ALE/Breakout-v5'),
        lambda: _make_atari('ALE/Breakout-v5', render=True),
        lambda: _make_atari('ALE/Breakout-v5', silent=True),
    ),
    'ALE/SpaceInvaders-v5': (
        lambda: _make_atari('ALE/SpaceInvaders-v5'),
        lambda: _make_atari('ALE/SpaceInvaders-v5', render=True),
        lambda: _make_atari('ALE/SpaceInvaders-v5', silent=True),
    ),
    'ALE/Asteroids-v5': (
        lambda: _make_atari('ALE/Asteroids-v5'),
        lambda: _make_atari('ALE/Asteroids-v5', render=True),
        lambda: _make_atari('ALE/Asteroids-v5', silent=True),
    ),
    'ALE/Pitfall-v5': (
        lambda: _make_atari('ALE/Pitfall-v5'),
        lambda: _make_atari('ALE/Pitfall-v5', render=True),
        lambda: _make_atari('ALE/Pitfall-v5', silent=True),
    ),
    'ALE/Centipede-v5': (
        lambda: _make_atari('ALE/Centipede-v5'),
        lambda: _make_atari('ALE/Centipede-v5', render=True),
        lambda: _make_atari('ALE/Centipede-v5', silent=True),
    ),
    'ALE/DonkeyKong-v5': (
        lambda: _make_atari('ALE/DonkeyKong-v5'),
        lambda: _make_atari('ALE/DonkeyKong-v5', render=True),
        lambda: _make_atari('ALE/DonkeyKong-v5', silent=True),
    ),

    # ── CarRacing ─────────────────────────────────────────────────────────────
    'CarRacing-v3': (
        lambda: _make('CarRacing-v3', fmt_fn=_fmt_reward),
        lambda: _make('CarRacing-v3', render=True),
        lambda: _make('CarRacing-v3', silent=True),
    ),

    # ── Classic control ───────────────────────────────────────────────────────
    'CartPole-v1': _factory(
        'CartPole-v1', strip_tl=True, fmt_fn=_fmt_length),

    'Acrobot-v1': _factory(
        'Acrobot-v1', fmt_fn=_fmt_length),

    'MountainCarContinuous-v0': _factory(
        'MountainCarContinuous-v0', fmt_fn=_fmt_reward),

    # ── Pendulum ──────────────────────────────────────────────────────────────
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

    # ── MuJoCo locomotion ─────────────────────────────────────────────────────
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

    # ── MuJoCo balance ────────────────────────────────────────────────────────
    'InvertedPendulum-v4': _factory(
        'InvertedPendulum-v4', normalize_obs=True, strip_tl=True, fmt_fn=_fmt_length),

    'InvertedDoublePendulum-v4': _factory(
        'InvertedDoublePendulum-v4', normalize_obs=True, strip_tl=True, fmt_fn=_fmt_length),
}
# ──────────────────────────────────────────────────────────────────────────────


def run_train(env_id, checkpoint=None, total_steps=None):
    path = config_path(env_id)

    if os.path.exists(path):
        with open(path) as f:
            cfg = json.load(f)
        params = cfg['params']
        print(f'  Loaded config for {env_id} (tuned score: {cfg["score"]:.1f})')
    else:
        print(f'  No config found for {env_id} — tuning first...')
        _, _, tune_factory = FACTORIES.get(env_id, (None, None, None))
        params, _ = tune(env_id, env_factory=tune_factory)

    env_factory, render_factory, _ = FACTORIES.get(
        env_id,
        (lambda: gym.make(env_id),
         lambda: gym.make(env_id, render_mode='human'),
         None)
    )

    train(
        env_id          = env_id,
        total_steps     = total_steps or float('inf'),  # type: ignore
        render          = True,
        checkpoint      = checkpoint,
        env_factory     = env_factory,
        render_factory  = render_factory,
        silent_episodes = env_id in FACTORIES,
        use_lstm        = True,
        gamma           = params.pop('gamma',         0.99),
        gae_lambda      = params.pop('gae_lambda',    0.95),
        rollout_steps   = params.pop('rollout_steps', 2048),
        hidden          = params.pop('hidden',        64),
        lr              = params.pop('lr',            3e-4),
        **params,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--tune',        action='store_true')
    parser.add_argument('--env',         type=str, default=ENV_ID)
    parser.add_argument('--trials',      type=int, default=50)
    parser.add_argument('--checkpoint',  type=str, default=None)
    parser.add_argument('--total-steps', type=int, default=None)
    args = parser.parse_args()

    if args.tune:
        _, _, tune_factory = FACTORIES.get(args.env, (None, None, None))
        tune(args.env, args.trials, env_factory=tune_factory, total_steps=args.total_steps)
    else:
        run_train(args.env, args.checkpoint, args.total_steps)