import argparse
import json
import os
import gymnasium as gym # type: ignore
from ppo import train, tune
from wrappers import PhaseSwitch, MetricsPrinter, strip_timelimit

ENV_ID = 'CartPole-v1'

def config_path(env_id):
    return f'configs/{env_id.replace("/", "_")}.json'


# ── Env factories ──────────────────────────────────────────────────────────────
def make_cartpole():
    env = gym.make('CartPole-v1')
    env = strip_timelimit(env)
    return MetricsPrinter(env,
        lambda c, l, r: f'  Episode {c} | length {l}')

def make_cartpole_render():
    env = gym.make('CartPole-v1', render_mode='human')
    return strip_timelimit(env)

def make_pendulum():
    env = gym.make('Pendulum-v1')
    env = PhaseSwitch(env, threshold=-200)
    return MetricsPrinter(env,
        lambda c, l, r: f'  Episode {c} | reward {r:.1f}')

def make_pendulum_render():
    env = gym.make('Pendulum-v1', render_mode='human')
    return PhaseSwitch(env, threshold=-200)

def make_lunarlander():
    env = gym.make('LunarLander-v3')
    return MetricsPrinter(env,
        lambda c, l, r: f'  Episode {c} | reward {r:.1f} | {"LANDED ✓" if r > 100 else "CRASHED ✗" if r < -50 else "ok"}')

def make_lunarlander_render():
    env = gym.make('LunarLander-v3', render_mode='human')
    return env

# Default — plain gym.make, no custom printing
def default_factory(env_id):
    return lambda: gym.make(env_id)

def default_render_factory(env_id):
    return lambda: gym.make(env_id, render_mode='human')

FACTORIES = {
    'CartPole-v1':    (make_cartpole,    make_cartpole_render),
    'Pendulum-v1':    (make_pendulum,    make_pendulum_render),
    'LunarLander-v3': (make_lunarlander, make_lunarlander_render),
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
        (default_factory(env_id), default_render_factory(env_id))
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