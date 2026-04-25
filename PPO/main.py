import argparse
import json
import os
from ppo import train, tune

ENV_ID = 'CartPole-v1'

def config_path(env_id):
    return f'configs/{env_id.replace("/", "_")}.json'

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

    train(
        env_id = env_id,
        total_steps = float('inf'),  # type: ignore
        render = True,
        checkpoint = checkpoint,
        gamma = params.pop('gamma', .99),
        gae_lambda = params.pop('gae_lambda', 0.95),
        rollout_steps = params.pop('rollout_steps', 2048),
        hidden = params.pop('hidden', 64),
        lr = params.pop('lr', 3e-4),
        **params,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--tune', action='store_true')
    parser.add_argument('--env', type=str, default=ENV_ID)
    parser.add_argument('--trials', type=int, default=50)
    parser.add_argument('--checkpoint', type=str, default=None)
    args = parser.parse_args()

    if args.tune:
        tune(args.env, args.trials)
    else:
        run_train(args.env, args.checkpoint)