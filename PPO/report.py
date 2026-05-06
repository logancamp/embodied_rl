"""
report.py — comparison report across PPO variants and environments.

Each training run saves a metrics JSON to metrics/<variant>/<env_id>.json.
This script reads all of them and prints a comparison table.

Run with: make report
"""

import json
import os
from pathlib import Path


METRICS_DIR = Path('metrics')


def load_all():
    """Load all metrics files into {variant: {env_id: metrics}}."""
    data = {}
    if not METRICS_DIR.exists():
        return data
    for variant_dir in sorted(METRICS_DIR.iterdir()):
        if not variant_dir.is_dir():
            continue
        variant = variant_dir.name
        data[variant] = {}
        for f in sorted(variant_dir.glob('*.json')):
            env_id = f.stem.replace('_', '/', 1) if f.stem.startswith('ALE') else f.stem
            with open(f) as fp:
                data[variant][env_id] = json.load(fp)
    return data


def print_table(data):
    if not data:
        print('  No metrics found. Run training first to generate metrics.')
        print('  Metrics are saved to metrics/<variant>/<env>.json')
        return

    variants = list(data.keys())
    all_envs = sorted({env for v in data.values() for env in v})

    col_w = 24

    # Header
    print()
    print('  PPO Variant Comparison Report')
    print('  ' + '─' * (col_w + len(variants) * 20))
    header = f'  {"Environment":<{col_w}}' + ''.join(f'{v:>20}' for v in variants)
    print(header)
    print('  ' + '─' * (col_w + len(variants) * 20))

    metrics_to_show = [
        ('mean_reward',        'Mean Reward'),
        ('std_reward',         'Std Reward'),
        ('max_reward',         'Max Reward'),
        ('steps_to_threshold', 'Steps→Threshold'),
        ('total_steps',        'Total Steps'),
        ('wall_time_s',        'Wall Time (s)'),
    ]

    for env in all_envs:
        print(f'\n  {env}')
        for key, label in metrics_to_show:
            row = f'    {label:<{col_w-4}}'
            for variant in variants:
                val = data[variant].get(env, {}).get(key)
                if val is None:
                    row += f'{"—":>20}'
                elif isinstance(val, float):
                    row += f'{val:>20.1f}'
                else:
                    row += f'{str(val):>20}'
            print(row)

    print()
    print('  ' + '─' * (col_w + len(variants) * 20))
    print()


if __name__ == '__main__':
    data = load_all()
    print_table(data)