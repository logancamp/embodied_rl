"""
report.py — comparison report across PPO variants and environments.

Each training run saves a metrics JSON to metrics/<variant>/<env_id>.json.
This script reads all of them and prints + saves a comparison table.

Run with: make report
Optional: point at a combined metrics folder with --metrics-dir
Example:  python report.py --metrics-dir ~/embodied_rl/combined_metrics
"""

import json
import os
import argparse
from pathlib import Path
from datetime import datetime


def load_all(metrics_dir):
    """Load all metrics files into {variant: {env_id: metrics}}."""
    data = {}
    metrics_path = Path(metrics_dir)
    if not metrics_path.exists():
        return data
    for variant_dir in sorted(metrics_path.iterdir()):
        if not variant_dir.is_dir():
            continue
        variant = variant_dir.name
        data[variant] = {}
        for f in sorted(variant_dir.glob('*.json')):
            env_id = f.stem.replace('_', '/', 1) if f.stem.startswith('ALE') else f.stem
            with open(f) as fp:
                data[variant][env_id] = json.load(fp)
    return data


def build_table(data):
    """Build report lines and return as a list of strings."""
    lines = []
    if not data:
        lines.append('  No metrics found. Run training first to generate metrics.')
        lines.append('  Metrics are saved to metrics/<variant>/<env>.json')
        return lines

    variants = list(data.keys())
    all_envs = sorted({env for v in data.values() for env in v})

    col_w = 24

    lines.append('')
    lines.append('  PPO Variant Comparison Report')
    lines.append(f'  Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    lines.append('  ' + '─' * (col_w + len(variants) * 20))
    lines.append(f'  {"Environment":<{col_w}}' + ''.join(f'{v:>20}' for v in variants))
    lines.append('  ' + '─' * (col_w + len(variants) * 20))

    metrics_to_show = [
        ('mean_reward',        'Mean Reward'),
        ('std_reward',         'Std Reward'),
        ('max_reward',         'Max Reward'),
        ('steps_to_threshold', 'Steps→Threshold'),
        ('total_steps',        'Total Steps'),
        ('wall_time_s',        'Wall Time (s)'),
    ]

    for env in all_envs:
        lines.append(f'\n  {env}')
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
            lines.append(row)

    lines.append('')
    lines.append('  ' + '─' * (col_w + len(variants) * 20))
    lines.append('')
    return lines


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--metrics-dir', type=str, default='metrics',
                        help='Path to metrics folder (default: ./metrics)')
    parser.add_argument('--output',      type=str, default=None,
                        help='Save report to this file (default: reports/report_<timestamp>.txt)')
    args = parser.parse_args()

    data  = load_all(args.metrics_dir)
    lines = build_table(data)

    # print to terminal
    for line in lines:
        print(line)

    # save to file
    if data:
        out_dir = Path('reports')
        out_dir.mkdir(exist_ok=True)
        out_path = args.output or str(out_dir / f'report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')
        with open(out_path, 'w') as f:
            f.write('\n'.join(lines))
        print(f'  Report saved → {out_path}')