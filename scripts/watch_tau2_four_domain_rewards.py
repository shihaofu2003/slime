"""Refresh reward plots when completed optimizer updates change (run on the cluster)."""

import argparse
from datetime import datetime
from pathlib import Path
import re
import subprocess
import sys
import time


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-stamp', default='20260915_092700')
    parser.add_argument('--experiment-dir', type=Path,
                        default=root / 'output/experiments/tau2-async-db-count-four-domain')
    parser.add_argument('--interval', type=float, default=60)
    parser.add_argument('--target-updates', type=int, default=556)
    parser.add_argument('--run-name', action='append', help='Explicit run directory name; repeat for multiple runs')
    parser.add_argument('--batch-size', type=int, default=40)
    parser.add_argument('--title', default='Four-domain asynchronous')
    parser.add_argument('--once', action='store_true', help='Check and refresh once, then exit')
    args = parser.parse_args()
    if args.interval <= 0:
        parser.error('--interval must be positive')
    recipes = args.run_name or ('vanilla-grpo', 'progress-db-count-v1')
    offsets = dict.fromkeys(recipes, 0)
    pending = dict.fromkeys(recipes, b'')
    completed = {recipe: set() for recipe in recipes}
    plotted = {recipe: set() for recipe in recipes}
    plotted_files = dict.fromkeys(recipes)
    pattern = re.compile(rb'tau2_trainer batch=(\d+) start=[\d.]+ end=[\d.]+')
    while True:
        started = time.monotonic()
        for recipe in recipes:
            run_name = recipe if args.run_name else f'{args.run_stamp}-{recipe}-train'
            run = args.experiment_dir / 'arms/async' / run_name
            log = run / 'run.log'
            if not log.exists():
                print(f'{datetime.now().isoformat(timespec="seconds")} {recipe}: waiting for log', flush=True)
                continue
            if log.stat().st_size < offsets[recipe]:
                offsets[recipe] = 0
                pending[recipe] = b''
                completed[recipe].clear()
                plotted[recipe].clear()
            with log.open('rb') as stream:
                stream.seek(offsets[recipe])
                while chunk := stream.read(1024 * 1024):
                    lines = (pending[recipe] + chunk).split(b'\n')
                    pending[recipe] = lines.pop()
                    for line in lines:
                        completed[recipe].update(int(match[1]) for match in pattern.finditer(line))
                offsets[recipe] = stream.tell()
            print(f'{datetime.now().isoformat(timespec="seconds")} {recipe}: '
                  f'{len(completed[recipe])}/{args.target_updates} completed updates', flush=True)
            trajectory_files = tuple((str(p), p.stat().st_size) for p in
                                     sorted((run / 'trajectories').glob('*.jsonl')))
            if completed[recipe] != plotted[recipe] or trajectory_files != plotted_files[recipe]:
                output = (args.experiment_dir / 'plots' / recipe if args.run_name else
                          args.experiment_dir / 'plots' / args.run_stamp / recipe)
                result = subprocess.run([
                    sys.executable, str(root / 'scripts/plot_tau2_four_domain_rewards.py'),
                    '--run-dir', str(run), '--output-dir', str(output), '--label', recipe,
                    '--batch-size', str(args.batch_size), '--title', args.title,
                ])
                if result.returncode == 0:
                    plotted[recipe] = completed[recipe].copy()
                    plotted_files[recipe] = trajectory_files
                else:
                    print(f'{recipe}: plotting failed; will retry on the next check', flush=True)
                    if args.once:
                        return result.returncode
        if args.once or all(len(plotted[r]) >= args.target_updates for r in recipes):
            return 0
        time.sleep(max(0, args.interval - (time.monotonic() - started)))


if __name__ == '__main__':
    sys.exit(main())
