"""Summarize completed SFT updates, loss, timing and sample/token coverage."""
import argparse
import json
import math
import re
from pathlib import Path


def read_training_log(text, expected_updates):
    losses = {}
    for line in text.splitlines():
        step = re.search(r"'train/step': (\d+)", line)
        loss = re.search(r"'train/loss': ([^,}]+)", line)
        if step and loss:
            losses[int(step[1])] = float(loss[1])
    if set(losses) != set(range(expected_updates)):
        raise ValueError(f'expected {expected_updates} logged updates, found {len(losses)}')
    if not all(math.isfinite(value) for value in losses.values()):
        raise ValueError('non-finite training loss')
    elapsed = re.search(r'training_elapsed_seconds=(\d+)', text)
    return dict(updates=len(losses), first_loss=losses[0], final_loss=losses[expected_updates-1],
                minimum_loss=min(losses.values()), maximum_loss=max(losses.values()),
                elapsed_seconds=int(elapsed[1]) if elapsed else None,
                loss_by_update=[losses[i] for i in range(expected_updates)])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    parser.add_argument('arm')
    parser.add_argument('mode', choices=('smoke','full'))
    parser.add_argument('--run-log', type=Path)
    args = parser.parse_args()
    run = args.root / args.arm / args.mode
    log = args.run_log or run / 'train.log'
    stats = json.loads((args.root/'data/token_stats.json').read_text())['smoke' if args.mode == 'smoke' else args.arm]
    report = read_training_log(log.read_text(errors='replace'), stats['updates'])
    iteration = int((run/'checkpoints/latest_checkpointed_iteration.txt').read_text())
    if iteration != stats['updates']-1:
        raise ValueError(f'final checkpoint iteration {iteration}, expected {stats["updates"]-1}')
    report.update(arm=args.arm, mode=args.mode, coverage=stats, final_iteration=iteration, run_log=str(log))
    (run/'training_report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k != 'loss_by_update'},indent=2))


if __name__ == '__main__':
    main()
