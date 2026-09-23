"""Plot official rewards for completed optimizer batches from live trajectory logs."""
import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path


def plot_prefilter(records, output_dir, label):
    """Include every dumped final attempt, regardless of group acceptance."""
    values = list(records.values())
    if not values:
        return
    rows = []
    total = 0.0
    for start in range(0, len(values), 40):
        end = min(start + 40, len(values))
        batch = values[start:end]
        total += sum(batch)
        window = values[max(0, end - 400):end]
        rows.append(dict(trajectories=end, window_trajectories=len(batch),
                         reward=sum(batch) / len(batch),
                         moving_average_400=sum(window) / len(window),
                         cumulative_mean=total / end))
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / 'reward_prefilter.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 4.8))
    x = [r['trajectories'] for r in rows]
    ax.plot(x, [r['reward'] for r in rows], color='#96b6d9', alpha=.6,
            label='40-trajectory mean (last window may be partial)')
    ax.plot(x, [r['moving_average_400'] for r in rows], color='#155fa0',
            linewidth=2, label='Trailing 400-trajectory mean')
    ax.plot(x, [r['cumulative_mean'] for r in rows], color='#d77720',
            linewidth=1.5, label='Cumulative mean')
    ax.set(title=f'{label}: reward before group filtering',
           xlabel='Completed trajectories in log order',
           ylabel='Mean official terminal reward', ylim=(0, 1))
    ax.grid(alpha=.2)
    ax.legend(frameon=False)
    fig.text(.1, .015, 'All dumped final attempts, including rejected groups; intermediate retries excluded.', fontsize=9)
    fig.tight_layout(rect=(0, .045, 1, 1))
    fig.savefig(output_dir / 'reward_prefilter.png', dpi=180)
    plt.close(fig)
    print(json.dumps(dict(label=label, prefilter_trajectories=len(values),
                          prefilter_mean=rows[-1]['cumulative_mean'],
                          prefilter_last400=rows[-1]['moving_average_400']), indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--label', default='Vanilla GRPO')
    parser.add_argument('--batch-size', type=int, default=40)
    parser.add_argument('--title', default='Four-domain asynchronous')
    args = parser.parse_args()
    batches, completed = {}, set()
    with (args.run_dir / 'run.log').open(errors='replace') as stream:
        for line in stream:
            match = re.search(r'tau2_pool_batch update=(\d+) groups=(\[[^\]]*\])', line)
            if match:
                batches[int(match[1])] = json.loads(match[2])
            match = re.search(r'tau2_trainer batch=(\d+) start=[\d.]+ end=[\d.]+', line)
            if match:
                completed.add(int(match[1]))
    rewards = defaultdict(dict)
    prefilter = {}
    for path in sorted((args.run_dir / 'trajectories').glob('*.jsonl')):
        with path.open() as stream:
            for line in stream:
                if not line.endswith('\n'):
                    continue  # A live writer may still be appending the final record.
                row = json.loads(line)
                rewards[row['group_index']][row['sample_index']] = float(row['reward'])
                prefilter[(row['group_index'], row['sample_index'])] = float(
                    row['metadata']['tau2_reward_info']['reward'])
    plot_prefilter(prefilter, args.output_dir, args.label)
    rows = []
    for iteration in sorted(completed):
        values = [reward for group in batches[iteration] for reward in rewards[group].values()]
        if len(values) != args.batch_size:
            raise ValueError(f'iter{iteration}: expected {args.batch_size} accepted trajectories, got {len(values)}')
        rows.append({'iteration': iteration, 'updates': iteration + 1,
                     'reward': sum(values) / len(values), 'trajectories': len(values)})
    if not rows:
        print('No completed optimizer batches; only prefilter statistics available')
        return
    for i, row in enumerate(rows):
        window = rows[max(0, i-9):i+1]
        row['moving_average_10'] = sum(r['reward'] for r in window) / len(window)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / 'reward_curves.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 4.8))
    x = [r['updates'] for r in rows]
    ax.plot(x, [r['reward'] for r in rows], color='#96b6d9', linewidth=1.2,
            marker='o', markersize=2.5, label=f'Accepted batch mean ({args.batch_size} trajectories)')
    ax.plot(x, [r['moving_average_10'] for r in rows], color='#155fa0', linewidth=2.3,
            label='Trailing 10-update mean')
    ax.set(title=f'{args.title} {args.label}: training reward', xlabel='Completed optimizer updates',
           ylabel='Mean official terminal reward', ylim=(0, 1))
    ax.grid(alpha=.2)
    ax.legend(loc='best', frameon=False)
    fig.text(.1, .015, f'{args.label} | {len(rows)} updates | {sum(r["trajectories"] for r in rows):,} accepted trajectories | Training rewards, not held-out evaluation', fontsize=9)
    fig.tight_layout(rect=(0, .045, 1, 1))
    fig.savefig(args.output_dir / 'reward_curves.png', dpi=180)
    plt.close(fig)
    print(json.dumps({'updates': len(rows), 'latest_iteration': rows[-1]['iteration'],
                      'overall_mean': sum(r['reward'] for r in rows)/len(rows),
                      'first_10_mean': sum(r['reward'] for r in rows[:10])/len(rows[:10]),
                      'last_10_mean': rows[-1]['moving_average_10'],
                      'output': str(args.output_dir)}, indent=2))


if __name__ == '__main__':
    main()
