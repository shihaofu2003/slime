"""Plot prefilter task rewards and credit retention from compact producer logs."""

import argparse
import csv
import json
import math
import re
import time
from collections import Counter
from pathlib import Path


def read_log(path):
    groups, batches, completed = {}, {}, set()
    if not path.exists():
        return groups, batches, completed
    with path.open(errors="replace") as stream:
        for line in stream:
            if not line.endswith("\n"):
                continue
            match = re.search(r"tau2_credit_group (\{.*\})", line)
            if match:
                row = json.loads(match[1])
                groups[row["group"]] = row
            match = re.search(r"tau2_pool_batch update=(\d+) groups=(\[[^\]]*\])", line)
            if match:
                batches[int(match[1])] = json.loads(match[2])
            match = re.search(r"tau2_trainer batch=(\d+) start=[\d.]+ end=[\d.]+", line)
            if match:
                completed.add(int(match[1]))
    return groups, batches, completed


def make_tables(groups, batches, completed, weights, window_groups):
    values = list(groups.values())
    windows, training = [], []
    for end in sorted(set(range(20, len(values) + 1, 20)) | {len(values)}):
        if not end:
            continue
        window = values[max(0, end - window_groups):end]
        row = dict(completed_groups=end, window_groups=len(window),
                   reward=sum(g["reward"] for g in window) / len(window),
                   kept_rate=sum(g["kept"] for g in window) / len(window),
                   all_zero_rate=sum(g["outcome"] == "all_zero" for g in window) / len(window),
                   all_one_rate=sum(g["outcome"] == "all_one" for g in window) / len(window),
                   zero_signal_rate=sum(g["reason"] == "credit_zero_signal" for g in window) / len(window),
                   unusable_rate=sum(not g["kept"] and g["reason"] != "credit_zero_signal" for g in window) / len(window))
        weighted, variance = 0.0, 0.0
        for domain, weight in weights.items():
            domain_groups = [g for g in window if g["domain"] == domain]
            rewards = [g["reward"] for g in domain_groups]
            mean = sum(rewards) / len(rewards) if rewards else float("nan")
            row[domain + "_reward"] = mean
            row[domain + "_groups"] = len(rewards)
            row[domain + "_kept_groups"] = sum(g["kept"] for g in domain_groups)
            weighted += weight * mean
            if len(rewards) > 1:
                variance += weight ** 2 * sum((v - mean) ** 2 for v in rewards) / (len(rewards) * (len(rewards) - 1))
            else:
                variance = float("nan")
        row.update(fixed_domain_reward=weighted, group_standard_error=math.sqrt(variance))
        windows.append(row)
    for update in sorted(completed):
        keys = batches.get(update, [])
        if not keys or any(key not in groups for key in keys):
            continue  # A live Ray log may not have flushed every group's record yet.
        used = [groups[key] for key in keys]
        n = sum(g["trajectories"] for g in used)
        row = dict(update=update, trajectories=n,
                   reward=sum(g["reward"] * g["trajectories"] for g in used) / n)
        row.update({domain + "_groups": sum(g["domain"] == domain for g in used) for domain in weights})
        recent = [r["reward"] for r in training[-9:]] + [row["reward"]]
        row["moving_average_10"] = sum(recent) / len(recent)
        training.append(row)
    return windows, training


def render(run_dir, output_dir, weights, window_groups):
    groups, batches, completed = read_log(run_dir / "run.log")
    windows, training = make_tables(groups, batches, completed, weights, window_groups)
    if not windows:
        print("Waiting for complete credit groups", flush=True)
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename, rows in [("groups.csv", list(groups.values())),
                           ("reward_prefilter_by_domain.csv", windows),
                           ("reward_accepted.csv", training)]:
        if rows:
            with (output_dir / filename).open("w") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    x = [r["completed_groups"] for r in windows]
    y = np.array([r["fixed_domain_reward"] for r in windows])
    error = 1.96 * np.array([r["group_standard_error"] for r in windows])
    axes[0, 0].plot(x, y, color="#1764ab", label="Fixed task-pool domain weights")
    axes[0, 0].fill_between(x, np.clip(y - error, 0, 1), np.clip(y + error, 0, 1), alpha=.15)
    axes[0, 0].set(title=f"Before filtering: trailing {window_groups} complete groups",
                   xlabel="Completed groups", ylabel="Official reward")
    for domain in weights:
        axes[0, 1].plot(x, [r[domain + "_reward"] for r in windows], label=domain)
    axes[0, 1].set(title="Before filtering: per-domain reward", xlabel="Completed groups")
    if training:
        updates = [r["update"] + 1 for r in training]
        axes[1, 0].plot(updates, [r["reward"] for r in training], alpha=.3, color="#1764ab", label="Accepted batch")
        axes[1, 0].plot(updates, [r["moving_average_10"] for r in training], color="#1764ab", label="Trailing 10 updates")
    axes[1, 0].set(title="Accepted groups: terminal reward", xlabel="Optimizer updates", ylabel="Official reward")
    for key, label in [("all_zero_rate", "All failed"), ("all_one_rate", "All succeeded"),
                       ("kept_rate", "Kept for credit"), ("unusable_rate", "Unusable")]:
        axes[1, 1].plot(x, [r[key] for r in windows], label=label)
    axes[1, 1].set(title="Outcomes and retention (overlapping rates)", xlabel="Completed groups")
    for ax in axes.flat:
        ax.set_ylim(0, 1)
        ax.grid(alpha=.2)
        handles, _ = ax.get_legend_handles_labels()
        if handles:
            ax.legend(fontsize=8, frameon=False)
    fig.suptitle("Credit-signal filtering | Training diagnostics, not held-out evaluation")
    fig.savefig(output_dir / "reward_diagnostics.png", dpi=160)
    plt.close(fig)
    print(json.dumps(dict(groups=len(groups), completed_updates=len(training),
                          fixed_domain_reward=windows[-1]["fixed_domain_reward"])), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prompt-data", type=Path, required=True)
    parser.add_argument("--window-groups", type=int, default=400)
    parser.add_argument("--watch", type=float, default=0, help="Refresh interval in seconds; 0 renders once")
    args = parser.parse_args()
    if args.window_groups <= 0 or args.watch < 0:
        parser.error("window-groups must be positive and watch must be nonnegative")
    with args.prompt_data.open() as stream:
        counts = Counter(json.loads(line)["metadata"]["domain"] for line in stream if line.strip())
    weights = {domain: n / sum(counts.values()) for domain, n in counts.items()}
    while True:
        render(args.run_dir, args.output_dir, weights, args.window_groups)
        if not args.watch:
            return
        time.sleep(args.watch)


if __name__ == "__main__":
    main()
