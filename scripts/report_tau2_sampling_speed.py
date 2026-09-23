"""Compare update wall time and sampled GPU utilization from Tau2 run logs."""

import argparse
import csv
import datetime as dt
import json
import re
import statistics
from pathlib import Path


def read_run(root):
    log = (root / "run.log").read_text(errors="replace").replace("\0", "")
    admission = re.search(r"tau2_pool_admission time=([\d.]+)", log)
    starts = {int(i): (float(a), float(b)) for i, a, b in re.findall(
        r"tau2_trainer batch=(\d+) start=([\d.]+) end=([\d.]+)", log)}
    ready = {int(i): (dt.datetime.strptime(t, "%Y-%m-%d %H:%M:%S").replace(
        tzinfo=dt.timezone.utc).timestamp(), float(wait)) for t, i, wait in re.findall(
        r"\[(2026-\d\d-\d\d \d\d:\d\d:\d\d)\].*tau2_pool_batch update=(\d+).*?ready_wait=([\d.]+)", log)}
    sync = {int(i): dict(drain=float(a), apply=float(b), resume=float(c))
            for i, a, b, c in re.findall(
                r"tau2_weight_update_phases batch=(\d+).*drain_seconds=([\d.]+) "
                r"transfer_apply_seconds=([\d.]+) resume_seconds=([\d.]+)", log)}
    saves = {int(i): float(b) - float(a) for i, a, b in re.findall(
        r"tau2_checkpoint batch=(\d+) start=([\d.]+) end=([\d.]+)", log)}
    groups = {int(i): (float(a), float(b), keep == "True") for i, a, b, keep in re.findall(
        r"tau2_producer group=(\d+) start=([\d.]+) end=([\d.]+) accepted=(True|False)", log)}
    used_groups = set()
    for i, values in re.findall(r"tau2_pool_batch update=(\d+) groups=(\[[^\]]*\])", log):
        if int(i) in starts:
            used_groups.update(json.loads(values))
    first = float(admission[1]) if admission else None
    updates = {}
    for i, (start, end) in sorted(starts.items()):
        previous = starts[i - 1][1] if i - 1 in starts else first
        updates[i] = dict(start=start, end=end, training_seconds=end - start,
                          interval_seconds=end - previous if previous is not None else None,
                          ready_wait_seconds=ready[i][1],
                          ready_to_trainer_seconds=start - ready[i][0],
                          checkpoint_after_seconds=saves.get(i, 0),
                          sync_after_seconds=sum(sync.get(i, {}).values()))
    gpu = []
    gpu_names = set()
    for path in sorted(root.glob("gpu_*.csv")):
        lines = (line.replace("\0", "") for line in path.open())
        for row in csv.DictReader(lines, skipinitialspace=True):
            # Ignore only an incomplete last row while nvidia-smi is appending.
            if not row.get("memory.total [MiB]"):
                continue
            timestamp = dt.datetime.strptime(row["timestamp"], "%Y/%m/%d %H:%M:%S.%f").replace(
                tzinfo=dt.timezone.utc).timestamp()
            gpu.append((timestamp, int(row["index"]), float(row["utilization.gpu [%]"].split()[0])))
            gpu_names.add(row["name"])
    return dict(name=root.name, completed_updates=len(updates), sampling_start=first,
                updates=updates, gpu_names=sorted(gpu_names), gpu=gpu,
                completed_groups=len(groups), retained_groups=sum(g[2] for g in groups.values()),
                filtered_groups=sum(not g[2] for g in groups.values()),
                trained_groups=len(used_groups),
                retained_unused_groups=sum(g[2] and i not in used_groups for i, g in groups.items()))


def gpu_window(rows, start, end):
    per_gpu = {}
    for timestamp, index, value in rows:
        if start <= timestamp <= end:
            per_gpu.setdefault(index, []).append(value)
    return dict(samples={i: len(v) for i, v in per_gpu.items()},
                per_gpu_mean={i: statistics.mean(v) for i, v in per_gpu.items()},
                trainer_mean=statistics.mean(v for i, vs in per_gpu.items() if i < 2 for v in vs)
                if any(i < 2 for i in per_gpu) else None,
                generator_mean=statistics.mean(v for i, vs in per_gpu.items() if i >= 2 for v in vs)
                if any(i >= 2 for i in per_gpu) else None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-updates", type=int, default=30)
    args = parser.parse_args()
    baseline, current = read_run(args.baseline), read_run(args.run)
    common = sorted(set(baseline["updates"]) & set(current["updates"]))
    # Omit startup and the baseline's last two updates. Apply exactly the same
    # update indices to both runs, even while the long run is still in progress.
    stable = [i for i in common if 1 <= i <= len(baseline["updates"]) - 3]
    comparisons = {}
    for label, indices in (("matched", common), ("stable", stable)):
        if not indices:
            continue
        comparison = {"zero_based_updates": indices}
        for name, run in (("baseline", baseline), ("current", current)):
            start = run["updates"][indices[0] - 1]["end"] if indices[0] else run["sampling_start"]
            end = run["updates"][indices[-1]]["end"]
            comparison[name] = dict(elapsed_seconds=end - start,
                mean_interval_seconds=statistics.mean(run["updates"][i]["interval_seconds"] for i in indices),
                mean_ready_wait_seconds=statistics.mean(run["updates"][i]["ready_wait_seconds"] for i in indices),
                mean_ready_to_trainer_seconds=statistics.mean(run["updates"][i]["ready_to_trainer_seconds"] for i in indices),
                mean_training_seconds=statistics.mean(run["updates"][i]["training_seconds"] for i in indices),
                gpu=gpu_window(run["gpu"], start, end))
        comparison["speedup"] = comparison["baseline"]["elapsed_seconds"] / comparison["current"]["elapsed_seconds"]
        comparisons[label] = comparison
    for run in (baseline, current):
        run["five_update_windows"] = []
        for first in range(0, len(run["updates"]), 5):
            indices = [i for i in range(first, first + 5) if i in run["updates"]]
            run["five_update_windows"].append(dict(updates=[i + 1 for i in indices],
                mean_interval_seconds=statistics.mean(run["updates"][i]["interval_seconds"] for i in indices),
                mean_ready_wait_seconds=statistics.mean(run["updates"][i]["ready_wait_seconds"] for i in indices)))
        if run["updates"]:
            end = run["updates"][max(run["updates"])]["end"]
            run["sampling_to_last_training_seconds"] = end - run["sampling_start"]
            run["gpu_during_training_run"] = gpu_window(run["gpu"], run["sampling_start"], end)
        del run["gpu"]
    report = dict(updated_at=dt.datetime.now(dt.timezone.utc).isoformat(),
                  expected_updates=args.expected_updates, baseline=baseline, current=current,
                  comparisons=comparisons)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "speed_comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    text = ["# Sampling long-run speed comparison", "",
            f"Purpose: verify sustained speed with the completed sampling fixes. Run: `{current['name']}`.", "",
            f"Completed updates: **{current['completed_updates']}/{args.expected_updates}**. "
            f"Reference: `{baseline['name']}` ({baseline['completed_updates']} updates).", "",
            "Both use SFT4673, LR2e-6, batch128/K8, progress1, Trainer2+Generator6 and external User. "
            "Task selection is recorded in each run's launcher; asynchronous completion can change which groups train. "
            "This reports observed end-to-end speed.", "",
            "| Window | Updates compared | Baseline min/update | Current min/update | Speedup |",
            "|---|---:|---:|---:|---:|"]
    for label, row in comparisons.items():
        text.append(f"| {label} | {len(row['zero_based_updates'])} | "
                    f"{row['baseline']['mean_interval_seconds']/60:.2f} | "
                    f"{row['current']['mean_interval_seconds']/60:.2f} | {row['speedup']:.2f}x |")
    text += ["", "Stable excludes the reference's first and last two updates; intervals run between "
             "consecutive optimizer completions and include intervening sampling, checkpoint and synchronization time. "
             "Matched total starts at initial pool admission. Model startup and the final checkpoint are excluded."]
    if "matched" in comparisons:
        before, after = comparisons["matched"]["baseline"], comparisons["matched"]["current"]
        text += ["", "| Metric over the matched window | Baseline | Current |", "|---|---:|---:|"]
        for label, key in [("Batch ready wait, seconds/update", "mean_ready_wait_seconds"),
                           ("Batch ready to trainer start, seconds", "mean_ready_to_trainer_seconds"),
                           ("Training compute, seconds/update", "mean_training_seconds")]:
            text.append(f"| {label} | {before[key]:.2f} | {after[key]:.2f} |")
        for label, key in [("Trainer GPU utilization", "trainer_mean"), ("Generator GPU utilization", "generator_mean")]:
            text.append(f"| {label} | {before['gpu'][key]:.2f}% | {after['gpu'][key]:.2f}% |")
        text += ["", "GPU utilization is the mean of 5-second samples across the respective GPUs. "
                 "Batch-ready log timestamps have one-second precision; phase timers overlap and must not be summed."]
    text += ["",
             "| Current update window | Mean min/update | Mean ready wait, seconds |",
             "|---|---:|---:|"]
    for row in current["five_update_windows"]:
        text.append(f"| {row['updates'][0]}–{row['updates'][-1]} | {row['mean_interval_seconds']/60:.2f} | "
                    f"{row['mean_ready_wait_seconds']:.2f} |")
    text += ["", "GPU samples, per-update times, group-filter counts and completed-but-unused groups are in "
             "[the JSON report](speed_comparison.json). Training reward curves are separate from held-out evaluation.", ""]
    (args.output / "README.md").write_text("\n".join(text))
    print(json.dumps(dict(completed=current["completed_updates"], comparisons=comparisons), indent=2))


if __name__ == "__main__":
    main()
