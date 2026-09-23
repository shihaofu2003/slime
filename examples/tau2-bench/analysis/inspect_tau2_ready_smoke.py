"""Inspect accepted ready-pool groups and measured sampling/training overlap."""

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from datetime import datetime, timezone
from statistics import mean


def inspect(logs, trajectories, gpu_csv=None, max_policy_lag=None):
    text = "\n".join(p.read_text(errors="replace").replace("\x00", "") for p in logs)
    if max_policy_lag is None:
        configured_lags = re.findall(r"tau2_max_policy_lag\s+\.+\s+(-?\d+)", text)
        max_policy_lag = int(configured_lags[-1]) if configured_lags else 1
    batches = [(int(update), json.loads(groups)) for update, groups in re.findall(
        r"tau2_pool_batch update=(\d+) groups=(\[[\d, ]+\])", text)]
    trained_updates = {int(u) for u in re.findall(r"tau2_trainer batch=(\d+) start=[\d.]+ end=[\d.]+", text)}
    selected_but_untrained = [u for u, _ in batches if u not in trained_updates]
    batches = [(u, keys) for u, keys in batches if u in trained_updates]
    selected = {key for _, groups in batches for key in groups}
    records = defaultdict(list)
    dump_count = 0
    with trajectories.open() as source:
        for line in source:
            row = json.loads(line)
            dump_count += 1
            if row["group_index"] in selected:
                records[row["group_index"]].append(row)
    violations, per_update = [], []
    if not batches:
        violations.append("No completed optimizer updates with ready-pool group records")
    all_ids = [key for _, groups in batches for key in groups]
    if len(all_ids) != len(set(all_ids)):
        violations.append("A group was consumed by more than one optimizer update")
    turn_count = cross_version = 0
    segment_count = training_tokens = unmerged_tokens = retry_count = 0
    domain_counts = Counter()
    for update, keys in batches:
        domains, lag, trajectory_count = Counter(), [], 0
        if len(keys) != 5:
            violations.append(f"Update {update} did not consume five groups")
        for key in keys:
            rows = records[key]
            if len(rows) != 8 or len({r["sample_index"] for r in rows}) != 8:
                violations.append(f"Group {key} does not have eight independent trajectories")
            if len({r["task_id"] for r in rows}) != 1:
                violations.append(f"Group {key} mixes tasks or is missing")
            if not rows:
                continue
            domain = rows[0]["metadata"]["domain"]
            domains[domain] += 1
            domain_counts[domain] += 1
            trajectory_count += len(rows)
            for row in rows:
                if row["remove_sample"] or row["reward"] not in (0, 1):
                    violations.append(f"Untrainable/nonbinary trajectory {row['sample_index']} was consumed")
                turns = [(m.get("raw_data") or {}).get("meta_info", {}).get("behavior_turn")
                         for m in row["simulation"]["messages"]]
                turns = [t for t in turns if t]
                if not turns:
                    violations.append(f"Trajectory {row['sample_index']} has no recorded Agent turn")
                    continue
                segment_count += row["metadata"]["tau2_training_segments"]
                training_tokens += row["metadata"]["tau2_train_total_tokens"]
                unmerged_tokens += sum(len(t["prompt_token_ids"]) + len(t["output_token_ids"]) for t in turns)
                retry_count += row["metadata"]["tau2_rollout_attempts"] - 1
                versions = [t["policy_version"] for t in turns]
                earliest = min(versions)
                lag.append(update - earliest)
                cross_version += int(earliest != max(versions))
                if earliest != row["metadata"]["tau2_earliest_policy_version"]:
                    violations.append(f"Incorrect earliest version for trajectory {row['sample_index']}")
                for turn in turns:
                    turn_count += 1
                    ids, probs = turn["output_token_ids"], turn["log_probs"]
                    if not ids or len(ids) != len(probs) or not all(math.isfinite(p) for p in probs):
                        violations.append(f"Missing/misaligned behavior probabilities in trajectory {row['sample_index']}")
                    stop = (turn.get("finish_reason") or {}).get("matched")
                    if isinstance(stop, int) and ids[-1] != stop:
                        violations.append(f"Sampled stop token missing in trajectory {row['sample_index']}")
        if lag and (min(lag) < 0 or (max_policy_lag >= 0 and max(lag) > max_policy_lag)):
            violations.append(f"Update {update} has earliest-turn lag outside configured limit {max_policy_lag}")
        per_update.append(dict(update=update, groups=keys, domains=dict(domains),
                               trajectories=trajectory_count, lag_min=min(lag) if lag else None,
                               lag_max=max(lag) if lag else None))
    training = [(int(u), float(a), float(b)) for u, a, b in re.findall(
        r"tau2_trainer batch=(\d+) start=([\d.]+) end=([\d.]+)", text)]
    overlap, request_stats = {}, {}
    for role, pattern in (
        ("agent", r"tau2_agent_turn version=\d+ queued=[\d.]+ start=([\d.]+) end=([\d.]+)"),
        ("user", r"tau2_user_turn sample=\d+ attempt=\d+ start=([\d.]+) end=([\d.]+)"),
    ):
        requests = [(float(a), float(b)) for a, b in re.findall(pattern, text)]
        overlap[role] = {u: sum(a < end and b > start for a, b in requests) for u, start, end in training}
        durations = sorted(b - a for a, b in requests)
        active = peak = 0
        for _, delta in sorted([(a, 1) for a, _ in requests] + [(b, -1) for _, b in requests]):
            active += delta
            peak = max(peak, active)
        request_stats[role] = dict(count=len(requests), peak_outstanding=peak,
                                   mean_seconds=mean(durations) if durations else None,
                                   p95_seconds=durations[math.ceil(len(durations) * .95) - 1] if durations else None)
    gpu_stats = {}
    if gpu_csv:
        intervals = [(float(a), float(b)) for a, b in re.findall(
            r"tau2_training_elapsed start=([\d.]+) end=([\d.]+)", text)]
        values = defaultdict(list)
        with gpu_csv.open() as source:
            for row in csv.DictReader(source, skipinitialspace=True):
                moment = datetime.strptime(row["timestamp"], "%Y/%m/%d %H:%M:%S.%f").replace(tzinfo=timezone.utc).timestamp()
                if any(a <= moment <= b for a, b in intervals):
                    index = int(row["index"])
                    role = "trainer" if index < 2 else "agent" if index < 6 else "user"
                    values[role].append(float(row["utilization.gpu [%]"].split()[0]))
        gpu_stats = {role: dict(mean_utilization_percent=mean(samples), observations=len(samples))
                     for role, samples in values.items()}
    return dict(dump_records=dump_count, accepted_groups=len(all_ids), accepted_trajectories=sum(
        row["trajectories"] for row in per_update), groups_by_domain=dict(domain_counts), updates=per_update,
        recorded_agent_turns=turn_count, cross_version_trajectories=cross_version,
        training_segments=segment_count, training_tokens=training_tokens, unmerged_tokens=unmerged_tokens,
        trajectory_retries=retry_count, request_stats=request_stats, gpu_stats_during_loop=gpu_stats,
        requests_overlapping_training=overlap, selected_but_untrained_updates=selected_but_untrained, violations=violations)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logs", nargs="+", type=Path, required=True)
    parser.add_argument("--trajectories", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu-csv", type=Path)
    parser.add_argument("--max-policy-lag", type=int, default=None, help="Default: read the training configuration from logs")
    args = parser.parse_args()
    result = inspect(args.logs, args.trajectories, args.gpu_csv, args.max_policy_lag)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
