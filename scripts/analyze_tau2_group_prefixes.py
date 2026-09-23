"""Measure prefix-screening errors and trajectory occupancy from a completed run."""

import argparse
import collections
import datetime
import json
import re
import statistics
from pathlib import Path


def timestamp(value):
    if isinstance(value, (int, float)):
        return value
    return datetime.datetime.fromisoformat(value.replace("Z", "+00:00")).replace(
        tzinfo=datetime.timezone.utc
    ).timestamp()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    log = (args.run_dir / "run.log").read_text(errors="replace")
    groups = {}
    for key, start, end, keep in re.findall(
        r"tau2_producer group=(\d+) start=([\d.]+) end=([\d.]+) accepted=(True|False)", log
    ):
        groups[int(key)] = dict(start=float(start), end=float(end), accepted=keep == "True", samples=[])
    for order, line in enumerate((args.run_dir / "trajectories/train.jsonl").open()):
        row = json.loads(line)
        sim, meta = row["simulation"], row["metadata"]
        group = groups[row["group_index"]]
        group["task"] = row["task_id"]
        group["samples"].append(dict(
            index=row["sample_index"], order=order, reward=row["reward"],
            start=timestamp(sim["start_time"]), end=timestamp(sim["end_time"]),
            duration=sim["duration"], scoring=meta.get("tau2_progress_scoring_seconds", 0),
            removed=row["remove_sample"], termination=sim["termination_reason"],
        ))
    report = {"groups": len(groups), "accepted": sum(g["accepted"] for g in groups.values())}
    for ordering, key in [("completion", "end"), ("launch", "index"), ("dump", "order")]:
        counts = collections.Counter()
        saved = []
        for group in groups.values():
            samples = sorted(group["samples"], key=lambda s: s[key])
            assert len(samples) == 8, (group["task"], len(samples))
            first = sum(s["reward"] for s in samples[:4])
            last = sum(s["reward"] for s in samples[4:])
            counts[(int(first), int(last), group["accepted"])] += 1
            if first in (0, 4):
                cutoff = samples[3]["end"]
                saved.append(sum(max(0, s["end"] - cutoff) for s in samples[4:]))
        report[ordering] = [dict(first4=first, last4=last, accepted=accepted, count=count)
                            for (first, last, accepted), count in sorted(counts.items())]
        if ordering == "completion":
            report["cancel_after_first4_remaining_trajectory_seconds"] = sum(saved)
    events = []
    for group in groups.values():
        for sample in group["samples"]:
            # Slot reservation until trajectory end; includes thread startup/queue wait.
            events.extend([(group["start"], 1), (sample["end"], -1)])
    events.sort()
    active = area = 0
    previous = events[0][0]
    for time, delta in events:
        area += active * (time - previous)
        active += delta
        previous = time
    report["mean_unfinished_trajectories"] = area / (events[-1][0] - events[0][0])
    for label, predicate in [("accepted", lambda g: g["accepted"]),
                             ("rejected", lambda g: not g["accepted"])]:
        chosen = [g for g in groups.values() if predicate(g)]
        report[label + "_stats"] = dict(
            groups=len(chosen),
            mean_group_seconds=statistics.mean(g["end"] - g["start"] for g in chosen),
            all_zero=sum(all(s["reward"] == 0 for s in g["samples"]) for g in chosen),
            all_one=sum(all(s["reward"] == 1 for s in g["samples"]) for g in chosen),
            removed_samples=sum(s["removed"] for g in chosen for s in g["samples"]),
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    args.output.with_suffix(".samples.json").write_text(json.dumps(groups) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
