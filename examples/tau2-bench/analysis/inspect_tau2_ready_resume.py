"""Check regenerated pending inputs and the continued uniform task RNG stream."""

import argparse
import json
import random
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial-state", type=Path, required=True)
    parser.add_argument("--final-state", type=Path, required=True)
    parser.add_argument("--trajectories", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--resume-version", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    initial = torch.load(args.initial_state, weights_only=False)
    final = torch.load(args.final_state, weights_only=False)
    pending = {sample.index: sample for group in initial["metadata"]["tau2_pending_groups"].values() for sample in group}
    observed, groups = {}, {}
    with args.trajectories.open() as source:
        for line in source:
            row = json.loads(line)
            observed[row["sample_index"]] = row
            groups.setdefault(row["group_index"], row)
    violations = []
    for index, sample in pending.items():
        row = observed.get(index)
        if row is None:
            violations.append(f"Pending sample {index} was not regenerated")
            continue
        if row["metadata"]["task"] != sample.metadata["task"] or row["metadata"]["db_path"] != sample.metadata["db_path"]:
            violations.append(f"Pending sample {index} used a different task or DB")
        if not row["remove_sample"] and row["metadata"]["tau2_earliest_policy_version"] < args.resume_version:
            violations.append(f"Pending sample {index} reused pre-resume behavior tokens")
    repeated_old = [index for index in observed if index < initial["sample_index"] and index not in pending]
    if repeated_old:
        violations.append("Previously consumed or rejected samples appeared again")
    rng = random.Random()
    rng.setstate(initial["metadata"]["tau2_task_rng_state"])
    with args.dataset.open() as source:
        tasks = [json.loads(line) for line in source]
    new_draws = final["sample_group_index"] - initial["sample_group_index"]
    for key in range(initial["sample_group_index"], final["sample_group_index"]):
        task = rng.choice(tasks)
        row = groups.get(key)
        if row is None or row["metadata"]["task"] != task["metadata"]["task"] or row["metadata"]["db_path"] != task["metadata"]["db_path"]:
            violations.append(f"New group {key} does not match the restored task RNG stream")
    if rng.getstate() != final["metadata"]["tau2_task_rng_state"]:
        violations.append("Final task RNG state differs from continued draws")
    if final["metadata"]["tau2_pending_groups"]:
        violations.append("Final checkpoint still contains unconsumed groups")
    before_count = sum(initial["metadata"]["tau2_task_trained_counts"].values())
    after_count = sum(final["metadata"]["tau2_task_trained_counts"].values())
    result = dict(pending_inputs=len(pending), regenerated_pending_inputs=len(set(pending) & set(observed)),
                  repeated_old_sample_ids=repeated_old, new_random_draws=new_draws,
                  consumed_groups_before_resume=before_count, consumed_groups_after_resume=after_count,
                  consumed_groups_in_recovery_run=after_count - before_count,
                  final_draw_counts=final["metadata"]["tau2_task_draw_counts"],
                  final_trained_counts=final["metadata"]["tau2_task_trained_counts"], violations=violations)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
