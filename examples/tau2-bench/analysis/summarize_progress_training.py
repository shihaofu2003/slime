"""Summarize accepted progress credit separately from all generated trajectories."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import fmean

from summarize_single_call_credit_training import summarize_logs


def summarize(stage_dir, logs):
    with (stage_dir / "credit.jsonl").open() as stream:
        accepted = [json.loads(line) for line in stream]
    groups = {row["group_index"]: row for row in accepted}
    accepted_ids = {row["sample_index"] for row in accepted}
    turns = [turn for row in accepted for turn in row["turns"]]
    raw = [turn["raw_rerender"] for turn in turns if turn.get("raw_rerender", {}).get("available")]
    raw_summary = {"observed_turns": len(raw), "body_mismatches": sum(not row["body_equal"] for row in raw),
                   "diagnostic_versions": dict(Counter(row.get("diagnostic_version", 1) for row in raw))}
    replay_path = stage_dir / "tokenization_audit.json"
    if replay_path.exists():
        replay = json.loads(replay_path.read_text())
        raw_summary = {"observed_turns": replay["counts"]["turns"],
                       "body_mismatches": replay["counts"]["body_mismatches"],
                       "diagnostic_version": replay["diagnostic_version"], "source": str(replay_path)}
    candidates = Counter()
    accepted_terminations = Counter()
    accepted_errors = Counter()
    components = {}
    final_state_checks = Counter()
    for path in sorted((stage_dir / "trajectories").glob("*.jsonl")):
        with path.open() as stream:
            for line in stream:
                row = json.loads(line)
                metadata = row["metadata"]
                candidates["trajectories"] += 1
                candidates["extra_resample_attempts"] += metadata.get("tau2_rollout_attempts", 1) - 1
                candidates["removed_samples"] += bool(row["remove_sample"])
                if row["sample_index"] in accepted_ids:
                    accepted_terminations[metadata["tau2_termination_reason"]] += 1
                    signals = metadata["tau2_field_reward_signals"]
                    accepted_errors.update(signals["counts"])
                    progress = metadata["tau2_progress"]
                    reward_info = metadata.get("tau2_reward_info") or {}
                    basis = (metadata["task"].get("evaluation_criteria") or {}).get("reward_basis", ["DB", "COMMUNICATE"])
                    if not progress["unavailable_reason"]:
                        score = progress["scores"][-1]
                        if "ENV_ASSERTION" in basis and reward_info.get("env_assertions"):
                            assertions = reward_info["env_assertions"]
                            expected = sum(a["met"] for a in assertions) / len(assertions)
                            final_state_checks["env_checked"] += 1
                            final_state_checks["env_mismatch"] += score != expected
                        elif "DB" in basis and reward_info.get("db_check"):
                            final_state_checks["db_checked"] += 1
                            final_state_checks["db_mismatch"] += (score == 0) != reward_info["db_check"]["db_match"]
                    for key, value in signals.get("components", {}).items():
                        if isinstance(value, (int, float)):
                            components.setdefault(key, []).append(value)
    result = {
        "accepted_trajectories": len(accepted), "accepted_groups": len(groups),
        "accepted_domain_trajectories": dict(Counter(row["domain"] for row in accepted)),
        "binary_zero_variance_group_rate": fmean(row["binary_zero_variance_group"] for row in groups.values()),
        "final_zero_signal_group_rate": fmean(not row["group_has_signal"] for row in groups.values()),
        "auxiliary_unavailable_groups": sum(bool(row["group_unavailable_reasons"]) for row in groups.values()),
        "auxiliary_unavailable_reasons": dict(Counter(reason for row in groups.values() for reason in row["group_unavailable_reasons"])),
        "scoring_seconds_per_trajectory": fmean(row["progress"]["scoring_seconds"] for row in accepted),
        "generated_candidates": dict(candidates), "accepted_termination_counts": dict(accepted_terminations),
        "accepted_error_counts": dict(accepted_errors),
        "field_diagnostic_means": {key: fmean(values) for key, values in components.items()},
        "final_state_vs_official_evaluator": dict(final_state_checks),
        "raw_rerender": raw_summary,
        "training_logs": summarize_logs(logs),
    }
    output = stage_dir / "training_summary.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-dir", type=Path, required=True)
    parser.add_argument("--log", type=Path, action="append", required=True)
    args = parser.parse_args()
    print(json.dumps(summarize(args.stage_dir, args.log), ensure_ascii=False, indent=2))
