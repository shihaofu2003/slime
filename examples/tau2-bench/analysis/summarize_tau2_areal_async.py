"""Paired official evaluation and measured timing for Tau2 asynchronous GRPO."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from summarize_raw_processed_vanilla_grpo import (
    PASS_METRICS,
    aggregate_evaluations,
    compare_models,
    load_evaluation,
)

HISTORICAL_RL = dict(zip(PASS_METRICS, (0.5075, 0.8, 0.225), strict=True))
PROCESSED_SFT = dict(zip(PASS_METRICS, (0.5575, 0.81, 0.28), strict=True))


def training_timing(paths):
    updates, process_seconds, loop_seconds, end_to_end_seconds = [], [], [], []
    trainer_seconds = update_seconds = checkpoint_seconds = 0.0
    agent_read_retries = 0
    diagnostics = {}
    for path in paths:
        text = path.read_text(errors="replace")
        agent_read_retries += len(re.findall(r"tau2_agent_read_retry version=\d+:", text))
        end_to_end_seconds.extend(float(s) for s in re.findall(r"tau2_training_end_to_end start=\S+ end=\S+ seconds=([\d.]+)", text))
        process_seconds.extend(float(s) for s in re.findall(r"tau2_training_process start=\S+ end=\S+ seconds=([\d.]+)", text))
        loop_seconds.extend(float(s) for s in re.findall(r"tau2_training_elapsed start=\S+ end=\S+ seconds=([\d.]+)", text))
        for key, value in re.findall(r"['\"](?:train/|rollout/)(tis_weight_squared|tis_weight|tis_clipfrac|approx_kl|ppo_kl|kl|kl_loss|policy_lag_max|policy_lag_mean|zero_variance_group_rate|truncated|truncated_ratio|producer_duration_seconds|producer_wait_seconds|ready_wait_seconds|lag_wait_seconds|sync_barrier_seconds|trained_groups_(?:airline|retail|telecom|banking|banking_knowledge)|drawn_groups_(?:airline|retail|telecom|banking|banking_knowledge))['\"]:\s*([-+\d.eE]+)", text):
            diagnostics.setdefault(key, []).append(float(value))
        for batch, start, end in re.findall(r"tau2_trainer batch=(\d+) start=([\d.]+) end=([\d.]+)", text):
            updates.append(int(batch))
            trainer_seconds += float(end) - float(start)
        for start, end in re.findall(r"tau2_weight_update batch=\d+ start=([\d.]+) end=([\d.]+)", text):
            update_seconds += float(end) - float(start)
        for start, end in re.findall(r"tau2_checkpoint batch=\d+ start=([\d.]+) end=([\d.]+)", text):
            checkpoint_seconds += float(end) - float(start)
    return {
        "diagnostics": diagnostics,
        "trained_groups_by_domain": {d: sum(diagnostics[f"trained_groups_{d}"]) for d in ("airline", "retail", "telecom", "banking", "banking_knowledge") if f"trained_groups_{d}" in diagnostics},
        "drawn_groups_by_domain": {d: max(diagnostics[f"drawn_groups_{d}"]) for d in ("airline", "retail", "telecom", "banking", "banking_knowledge") if f"drawn_groups_{d}" in diagnostics},
        "trajectory_weighted_tis_effective_fraction": [mean * mean / second if second else 0.0 for mean, second in zip(diagnostics.get("tis_weight", []), diagnostics.get("tis_weight_squared", []), strict=False)],
        "logs": [str(p) for p in paths],
        "updates": updates,
        "process_seconds": sum(process_seconds) if process_seconds else None,
        "end_to_end_seconds": sum(end_to_end_seconds) if end_to_end_seconds else None,
        "loop_seconds": sum(loop_seconds) if loop_seconds else None,
        "trainer_seconds": trainer_seconds,
        "weight_update_seconds": update_seconds,
        "checkpoint_seconds": checkpoint_seconds,
        "agent_read_retries": agent_read_retries,
        "complete_100_updates": sorted(updates) == list(range(100)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for arm in ("sync", "async"):
        parser.add_argument(f"--{arm}-logs", nargs="+", type=Path, required=True)
        parser.add_argument(f"--{arm}-eval", nargs=2, type=Path, required=True, metavar=("SEED300", "SEED301"))
    parser.add_argument("--historical-eval", nargs=2, type=Path, metavar=("SEED300", "SEED301"))
    parser.add_argument("--results-root", type=Path, default=Path(__file__).resolve().parents[4] / "tau2-bench")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows, metrics, timings = {}, {}, {}
    for arm in ("sync", "async", "historical"):
        paths = getattr(args, f"{arm}_eval")
        if paths is None:
            continue
        rows[arm] = {seed: load_evaluation(path, expected_seed=seed, results_root=args.results_root, label=arm) for seed, path in zip((300, 301), paths, strict=True)}
        metrics[arm] = aggregate_evaluations(rows[arm])
        if arm != "historical":
            timings[arm] = training_timing(getattr(args, f"{arm}_logs"))
    comparisons = {arm: compare_models(key=f"async_vs_{arm}", candidate=rows["async"], reference=rows[arm], candidate_metrics=metrics["async"], reference_metrics=metrics[arm], samples=20000, bootstrap_seed=1234) for arm in ("sync", "historical") if arm in rows}
    async_metrics = metrics["async"]["overall"]
    historical = metrics.get("historical", {}).get("overall", HISTORICAL_RL)
    deltas = {arm: {m: async_metrics[m] - reference[m] for m in PASS_METRICS} for arm, reference in (("sync", metrics["sync"]["overall"]), ("historical", historical), ("sft", PROCESSED_SFT))}
    enough_time_data = all(t["complete_100_updates"] and t["end_to_end_seconds"] for t in timings.values())
    speedup = timings["sync"]["end_to_end_seconds"] / timings["async"]["end_to_end_seconds"] if enough_time_data else None
    # An exact five-percentage-point difference can round just below -0.05.
    effect_pass = all(deltas[arm][m] >= -0.05 - 1e-12 for arm in ("sync", "historical") for m in PASS_METRICS)
    ci_support = {arm: all(row["ci95"][0] >= -0.05 - 1e-12 for row in comparison["paired_bootstrap"] if row["scope"] == "overall") for arm, comparison in comparisons.items()}
    report = {
        "timing": timings,
        "speedup": speedup,
        "evaluation": metrics,
        "deltas": deltas,
        "paired_comparisons": comparisons,
        "empirical_effect_pass": effect_pass,
        "acceptance": "pending" if speedup is None else ("pass" if speedup >= 2 and effect_pass else "fail"),
        "ci95_supports_5pp_noninferiority": ci_support,
        "historical_paired_ci_available": "historical" in comparisons,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("speedup", "acceptance", "ci95_supports_5pp_noninferiority")}, indent=2))


FOUR_DOMAIN_TASKS = {"airline": 20, "retail": 40, "telecom": 40, "banking_knowledge": 97}


def credit_diagnostics(path):
    from collections import defaultdict
    groups = defaultdict(list)
    if not path.exists():
        return None
    for line in path.open():
        row = json.loads(line)
        groups[row["group_index"]].append(row)
    uniform = {"all_correct": [], "all_wrong": []}
    for group in groups.values():
        for label, reward in (("all_correct", 1), ("all_wrong", 0)):
            if all(row["reward"] == reward for row in group):
                uniform[label].append(any(row["group_has_signal"] for row in group))
    rows = [row for group in groups.values() for row in group]
    return {
        "accepted_groups": len(groups), "accepted_trajectories": len(rows),
        "uniform_group_nonzero_turn_advantage_rate": {
            key: sum(values) / len(values) if values else None for key, values in uniform.items()},
        "uniform_group_counts": {key: len(values) for key, values in uniform.items()},
        "final_zero_signal_group_rate": sum(not any(r["group_has_signal"] for r in group)
                                             for group in groups.values()) / len(groups) if groups else None,
        "accepted_scoring_seconds": sum(row.get("scoring_seconds", 0) for row in rows),
        "accepted_resample_attempts": sum(row.get("rollout_attempts", 1) - 1 for row in rows),
        "unique_accepted_tasks": len({(row["domain"], row["task_id"]) for row in rows}),
    }


def trajectory_diagnostics(paths, accepted_groups):
    generated = accepted = attempts = 0
    seconds = 0.0
    unique, unique_accepted = set(), set()
    rewards = {}
    for path in paths:
        for line in path.open():
            row = json.loads(line)
            metadata = row["metadata"]
            key = (metadata["tau2_domain"], row["task_id"])
            generated += 1
            attempts += metadata.get("tau2_rollout_attempts", 1)
            seconds += metadata.get("tau2_progress_scoring_seconds", metadata.get("tau2_progress", {}).get("scoring_seconds", 0))
            unique.add(key)
            if row["group_index"] in accepted_groups:
                accepted += 1
                unique_accepted.add(key)
                rewards.setdefault(row["group_index"], []).append(row["reward"])
    return {"generated_completed_trajectories": generated, "generated_attempts": attempts,
            "resample_attempts": attempts - generated, "accepted_trajectories": accepted,
            "unique_generated_tasks": len(unique), "unique_accepted_tasks": len(unique_accepted),
            "unique_task_coverage": len(unique) / 2524, "scoring_seconds": seconds}, rewards


def four_domain_main():
    import csv
    from itertools import combinations
    parser = argparse.ArgumentParser(description="Four-domain seed300 controlled comparison")
    parser.add_argument("--four-domain", action="store_true")
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, default=Path(__file__).resolve().parents[4] / "tau2-bench")
    labels = ("sft", "vanilla-grpo", "progress-db-count-v1")
    for label in labels:
        parser.add_argument(f"--{label}-eval", type=Path, required=True)
        if label != "sft":
            parser.add_argument(f"--{label}-logs", type=Path, nargs="+", required=True)
            parser.add_argument(f"--{label}-run-dir", type=Path, required=True)
    args = parser.parse_args()
    evaluations, metrics, efficiency, curves = {}, {}, {}, []
    for label in labels:
        evaluations[label] = {300: load_evaluation(getattr(args, f"{label.replace('-', '_')}_eval"),
            expected_seed=300, results_root=args.results_root, label=label, expected_tasks=FOUR_DOMAIN_TASKS)}
        metrics[label] = aggregate_evaluations(evaluations[label], seeds=(300,), domains=tuple(FOUR_DOMAIN_TASKS))
        if label == "sft":
            continue
        logs = getattr(args, f"{label.replace('-', '_')}_logs")
        run_dir = getattr(args, f"{label.replace('-', '_')}_run_dir")
        timing = training_timing(logs)
        batches = {}
        for path in logs:
            for update, groups in re.findall(r"tau2_pool_batch update=(\d+) groups=(\[[^\]]*\])", path.read_text(errors="replace")):
                batches[int(update)] = json.loads(groups)
        accepted_groups = {group for groups in batches.values() for group in groups}
        trajectory, rewards = trajectory_diagnostics(sorted((run_dir / "trajectories").glob("*.jsonl")), accepted_groups)
        for update, groups in sorted(batches.items()):
            values = [r for group in groups for r in rewards.get(group, [])]
            if values:
                curves.append({"recipe": label, "update": update, "reward": sum(values) / len(values), "trajectories": len(values)})
        credit = credit_diagnostics(run_dir / "credit.jsonl")
        zero_signal = (credit["final_zero_signal_group_rate"] if credit is not None else
                       sum(len(set(values)) == 1 for values in rewards.values()) / len(rewards) if rewards else None)
        efficiency[label] = {**timing, **trajectory, "credit": credit,
                             "final_zero_signal_group_rate": zero_signal,
                             "complete_556_updates": sorted(batches) == list(range(556)),
                             "training_gpu_hours": timing["end_to_end_seconds"] * 8 / 3600 if timing["end_to_end_seconds"] else None}
    comparisons = {}
    for reference, candidate in combinations(labels, 2):
        key = f"{candidate}_vs_{reference}"
        comparisons[key] = compare_models(key=key, candidate=evaluations[candidate], reference=evaluations[reference],
            candidate_metrics=metrics[candidate], reference_metrics=metrics[reference], samples=20000,
            bootstrap_seed=1234, seeds=(300,), domains=tuple(FOUR_DOMAIN_TASKS))
    reports, plots = args.experiment_dir / "reports", args.experiment_dir / "plots"
    reports.mkdir(parents=True, exist_ok=True)
    plots.mkdir(parents=True, exist_ok=True)
    (reports / "efficiency.json").write_text(json.dumps(efficiency, indent=2) + "\n")
    (reports / "comparison.json").write_text(json.dumps({"evaluation": metrics, "paired_comparisons": comparisons}, indent=2) + "\n")
    lines = ["# Four-domain asynchronous GRPO", "", "Seed300, four trials per task; 197 tasks and 788 trajectories per model.", "",
             "| Model | Domain | pass@1 | pass@4(any) | pass^4 | Action accuracy | DB accuracy |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for label, result in metrics.items():
        for domain, row in [("overall", result["overall"]), *result["by_domain"].items()]:
            values = [row[key] for key in (*PASS_METRICS, "action_accuracy", "db_accuracy")]
            lines.append(f"| {label} | {domain} | " + " | ".join(f"{v:.2%}" if v is not None else "n/a" for v in values) + " |")
    lines.extend(["", "## Paired task bootstrap", "", "| Comparison | Domain | Metric | Delta (pp) | 95% CI (pp) |", "|---|---|---|---:|---|"])
    for key, comparison in comparisons.items():
        for row in comparison["paired_bootstrap"]:
            low, high = row["ci95"]
            lines.append(f"| {key} | {row['scope']} | {row['metric']} | {100 * row['delta']:+.2f} | [{100 * low:+.2f}, {100 * high:+.2f}] |")
    (reports / "comparison.md").write_text("\n".join(lines) + "\n")
    with (plots / "reward_curves.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=["recipe", "update", "reward", "trajectories"])
        writer.writeheader()
        writer.writerows(curves)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for label in labels[1:]:
        rows = [row for row in curves if row["recipe"] == label]
        plt.plot([r["update"] for r in rows], [r["reward"] for r in rows], label=label, alpha=0.75)
    plt.xlabel("Optimizer update")
    plt.ylabel("Accepted trajectory mean official reward")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots / "reward_curves.png", dpi=180)
    plt.close()


if __name__ == "__main__":
    import sys
    four_domain_main() if "--four-domain" in sys.argv else main()
