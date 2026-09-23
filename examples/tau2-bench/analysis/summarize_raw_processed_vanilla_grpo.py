#!/usr/bin/env python3
"""Summarize the paired raw-all versus processed vanilla-GRPO experiment."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


SERVICE_AGENT_ROOT = Path("/mnt/afs/users/fush/projects/ServiceAgent")
PROJECT_ROOT = SERVICE_AGENT_ROOT / "slime"
EXPERIMENT_DIR = (
    PROJECT_ROOT
    / "output/experiments/tau2-rl-sft-raw-vs-processed-vanilla-grpo"
)
RESULTS_ROOT = SERVICE_AGENT_ROOT / "tau2-bench"
DOMAINS = ("airline", "retail", "telecom")
EXPECTED_TASKS = {"airline": 20, "retail": 40, "telecom": 40}
ARMS = ("raw-all", "processed")
ITERATIONS = (9, 39, 69, 99)
SEEDS = (300, 301)
PASS_METRICS = ("pass_at_1", "pass_at_4_any", "pass_power_4")
METRIC_LABELS = {
    "pass_at_1": "pass@1",
    "pass_at_4_any": "pass@4(any)",
    "pass_power_4": "pass^4",
}
SFT_SUMMARIES = {
    ("raw-all", 300): PROJECT_ROOT
    / "output/experiments/tau2-sft-raw-all-max16384/eval/final/seed300_0903_035744_summary.json",
    ("raw-all", 301): PROJECT_ROOT
    / "output/experiments/tau2-sft-raw-all-max16384/eval/final/seed301_0903_035747_summary.json",
    ("processed", 300): PROJECT_ROOT
    / "output/experiments/tau2-sft-official-native-expanded/eval/checkpoint-iter_0003795/seed300_0902_224647_summary.json",
    ("processed", 301): PROJECT_ROOT
    / "output/experiments/tau2-sft-official-native-expanded/eval/checkpoint-iter_0003795/seed301_0902_233656_summary.json",
}


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def assert_close(actual: Any, expected: float, label: str) -> None:
    if not isinstance(actual, (int, float)) or not math.isclose(
        float(actual), expected, abs_tol=1e-12
    ):
        raise ValueError(f"{label}: recorded {actual!r}, computed {expected}")


def task_metrics(rewards: tuple[float, ...]) -> dict[str, float]:
    if len(rewards) != 4 or any(reward not in (0.0, 1.0) for reward in rewards):
        raise ValueError(f"expected four binary trials, found {rewards}")
    return {
        "pass_at_1": sum(rewards) / 4,
        "pass_at_4_any": float(any(rewards)),
        "pass_power_4": float(all(rewards)),
    }


def aggregate_pass(
    task_rewards: dict[tuple[str, str], tuple[float, ...]],
    *,
    domain: str | None = None,
) -> dict[str, float]:
    rows = [
        task_metrics(rewards)
        for (task_domain, _), rewards in sorted(task_rewards.items())
        if domain is None or task_domain == domain
    ]
    if not rows:
        raise ValueError(f"no task rewards for domain={domain!r}")
    return {
        metric: sum(row[metric] for row in rows) / len(rows)
        for metric in PASS_METRICS
    }


def accuracy(counts: dict[str, int]) -> dict[str, float | None]:
    return {
        "action_accuracy": (
            counts["correct_actions"] / counts["total_actions"]
            if counts["total_actions"]
            else None
        ),
        "db_accuracy": (
            counts["db_matches"] / (counts["db_matches"] + counts["db_mismatches"])
            if counts["db_matches"] + counts["db_mismatches"]
            else None
        ),
    }


def load_evaluation(
    path: Path,
    *,
    expected_seed: int,
    results_root: Path,
    label: str,
    expected_tasks: dict[str, int] = EXPECTED_TASKS,
) -> dict[str, Any]:
    summary = read_json(path)
    expected_header = {
        "task_split_name": "test",
        "num_trials": 4,
        "seed": expected_seed,
        "max_steps": 200,
        "agent_eval_mode": "official-native",
    }
    for key, expected in expected_header.items():
        if summary.get(key) != expected:
            raise ValueError(
                f"{label}: expected {key}={expected!r}, found {summary.get(key)!r}"
            )

    all_rewards: dict[tuple[str, str], tuple[float, ...]] = {}
    all_counts = defaultdict(int)
    by_domain = {}
    for domain in expected_tasks:
        domain_summary = (summary.get("domains") or {}).get(domain) or {}
        relative_path = Path(str(domain_summary.get("results_file") or ""))
        result_path = (
            relative_path if relative_path.is_absolute() else results_root / relative_path
        )
        result = read_json(result_path)
        simulations = result.get("simulations") or []
        expected_simulations = expected_tasks[domain] * 4
        if len(simulations) != expected_simulations:
            raise ValueError(
                f"{label}/{domain}: expected {expected_simulations} simulations, "
                f"found {len(simulations)}"
            )

        trials: dict[str, dict[int, float]] = defaultdict(dict)
        for simulation in simulations:
            task_id = str(simulation.get("task_id") or "")
            trial = simulation.get("trial")
            reward = (simulation.get("reward_info") or {}).get("reward")
            if (
                not task_id
                or not isinstance(trial, int)
                or trial in trials[task_id]
            ):
                raise ValueError(f"{label}/{domain}: malformed or duplicate task/trial")
            if reward not in (0, 0.0, 1, 1.0):
                raise ValueError(f"{label}/{domain}/{task_id}: non-binary reward")
            if simulation.get("termination_reason") == "infrastructure_error":
                raise ValueError(f"{label}/{domain}/{task_id}: infrastructure error")
            trials[task_id][trial] = float(reward)
        if len(trials) != expected_tasks[domain]:
            raise ValueError(
                f"{label}/{domain}: expected {expected_tasks[domain]} tasks, "
                f"found {len(trials)}"
            )
        domain_rewards = {}
        for task_id, task_trials in trials.items():
            if set(task_trials) != {0, 1, 2, 3}:
                raise ValueError(f"{label}/{domain}/{task_id}: incomplete trials")
            rewards = tuple(task_trials[index] for index in range(4))
            domain_rewards[(domain, task_id)] = rewards
        all_rewards.update(domain_rewards)

        computed = aggregate_pass(domain_rewards)
        recorded = domain_summary.get("pass_metrics") or {}
        for metric, value in computed.items():
            assert_close(recorded.get(metric), value, f"{label}/{domain}/{metric}")
        if recorded.get("tasks") != expected_tasks[domain] or recorded.get(
            "simulations"
        ) != expected_simulations:
            raise ValueError(f"{label}/{domain}: summary coverage mismatch")

        raw_counts = domain_summary.get("metrics") or {}
        if raw_counts.get("infra_error_count") != 0:
            raise ValueError(f"{label}/{domain}: infra_error_count is not zero")
        counts = {
            "correct_actions": int(raw_counts.get("correct_read_actions") or 0)
            + int(raw_counts.get("correct_write_actions") or 0),
            "total_actions": int(raw_counts.get("total_read_actions") or 0)
            + int(raw_counts.get("total_write_actions") or 0),
            "db_matches": int(raw_counts.get("db_match_count") or 0),
            "db_mismatches": int(raw_counts.get("db_mismatch_count") or 0),
        }
        for key, value in counts.items():
            all_counts[key] += value
        by_domain[domain] = {**computed, **accuracy(counts), "counts": counts}

    overall = aggregate_pass(all_rewards)
    recorded_overall = summary.get("overall") or {}
    for metric, value in overall.items():
        assert_close(recorded_overall.get(metric), value, f"{label}/overall/{metric}")
    if recorded_overall.get("tasks") != sum(expected_tasks.values()) or recorded_overall.get(
        "simulations"
    ) != 4 * sum(expected_tasks.values()):
        raise ValueError(f"{label}: overall coverage mismatch")
    if (recorded_overall.get("diagnostics") or {}).get("infrastructure_errors") != 0:
        raise ValueError(f"{label}: overall infrastructure errors are not zero")

    return {
        "label": label,
        "seed": expected_seed,
        "summary_path": str(path.resolve()),
        "overall": {**overall, **accuracy(all_counts), "counts": dict(all_counts)},
        "by_domain": by_domain,
        "task_rewards": all_rewards,
    }


def discover_rl_summary(experiment_dir: Path, arm: str, iteration: int, seed: int) -> Path:
    directory = experiment_dir / "eval" / f"{arm}-iter{iteration}"
    paths = sorted(directory.glob(f"seed{seed}_*_summary.json"))
    if len(paths) != 1:
        raise ValueError(
            f"expected one seed{seed} summary below {directory}, found {len(paths)}"
        )
    return paths[0]


def aggregate_evaluations(rows: dict[int, dict[str, Any]], *, seeds=SEEDS, domains=DOMAINS) -> dict[str, Any]:
    if set(rows) != set(seeds):
        raise ValueError(f"aggregate requires seeds {seeds}")
    task_sets = [set(row["task_rewards"]) for row in rows.values()]
    if any(tasks != task_sets[0] for tasks in task_sets[1:]):
        raise ValueError("evaluation seeds do not contain the same tasks")

    def scope_metrics(domain: str | None) -> dict[str, Any]:
        tasks = [
            key
            for key in sorted(task_sets[0])
            if domain is None or key[0] == domain
        ]
        values = {
            metric: sum(
                task_metrics(rows[seed]["task_rewards"][key])[metric]
                for seed in seeds
                for key in tasks
            )
            / (len(seeds) * len(tasks))
            for metric in PASS_METRICS
        }
        counts = defaultdict(int)
        for seed in seeds:
            source = (
                rows[seed]["overall"]
                if domain is None
                else rows[seed]["by_domain"][domain]
            )["counts"]
            for key, value in source.items():
                counts[key] += value
        return {**values, **accuracy(counts), "counts": dict(counts)}

    return {
        "seeds": list(seeds),
        "overall": scope_metrics(None),
        "by_domain": {domain: scope_metrics(domain) for domain in domains},
        "per_seed": {
            str(seed): {
                "overall": rows[seed]["overall"],
                "by_domain": rows[seed]["by_domain"],
            }
            for seed in seeds
        },
    }


def paired_bootstrap_ci(
    deltas: list[float], *, samples: int, seed: int
) -> tuple[float, float, float]:
    import numpy as np

    values = np.asarray(deltas, dtype=np.float64)
    observed = float(values.mean())
    rng = np.random.default_rng(seed)
    draws = np.empty(samples, dtype=np.float64)
    for start in range(0, samples, 10_000):
        stop = min(start + 10_000, samples)
        indices = rng.integers(0, len(values), size=(stop - start, len(values)))
        draws[start:stop] = values[indices].mean(axis=1)
    low, high = np.quantile(draws, [0.025, 0.975], method="linear")
    return observed, float(low), float(high)


def compare_models(
    *,
    key: str,
    candidate: dict[int, dict[str, Any]],
    reference: dict[int, dict[str, Any]],
    candidate_metrics: dict[str, Any],
    reference_metrics: dict[str, Any],
    samples: int,
    bootstrap_seed: int,
    seeds=SEEDS,
    domains=DOMAINS,
) -> dict[str, Any]:
    task_sets = [
        set(row["task_rewards"])
        for row in (*candidate.values(), *reference.values())
    ]
    if any(tasks != task_sets[0] for tasks in task_sets[1:]):
        raise ValueError(f"{key}: task sets differ")
    rows = []
    scopes = (("overall", None), *((domain, domain) for domain in domains))
    for scope_index, (scope, domain) in enumerate(scopes):
        tasks = [
            task
            for task in sorted(task_sets[0])
            if domain is None or task[0] == domain
        ]
        for metric_index, metric in enumerate(PASS_METRICS):
            deltas = [
                sum(
                    task_metrics(candidate[seed]["task_rewards"][task])[metric]
                    - task_metrics(reference[seed]["task_rewards"][task])[metric]
                    for seed in seeds
                )
                / len(seeds)
                for task in tasks
            ]
            observed, low, high = paired_bootstrap_ci(
                deltas,
                samples=samples,
                seed=bootstrap_seed + scope_index * 10 + metric_index,
            )
            rows.append(
                {
                    "scope": scope,
                    "metric": metric,
                    "tasks": len(tasks),
                    "delta": observed,
                    "ci95": [low, high],
                    "ci_excludes_zero": low > 0 or high < 0,
                }
            )

    metric_deltas = {}
    for scope, candidate_scope, reference_scope in (
        ("overall", candidate_metrics["overall"], reference_metrics["overall"]),
        *(
            (
                domain,
                candidate_metrics["by_domain"][domain],
                reference_metrics["by_domain"][domain],
            )
            for domain in domains
        ),
    ):
        metric_deltas[scope] = {
            metric: candidate_scope[metric] - reference_scope[metric]
            for metric in (*PASS_METRICS, "action_accuracy", "db_accuracy")
            if candidate_scope[metric] is not None and reference_scope[metric] is not None
        }
    return {
        "key": key,
        "metric_deltas": metric_deltas,
        "per_seed_pass_at_1_delta": {
            str(seed): candidate[seed]["overall"]["pass_at_1"]
            - reference[seed]["overall"]["pass_at_1"]
            for seed in seeds
        },
        "paired_bootstrap": rows,
    }


def compact_training(path: Path, arm: str) -> dict[str, Any]:
    report = read_json(path)
    logs = report.get("run_log_summary") or {}
    trajectory = report.get("trajectory_summary") or {}
    if logs.get("observed_steps") != list(range(100)):
        raise ValueError(f"{arm}: training diagnostics do not cover steps 0..99")
    if logs.get("metric_parse_error_lines"):
        raise ValueError(f"{arm}: non-finite or malformed training metric line")
    quota = logs.get("sparse_count_totals") or {}
    expected_quota = {"airline": 100.0, "retail": 200.0, "telecom": 200.0}
    for domain, expected in expected_quota.items():
        actual = quota.get(f"rollout/domain_quota/{domain}/accepted")
        if actual != expected:
            raise ValueError(f"{arm}: accepted {domain} groups {actual}, expected {expected}")
    policy_groups = trajectory.get("policy_signal_groups") or {}
    metrics = logs.get("metrics") or {}
    for metric in (
        "train/loss",
        "train/grad_norm",
        "rollout/raw_reward",
        "train/kl_loss",
    ):
        if (metrics.get(metric) or {}).get("updates") != 100:
            raise ValueError(f"{arm}: {metric} does not cover all 100 updates")
    global_batch_size = metrics.get("train/global_batch_size") or {}
    if (
        global_batch_size.get("updates") != 100
        or global_batch_size.get("min") != 40
        or global_batch_size.get("max") != 40
    ):
        raise ValueError(f"{arm}: every update must train global batch 40")
    zero_std_counts = logs.get("zero_std_count_totals") or {}
    accepted_zero_std_groups = int(sum(zero_std_counts.values()))
    sampling = trajectory.get("sampling") or {}
    dumped_trajectories = (trajectory.get("trajectory_counts") or {}).get("dumped")
    series = logs.get("series") or {}
    series_by_key = {
        key: {int(item["step"]): item["value"] for item in values}
        for key, values in series.items()
    }
    per_step_quota = {"airline": 1.0, "retail": 2.0, "telecom": 2.0}
    for domain, expected_per_step in per_step_quota.items():
        values = series_by_key[f"rollout/domain_quota/{domain}/accepted"]
        if set(values) != set(range(100)) or any(
            value != expected_per_step for value in values.values()
        ):
            raise ValueError(
                f"{arm}: {domain} accepted quota drifted from "
                f"{expected_per_step:g} per update"
            )
    training_curve = []
    for step in range(100):
        training_curve.append(
            {
                "step": step,
                "raw_reward": series_by_key["rollout/raw_reward"][step],
                "kl_drift": series_by_key["train/kl_loss"][step],
                "truncated_ratio": series_by_key["rollout/truncated_ratio"][step],
                "loss": series_by_key["train/loss"][step],
                "grad_norm": series_by_key["train/grad_norm"][step],
                "zero_std_groups": sum(
                    values.get(step, 0.0)
                    for key, values in series_by_key.items()
                    if key.startswith("rollout/zero_std/count_")
                ),
            }
        )
    return {
        "path": str(path.resolve()),
        "updates": 100,
        "accepted_prompt_groups": 500,
        "accepted_trajectories": 4000,
        "accepted_domain_groups": expected_quota,
        "global_batch_size": global_batch_size,
        "raw_reward": metrics.get("rollout/raw_reward"),
        "kl_drift": metrics.get("train/kl_loss"),
        "loss": metrics.get("train/loss"),
        "grad_norm": metrics.get("train/grad_norm"),
        "truncated": metrics.get("rollout/truncated_ratio"),
        "zero_std_count_totals": zero_std_counts,
        "accepted_zero_std_groups": accepted_zero_std_groups,
        "accepted_zero_std_rate": accepted_zero_std_groups / 500,
        "zero_signal_groups": policy_groups.get("zero_signal"),
        "zero_signal_rate": policy_groups.get("zero_signal_rate"),
        "viable_candidate_groups_before_final_selection": policy_groups.get(
            "trainable_candidates"
        ),
        "dumped_trajectories": dumped_trajectories,
        "sampling": sampling,
        "retry_rate": (
            sampling.get("retried_trajectories", 0) / dumped_trajectories
            if dumped_trajectories
            else None
        ),
        "permanently_overlong_rate": (
            sampling.get("permanently_too_long", 0) / dumped_trajectories
            if dumped_trajectories
            else None
        ),
        "termination": trajectory.get("termination"),
        "curve": training_curve,
    }


def public_evaluation(row: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if key != "task_rewards"}


def build_report(
    *,
    experiment_dir: Path,
    results_root: Path,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    sft = {
        arm: {
            seed: load_evaluation(
                SFT_SUMMARIES[(arm, seed)],
                expected_seed=seed,
                results_root=results_root,
                label=f"{arm}-sft-seed{seed}",
            )
            for seed in SEEDS
        }
        for arm in ARMS
    }
    rl = {
        arm: {
            iteration: {
                seed: load_evaluation(
                    discover_rl_summary(experiment_dir, arm, iteration, seed),
                    expected_seed=seed,
                    results_root=results_root,
                    label=f"{arm}-iter{iteration}-seed{seed}",
                )
                for seed in ((300, 301) if iteration == 99 else (300,))
            }
            for iteration in ITERATIONS
        }
        for arm in ARMS
    }
    final_models = {}
    for arm in ARMS:
        final_models[f"{arm}-sft"] = aggregate_evaluations(sft[arm])
        final_models[f"{arm}-rl-iter99"] = aggregate_evaluations(rl[arm][99])

    comparisons = [
        compare_models(
            key="raw-all-rl-minus-sft",
            candidate=rl["raw-all"][99],
            reference=sft["raw-all"],
            candidate_metrics=final_models["raw-all-rl-iter99"],
            reference_metrics=final_models["raw-all-sft"],
            samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed,
        ),
        compare_models(
            key="processed-rl-minus-sft",
            candidate=rl["processed"][99],
            reference=sft["processed"],
            candidate_metrics=final_models["processed-rl-iter99"],
            reference_metrics=final_models["processed-sft"],
            samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed + 100,
        ),
        compare_models(
            key="processed-rl-minus-raw-all-rl",
            candidate=rl["processed"][99],
            reference=rl["raw-all"][99],
            candidate_metrics=final_models["processed-rl-iter99"],
            reference_metrics=final_models["raw-all-rl-iter99"],
            samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed + 200,
        ),
    ]
    primary = next(
        row
        for row in comparisons[-1]["paired_bootstrap"]
        if row["scope"] == "overall" and row["metric"] == "pass_at_1"
    )
    if primary["ci95"][0] > 0:
        decision = "processed_rl_leads_with_ci_excluding_zero"
    elif primary["ci95"][1] < 0:
        decision = "raw_all_rl_leads_with_ci_excluding_zero"
    elif primary["delta"] > 0:
        decision = "processed_rl_point_estimate_leads_but_evidence_is_insufficient"
    elif primary["delta"] < 0:
        decision = "raw_all_rl_point_estimate_leads_but_evidence_is_insufficient"
    else:
        decision = "iter99_two_seed_pass_at_1_point_estimate_tied"

    training = {
        arm: compact_training(
            experiment_dir / "arms" / arm / "TRAINING_DIAGNOSTICS.json", arm
        )
        for arm in ARMS
    }
    return {
        "experiment": "tau2-rl-sft-raw-vs-processed-vanilla-grpo",
        "decision": decision,
        "primary_comparison": primary,
        "protocol": {
            "training_seed": 1234,
            "rollout_seed": 42,
            "evaluation_seeds": list(SEEDS),
            "selection_checkpoint": 99,
            "intermediate_checkpoints_are_diagnostic_only": True,
        },
        "iter99_two_seed": final_models,
        "comparisons": comparisons,
        "seed300_curve": {
            arm: {
                str(iteration): public_evaluation(rl[arm][iteration][300])
                for iteration in ITERATIONS
            }
            for arm in ARMS
        },
        "training": training,
        "bootstrap": {
            "unit": "task; the two evaluation seeds are averaged within task",
            "samples": bootstrap_samples,
            "seed": bootstrap_seed,
            "interval": "percentile_95",
        },
        "scope": (
            "One paired training seed; evidence describes vanilla-GRPO trainability "
            "and final capability, not training-seed-invariant superiority."
        ),
    }


def percent(value: float | None) -> str:
    return "n/a" if value is None else f"{100 * value:.2f}%"


def pp(value: float | None) -> str:
    return "n/a" if value is None else f"{100 * value:+.2f} pp"


def render_markdown(report: dict[str, Any]) -> str:
    model_labels = (
        ("raw-all-sft", "Raw-all SFT"),
        ("raw-all-rl-iter99", "Raw-all RL iter99"),
        ("processed-sft", "Processed SFT"),
        ("processed-rl-iter99", "Processed RL iter99"),
    )
    decision_text = report["decision"].replace("_", " ")
    lines = [
        "# Raw-all versus Processed vanilla GRPO results",
        "",
        "## Conclusion",
        "",
        f"Primary result: `{decision_text}`. The decision uses the two-seed iter99 "
        "overall pass@1 mean; intermediate checkpoints are diagnostic only.",
        "",
        "## Iter99 two-seed results",
        "",
        "| Model | pass@1 | pass@4(any) | pass^4 | Action accuracy | DB accuracy |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for key, label in model_labels:
        metrics = report["iter99_two_seed"][key]["overall"]
        lines.append(
            f"| {label} | {percent(metrics['pass_at_1'])} | "
            f"{percent(metrics['pass_at_4_any'])} | {percent(metrics['pass_power_4'])} | "
            f"{percent(metrics['action_accuracy'])} | {percent(metrics['db_accuracy'])} |"
        )

    lines.extend(
        [
            "",
            "## Final deltas",
            "",
            "| Comparison | pass@1 | pass@4(any) | pass^4 | Action accuracy | DB accuracy |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for comparison in report["comparisons"]:
        delta = comparison["metric_deltas"]["overall"]
        lines.append(
            f"| {comparison['key']} | {pp(delta['pass_at_1'])} | "
            f"{pp(delta['pass_at_4_any'])} | {pp(delta['pass_power_4'])} | "
            f"{pp(delta['action_accuracy'])} | {pp(delta['db_accuracy'])} |"
        )

    lines.extend(
        [
            "",
            "## Paired task bootstrap",
            "",
            f"Intervals use {report['bootstrap']['samples']:,} task-level resamples; "
            "seed 300 and 301 deltas are averaged within each task before resampling.",
            "",
            "| Comparison | Metric | Delta | 95% CI | Excludes zero |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for comparison in report["comparisons"]:
        for row in comparison["paired_bootstrap"]:
            if row["scope"] != "overall":
                continue
            lines.append(
                f"| {comparison['key']} | {METRIC_LABELS[row['metric']]} | "
                f"{pp(row['delta'])} | [{pp(row['ci95'][0])}, {pp(row['ci95'][1])}] | "
                f"{'yes' if row['ci_excludes_zero'] else 'no'} |"
            )

    direct = report["comparisons"][-1]["per_seed_pass_at_1_delta"]
    lines.extend(
        [
            "",
            "## Per-seed direction",
            "",
            "| Seed | Processed RL - Raw-all RL pass@1 |",
            "|---:|---:|",
            *[f"| {seed} | {pp(direct[str(seed)])} |" for seed in SEEDS],
            "",
            "## Seed-300 checkpoint curve",
            "",
            "| Arm | Iteration | pass@1 | pass@4(any) | pass^4 | Action accuracy | DB accuracy |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for arm in ARMS:
        for iteration in ITERATIONS:
            metrics = report["seed300_curve"][arm][str(iteration)]["overall"]
            lines.append(
                f"| {arm} | {iteration} | {percent(metrics['pass_at_1'])} | "
                f"{percent(metrics['pass_at_4_any'])} | {percent(metrics['pass_power_4'])} | "
                f"{percent(metrics['action_accuracy'])} | {percent(metrics['db_accuracy'])} |"
            )

    lines.extend(
        [
            "",
            "## Training diagnostics",
            "",
            "| Arm | Groups / trajectories | Raw reward mean | KL mean / max | "
            "Accepted binary zero-variance groups | Truncation mean / max | "
            "Over-cap retried / rate | Permanently over-cap / rate |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for arm in ARMS:
        row = report["training"][arm]
        retry = row["sampling"] or {}
        lines.append(
            f"| {arm} | {row['accepted_prompt_groups']} / {row['accepted_trajectories']} | "
            f"{row['raw_reward']['mean']:.4f} | {row['kl_drift']['mean']:.6f} / "
            f"{row['kl_drift']['max']:.6f} | {row['accepted_zero_std_groups']} "
            f"({percent(row['accepted_zero_std_rate'])}) | {row['truncated']['mean']:.4f} / "
            f"{row['truncated']['max']:.4f} | {retry.get('retried_trajectories', 0)} / "
            f"{percent(row['retry_rate'])} | {retry.get('permanently_too_long', 0)} / "
            f"{percent(row['permanently_overlong_rate'])} |"
        )
    lines.extend(
        [
            "",
            "Observation points from the full per-update curves stored in `RESULTS.json`:",
            "",
            "| Arm | Step | Raw reward | K2 drift | Grad norm | Zero-variance groups | Truncation |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for arm in ARMS:
        curve = {row["step"]: row for row in report["training"][arm]["curve"]}
        for step in (0, 9, 39, 69, 99):
            point = curve[step]
            lines.append(
                f"| {arm} | {step} | {point['raw_reward']:.4f} | "
                f"{point['kl_drift']:.6f} | {point['grad_norm']:.4f} | "
                f"{int(point['zero_std_groups'])} | {point['truncated_ratio']:.4f} |"
            )
    lines.extend(
        [
            "",
            "Both arms completed 100 finite updates and accepted Airline/Retail/Telecom "
            "groups in the exact 100/200/200 totals. KL is a coefficient-zero K2 drift "
            "diagnostic and entropy does not enter the loss.",
            "",
            "## Scope",
            "",
            report["scope"],
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, default=EXPERIMENT_DIR)
    parser.add_argument("--results-root", type=Path, default=RESULTS_ROOT)
    parser.add_argument("--bootstrap-samples", type=int, default=100_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260903)
    parser.add_argument(
        "--json-output", type=Path, default=EXPERIMENT_DIR / "RESULTS.json"
    )
    parser.add_argument(
        "--markdown-output", type=Path, default=EXPERIMENT_DIR / "RESULTS.md"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.bootstrap_samples <= 0:
        raise ValueError("--bootstrap-samples must be positive")
    report = build_report(
        experiment_dir=args.experiment_dir,
        results_root=args.results_root,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    args.json_output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    args.markdown_output.write_text(render_markdown(report), encoding="utf-8")


if __name__ == "__main__":
    main()
