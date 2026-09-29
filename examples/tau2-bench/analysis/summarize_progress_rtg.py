"""Paired official evaluation of progress-rtg-v1 against the fixed historical controls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from summarize_raw_processed_vanilla_grpo import (
    DOMAINS, PASS_METRICS, METRIC_LABELS, RESULTS_ROOT, EXPERIMENT_DIR, SFT_SUMMARIES,
    load_evaluation, discover_rl_summary, aggregate_evaluations, task_metrics, paired_bootstrap_ci,
)


def compare(candidate, reference, bootstrap_samples):
    seeds = sorted(candidate)
    tasks = set(candidate[seeds[0]]["task_rewards"])
    if set(reference) != set(seeds) or any(set(row["task_rewards"]) != tasks
                                         for row in [*candidate.values(), *reference.values()]):
        raise ValueError("Paired comparison needs identical task and seed sets")
    rows = []
    for scope in ("overall", *DOMAINS):
        selected = sorted(task for task in tasks if scope == "overall" or task[0] == scope)
        for metric in PASS_METRICS:
            deltas = [sum(task_metrics(candidate[seed]["task_rewards"][task])[metric]
                          - task_metrics(reference[seed]["task_rewards"][task])[metric]
                          for seed in seeds) / len(seeds) for task in selected]
            delta, low, high = paired_bootstrap_ci(deltas, samples=bootstrap_samples, seed=1234)
            rows.append({"scope": scope, "metric": metric, "delta": delta, "ci95": [low, high]})
    return rows


def summarize(run_dir, iteration, bootstrap_samples=100_000):
    seeds = (300,) if iteration == 39 else (300, 301)
    evaluations = {name: {} for name in ("progress", "vanilla", "sft")}
    for seed in seeds:
        paths = {
            "progress": run_dir / f"eval/progress-rtg-v1-iter{iteration}/seed{seed}/summary.json",
            "vanilla": discover_rl_summary(EXPERIMENT_DIR, "processed", iteration, seed),
            "sft": SFT_SUMMARIES[("processed", seed)],
        }
        for name, path in paths.items():
            evaluations[name][seed] = load_evaluation(path, expected_seed=seed, results_root=RESULTS_ROOT, label=name)
    aggregates = {}
    for name, rows in evaluations.items():
        aggregates[name] = aggregate_evaluations(rows) if len(seeds) == 2 else {
            key: rows[300][key] for key in ("overall", "by_domain")}
    comparisons = {name: compare(evaluations["progress"], evaluations[name], bootstrap_samples)
                   for name in ("vanilla", "sft")}
    metrics = aggregates["progress"]["overall"]
    continue_training = iteration == 39 and metrics["pass_at_1"] > 0.5725 and metrics["pass_power_4"] >= 0.29
    report = {
        "iteration": iteration, "seeds": seeds, "bootstrap_samples": bootstrap_samples,
        "models": aggregates, "comparisons": comparisons,
        "summary_paths": {name: {seed: row["summary_path"] for seed, row in rows.items()}
                          for name, rows in evaluations.items()},
        "continue_to_100": continue_training if iteration == 39 else None,
    }
    lines = [f"# Progress RTG iter{iteration} official evaluation", "",
             "Processed SFT init; fixed Qwen3.6 User and official binary success. Training seed 1234.", "",
             "| Model / domain | pass@1 | pass@4(any) | pass^4 | Action accuracy | DB accuracy |",
             "|---|---:|---:|---:|---:|---:|"]
    for name, model in aggregates.items():
        for scope, values in [("overall", model["overall"]), *model["by_domain"].items()]:
            cells = ["n/a" if values.get(key) is None else f"{values[key] * 100:.2f}%"
                     for key in (*PASS_METRICS, "action_accuracy", "db_accuracy")]
            lines.append(f"| {name} / {scope} | " + " | ".join(cells) + " |")
    lines += ["", "## Paired task bootstrap", "",
              f"{bootstrap_samples:,} task resamples; evaluation seeds averaged within each task before resampling.", "",
              "| Progress minus control | Domain | Metric | Delta (pp) | 95% CI (pp) |",
              "|---|---|---|---:|---:|"]
    for name, rows in comparisons.items():
        for row in rows:
            low, high = row["ci95"]
            lines.append(f"| {name} | {row['scope']} | {METRIC_LABELS[row['metric']]} | "
                         f"{100 * row['delta']:+.2f} | [{100 * low:+.2f}, {100 * high:+.2f}] |")
    lines += ["", "## Decision", ""]
    if iteration == 39:
        lines.append("Continue to 100 updates." if continue_training else "Stop at iter39; the predeclared trend criterion was not met.")
        lines.append("Criterion: pass@1 > 57.25% and pass^4 ≥ 29.00%. Passing only supports continuing the experiment.")
    else:
        primary = comparisons["vanilla"][0]
        lines.append(f"Primary two-seed pass@1 delta versus vanilla GRPO: {100 * primary['delta']:+.2f} pp.")
    lines.append("Intervals containing zero do not establish improvement. Conclusions are limited to this one training seed.")
    destination = run_dir / f"iter{iteration}_results"
    destination.with_suffix(".json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    destination.with_suffix(".md").write_text("\n".join(lines) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--iteration", type=int, choices=(39, 99), required=True)
    args = parser.parse_args()
    result = summarize(args.run_dir, args.iteration)
    print(json.dumps({"iteration": result["iteration"], "continue_to_100": result["continue_to_100"]}))
