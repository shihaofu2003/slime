"""Summarize the completed five-model, two-seed OPD comparison."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

EXP = Path(__file__).resolve().parent
MODELS = ("sft", "airline", "retail", "A", "B")
DOMAINS = {"airline": 20, "retail": 40, "telecom": 40}
SEEDS = (300, 301)
METRICS = ("pass_at_1", "pass_at_4_any", "pass_power_4")


def summarize():
    paths = {(model, seed): EXP / "eval" / model / f"seed{seed}_summary.json"
             for model in MODELS for seed in SEEDS}
    missing = [f"{model}/seed{seed}" for (model, seed), path in paths.items() if not path.exists()]
    if missing:
        print("Waiting for evaluation summaries: " + ", ".join(missing))
        return

    scores, diagnostics, task_metrics = {}, {}, {}
    for (model, seed), path in paths.items():
        summary = json.loads(path.read_text())
        for domain, expected_tasks in DOMAINS.items():
            row = summary["domains"][domain]
            if row["pass_metrics"]["simulations"] != expected_tasks * 4 or row["metrics"]["infra_error_count"]:
                print(f"Incomplete or infrastructure-failed evaluation: {model}/seed{seed}/{domain}")
                return
            scores[model, seed, domain] = [row["pass_metrics"][metric] for metric in METRICS]
            diagnostics[model, seed, domain] = row["diagnostics"]
            results = json.loads(Path(row["results_file"]).read_text())
            by_task = defaultdict(list)
            for simulation in results["simulations"]:
                by_task[simulation["task_id"]].append(float(simulation["reward_info"]["reward"]))
            task_metrics[model, seed, domain] = {
                task: np.array([np.mean(values), float(any(values)), float(all(values))])
                for task, values in by_task.items()
            }

    comparisons = {}
    for candidate, reference, domain in (
        ("A", "B", "airline"), ("A", "B", "retail"), ("A", "B", "telecom"),
        ("A", "sft", "airline"), ("A", "sft", "retail"),
        ("A", "airline", "airline"), ("A", "retail", "retail"),
        ("B", "sft", "airline"), ("B", "sft", "retail"),
        ("B", "airline", "airline"), ("B", "retail", "retail"),
        ("airline", "sft", "airline"), ("retail", "sft", "retail"),
        ("airline", "retail", "retail"),
    ):
        task_ids = sorted(task_metrics[candidate, SEEDS[0], domain])
        # Keep both seeds and all four trials together for each resampled task.
        delta = np.array([
            np.mean([task_metrics[candidate, seed, domain][task]
                     - task_metrics[reference, seed, domain][task] for seed in SEEDS], axis=0)
            for task in task_ids
        ]) * 100
        rng = np.random.default_rng(20260922)
        bootstrap = delta[rng.integers(0, len(task_ids), (20000, len(task_ids)))].mean(axis=1)
        comparisons[f"{candidate}-{reference}/{domain}"] = {
            "delta_pp": delta.mean(axis=0).tolist(),
            "task_paired_ci95_pp": np.quantile(bootstrap, [0.025, 0.975], axis=0).T.tolist(),
            "per_seed_delta_pp": {
                seed: ((np.array(scores[candidate, seed, domain]) - scores[reference, seed, domain]) * 100).tolist()
                for seed in SEEDS
            },
        }

    means = {model: {domain: np.mean([scores[model, seed, domain] for seed in SEEDS], axis=0).tolist()
                     for domain in DOMAINS} for model in MODELS}
    output = {
        "seeds": list(SEEDS), "metric_order": list(METRICS), "means": means,
        "per_seed": {f"{model}/seed{seed}/{domain}": value for (model, seed, domain), value in scores.items()},
        "diagnostics": {f"{model}/seed{seed}/{domain}": value for (model, seed, domain), value in diagnostics.items()},
        "comparisons": comparisons,
    }
    (EXP / "comparison.json").write_text(json.dumps(output, indent=2) + "\n")
    lines = [
        "# OPD buffer A/B official results", "",
        "Purpose: compare total buffer32 (A) with unbounded prefetch (B), both LR2e-6/60 updates.", "",
        "All five models: native three-domain test, seeds300/301, four trials, identical Agent/User and concurrency settings.",
        "B resumed from iter49 after a compute-cluster crash. Pending trajectories were regenerated at the restored policy; its final ten updates are not an uninterrupted unbounded-prefetch control.",
        "Cells are pass@1 / pass@4(any) / pass^4 (%), averaged over both seeds.", "",
        "| Model | Airline | Retail | Telecom |", "|---|---|---|---|",
    ]
    for model in MODELS:
        cells = [" / ".join(f"{v * 100:.2f}" for v in means[model][domain]) for domain in DOMAINS]
        lines.append("| " + " | ".join([model, *cells]) + " |")
    lines += ["", "## Paired task uncertainty", "",
              "20,000 task-cluster bootstrap draws; seeds/trials stay within their task. These intervals do not cover training-seed uncertainty.", "",
              "| Comparison/domain | Δ pass@1 pp | 95% CI pp | Δ pass@4(any) pp | Δ pass^4 pp |",
              "|---|---:|---|---:|---:|"]
    for name, row in comparisons.items():
        a, b, c = row["delta_pp"]
        lo, hi = row["task_paired_ci95_pp"][0]
        lines.append(f"| {name} | {a:+.2f} | [{lo:+.2f}, {hi:+.2f}] | {b:+.2f} | {c:+.2f} |")
    lines += ["", "[Per-seed metrics, action/DB diagnostics and all intervals](comparison.json).",
              "Interpret jointly with the consumed-domain counts and lag in the A/B training logs; no automatic recipe winner is declared.", ""]
    (EXP / "RESULTS.md").write_text("\n".join(lines))
    print(EXP / "RESULTS.md")


if __name__ == "__main__":
    summarize()
