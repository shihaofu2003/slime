"""Recompute the bounded OPD learning-rate comparison from saved official results."""

import ast
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np


EXP = Path(__file__).resolve().parent
PREVIOUS = EXP.parent / "tau2-opd-current-buffer-ab"
MODELS = {
    "sft": PREVIOUS / "eval/sft",
    "airline_teacher": PREVIOUS / "eval/airline",
    "retail_teacher": PREVIOUS / "eval/retail",
    "old2e6": PREVIOUS / "eval/A",
    "lr2e6": EXP / "eval/lr2e6",
    "lr5e6": EXP / "eval/lr5e6",
}
DOMAINS = {"airline": 20, "retail": 40, "telecom": 40}
SEEDS = (300, 301)
METRICS = ("pass_at_1", "pass_at_4_any", "pass_power_4")


def protocol(info):
    result = {k: info[k] for k in ("num_trials", "max_steps", "max_errors", "seed", "environment_info")}
    for role in ("agent_info", "user_info"):
        result[role] = dict(info[role])
        result[role]["llm_args"] = {
            k: v for k, v in info[role]["llm_args"].items() if k not in ("api_base", "api_key")
        }
    return result


def main():
    scores, task_metrics, counts, references = {}, {}, {}, {}
    protocol_differences = []
    total_simulations = 0
    for model, directory in MODELS.items():
        for seed in SEEDS:
            summary = json.loads((directory / f"seed{seed}_summary.json").read_text())
            for domain, expected_tasks in DOMAINS.items():
                row = summary["domains"][domain]
                raw = json.loads(Path(row["results_file"]).read_text())
                simulations = raw["simulations"]
                identities = [(s["task_id"], s["trial"]) for s in simulations]
                assert len(identities) == len(set(identities)) == expected_tasks * 4, (model, seed, domain)
                groups = defaultdict(list)
                for simulation in simulations:
                    reward = simulation["reward_info"]["reward"]
                    assert reward in (0, 1), (model, seed, domain, reward)
                    groups[simulation["task_id"]].append(reward)
                assert len(groups) == expected_tasks and all(len(v) == 4 for v in groups.values())
                per_task = {
                    task: np.array([np.mean(values), float(any(values)), float(all(values))])
                    for task, values in groups.items()
                }
                recomputed = np.mean(list(per_task.values()), axis=0)
                assert np.allclose(recomputed, [row["pass_metrics"][k] for k in METRICS])
                scores[model, seed, domain] = recomputed
                task_metrics[model, seed, domain] = per_task
                counts[model, seed, domain] = row["metrics"]
                total_simulations += len(simulations)
                comparable = {
                    "protocol": protocol(raw["info"]),
                    "tasks": {task["id"]: task for task in raw["tasks"]},
                    "trial_seeds": {(s["task_id"], s["trial"]): s["seed"] for s in simulations},
                }
                if model == "sft":
                    references[seed, domain] = comparable
                else:
                    for field, value in comparable.items():
                        if value != references[seed, domain][field]:
                            protocol_differences.append(f"{model}/seed{seed}/{domain}/{field}")

    means = {}
    diagnostics = {}
    for model in MODELS:
        means[model], diagnostics[model] = {}, {}
        for domain in DOMAINS:
            means[model][domain] = np.mean([scores[model, seed, domain] for seed in SEEDS], axis=0).tolist()
            pooled = {
                key: sum(counts[model, seed, domain][key] for seed in SEEDS)
                for key in (
                    "total_read_actions", "correct_read_actions", "total_write_actions", "correct_write_actions",
                    "db_match_count", "db_mismatch_count", "db_not_checked", "infra_error_count",
                    "termination_max_steps", "termination_error", "total_simulations",
                )
            }
            pooled["action_accuracy"] = (pooled["correct_read_actions"] + pooled["correct_write_actions"]) / (
                pooled["total_read_actions"] + pooled["total_write_actions"]
            )
            pooled["db_accuracy"] = pooled["db_match_count"] / (pooled["db_match_count"] + pooled["db_mismatch_count"])
            diagnostics[model][domain] = pooled
        for label, domains in (("targets", ("airline", "retail")), ("overall", tuple(DOMAINS))):
            means[model][label] = np.average(
                [means[model][d] for d in domains], axis=0, weights=[DOMAINS[d] for d in domains]
            ).tolist()

    comparisons = {}
    for candidate, reference in (
        ("lr2e6", "lr5e6"), ("lr2e6", "sft"), ("lr2e6", "old2e6"),
        ("lr2e6", "airline_teacher"), ("lr2e6", "retail_teacher"),
    ):
        draws, deltas, per_seed = {}, {}, {}
        for index, domain in enumerate(DOMAINS):
            tasks = sorted(task_metrics[candidate, SEEDS[0], domain])
            delta = np.array([
                np.mean([task_metrics[candidate, seed, domain][task]
                         - task_metrics[reference, seed, domain][task] for seed in SEEDS], axis=0)
                for task in tasks
            ]) * 100
            rng = np.random.default_rng(20260922 + index)
            draws[domain] = delta[rng.integers(0, len(tasks), (20000, len(tasks)))].mean(axis=1)
            deltas[domain] = delta.mean(axis=0)
            per_seed[domain] = {
                seed: (scores[candidate, seed, domain] - scores[reference, seed, domain]) * 100 for seed in SEEDS
            }
        for label, domains in (("targets", ("airline", "retail")), ("overall", tuple(DOMAINS))):
            weights = np.array([DOMAINS[d] for d in domains], dtype=float)
            weights /= weights.sum()
            draws[label] = sum(draws[d] * w for d, w in zip(domains, weights))
            deltas[label] = sum(deltas[d] * w for d, w in zip(domains, weights))
            per_seed[label] = {
                seed: sum(per_seed[d][seed] * w for d, w in zip(domains, weights)) for seed in SEEDS
            }
        for domain in deltas:
            comparisons[f"{candidate}-{reference}/{domain}"] = {
                "delta_pp": deltas[domain].tolist(),
                "task_paired_ci95_pp": np.quantile(draws[domain], [0.025, 0.975], axis=0).T.tolist(),
                "per_seed_delta_pp": {seed: value.tolist() for seed, value in per_seed[domain].items()},
            }

    training = {}
    for arm in ("lr2e6", "lr5e6"):
        root = EXP / f"pilot-{arm}-seed1235-43"
        log = root.joinpath("run.log").read_text(errors="replace").replace("\x00", "")
        log = re.sub(r"\x1b\[[0-9;]*m", "", log)
        rows = defaultdict(dict)
        for match in re.finditer(r"\b(step|perf|rollout|opd) (\d+): (\{[^\n]+\})", log):
            rows[int(match.group(2))].update(ast.literal_eval(match.group(3)))
        updates = [index for index, row in rows.items() if "train/loss" in row]
        finite = all(math.isfinite(row[key]) for row in rows.values() for key in ("train/loss", "train/grad_norm"))
        stats = {}
        for key in (
            "rollout/policy_lag_mean", "rollout/policy_lag_max", "rollout/buffered_groups",
            "rollout/raw_reward", "rollout/truncated_ratio", "rollout/repetition_frac",
            "train/grad_norm", "train/kl_loss", "train/opd_reverse_kl", "train/tis_clipfrac",
            "opd/post_update_logprob_abs_diff", "opd/post_update_tis_weighted_k2",
            "perf/step_time", "perf/train_wait_time",
        ):
            values = [row[key] for _, row in sorted(rows.items()) if key in row]
            stats[key] = {
                "count": len(values), "mean": float(np.mean(values)), "max": float(np.max(values)),
                "first10_mean": float(np.mean(values[:10])), "last10_mean": float(np.mean(values[-10:])),
            }
        training[arm] = {
            "updates": updates, "loss_and_grad_finite": finite,
            "checkpoint_iteration": int(root.joinpath("checkpoints/latest_checkpointed_iteration.txt").read_text()),
            "hf_index_exists": root.joinpath("checkpoints/iter_0000059_hf/model.safetensors.index.json").exists(),
            "consumed_domains": {d: sum(row[f"rollout/trained_groups_{d}"] for row in rows.values())
                                 for d in ("airline", "retail")},
            "domain_raw_successes": {
                d: sum(row[f"rollout/trained_groups_{d}"] * row.get(f"rollout/opd/{d}/raw_reward", 0)
                       for row in rows.values()) for d in ("airline", "retail")
            },
            "stats": stats,
        }

    output = {
        "new_simulations": sum(diagnostics[m][d]["total_simulations"] for m in ("lr2e6", "lr5e6") for d in DOMAINS),
        "all_simulations_including_controls": total_simulations,
        "protocol_differences": protocol_differences,
        "seeds": list(SEEDS), "metric_order": list(METRICS), "means": means,
        "per_seed": {f"{m}/seed{s}/{d}": value.tolist() for (m, s, d), value in scores.items()},
        "diagnostics": diagnostics, "training": training, "comparisons": comparisons,
        "ci_method": "20,000 paired task-cluster bootstrap draws, retaining both seeds and four trials per task; pooled metrics are stratified by domain with task-count weights. No training-seed uncertainty or multiple-comparison adjustment.",
    }
    EXP.joinpath("comparison.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({
        "new_simulations": output["new_simulations"], "total_simulations": total_simulations,
        "protocol_differences": protocol_differences,
        "new_infra_errors": sum(diagnostics[m][d]["infra_error_count"] for m in ("lr2e6", "lr5e6") for d in DOMAINS),
        "means": means,
    }, indent=2))


if __name__ == "__main__":
    main()
