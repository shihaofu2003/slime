#!/usr/bin/env python3
"""Compare identical-protocol tau2 official evals with paired task CIs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

SHARED_DIR = Path(__file__).resolve().parents[1] / "shared"
if str(SHARED_DIR) not in sys.path:
    sys.path.insert(0, str(SHARED_DIR))

from protocol_profiles import (  # noqa: E402
    PROTOCOL_AGENT_OWNED_DEPENDENCY_SAFE_MULTI,
    protocol_signature,
)


METRIC_LABELS = {
    "pass_at_1": "pass@1 / pass^1",
    "pass_at_4_any": "pass@4(any)",
    "pass_power_4": "pass^4",
}
EXPECTED_TEST_TASKS = {"airline": 20, "retail": 40, "telecom": 40}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_finite(value: Any) -> Any:
    """Replace non-finite diagnostic numbers with JSON null recursively."""

    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: json_finite(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_finite(item) for item in value]
    return value


def task_metrics(rewards: list[float]) -> dict[str, float]:
    if not rewards:
        raise ValueError("a task must contain at least one trial")
    if any(reward not in (0.0, 1.0) for reward in rewards):
        raise ValueError(f"tau2 task rewards must be binary: {rewards}")
    return {
        "pass_at_1": sum(rewards) / len(rewards),
        "pass_at_4_any": float(any(rewards)),
        "pass_power_4": float(all(rewards)),
    }


def aggregate_metrics(
    task_rewards: dict[tuple[str, str], list[float]],
    *,
    domain: str | None = None,
) -> dict[str, float]:
    selected = [
        task_metrics(rewards)
        for (task_domain, _), rewards in sorted(task_rewards.items())
        if domain is None or task_domain == domain
    ]
    if not selected:
        raise ValueError(f"no tasks available for domain={domain!r}")
    return {
        metric: sum(row[metric] for row in selected) / len(selected)
        for metric in METRIC_LABELS
    }


def trajectory_has_multi_tool_turn(messages: list[dict[str, Any]]) -> bool:
    return any(
        message.get("role") == "assistant"
        and isinstance(message.get("tool_calls"), list)
        and len(message["tool_calls"]) > 1
        for message in messages
    )


def aggregate_behavior(
    simulations: list[dict[str, Any]],
    *,
    domain: str | None = None,
) -> dict[str, Any]:
    selected = [
        simulation
        for simulation in simulations
        if domain is None or simulation["domain"] == domain
    ]
    if not selected:
        raise ValueError(f"no simulations available for domain={domain!r}")
    by_mode = {}
    for mode in ("multi_exposed", "non_multi_exposed"):
        rows = [simulation for simulation in selected if simulation["call_mode"] == mode]
        successes = sum(simulation["reward"] for simulation in rows)
        by_mode[mode] = {
            "simulations": len(rows),
            "successes": successes,
            "success_rate": successes / len(rows) if rows else None,
            "too_many_errors": sum(
                simulation["termination_reason"] == "too_many_errors"
                for simulation in rows
            ),
            "max_steps": sum(
                simulation["termination_reason"] == "max_steps"
                for simulation in rows
            ),
        }
    return {
        "by_call_mode": by_mode,
        "termination_reasons": dict(
            sorted(Counter(row["termination_reason"] for row in selected).items())
        ),
    }


def paired_bootstrap_ci(
    deltas: list[float],
    *,
    samples: int,
    seed: int,
) -> tuple[float, float, float]:
    if not deltas:
        raise ValueError("paired bootstrap requires at least one task")
    if samples < 1:
        raise ValueError("bootstrap sample count must be positive")
    import numpy as np

    values = np.asarray(deltas, dtype=np.float64)
    observed = float(values.mean())
    rng = np.random.default_rng(seed)
    draws = np.empty(samples, dtype=np.float64)
    batch_size = 10_000
    for start in range(0, samples, batch_size):
        stop = min(start + batch_size, samples)
        indices = rng.integers(0, len(values), size=(stop - start, len(values)))
        draws[start:stop] = values[indices].mean(axis=1)
    low, high = np.quantile(draws, [0.025, 0.975], method="linear")
    return observed, float(low), float(high)


def _selected_args(info: dict[str, Any], side: str) -> dict[str, Any]:
    section = info.get(f"{side}_info") or {}
    llm_args = section.get("llm_args") or {}
    keys = (
        "temperature",
        "top_p",
        "max_tokens",
        "protocol_profile",
        "protocol_signature",
    )
    return {
        "implementation": section.get("implementation"),
        "llm": section.get("llm") if side == "user" else None,
        "generation": {key: llm_args.get(key) for key in keys},
    }


def _protocol_signature(
    summary: dict[str, Any],
    result_info: dict[str, Any],
) -> dict[str, Any]:
    return {
        "task_split_name": summary.get("task_split_name"),
        "num_trials": summary.get("num_trials"),
        "agent_protocol_profile": summary.get("agent_protocol_profile"),
        "seed": result_info.get("seed"),
        "max_steps": result_info.get("max_steps"),
        "max_errors": result_info.get("max_errors"),
        "agent": _selected_args(result_info, "agent"),
        "user": _selected_args(result_info, "user"),
    }


def _without_agent_protocol(signature: dict[str, Any]) -> dict[str, Any]:
    """Keep every eval setting except the Agent prompt-protocol fields."""

    normalized = json.loads(json.dumps(signature))
    normalized["agent_protocol_profile"] = None
    generation = normalized["agent"]["generation"]
    generation["protocol_profile"] = None
    generation["protocol_signature"] = None
    return normalized


def load_model_eval(
    *,
    label: str,
    summary_path: Path,
    results_root: Path,
    expected_profile: str,
    allow_legacy_unsigned_protocol: bool = False,
) -> dict[str, Any]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary_profile = summary.get("agent_protocol_profile")
    expected_agent_signature = protocol_signature(expected_profile)
    summary_agent_signature = summary.get("agent_protocol_signature")
    if not allow_legacy_unsigned_protocol and summary_profile != expected_profile:
        raise ValueError(
            f"{label}: protocol profile {summary_profile!r} "
            f"does not match {expected_profile!r}"
        )
    if allow_legacy_unsigned_protocol and summary_profile not in (
        None,
        expected_profile,
    ):
        raise ValueError(f"{label}: unsupported legacy protocol profile {summary_profile!r}")
    num_trials = summary.get("num_trials")
    if num_trials != 4:
        raise ValueError(f"{label}: expected exactly four trials, found {num_trials}")
    if (
        not allow_legacy_unsigned_protocol
        and summary_agent_signature != expected_agent_signature
    ):
        raise ValueError(
            f"{label}: agent protocol content signature "
            f"{summary_agent_signature!r} does not match "
            f"{expected_agent_signature!r}"
        )
    if allow_legacy_unsigned_protocol and summary_agent_signature not in (
        None,
        expected_agent_signature,
    ):
        raise ValueError(
            f"{label}: unsupported legacy protocol content signature "
            f"{summary_agent_signature!r}"
        )
    domains = summary.get("domains")
    if not isinstance(domains, dict) or not domains:
        raise ValueError(f"{label}: summary has no domains")

    task_trials: dict[tuple[str, str], dict[int, float]] = defaultdict(dict)
    result_artifacts = {}
    simulation_rows = []
    signatures = []
    contract_signatures: dict[str, str] = {}
    official_metrics_by_domain: dict[str, dict[str, Any]] = {}
    for domain, domain_summary in sorted(domains.items()):
        metrics = domain_summary.get("metrics") or {}
        official_metrics_by_domain[domain] = json_finite(metrics)
        if metrics.get("infra_error_count") != 0:
            raise ValueError(
                f"{label}/{domain}: infra_error_count={metrics.get('infra_error_count')}"
            )
        relative_result = Path(str(domain_summary["results_file"]))
        result_path = relative_result if relative_result.is_absolute() else results_root / relative_result
        result_data = json.loads(result_path.read_text(encoding="utf-8"))
        result_info = result_data.get("info") or {}
        signature = _protocol_signature(summary, result_info)
        if expected_profile == PROTOCOL_AGENT_OWNED_DEPENDENCY_SAFE_MULTI:
            contract_metadata = domain_summary.get("agent_contract") or {}
            contract_signature = contract_metadata.get("agent_contract_signature")
            result_contract_signature = (
                (result_info.get("agent_info") or {}).get("llm_args") or {}
            ).get("agent_contract_signature")
            if (
                not isinstance(contract_signature, str)
                or not contract_signature.startswith("sha256:")
                or result_contract_signature != contract_signature
            ):
                raise ValueError(
                    f"{label}/{domain}: missing or mismatched Agent contract signature"
                )
            contract_signatures[domain] = contract_signature
        if result_info.get("num_trials") != num_trials:
            raise ValueError(f"{label}/{domain}: result num_trials mismatch")
        result_profile = signature["agent"]["generation"].get("protocol_profile")
        result_agent_signature = signature["agent"]["generation"].get(
            "protocol_signature"
        )
        if not allow_legacy_unsigned_protocol and result_profile != expected_profile:
            raise ValueError(f"{label}/{domain}: result omitted the protocol profile")
        if not allow_legacy_unsigned_protocol and (
            result_agent_signature != expected_agent_signature
        ):
            raise ValueError(
                f"{label}/{domain}: result omitted or mismatched the agent "
                "protocol content signature"
            )
        if allow_legacy_unsigned_protocol:
            if result_profile not in (None, expected_profile):
                raise ValueError(
                    f"{label}/{domain}: unsupported legacy result protocol profile "
                    f"{result_profile!r}"
                )
            if result_agent_signature not in (None, expected_agent_signature):
                raise ValueError(
                    f"{label}/{domain}: unsupported legacy result protocol signature "
                    f"{result_agent_signature!r}"
                )
            if (result_profile, result_agent_signature) != (
                summary_profile,
                summary_agent_signature,
            ):
                raise ValueError(
                    f"{label}/{domain}: summary/result legacy protocol provenance differs"
                )
        signatures.append(signature)
        simulations = result_data.get("simulations") or []
        infrastructure_terminations = 0
        for simulation in simulations:
            task_id = str(simulation.get("task_id") or "")
            trial = simulation.get("trial")
            reward = ((simulation.get("reward_info") or {}).get("reward"))
            if not task_id or not isinstance(reward, (int, float)) or not math.isfinite(reward):
                raise ValueError(f"{label}/{domain}: malformed simulation reward")
            if not isinstance(trial, int) or not 0 <= trial < num_trials:
                raise ValueError(
                    f"{label}/{domain}/{task_id}: invalid trial index {trial!r}"
                )
            task_key = (domain, task_id)
            if trial in task_trials[task_key]:
                raise ValueError(
                    f"{label}/{domain}/{task_id}: duplicate trial index {trial}"
                )
            task_trials[task_key][trial] = float(reward)
            termination_reason = str(
                simulation.get("termination_reason") or "unknown"
            )
            infrastructure_terminations += int(
                termination_reason == "infrastructure_error"
            )
            simulation_rows.append(
                {
                    "domain": domain,
                    "task_id": task_id,
                    "trial": trial,
                    "reward": float(reward),
                    "call_mode": (
                        "multi_exposed"
                        if trajectory_has_multi_tool_turn(simulation.get("messages") or [])
                        else "non_multi_exposed"
                    ),
                    "termination_reason": termination_reason,
                }
            )
        if infrastructure_terminations:
            raise ValueError(
                f"{label}/{domain}: found {infrastructure_terminations} "
                "infrastructure-error simulations"
            )
        result_artifacts[domain] = {
            "path": str(result_path),
            "sha256": file_sha256(result_path),
            "simulations": len(simulations),
        }

    if len({json.dumps(signature, sort_keys=True) for signature in signatures}) != 1:
        raise ValueError(f"{label}: protocol configuration differs across domains")
    if (
        summary_profile == expected_profile
        and summary_agent_signature == expected_agent_signature
    ):
        protocol_provenance = "signed-current"
    elif summary_profile == expected_profile and summary_agent_signature is None:
        protocol_provenance = "legacy-unsigned-declared"
    elif summary_profile is None and summary_agent_signature is None:
        protocol_provenance = "legacy-implicit-current-single"
    else:
        raise ValueError(f"{label}: unsupported protocol provenance")
    expected_trial_indices = set(range(num_trials))
    wrong_trials = {
        f"{domain}:{task_id}": sorted(trials)
        for (domain, task_id), trials in task_trials.items()
        if set(trials) != expected_trial_indices
    }
    if wrong_trials:
        raise ValueError(f"{label}: incomplete task trials: {wrong_trials}")
    task_rewards = {
        key: [trials[index] for index in range(num_trials)]
        for key, trials in task_trials.items()
    }
    for rewards in task_rewards.values():
        task_metrics(rewards)
    return {
        "label": label,
        "summary_path": str(summary_path),
        "summary_sha256": file_sha256(summary_path),
        "result_artifacts": result_artifacts,
        "signature": signatures[0],
        "protocol_provenance": protocol_provenance,
        "agent_contract_signatures": contract_signatures,
        "official_metrics_by_domain": official_metrics_by_domain,
        "task_rewards": dict(task_rewards),
        "simulation_rows": simulation_rows,
    }


def compare(args: argparse.Namespace) -> dict[str, Any]:
    parsed_models = []
    seen_labels = set()
    for spec in args.model:
        if "=" not in spec:
            raise ValueError(f"--model must be LABEL=SUMMARY.json: {spec}")
        label, raw_path = spec.split("=", 1)
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", label) or label in seen_labels:
            raise ValueError(f"invalid or duplicate model label: {label!r}")
        seen_labels.add(label)
        parsed_models.append((label, Path(raw_path).resolve()))
    if args.candidate not in seen_labels:
        raise ValueError(f"candidate {args.candidate!r} is absent from --model")
    if len(parsed_models) < 2:
        raise ValueError("at least two --model entries are required")
    legacy_labels = set(getattr(args, "legacy_model", []) or [])
    unknown_legacy_labels = legacy_labels - seen_labels
    if unknown_legacy_labels:
        raise ValueError(f"legacy model labels are absent from --model: {unknown_legacy_labels}")
    if args.candidate in legacy_labels:
        raise ValueError("the candidate cannot be marked as a legacy model")

    loaded = {
        label: load_model_eval(
            label=label,
            summary_path=path,
            results_root=args.results_root.resolve(),
            expected_profile=args.protocol_profile,
            allow_legacy_unsigned_protocol=label in legacy_labels,
        )
        for label, path in parsed_models
    }
    contract_signature_maps = {
        label: item.get("agent_contract_signatures", {})
        for label, item in loaded.items()
    }
    if args.protocol_profile == PROTOCOL_AGENT_OWNED_DEPENDENCY_SAFE_MULTI:
        if len(
            {
                json.dumps(value, sort_keys=True)
                for value in contract_signature_maps.values()
            }
        ) != 1:
            raise ValueError(
                f"Agent contract signatures differ across models: {contract_signature_maps}"
            )
    task_sets = {label: set(item["task_rewards"]) for label, item in loaded.items()}
    expected_tasks = task_sets[args.candidate]
    mismatches = {
        label: {
            "missing": sorted(expected_tasks - tasks),
            "unexpected": sorted(tasks - expected_tasks),
        }
        for label, tasks in task_sets.items()
        if tasks != expected_tasks
    }
    if mismatches:
        raise ValueError(f"model task sets differ: {mismatches}")
    domain_task_counts = Counter(domain for domain, _ in expected_tasks)
    if dict(sorted(domain_task_counts.items())) != EXPECTED_TEST_TASKS:
        raise ValueError(
            "official test task coverage mismatch: "
            f"{dict(sorted(domain_task_counts.items()))} != {EXPECTED_TEST_TASKS}"
        )
    signatures = {
        label: item["signature"]
        for label, item in loaded.items()
    }
    rendered_signatures = {
        json.dumps(value, sort_keys=True) for value in signatures.values()
    }
    if not legacy_labels and len(rendered_signatures) != 1:
        raise ValueError(f"evaluation protocol signatures differ: {signatures}")
    protocol_agnostic_signatures = {
        label: _without_agent_protocol(signature)
        for label, signature in signatures.items()
    }
    if len(
        {
            json.dumps(value, sort_keys=True)
            for value in protocol_agnostic_signatures.values()
        }
    ) != 1:
        raise ValueError(
            "non-protocol evaluation settings differ: "
            f"{protocol_agnostic_signatures}"
        )

    domains = sorted({domain for domain, _ in expected_tasks})
    models = {}
    for label, item in loaded.items():
        rewards = item["task_rewards"]
        models[label] = {
            "tasks": len(rewards),
            "simulations": sum(len(values) for values in rewards.values()),
            "overall": aggregate_metrics(rewards),
            "by_domain": {
                domain: aggregate_metrics(rewards, domain=domain)
                for domain in domains
            },
            "behavior": {
                "overall": aggregate_behavior(item["simulation_rows"]),
                "by_domain": {
                    domain: aggregate_behavior(item["simulation_rows"], domain=domain)
                    for domain in domains
                },
                # Preserve every metric emitted by the official evaluator. These
                # diagnostics are deliberately reported but never consumed by a
                # capability or promotion decision.
                "official_metrics_by_domain": item.get(
                    "official_metrics_by_domain", {}
                ),
            },
        }

    candidate_rewards = loaded[args.candidate]["task_rewards"]
    paired = []
    for reference, _ in parsed_models:
        if reference == args.candidate:
            continue
        reference_rewards = loaded[reference]["task_rewards"]
        scopes = [("overall", None)] + [(domain, domain) for domain in domains]
        for scope, selected_domain in scopes:
            selected_tasks = [
                key
                for key in sorted(expected_tasks)
                if selected_domain is None or key[0] == selected_domain
            ]
            for metric in METRIC_LABELS:
                deltas = [
                    task_metrics(candidate_rewards[key])[metric]
                    - task_metrics(reference_rewards[key])[metric]
                    for key in selected_tasks
                ]
                salt_key = (
                    f"{reference}:{args.candidate}:{metric}"
                    if scope == "overall"
                    else f"{reference}:{args.candidate}:{scope}:{metric}"
                )
                salt = int.from_bytes(
                    hashlib.sha256(salt_key.encode()).digest()[:4],
                    "big",
                )
                observed, low, high = paired_bootstrap_ci(
                    deltas,
                    samples=args.bootstrap_samples,
                    seed=args.bootstrap_seed + salt,
                )
                paired.append(
                    {
                        "comparison": f"{args.candidate}-{reference}",
                        "reference": reference,
                        "candidate": args.candidate,
                        "scope": scope,
                        "metric": metric,
                        "tasks": len(deltas),
                        "delta": observed,
                        "ci95": [low, high],
                        "ci_excludes_zero": low > 0 or high < 0,
                    }
                )

    return {
        "status": "pass",
        "comparison_mode": (
            "legacy-baseline-cross-protocol" if legacy_labels else "strict-signed"
        ),
        "protocol_profile": args.protocol_profile,
        "agent_protocol_signature": protocol_signature(args.protocol_profile),
        "agent_contract_signatures": contract_signature_maps[args.candidate],
        "protocol_signature": signatures[args.candidate],
        "legacy_models": {
            label: loaded[label]["protocol_provenance"]
            for label in sorted(legacy_labels)
        },
        "comparability_warning": (
            "Legacy baselines lack the candidate's signed Agent protocol. Point "
            "estimates and paired intervals are descriptive, not a controlled "
            "single-variable comparison."
            if legacy_labels
            else None
        ),
        "candidate": args.candidate,
        "bootstrap": {
            "unit": "domain/task_id",
            "samples": args.bootstrap_samples,
            "seed": args.bootstrap_seed,
            "interval": "percentile_95",
        },
        "artifacts": {
            label: {
                key: value
                for key, value in item.items()
                if key
                not in (
                    "task_rewards",
                    "simulation_rows",
                    "signature",
                    "label",
                    "official_metrics_by_domain",
                )
            }
            for label, item in loaded.items()
        },
        "models": models,
        "paired_deltas": paired,
    }


def _percent(value: float) -> str:
    return f"{100 * value:.2f}%"


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# tau2 official evaluation comparison",
        "",
        "## Protocol",
        "",
        f"Profile: `{report['protocol_profile']}`; paired bootstrap unit: task; "
        f"samples: {report['bootstrap']['samples']:,}.",
    ]
    if report.get("comparability_warning"):
        lines.extend(["", f"Warning: {report['comparability_warning']}"])
    lines.extend(
        [
            "",
            "## Overall results",
            "",
            "| Model | Tasks | pass@1 / pass^1 | pass@4(any) | pass^4 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for label, model in report["models"].items():
        metrics = model["overall"]
        lines.append(
            f"| {label} | {model['tasks']} | {_percent(metrics['pass_at_1'])} | "
            f"{_percent(metrics['pass_at_4_any'])} | {_percent(metrics['pass_power_4'])} |"
        )
    lines.extend(
        [
            "",
            "## Domain results",
            "",
            "| Model | Domain | pass@1 / pass^1 | pass@4(any) | pass^4 |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for label, model in report["models"].items():
        for domain, metrics in model["by_domain"].items():
            lines.append(
                f"| {label} | {domain} | {_percent(metrics['pass_at_1'])} | "
                f"{_percent(metrics['pass_at_4_any'])} | {_percent(metrics['pass_power_4'])} |"
            )
    lines.extend(
        [
            "",
            "## Paired deltas",
            "",
            "Positive values favor the candidate. Intervals resample the 100 matched tasks.",
            "",
            "| Comparison | Scope | Metric | Delta | 95% paired bootstrap CI | Excludes 0 |",
            "|---|---|---|---:|---:|---:|",
        ]
    )
    for row in report["paired_deltas"]:
        lines.append(
            f"| {row['comparison']} | {row.get('scope', 'overall')} | "
            f"{METRIC_LABELS[row['metric']]} | "
            f"{100 * row['delta']:+.2f} pp | "
            f"[{100 * row['ci95'][0]:+.2f}, {100 * row['ci95'][1]:+.2f}] pp | "
            f"{'yes' if row['ci_excludes_zero'] else 'no'} |"
        )
    lines.extend(
        [
            "",
            "## Trajectory tool-mode diagnostic",
            "",
            "A trajectory is `multi_exposed` when at least one assistant turn contains more than "
            "one tool call. These success rates are observational and are not causal effects of "
            "batching.",
            "",
            "| Model | Exposure | Simulations | Success rate | Too many errors | Max steps |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for label, model in report["models"].items():
        for mode, diagnostics in model["behavior"]["overall"]["by_call_mode"].items():
            success_rate = diagnostics["success_rate"]
            rendered_rate = "n/a" if success_rate is None else _percent(success_rate)
            lines.append(
                f"| {label} | {mode} | {diagnostics['simulations']} | {rendered_rate} | "
                f"{diagnostics['too_many_errors']} | {diagnostics['max_steps']} |"
            )
    return "\n".join(lines) + "\n"


def write_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="append", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--protocol-profile", default="dependency-safe-multi")
    parser.add_argument(
        "--legacy-model",
        action="append",
        default=[],
        help=(
            "Explicit model label allowed to reuse an unsigned historical protocol "
            "artifact. The candidate may not be legacy."
        ),
    )
    parser.add_argument("--bootstrap-samples", type=int, default=100_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260803)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = compare(args)
    write_atomic(
        args.json_output,
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
    )
    write_atomic(args.markdown_output, render_markdown(report))
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
