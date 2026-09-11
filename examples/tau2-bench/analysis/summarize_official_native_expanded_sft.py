#!/usr/bin/env python3
"""Validate and summarize the official-native expanded SFT checkpoint curve."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


CHECKPOINTS = (
    ("iter_0000399", 400),
    ("iter_0000799", 800),
    ("iter_0001199", 1200),
    ("iter_0001599", 1600),
    ("iter_0001897", 1898),
    ("iter_0001999", 2000),
    ("iter_0002399", 2400),
    ("iter_0002799", 2800),
    ("iter_0003199", 3200),
    ("iter_0003599", 3600),
    ("iter_0003795", 3796),
)
EXPECTED_TASKS = {"airline": 20, "retail": 40, "telecom": 40}
FINAL_CHECKPOINT = CHECKPOINTS[-1][0]


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _task_metrics(task_trials: dict[str, dict[int, float]]) -> dict[str, float]:
    rows = []
    for task_id, trials in sorted(task_trials.items()):
        if set(trials) != {0, 1, 2, 3}:
            raise ValueError(f"task {task_id} does not contain trials 0,1,2,3")
        rewards = [trials[index] for index in range(4)]
        if any(reward not in (0.0, 1.0) for reward in rewards):
            raise ValueError(f"task {task_id} has a non-binary reward")
        rows.append(
            {
                "pass_at_1": sum(rewards) / 4,
                "pass_at_4_any": float(any(rewards)),
                "pass_power_4": float(all(rewards)),
            }
        )
    return {
        key: sum(row[key] for row in rows) / len(rows)
        for key in ("pass_at_1", "pass_at_4_any", "pass_power_4")
    }


def _assert_close(actual: Any, expected: float, description: str) -> None:
    if not isinstance(actual, (int, float)) or not math.isclose(
        float(actual), expected, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError(f"{description}: {actual!r} != {expected!r}")


def _load_runtimes() -> dict[str, Any]:
    sft_dir = Path(__file__).resolve().parents[1] / "sft"
    if str(sft_dir) not in sys.path:
        sys.path.insert(0, str(sft_dir))
    from build_agent_official_native_expanded import load_domain_runtimes

    return load_domain_runtimes()


def _tool_diagnostics(simulations: list[dict[str, Any]], runtime: Any) -> dict[str, int]:
    nonexistent = invalid_arguments = result_errors = hallucination_retries = 0
    for simulation in simulations:
        hallucination_retries += int(simulation.get("hallucination_retries_used") or 0)
        for message in simulation.get("messages") or []:
            if message.get("role") == "tool" and message.get("requestor") == "assistant":
                result_errors += int(bool(message.get("error")))
            if message.get("role") != "assistant":
                continue
            for call in message.get("tool_calls") or []:
                name = call.get("name") if isinstance(call, dict) else None
                tool = runtime.agent_tools_by_name.get(name)
                if tool is None:
                    nonexistent += 1
                    continue
                arguments = call.get("arguments")
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except json.JSONDecodeError:
                        arguments = None
                try:
                    if not isinstance(arguments, dict):
                        raise ValueError("arguments are not an object")
                    if set(arguments) - set(tool.params.model_fields):
                        raise ValueError("unknown argument")
                    tool.params.model_validate_json(
                        json.dumps(arguments), strict=True
                    )
                except (TypeError, ValueError):
                    invalid_arguments += 1
    return {
        "nonexistent_tool_calls": nonexistent,
        "invalid_argument_calls": invalid_arguments,
        "tool_result_errors": result_errors,
        "hallucination_retries": hallucination_retries,
    }


def _validate_protocol(
    *,
    checkpoint: str,
    domain: str,
    result: dict[str, Any],
    expected_seed: int,
    expected_agent_llm: str | None = None,
    expected_agent_max_tokens: int = 1200,
) -> None:
    info = result.get("info") or {}
    agent = info.get("agent_info") or {}
    user = info.get("user_info") or {}
    agent_args = agent.get("llm_args") or {}
    user_args = user.get("llm_args") or {}
    if expected_agent_llm is None:
        expected_agent_llm = (
            "openai/Qwen3-4B-Instruct-2507-sft-official-native-expanded-"
            f"{checkpoint}"
        )
    checks = {
        "num_trials": (info.get("num_trials"), 4),
        "max_steps": (info.get("max_steps"), 200),
        "seed": (info.get("seed"), expected_seed),
        "agent implementation": (agent.get("implementation"), "llm_agent"),
        "agent model": (agent.get("llm"), expected_agent_llm),
        "agent temperature": (agent_args.get("temperature"), 0.6),
        "agent top_p": (agent_args.get("top_p"), 1.0),
        "agent max_tokens": (agent_args.get("max_tokens"), expected_agent_max_tokens),
        "user implementation": (
            user.get("implementation"),
            "slime_timed_user_simulator",
        ),
        "user model": (
            user.get("llm"),
            "openai/Qwen3.6-27B-tau2-user-nonthinking",
        ),
        "user temperature": (user_args.get("temperature"), 0.0),
        "user max_tokens": (user_args.get("max_tokens"), 512),
    }
    for name, (actual, expected) in checks.items():
        if actual != expected:
            raise ValueError(
                f"{checkpoint}/{domain}: {name} {actual!r} != {expected!r}"
            )


def load_checkpoint(
    checkpoint: str,
    summary_path: Path,
    *,
    results_root: Path,
    expected_seed: int,
    runtimes: dict[str, Any],
    expected_agent_llm: str | None = None,
    expected_agent_max_tokens: int = 1200,
    allow_extra_domains: bool = False,
) -> dict[str, Any]:
    summary = _read_json(summary_path)
    header_checks = {
        "task_split_name": "test",
        "num_trials": 4,
        "num_tasks": None,
        "seed": expected_seed,
        "max_steps": 200,
        "agent_eval_mode": "official-native",
        "agent": "llm_agent",
    }
    for key, expected in header_checks.items():
        if summary.get(key) != expected:
            raise ValueError(
                f"{checkpoint}: summary {key}={summary.get(key)!r}, expected {expected!r}"
            )
    domains = summary.get("domains") or {}
    if allow_extra_domains:
        domains_match = set(EXPECTED_TASKS).issubset(domains)
    else:
        domains_match = set(domains) == set(EXPECTED_TASKS)
    if not domains_match:
        raise ValueError(f"{checkpoint}: domains are {sorted(domains)}")

    by_domain = {}
    all_task_trials: dict[str, dict[int, float]] = {}
    total_tools: Counter[str] = Counter()
    all_terminations: Counter[str] = Counter()
    total_actions = 0
    correct_actions = 0
    db_matches = 0
    db_mismatches = 0
    for domain, expected_tasks in EXPECTED_TASKS.items():
        domain_summary = domains[domain]
        relative_path = Path(str(domain_summary.get("results_file") or ""))
        result_path = relative_path if relative_path.is_absolute() else results_root / relative_path
        result = _read_json(result_path)
        _validate_protocol(
            checkpoint=checkpoint,
            domain=domain,
            result=result,
            expected_seed=expected_seed,
            expected_agent_llm=expected_agent_llm,
            expected_agent_max_tokens=expected_agent_max_tokens,
        )
        simulations = result.get("simulations") or []
        if len(simulations) != expected_tasks * 4:
            raise ValueError(
                f"{checkpoint}/{domain}: expected {expected_tasks * 4} simulations, "
                f"found {len(simulations)}"
            )
        task_trials: dict[str, dict[int, float]] = defaultdict(dict)
        terminations: Counter[str] = Counter()
        for simulation in simulations:
            task_id = str(simulation.get("task_id") or "")
            trial = simulation.get("trial")
            reward = (simulation.get("reward_info") or {}).get("reward")
            if not task_id or not isinstance(trial, int) or trial in task_trials[task_id]:
                raise ValueError(f"{checkpoint}/{domain}: duplicate or malformed task trial")
            if not isinstance(reward, (int, float)) or not math.isfinite(float(reward)):
                raise ValueError(f"{checkpoint}/{domain}/{task_id}: malformed reward")
            task_trials[task_id][trial] = float(reward)
            terminations[str(simulation.get("termination_reason") or "unknown")] += 1
        if len(task_trials) != expected_tasks:
            raise ValueError(
                f"{checkpoint}/{domain}: expected {expected_tasks} tasks, found {len(task_trials)}"
            )
        if terminations.get("infrastructure_error", 0):
            raise ValueError(f"{checkpoint}/{domain}: infrastructure error trajectory")

        computed = _task_metrics(task_trials)
        recorded = domain_summary.get("pass_metrics") or {}
        for metric, value in computed.items():
            _assert_close(recorded.get(metric), value, f"{checkpoint}/{domain}/{metric}")
        if recorded.get("tasks") != expected_tasks or recorded.get("simulations") != expected_tasks * 4:
            raise ValueError(f"{checkpoint}/{domain}: summary coverage mismatch")
        metrics = domain_summary.get("metrics") or {}
        if metrics.get("infra_error_count") != 0:
            raise ValueError(f"{checkpoint}/{domain}: infra_error_count is not zero")
        total_actions += int(metrics.get("total_read_actions") or 0) + int(
            metrics.get("total_write_actions") or 0
        )
        correct_actions += int(metrics.get("correct_read_actions") or 0) + int(
            metrics.get("correct_write_actions") or 0
        )
        db_matches += int(metrics.get("db_match_count") or 0)
        db_mismatches += int(metrics.get("db_mismatch_count") or 0)

        tool_diagnostics = _tool_diagnostics(simulations, runtimes[domain])
        total_tools.update(tool_diagnostics)
        all_terminations.update(terminations)
        by_domain[domain] = {
            **computed,
            "action_accuracy": (domain_summary.get("diagnostics") or {}).get(
                "action_accuracy"
            ),
            "db_accuracy": (domain_summary.get("diagnostics") or {}).get(
                "db_accuracy"
            ),
            "terminations": dict(sorted(terminations.items())),
            "tool_diagnostics": tool_diagnostics,
            "wall_seconds": (domain_summary.get("timing") or {}).get("wall_seconds"),
        }
        all_task_trials.update(
            {f"{domain}/{task_id}": trials for task_id, trials in task_trials.items()}
        )

    overall_metrics = _task_metrics(all_task_trials)
    recorded_overall = summary.get("overall") or {}
    if allow_extra_domains:
        action_accuracy = correct_actions / total_actions if total_actions else None
        db_total = db_matches + db_mismatches
        db_accuracy = db_matches / db_total if db_total else None
        wall_seconds = max(
            row["wall_seconds"] for row in by_domain.values() if row["wall_seconds"] is not None
        )
    else:
        for metric, value in overall_metrics.items():
            _assert_close(recorded_overall.get(metric), value, f"{checkpoint}/overall/{metric}")
        if recorded_overall.get("tasks") != 100 or recorded_overall.get("simulations") != 400:
            raise ValueError(f"{checkpoint}: overall coverage mismatch")
        diagnostics = recorded_overall.get("diagnostics") or {}
        if diagnostics.get("infrastructure_errors") != 0:
            raise ValueError(f"{checkpoint}: overall infrastructure errors are not zero")
        action_accuracy = diagnostics.get("action_accuracy")
        db_accuracy = diagnostics.get("db_accuracy")
        wall_seconds = (recorded_overall.get("timing") or {}).get("wall_seconds")
    return {
        "checkpoint": checkpoint,
        "summary_path": str(summary_path.resolve()),
        "overall": {
            **overall_metrics,
            "action_accuracy": action_accuracy,
            "db_accuracy": db_accuracy,
            "terminations": dict(sorted(all_terminations.items())),
            "tool_diagnostics": dict(total_tools),
            "wall_seconds": wall_seconds,
        },
        "domains": by_domain,
    }


def select_best(models: dict[str, dict[str, Any]]) -> str:
    update_by_checkpoint = dict(CHECKPOINTS)
    return max(
        models,
        key=lambda checkpoint: (
            models[checkpoint]["overall"]["pass_at_1"],
            models[checkpoint]["overall"]["pass_power_4"],
            -update_by_checkpoint[checkpoint],
        ),
    )


def _parse_specs(values: list[str], *, option: str) -> dict[str, Path]:
    parsed = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"{option} must be CHECKPOINT=SUMMARY.json: {value}")
        checkpoint, path = value.split("=", 1)
        if checkpoint in parsed:
            raise ValueError(f"duplicate {option} checkpoint: {checkpoint}")
        parsed[checkpoint] = Path(path)
    return parsed


def build_report(
    *,
    seed300_specs: dict[str, Path],
    confirmation_specs: dict[str, Path],
    results_root: Path,
    baseline_summary: Path | None,
) -> dict[str, Any]:
    expected = [checkpoint for checkpoint, _ in CHECKPOINTS]
    if set(seed300_specs) != set(expected):
        raise ValueError(f"seed-300 summaries must contain exactly: {expected}")
    runtimes = _load_runtimes()
    seed300 = {
        checkpoint: load_checkpoint(
            checkpoint,
            seed300_specs[checkpoint],
            results_root=results_root,
            expected_seed=300,
            runtimes=runtimes,
        )
        for checkpoint in expected
    }
    best = select_best(seed300)
    required_confirmations = {best, FINAL_CHECKPOINT}
    if confirmation_specs and set(confirmation_specs) != required_confirmations:
        raise ValueError(
            "seed-301 confirmations must contain exactly the selected and final "
            f"checkpoints: {sorted(required_confirmations)}"
        )
    confirmations = {
        checkpoint: load_checkpoint(
            checkpoint,
            confirmation_specs[checkpoint],
            results_root=results_root,
            expected_seed=301,
            runtimes=runtimes,
        )
        for checkpoint in sorted(confirmation_specs, key=dict(CHECKPOINTS).__getitem__)
    }
    baseline = None
    if baseline_summary is not None:
        raw = _read_json(baseline_summary)
        baseline = {
            "job_id": 11878,
            "summary_path": str(baseline_summary.resolve()),
            "overall": raw.get("overall"),
            "strictly_comparable": raw.get("agent_eval_mode") == "official-native",
            "note": (
                "Job 11878 predates official-native llm_agent evaluation and is "
                "retained only as a historical raw-Instruct reference."
            ),
        }
    return {
        "selection_policy": "seed300 pass@1, then pass^4, then earlier checkpoint",
        "selected_checkpoint": best,
        "final_checkpoint": FINAL_CHECKPOINT,
        "seed300": seed300,
        "seed301": confirmations,
        "raw_instruct_reference": baseline,
    }


def _percent(value: Any) -> str:
    return "—" if value is None else f"{100 * float(value):.2f}%"


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# official-native expanded SFT checkpoint evaluation",
        "",
        "## Selection",
        "",
        f"Selected checkpoint: `{report['selected_checkpoint']}`. Policy: {report['selection_policy']}.",
        "",
        "## Seed 300 overall",
        "",
        "| Checkpoint | Updates | pass@1 | pass@4(any) | pass^4 | action acc. | DB acc. | wall |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for checkpoint, updates in CHECKPOINTS:
        overall = report["seed300"][checkpoint]["overall"]
        wall = overall.get("wall_seconds")
        lines.append(
            f"| `{checkpoint}` | {updates} | {_percent(overall['pass_at_1'])} | "
            f"{_percent(overall['pass_at_4_any'])} | {_percent(overall['pass_power_4'])} | "
            f"{_percent(overall.get('action_accuracy'))} | {_percent(overall.get('db_accuracy'))} | "
            f"{float(wall) / 3600:.2f} h |"
        )
    lines.extend(
        [
            "",
            "## Seed 300 domains",
            "",
            "| Checkpoint | Domain | pass@1 | pass@4(any) | pass^4 | action acc. | DB acc. |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for checkpoint, _ in CHECKPOINTS:
        for domain in EXPECTED_TASKS:
            row = report["seed300"][checkpoint]["domains"][domain]
            lines.append(
                f"| `{checkpoint}` | {domain} | {_percent(row['pass_at_1'])} | "
                f"{_percent(row['pass_at_4_any'])} | {_percent(row['pass_power_4'])} | "
                f"{_percent(row.get('action_accuracy'))} | {_percent(row.get('db_accuracy'))} |"
            )
    lines.extend(
        [
            "",
            "## Behavior diagnostics",
            "",
            "| Checkpoint | Terminations | nonexistent | invalid args | tool errors | parser retries |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for checkpoint, _ in CHECKPOINTS:
        overall = report["seed300"][checkpoint]["overall"]
        tools = overall["tool_diagnostics"]
        terminations = ", ".join(
            f"{name}:{count}" for name, count in overall["terminations"].items()
        )
        lines.append(
            f"| `{checkpoint}` | {terminations} | {tools['nonexistent_tool_calls']} | "
            f"{tools['invalid_argument_calls']} | {tools['tool_result_errors']} | "
            f"{tools['hallucination_retries']} |"
        )
    if report["seed301"]:
        lines.extend(
            [
                "",
                "## Seed 301 confirmation",
                "",
                "| Checkpoint | pass@1 | pass@4(any) | pass^4 | action acc. | DB acc. |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for checkpoint, model in report["seed301"].items():
            row = model["overall"]
            lines.append(
                f"| `{checkpoint}` | {_percent(row['pass_at_1'])} | "
                f"{_percent(row['pass_at_4_any'])} | {_percent(row['pass_power_4'])} | "
                f"{_percent(row.get('action_accuracy'))} | {_percent(row.get('db_accuracy'))} |"
            )
    if report["raw_instruct_reference"] is not None:
        baseline = report["raw_instruct_reference"]
        row = baseline["overall"] or {}
        lines.extend(
            [
                "",
                "## Raw Instruct reference",
                "",
                f"Job `11878`: pass@1 {_percent(row.get('pass_at_1'))}, "
                f"pass@4(any) {_percent(row.get('pass_at_4_any'))}, "
                f"pass^4 {_percent(row.get('pass_power_4'))}. {baseline['note']}",
            ]
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", action="append", default=[], required=True)
    parser.add_argument("--confirmation", action="append", default=[])
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--baseline-summary", type=Path)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = build_report(
            seed300_specs=_parse_specs(args.summary, option="--summary"),
            confirmation_specs=_parse_specs(
                args.confirmation, option="--confirmation"
            ),
            results_root=args.results_root.resolve(),
            baseline_summary=args.baseline_summary,
        )
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    args.markdown_output.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
