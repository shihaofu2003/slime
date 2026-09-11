#!/usr/bin/env python3
"""Validate raw-all final evals and compare them with processed SFT controls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from summarize_official_native_expanded_sft import _load_runtimes, load_checkpoint


SERVICE_AGENT_ROOT = Path("/mnt/afs/users/fush/projects/ServiceAgent")
PROJECT_ROOT = SERVICE_AGENT_ROOT / "slime"
EXPERIMENT_DIR = PROJECT_ROOT / "output/experiments/tau2-sft-raw-all-max16384"
RESULTS_ROOT = SERVICE_AGENT_ROOT / "tau2-bench"
PROCESSED_DIR = (
    PROJECT_ROOT
    / "output/experiments/tau2-sft-official-native-expanded/eval"
    / "checkpoint-iter_0003795"
)
RAW_INSTRUCT_SUMMARY = (
    PROJECT_ROOT
    / "output/experiments/tau2-eval-official-native-full/eval"
    / "four-domain-full-bm25"
    / "seed300_0902_060330_summary.json"
)
QWEN35_SUMMARY = (
    PROJECT_ROOT
    / "output/experiments/tau2-eval-qwen35-official-native-four-domain/eval"
    / "four-domain-full-bm25-thinking"
    / "seed300_0902_054146_summary.json"
)
DOMAINS = ("airline", "retail", "telecom")
PASS_METRICS = ("pass_at_1", "pass_at_4_any", "pass_power_4")


def _load(
    label: str,
    summary_path: Path,
    *,
    seed: int,
    runtimes,
    results_root: Path,
    agent_llm: str,
    agent_max_tokens: int = 1200,
    allow_extra_domains: bool = False,
) -> dict[str, Any]:
    return load_checkpoint(
        label,
        summary_path,
        results_root=results_root,
        expected_seed=seed,
        runtimes=runtimes,
        expected_agent_llm=agent_llm,
        expected_agent_max_tokens=agent_max_tokens,
        allow_extra_domains=allow_extra_domains,
    )


def _mean_rows(first: dict[str, Any], second: dict[str, Any]) -> dict[str, Any]:
    result = {}
    for key in (*PASS_METRICS, "action_accuracy", "db_accuracy", "wall_seconds"):
        values = [row.get(key) for row in (first, second)]
        result[key] = None if any(value is None for value in values) else sum(values) / 2
    return result


def _delta(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    return {
        key: None
        if left.get(key) is None or right.get(key) is None
        else left[key] - right[key]
        for key in (*PASS_METRICS, "action_accuracy", "db_accuracy")
    }


def build_report(
    *,
    raw_seed300: Path,
    raw_seed301: Path,
    processed_seed300: Path,
    processed_seed301: Path,
    raw_instruct_seed300: Path,
    qwen35_seed300: Path,
    results_root: Path,
) -> dict[str, Any]:
    runtimes = _load_runtimes()
    raw = {
        "300": _load(
            "raw-all-final",
            raw_seed300,
            seed=300,
            runtimes=runtimes,
            results_root=results_root,
            agent_llm="openai/Qwen3-4B-Instruct-2507-sft-raw-all-max16384-final",
        ),
        "301": _load(
            "raw-all-final",
            raw_seed301,
            seed=301,
            runtimes=runtimes,
            results_root=results_root,
            agent_llm="openai/Qwen3-4B-Instruct-2507-sft-raw-all-max16384-final",
        ),
    }
    processed = {
        "300": _load(
            "iter_0003795",
            processed_seed300,
            seed=300,
            runtimes=runtimes,
            results_root=results_root,
            agent_llm="openai/Qwen3-4B-Instruct-2507-sft-official-native-expanded-iter_0003795",
        ),
        "301": _load(
            "iter_0003795",
            processed_seed301,
            seed=301,
            runtimes=runtimes,
            results_root=results_root,
            agent_llm="openai/Qwen3-4B-Instruct-2507-sft-official-native-expanded-iter_0003795",
        ),
    }
    raw_instruct = _load(
        "raw-instruct",
        raw_instruct_seed300,
        seed=300,
        runtimes=runtimes,
        results_root=results_root,
        agent_llm="openai/tau2-agent",
        allow_extra_domains=True,
    )
    qwen35 = _load(
        "qwen3.5-4b-thinking",
        qwen35_seed300,
        seed=300,
        runtimes=runtimes,
        results_root=results_root,
        agent_llm="openai/tau2-agent",
        agent_max_tokens=8192,
        allow_extra_domains=True,
    )

    raw_mean = {
        "overall": _mean_rows(raw["300"]["overall"], raw["301"]["overall"]),
        "domains": {
            domain: _mean_rows(raw["300"]["domains"][domain], raw["301"]["domains"][domain])
            for domain in DOMAINS
        },
    }
    processed_mean = {
        "overall": _mean_rows(
            processed["300"]["overall"], processed["301"]["overall"]
        ),
        "domains": {
            domain: _mean_rows(
                processed["300"]["domains"][domain],
                processed["301"]["domains"][domain],
            )
            for domain in DOMAINS
        },
    }
    deltas = {
        seed: {
            "overall": _delta(processed[seed]["overall"], raw[seed]["overall"]),
            "domains": {
                domain: _delta(
                    processed[seed]["domains"][domain], raw[seed]["domains"][domain]
                )
                for domain in DOMAINS
            },
        }
        for seed in ("300", "301")
    }
    deltas["mean"] = {
        "overall": _delta(processed_mean["overall"], raw_mean["overall"]),
        "domains": {
            domain: _delta(processed_mean["domains"][domain], raw_mean["domains"][domain])
            for domain in DOMAINS
        },
    }
    return {
        "protocol": {
            "domains": {"airline": 20, "retail": 40, "telecom": 40},
            "trials_per_task": 4,
            "seeds": [300, 301],
            "agent_temperature": 0.6,
            "agent_top_p": 1.0,
            "agent_max_tokens": 1200,
            "max_steps": 200,
            "user": "Qwen3.6-27B non-thinking",
        },
        "raw_all": raw,
        "processed": processed,
        "two_seed_mean": {"raw_all": raw_mean, "processed": processed_mean},
        "processed_minus_raw_all": deltas,
        "seed300_four_way": {
            "raw_instruct": raw_instruct,
            "raw_all_sft": raw["300"],
            "processed_sft": processed["300"],
            "qwen3.5_4b_thinking": qwen35,
        },
        "baseline_note": (
            "Raw Instruct and Qwen3.5-4B are three-domain subsets of existing "
            "four-domain official-native seed-300 runs; Qwen3.5 thinking used "
            "an 8,192-token output cap."
        ),
    }


def _pct(value: Any) -> str:
    return "—" if value is None else f"{100 * float(value):.2f}%"


def _pp(value: Any) -> str:
    return "—" if value is None else f"{100 * float(value):+.2f}pp"


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Raw-all SFT control report",
        "",
        "## Seed-300 four-way comparison",
        "",
        "| Agent | pass@1 | pass@4(any) | pass^4 | action acc. | DB acc. |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    labels = {
        "raw_instruct": "Raw Qwen3-4B-Instruct-2507",
        "raw_all_sft": "Raw-all SFT final",
        "processed_sft": "Processed SFT final",
        "qwen3.5_4b_thinking": "Qwen3.5-4B thinking",
    }
    for key, label in labels.items():
        row = report["seed300_four_way"][key]["overall"]
        lines.append(
            f"| {label} | {_pct(row['pass_at_1'])} | {_pct(row['pass_at_4_any'])} | "
            f"{_pct(row['pass_power_4'])} | {_pct(row.get('action_accuracy'))} | "
            f"{_pct(row.get('db_accuracy'))} |"
        )
    lines.extend(["", report["baseline_note"], "", "## Raw-all versus processed overall", ""])
    lines.extend(
        [
            "| Seed | Agent | pass@1 | pass@4(any) | pass^4 | action acc. | DB acc. |",
            "|---:|---|---:|---:|---:|---:|---:|",
        ]
    )
    for seed in ("300", "301"):
        for key, label in (("raw_all", "Raw-all"), ("processed", "Processed")):
            row = report[key][seed]["overall"]
            lines.append(
                f"| {seed} | {label} | {_pct(row['pass_at_1'])} | "
                f"{_pct(row['pass_at_4_any'])} | {_pct(row['pass_power_4'])} | "
                f"{_pct(row.get('action_accuracy'))} | {_pct(row.get('db_accuracy'))} |"
            )
    for key, label in (("raw_all", "Raw-all mean"), ("processed", "Processed mean")):
        row = report["two_seed_mean"][key]["overall"]
        lines.append(
            f"| 300/301 | {label} | {_pct(row['pass_at_1'])} | "
            f"{_pct(row['pass_at_4_any'])} | {_pct(row['pass_power_4'])} | "
            f"{_pct(row.get('action_accuracy'))} | {_pct(row.get('db_accuracy'))} |"
        )
    lines.extend(
        [
            "",
            "## Processed minus raw-all",
            "",
            "| Seed | Scope | pass@1 | pass@4(any) | pass^4 | action acc. | DB acc. |",
            "|---:|---|---:|---:|---:|---:|---:|",
        ]
    )
    for seed in ("300", "301", "mean"):
        for scope in ("overall", *DOMAINS):
            row = (
                report["processed_minus_raw_all"][seed]["overall"]
                if scope == "overall"
                else report["processed_minus_raw_all"][seed]["domains"][scope]
            )
            lines.append(
                f"| {seed} | {scope} | {_pp(row['pass_at_1'])} | "
                f"{_pp(row['pass_at_4_any'])} | {_pp(row['pass_power_4'])} | "
                f"{_pp(row.get('action_accuracy'))} | {_pp(row.get('db_accuracy'))} |"
            )
    lines.extend(
        [
            "",
            "## Domain metrics",
            "",
            "| Seed | Agent | Domain | pass@1 | pass@4(any) | pass^4 | action acc. | DB acc. |",
            "|---:|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for seed in ("300", "301"):
        for key, label in (("raw_all", "Raw-all"), ("processed", "Processed")):
            for domain in DOMAINS:
                row = report[key][seed]["domains"][domain]
                lines.append(
                    f"| {seed} | {label} | {domain} | {_pct(row['pass_at_1'])} | "
                    f"{_pct(row['pass_at_4_any'])} | {_pct(row['pass_power_4'])} | "
                    f"{_pct(row.get('action_accuracy'))} | {_pct(row.get('db_accuracy'))} |"
                )
    lines.extend(
        [
            "",
            "## Behavior and runtime",
            "",
            "| Seed | Agent | Scope | Terminations | nonexistent | invalid args | tool errors | retries | wall |",
            "|---:|---|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for seed in ("300", "301"):
        for key, label in (("raw_all", "Raw-all"), ("processed", "Processed")):
            for scope in ("overall", *DOMAINS):
                row = (
                    report[key][seed]["overall"]
                    if scope == "overall"
                    else report[key][seed]["domains"][scope]
                )
                tools = row["tool_diagnostics"]
                terms = ", ".join(
                    f"{name}:{count}" for name, count in row["terminations"].items()
                )
                lines.append(
                    f"| {seed} | {label} | {scope} | {terms} | "
                    f"{tools['nonexistent_tool_calls']} | {tools['invalid_argument_calls']} | "
                    f"{tools['tool_result_errors']} | {tools['hallucination_retries']} | "
                    f"{float(row['wall_seconds']) / 60:.2f} min |"
                )
    delta = report["processed_minus_raw_all"]["mean"]["overall"]
    lines.extend(
        [
            "",
            "## Conclusion",
            "",
            "Processed SFT minus raw-all SFT on the two-seed mean is "
            f"`{_pp(delta['pass_at_1'])}/{_pp(delta['pass_at_4_any'])}/"
            f"{_pp(delta['pass_power_4'])}` for pass@1/pass@4(any)/pass^4, "
            f"with action/DB accuracy deltas of `{_pp(delta['action_accuracy'])}/"
            f"{_pp(delta['db_accuracy'])}`.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-seed300", type=Path, required=True)
    parser.add_argument("--raw-seed301", type=Path, required=True)
    parser.add_argument(
        "--processed-seed300",
        type=Path,
        default=PROCESSED_DIR / "seed300_0902_224647_summary.json",
    )
    parser.add_argument(
        "--processed-seed301",
        type=Path,
        default=PROCESSED_DIR / "seed301_0902_233656_summary.json",
    )
    parser.add_argument(
        "--raw-instruct-seed300", type=Path, default=RAW_INSTRUCT_SUMMARY
    )
    parser.add_argument("--qwen35-seed300", type=Path, default=QWEN35_SUMMARY)
    parser.add_argument("--results-root", type=Path, default=RESULTS_ROOT)
    parser.add_argument(
        "--json-output", type=Path, default=EXPERIMENT_DIR / "FINAL_REPORT.json"
    )
    parser.add_argument(
        "--markdown-output", type=Path, default=EXPERIMENT_DIR / "FINAL_REPORT.md"
    )
    args = parser.parse_args()
    report = build_report(
        raw_seed300=args.raw_seed300,
        raw_seed301=args.raw_seed301,
        processed_seed300=args.processed_seed300,
        processed_seed301=args.processed_seed301,
        raw_instruct_seed300=args.raw_instruct_seed300,
        qwen35_seed300=args.qwen35_seed300,
        results_root=args.results_root,
    )
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.markdown_output.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
