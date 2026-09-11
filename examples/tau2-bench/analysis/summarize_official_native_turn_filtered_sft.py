#!/usr/bin/env python3
"""Summarize the turn-filtered SFT curve and its Processed SFT ablation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from summarize_official_native_expanded_sft import _load_runtimes, load_checkpoint


CHECKPOINTS = (
    ("iter_0000399", 400),
    ("iter_0000799", 800),
    ("iter_0001199", 1200),
    ("iter_0001599", 1600),
    ("iter_0001855", 1856),
    ("iter_0001999", 2000),
    ("iter_0002399", 2400),
    ("iter_0002799", 2800),
    ("iter_0003199", 3200),
    ("iter_0003599", 3600),
    ("iter_0003711", 3712),
)
FINAL_CHECKPOINT = CHECKPOINTS[-1][0]
MATCHED_UPDATE_CHECKPOINT = "iter_0003599"
PROCESSED_FINAL_CHECKPOINT = "iter_0003795"
DOMAINS = ("airline", "retail", "telecom")
METRICS = (
    "pass_at_1",
    "pass_at_4_any",
    "pass_power_4",
    "action_accuracy",
    "db_accuracy",
)
SERVICE_AGENT_ROOT = Path("/mnt/afs/users/fush/projects/ServiceAgent")
PROJECT_ROOT = SERVICE_AGENT_ROOT / "slime"
EXPERIMENT_DIR = (
    PROJECT_ROOT
    / "output/experiments/tau2-sft-official-native-expanded-turn-filtered"
)
PROCESSED_EVAL_DIR = (
    PROJECT_ROOT
    / "output/experiments/tau2-sft-official-native-expanded/eval"
    / "checkpoint-iter_0003795"
)
PROCESSED_MATCHED_EVAL_DIR = (
    PROJECT_ROOT
    / "output/experiments/tau2-sft-official-native-expanded/eval"
    / "checkpoint-iter_0003599"
)


def _mean_rows(first: dict[str, Any], second: dict[str, Any]) -> dict[str, Any]:
    return {
        key: None
        if first.get(key) is None or second.get(key) is None
        else (first[key] + second[key]) / 2
        for key in METRICS
    }


def _delta(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    return {
        key: None
        if left.get(key) is None or right.get(key) is None
        else left[key] - right[key]
        for key in METRICS
    }


def _two_seed_mean(models: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return {
        "overall": _mean_rows(models["300"]["overall"], models["301"]["overall"]),
        "domains": {
            domain: _mean_rows(
                models["300"]["domains"][domain],
                models["301"]["domains"][domain],
            )
            for domain in DOMAINS
        },
    }


def _scope_delta(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    return {
        "overall": _delta(left["overall"], right["overall"]),
        "domains": {
            domain: _delta(left["domains"][domain], right["domains"][domain])
            for domain in DOMAINS
        },
    }


def _select_best(models: dict[str, dict[str, Any]]) -> str:
    updates = dict(CHECKPOINTS)
    return max(
        models,
        key=lambda checkpoint: (
            models[checkpoint]["overall"]["pass_at_1"],
            models[checkpoint]["overall"]["pass_power_4"],
            -updates[checkpoint],
        ),
    )


def _comparison(turn_filtered, processed):
    turn_filtered_mean = _two_seed_mean(turn_filtered)
    processed_mean = _two_seed_mean(processed)
    deltas = {
        seed: _scope_delta(turn_filtered[seed], processed[seed])
        for seed in ("300", "301")
    }
    deltas["mean"] = _scope_delta(turn_filtered_mean, processed_mean)
    return {
        "two_seed_mean": {
            "turn_filtered": turn_filtered_mean,
            "processed": processed_mean,
        },
        "turn_filtered_minus_processed": deltas,
    }


def build_ablation_report(*, seed300, seed301, processed, processed_matched=None):
    expected = {checkpoint for checkpoint, _ in CHECKPOINTS}
    if set(seed300) != expected:
        raise ValueError("seed300 must contain the complete turn-filtered curve")
    selected = _select_best(seed300)
    required_seed301 = {selected, FINAL_CHECKPOINT}
    if processed_matched is not None:
        required_seed301.add(MATCHED_UPDATE_CHECKPOINT)
    if seed301 and set(seed301) != required_seed301:
        raise ValueError("seed301 must contain the selected and final checkpoints")
    if set(processed) != {"300", "301"}:
        raise ValueError("processed must contain seeds 300 and 301")
    if processed_matched is not None and set(processed_matched) != {"300", "301"}:
        raise ValueError("processed_matched must contain seeds 300 and 301")

    selected_comparison = None
    final_comparison = None
    equal_update_comparison = None
    if seed301:
        selected_models = {
            "300": seed300[selected],
            "301": seed301[selected],
        }
        final_models = {
            "300": seed300[FINAL_CHECKPOINT],
            "301": seed301[FINAL_CHECKPOINT],
        }
        selected_comparison = _comparison(selected_models, processed)
        final_comparison = _comparison(final_models, processed)
        if processed_matched is not None:
            matched_models = {
                "300": seed300[MATCHED_UPDATE_CHECKPOINT],
                "301": seed301[MATCHED_UPDATE_CHECKPOINT],
            }
            equal_update_comparison = _comparison(
                matched_models,
                processed_matched,
            )
    return {
        "selection_policy": "seed300 pass@1, then pass^4, then earlier checkpoint",
        "checkpoint_schedule": [
            {"checkpoint": checkpoint, "updates": updates}
            for checkpoint, updates in CHECKPOINTS
        ],
        "selected_checkpoint": selected,
        "final_checkpoint": FINAL_CHECKPOINT,
        "matched_update_checkpoint": MATCHED_UPDATE_CHECKPOINT,
        "processed_final_checkpoint": PROCESSED_FINAL_CHECKPOINT,
        "seed300": seed300,
        "seed301": seed301,
        "processed": processed,
        "processed_matched": processed_matched,
        "equal_update_vs_processed": equal_update_comparison,
        "equal_epoch_final_vs_processed": final_comparison,
        "selected_vs_processed": selected_comparison,
    }


def _pct(value: Any) -> str:
    return "—" if value is None else f"{100 * float(value):.2f}%"


def _pp(value: Any) -> str:
    return "—" if value is None else f"{100 * float(value):+.2f}pp"


def _append_comparison(
    lines,
    *,
    heading,
    checkpoint_line,
    comparison,
    turn_filtered,
    processed,
):
    lines.extend(
        [
            "",
            f"## {heading}",
            "",
            checkpoint_line,
            "",
            "| Seed | Agent | pass@1 | pass@4(any) | pass^4 | action acc. | DB acc. |",
            "|---:|---|---:|---:|---:|---:|---:|",
        ]
    )
    for seed in ("300", "301"):
        for label, metrics in (
            ("Turn-filtered", turn_filtered[seed]["overall"]),
            ("Processed", processed[seed]["overall"]),
        ):
            lines.append(
                f"| {seed} | {label} | {_pct(metrics['pass_at_1'])} | "
                f"{_pct(metrics['pass_at_4_any'])} | {_pct(metrics['pass_power_4'])} | "
                f"{_pct(metrics.get('action_accuracy'))} | {_pct(metrics.get('db_accuracy'))} |"
            )
    for label, metrics in comparison["two_seed_mean"].items():
        lines.append(
            f"| 300/301 | {'Turn-filtered mean' if label == 'turn_filtered' else 'Processed mean'} | "
            f"{_pct(metrics['overall']['pass_at_1'])} | "
            f"{_pct(metrics['overall']['pass_at_4_any'])} | "
            f"{_pct(metrics['overall']['pass_power_4'])} | "
            f"{_pct(metrics['overall'].get('action_accuracy'))} | "
            f"{_pct(metrics['overall'].get('db_accuracy'))} |"
        )

    lines.extend(
        [
            "",
            "Turn-filtered minus Processed:",
            "",
            "| Seed | Scope | pass@1 | pass@4(any) | pass^4 | action acc. | DB acc. |",
            "|---:|---|---:|---:|---:|---:|---:|",
        ]
    )
    for seed in ("300", "301", "mean"):
        delta = comparison["turn_filtered_minus_processed"][seed]
        for scope in ("overall", *DOMAINS):
            metrics = (
                delta["overall"] if scope == "overall" else delta["domains"][scope]
            )
            lines.append(
                f"| {seed} | {scope} | {_pp(metrics['pass_at_1'])} | "
                f"{_pp(metrics['pass_at_4_any'])} | {_pp(metrics['pass_power_4'])} | "
                f"{_pp(metrics.get('action_accuracy'))} | {_pp(metrics.get('db_accuracy'))} |"
            )


def render_markdown(report):
    lines = [
        "# Turn-filtered SFT data ablation",
        "",
        "## Selection",
        "",
        f"Selected checkpoint: `{report['selected_checkpoint']}`. "
        f"Policy: {report['selection_policy']}.",
        "",
        "## Seed 300 checkpoint curve",
        "",
        "| Checkpoint | Updates | pass@1 | pass@4(any) | pass^4 | action acc. | DB acc. |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["checkpoint_schedule"]:
        checkpoint = row["checkpoint"]
        metrics = report["seed300"][checkpoint]["overall"]
        lines.append(
            f"| `{checkpoint}` | {row['updates']} | {_pct(metrics['pass_at_1'])} | "
            f"{_pct(metrics['pass_at_4_any'])} | {_pct(metrics['pass_power_4'])} | "
            f"{_pct(metrics.get('action_accuracy'))} | {_pct(metrics.get('db_accuracy'))} |"
        )

    selected_comparison = report["selected_vs_processed"]
    if selected_comparison is None:
        return "\n".join(lines) + "\n"

    selected = report["selected_checkpoint"]
    if report["equal_update_vs_processed"] is not None:
        matched = report["matched_update_checkpoint"]
        _append_comparison(
            lines,
            heading="Equal-update comparison",
            checkpoint_line=f"`{matched}` versus `{matched}`.",
            comparison=report["equal_update_vs_processed"],
            turn_filtered={
                "300": report["seed300"][matched],
                "301": report["seed301"][matched],
            },
            processed=report["processed_matched"],
        )
    _append_comparison(
        lines,
        heading="Equal-epoch final comparison",
        checkpoint_line=(
            f"`{report['final_checkpoint']}` versus "
            f"`{report['processed_final_checkpoint']}`."
        ),
        comparison=report["equal_epoch_final_vs_processed"],
        turn_filtered={
            "300": report["seed300"][report["final_checkpoint"]],
            "301": report["seed301"][report["final_checkpoint"]],
        },
        processed=report["processed"],
    )
    _append_comparison(
        lines,
        heading="Selected checkpoint comparison",
        checkpoint_line=(
            f"`{selected}` versus `{report['processed_final_checkpoint']}`."
        ),
        comparison=selected_comparison,
        turn_filtered={
            "300": report["seed300"][selected],
            "301": report["seed301"][selected],
        },
        processed=report["processed"],
    )
    return "\n".join(lines) + "\n"


def _parse_specs(values: list[str], option: str) -> dict[str, Path]:
    specs = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"{option} must be CHECKPOINT=SUMMARY.json: {value}")
        checkpoint, path = value.split("=", 1)
        if checkpoint in specs:
            raise ValueError(f"duplicate {option} checkpoint: {checkpoint}")
        specs[checkpoint] = Path(path)
    return specs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", action="append", default=[], required=True)
    parser.add_argument("--confirmation", action="append", default=[])
    parser.add_argument(
        "--processed-seed300",
        type=Path,
        default=PROCESSED_EVAL_DIR / "seed300_0902_224647_summary.json",
    )
    parser.add_argument(
        "--processed-seed301",
        type=Path,
        default=PROCESSED_EVAL_DIR / "seed301_0902_233656_summary.json",
    )
    parser.add_argument(
        "--processed-matched-seed300",
        type=Path,
        default=PROCESSED_MATCHED_EVAL_DIR / "seed300_0902_224547_summary.json",
    )
    parser.add_argument("--processed-matched-seed301", type=Path)
    parser.add_argument(
        "--results-root",
        type=Path,
        default=SERVICE_AGENT_ROOT / "tau2-bench",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=EXPERIMENT_DIR / "CHECKPOINT_EVAL.json",
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=EXPERIMENT_DIR / "CHECKPOINT_EVAL.md",
    )
    args = parser.parse_args()

    try:
        summary_specs = _parse_specs(args.summary, "--summary")
        confirmation_specs = _parse_specs(args.confirmation, "--confirmation")
        expected = {checkpoint for checkpoint, _ in CHECKPOINTS}
        if set(summary_specs) != expected:
            raise ValueError("--summary must contain the complete turn-filtered curve")
        runtimes = _load_runtimes()
        seed300 = {
            checkpoint: load_checkpoint(
                checkpoint,
                summary_specs[checkpoint],
                results_root=args.results_root,
                expected_seed=300,
                runtimes=runtimes,
                expected_agent_llm=(
                    "openai/Qwen3-4B-Instruct-2507-sft-official-native-expanded-"
                    f"turn-filtered-{checkpoint}"
                ),
            )
            for checkpoint, _ in CHECKPOINTS
        }
        seed301 = {
            checkpoint: load_checkpoint(
                checkpoint,
                path,
                results_root=args.results_root,
                expected_seed=301,
                runtimes=runtimes,
                expected_agent_llm=(
                    "openai/Qwen3-4B-Instruct-2507-sft-official-native-expanded-"
                    f"turn-filtered-{checkpoint}"
                ),
            )
            for checkpoint, path in confirmation_specs.items()
        }
        processed = {
            "300": load_checkpoint(
                "iter_0003795",
                args.processed_seed300,
                results_root=args.results_root,
                expected_seed=300,
                runtimes=runtimes,
                expected_agent_llm=(
                    "openai/Qwen3-4B-Instruct-2507-sft-official-native-expanded-"
                    "iter_0003795"
                ),
            ),
            "301": load_checkpoint(
                "iter_0003795",
                args.processed_seed301,
                results_root=args.results_root,
                expected_seed=301,
                runtimes=runtimes,
                expected_agent_llm=(
                    "openai/Qwen3-4B-Instruct-2507-sft-official-native-expanded-"
                    "iter_0003795"
                ),
            ),
        }
        processed_matched = None
        if seed301:
            if args.processed_matched_seed301 is None:
                raise ValueError(
                    "--processed-matched-seed301 is required with --confirmation"
                )
            processed_matched = {
                "300": load_checkpoint(
                    MATCHED_UPDATE_CHECKPOINT,
                    args.processed_matched_seed300,
                    results_root=args.results_root,
                    expected_seed=300,
                    runtimes=runtimes,
                    expected_agent_llm=(
                        "openai/Qwen3-4B-Instruct-2507-sft-official-native-expanded-"
                        f"{MATCHED_UPDATE_CHECKPOINT}"
                    ),
                ),
                "301": load_checkpoint(
                    MATCHED_UPDATE_CHECKPOINT,
                    args.processed_matched_seed301,
                    results_root=args.results_root,
                    expected_seed=301,
                    runtimes=runtimes,
                    expected_agent_llm=(
                        "openai/Qwen3-4B-Instruct-2507-sft-official-native-expanded-"
                        f"{MATCHED_UPDATE_CHECKPOINT}"
                    ),
                ),
            }
        report = build_ablation_report(
            seed300=seed300,
            seed301=seed301,
            processed=processed,
            processed_matched=processed_matched,
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
