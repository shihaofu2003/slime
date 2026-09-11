#!/usr/bin/env python3
"""Build raw Tau2 prompt-answer SFT data by dropping only over-cap rows."""

from __future__ import annotations

import argparse
import heapq
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from slime.rollout.prompt_answer_sft_rollout import build_prompt_answer_example


SERVICE_AGENT_ROOT = Path("/mnt/afs/users/fush/projects/ServiceAgent")
EXPERIMENT_DIR = SERVICE_AGENT_ROOT / "slime/output/experiments/tau2-sft-raw-all-max16384"
DEFAULT_INPUT = SERVICE_AGENT_ROOT / "datasets/AReaL-tau2-data/tau2_sft_train.jsonl"
DEFAULT_TOKENIZER = SERVICE_AGENT_ROOT / "models/Qwen3-4B-Instruct-2507"
DEFAULT_OUTPUT = EXPERIMENT_DIR / "data/raw_all_max16384.jsonl"
DEFAULT_SMOKE_OUTPUT = EXPERIMENT_DIR / "data/raw_all_max16384_longest32.jsonl"
DEFAULT_STATS_OUTPUT = EXPERIMENT_DIR / "data/raw_all_max16384.stats.json"
DOMAINS = ("airline", "retail", "telecom")
MAX_TOTAL_TOKENS = 16384


def _domain(row: Mapping[str, Any]) -> str:
    metadata = row.get("metadata")
    metadata = metadata if isinstance(metadata, Mapping) else {}
    explicit = str(metadata.get("domain") or "").lower()
    if explicit in DOMAINS:
        return explicit
    dialog_id = str(metadata.get("source_dialog_id") or "").lower()
    matches = [domain for domain in DOMAINS if domain in dialog_id]
    if len(matches) != 1:
        system_text = "\n".join(
            str(message.get("content") or "").lower()
            for message in row.get("messages") or []
            if isinstance(message, Mapping) and message.get("role") == "system"
        )
        matches = [
            domain for domain in DOMAINS if f"{domain} agent policy" in system_text
        ]
    if len(matches) != 1:
        raise ValueError(f"cannot infer Tau2 domain from metadata: {metadata}")
    return matches[0]


def _double_success(row: Mapping[str, Any]) -> bool:
    metadata = row.get("metadata")
    metadata = metadata if isinstance(metadata, Mapping) else {}
    return metadata.get("correct") in (1, 1.0, True) and metadata.get("reward") in (
        1,
        1.0,
        True,
    )


def _percentiles(lengths: Sequence[int]) -> dict[str, int]:
    if not lengths:
        return {}
    ordered = sorted(lengths)
    result = {
        label: ordered[round((len(ordered) - 1) * fraction)]
        for label, fraction in (("p50", 0.50), ("p90", 0.90), ("p95", 0.95), ("p99", 0.99))
    }
    result["max"] = ordered[-1]
    return result


def _load_generator(tokenizer_path: Path):
    from transformers import AutoTokenizer

    from slime.utils.mask_utils import MultiTurnLossMaskGenerator

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    generator = MultiTurnLossMaskGenerator(tokenizer, tokenizer_type="qwen3_full")
    return generator


def build_dataset(
    *,
    input_path: Path,
    output_path: Path,
    smoke_output_path: Path,
    stats_output_path: Path,
    max_tokens: int,
    smoke_size: int,
    mask_generator,
) -> dict[str, Any]:
    if max_tokens <= 0 or smoke_size <= 0:
        raise ValueError("max_tokens and smoke_size must be positive")
    if output_path == smoke_output_path:
        raise ValueError("full and smoke outputs must be different")

    domain_stats = {domain: Counter() for domain in DOMAINS}
    label_stats: Counter[str] = Counter()
    token_lengths: list[int] = []
    longest: list[tuple[int, int, str]] = []
    sequence = 0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    smoke_output_path.parent.mkdir(parents=True, exist_ok=True)
    stats_output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary_smoke = smoke_output_path.with_suffix(smoke_output_path.suffix + ".tmp")
    temporary_stats = stats_output_path.with_suffix(stats_output_path.suffix + ".tmp")

    try:
        with input_path.open(encoding="utf-8") as source, temporary_output.open(
            "w", encoding="utf-8"
        ) as output:
            for line_number, source_line in enumerate(source, start=1):
                if not source_line.strip():
                    continue
                try:
                    row = json.loads(source_line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{input_path}:{line_number}: invalid JSON") from exc
                if not isinstance(row, dict):
                    raise ValueError(f"{input_path}:{line_number}: expected an object")
                domain = _domain(row)
                domain_stats[domain]["source"] += 1

                _, token_ids, _ = build_prompt_answer_example(
                    row.get("messages"),
                    row.get("answer"),
                    mask_generator=mask_generator,
                    tools=row.get("tools"),
                )
                total_tokens = len(token_ids)
                if total_tokens > max_tokens:
                    domain_stats[domain]["dropped_over_cap"] += 1
                    continue

                original_line = source_line if source_line.endswith("\n") else source_line + "\n"
                output.write(original_line)
                domain_stats[domain]["kept"] += 1
                label_stats["double_success" if _double_success(row) else "has_failure"] += 1
                token_lengths.append(total_tokens)

                item = (total_tokens, sequence, original_line)
                sequence += 1
                if len(longest) < smoke_size:
                    heapq.heappush(longest, item)
                elif item[:2] > longest[0][:2]:
                    heapq.heapreplace(longest, item)

        with temporary_smoke.open("w", encoding="utf-8") as smoke_output:
            for _, _, line in sorted(longest, reverse=True):
                smoke_output.write(line)

        summary = {
            "input": str(input_path),
            "output": str(output_path),
            "smoke_output": str(smoke_output_path),
            "max_tokens": max_tokens,
            "smoke_rows": len(longest),
            "source_rows": sum(stats["source"] for stats in domain_stats.values()),
            "dropped_over_cap": sum(
                stats["dropped_over_cap"] for stats in domain_stats.values()
            ),
            "kept_rows": sum(stats["kept"] for stats in domain_stats.values()),
            "domains": {
                domain: {
                    key: domain_stats[domain][key]
                    for key in ("source", "dropped_over_cap", "kept")
                }
                for domain in DOMAINS
            },
            "kept_labels": {
                key: label_stats[key] for key in ("double_success", "has_failure")
            },
            "token_lengths": _percentiles(token_lengths),
        }
        temporary_stats.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary_output.replace(output_path)
        temporary_smoke.replace(smoke_output_path)
        temporary_stats.replace(stats_output_path)
        return summary
    finally:
        temporary_output.unlink(missing_ok=True)
        temporary_smoke.unlink(missing_ok=True)
        temporary_stats.unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--smoke-output", type=Path, default=DEFAULT_SMOKE_OUTPUT)
    parser.add_argument("--stats-output", type=Path, default=DEFAULT_STATS_OUTPUT)
    parser.add_argument("--max-tokens", type=int, default=MAX_TOTAL_TOKENS)
    parser.add_argument("--smoke-size", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generator = _load_generator(args.tokenizer)
    summary = build_dataset(
        input_path=args.input,
        output_path=args.output,
        smoke_output_path=args.smoke_output,
        stats_output_path=args.stats_output,
        max_tokens=args.max_tokens,
        smoke_size=args.smoke_size,
        mask_generator=generator,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
