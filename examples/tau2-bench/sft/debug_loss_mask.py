#!/usr/bin/env python3
"""Inspect tau2 SFT loss masks for real Qwen3 tokenizer output.

This is a read-only diagnostic. It loads a small set of SFT rows, generates the
same masks used by ``slime.rollout.sft_rollout.generate_rollout``, and prints
the text spans selected by ``loss_mask == 1``.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from slime.utils.mask_utils import MultiTurnLossMaskGenerator


SERVICE_AGENT_ROOT = Path("/mnt/afs/users/fush/projects/ServiceAgent")
DEFAULT_MODEL_PATH = SERVICE_AGENT_ROOT / "models/Qwen3-4B-Instruct-2507"
DEFAULT_DATA_PATH = SERVICE_AGENT_ROOT / "datasets/tau2-bench-sft/areal_tau2_sft_strict_no_thinking.jsonl"

TOOL_CALL_RE = re.compile(r"<tool_call>.*?</tool_call>", re.S)


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as f:
        for row_index, line in enumerate(f):
            line = line.strip()
            if line:
                row = json.loads(line)
                row["_debug_row_index"] = row_index
                yield row


def compact(text: str, limit: int) -> str:
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def contiguous_spans(token_ids: list[int], loss_mask: list[int]) -> list[list[int]]:
    spans: list[list[int]] = []
    current: list[int] = []
    for token_id, mask in zip(token_ids, loss_mask, strict=True):
        if mask:
            current.append(token_id)
        elif current:
            spans.append(current)
            current = []
    if current:
        spans.append(current)
    return spans


def row_kind(row: dict[str, Any]) -> str:
    messages = row.get("messages") or []
    assistant_count = sum(message.get("role") == "assistant" for message in messages)
    if any(
        message.get("role") == "assistant" and TOOL_CALL_RE.search(message.get("content") or "")
        for message in messages
    ):
        return "tool_call"
    if assistant_count > 1:
        return "multi_turn"
    return "plain"


def select_rows(path: Path, domains: list[str], samples_per_domain: int) -> dict[str, list[dict[str, Any]]]:
    selected: dict[str, list[dict[str, Any]]] = {domain: [] for domain in domains}
    seen_kinds: dict[str, set[str]] = defaultdict(set)

    for row in read_jsonl(path):
        metadata = row.get("metadata") or {}
        domain = metadata.get("domain")
        if domain not in selected or len(selected[domain]) >= samples_per_domain:
            continue

        kind = row_kind(row)
        if kind in seen_kinds[domain] and len(seen_kinds[domain]) < 3:
            continue

        selected[domain].append(row)
        seen_kinds[domain].add(kind)

        if all(len(rows) >= samples_per_domain for rows in selected.values()):
            break

    return selected


def normalized_contains(haystack: str, needle: str) -> bool:
    haystack = " ".join(haystack.split())
    needle = " ".join(needle.split())
    return bool(needle) and needle in haystack


def inspect_row(
    *,
    row: dict[str, Any],
    generator: MultiTurnLossMaskGenerator,
    preview_chars: int,
) -> tuple[int, int]:
    messages = row["messages"]
    token_ids, loss_mask = generator.get_loss_mask(messages)
    if len(token_ids) != len(loss_mask):
        raise RuntimeError(f"token/mask length mismatch: {len(token_ids)} != {len(loss_mask)}")

    response_length = generator.get_response_lengths([loss_mask])[0]
    mask_sum = sum(loss_mask)
    spans = contiguous_spans(token_ids, loss_mask)
    decoded_spans = [generator.tokenizer.decode(span) for span in spans]
    masked_text = "\n".join(decoded_spans)

    metadata = row.get("metadata") or {}
    print(
        "\n=== row "
        f"source_row={metadata.get('source_row')} jsonl_row={row.get('_debug_row_index')} "
        f"domain={metadata.get('domain')} kind={row_kind(row)} ==="
    )
    print(
        f"token_count={len(token_ids)} mask_len={len(loss_mask)} "
        f"mask_sum={mask_sum} response_length={response_length} spans={len(spans)}"
    )

    assistant_index = 0
    warning_count = 0
    for message in messages:
        role = message.get("role")
        content = message.get("content") or ""
        if role == "assistant":
            assistant_index += 1
            present = normalized_contains(masked_text, content)
            status = "masked" if present else "WARNING:not-found-in-masked-spans"
            warning_count += 0 if present else 1
            print(f"assistant[{assistant_index}] {status}: {compact(content, preview_chars)}")
        elif content:
            leak = normalized_contains(masked_text, content)
            if leak:
                warning_count += 1
                print(f"WARNING non-assistant text appears masked role={role}: {compact(content, preview_chars)}")

    print("masked_spans:")
    for span_index, text in enumerate(decoded_spans, start=1):
        print(f"  span[{span_index}]: {compact(text, preview_chars)}")

    return mask_sum, warning_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--domains", default="airline,retail,telecom")
    parser.add_argument("--samples-per-domain", type=int, default=3)
    parser.add_argument("--loss-mask-type", default="qwen3")
    parser.add_argument("--preview-chars", type=int, default=500)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    domains = [domain.strip() for domain in args.domains.split(",") if domain.strip()]
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    generator = MultiTurnLossMaskGenerator(tokenizer, tokenizer_type=args.loss_mask_type)
    selected = select_rows(args.data_path, domains, args.samples_per_domain)

    total_rows = 0
    total_masked_tokens = 0
    total_warnings = 0
    for domain in domains:
        rows = selected.get(domain, [])
        print(f"\n### domain={domain} selected_rows={len(rows)}")
        if not rows:
            total_warnings += 1
            print("WARNING no rows selected")
            continue
        for row in rows:
            total_rows += 1
            mask_sum, warnings = inspect_row(row=row, generator=generator, preview_chars=args.preview_chars)
            total_masked_tokens += mask_sum
            total_warnings += warnings

    print(
        "\nSUMMARY "
        f"rows={total_rows} masked_tokens={total_masked_tokens} warnings={total_warnings} "
        f"model_path={args.model_path} data_path={args.data_path}"
    )
    if total_rows == 0 or total_masked_tokens == 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
