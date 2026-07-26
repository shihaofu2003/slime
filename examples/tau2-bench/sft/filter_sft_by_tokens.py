#!/usr/bin/env python3
"""Filter slime SFT JSONL rows by tokenized sequence length."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer


SERVICE_AGENT_ROOT = Path("/mnt/afs/users/fush/projects/ServiceAgent")
DEFAULT_TOKENIZER_PATH = SERVICE_AGENT_ROOT / "models/Qwen3-4B-Instruct-2507"


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    tmp_path.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stats-output", type=Path, default=None)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER_PATH)
    parser.add_argument("--loss-mask-type", default="qwen3")
    parser.add_argument("--max-total-tokens", type=int, required=True)
    return parser.parse_args()


def find_all_sublist_indices(main_list: list[int], sublist: list[int]) -> list[int]:
    sublist_len = len(sublist)
    return [
        i
        for i in range(len(main_list) - sublist_len + 1)
        if main_list[i : i + sublist_len] == sublist
    ]


def get_qwen_system_message_length(tokenizer) -> int:
    test_string = "FOR TESTING ONLY"
    test_messages = [
        {"role": "user", "content": test_string},
        {"role": "user", "content": test_string},
    ]
    raw_token_ids = tokenizer(test_string, add_special_tokens=False)["input_ids"]
    chat_template_text = tokenizer.apply_chat_template(
        test_messages, add_special_tokens=False, tokenize=False
    )
    chat_template_token_ids = tokenizer(chat_template_text, add_special_tokens=False)["input_ids"]
    idx_1, idx_2 = find_all_sublist_indices(chat_template_token_ids, raw_token_ids)
    end_interval = len(chat_template_token_ids) - len(raw_token_ids) - idx_2
    return idx_1 - ((idx_2 - idx_1) - end_interval - len(raw_token_ids))


def get_qwen3_token_ids(
    tokenizer,
    messages: list[dict[str, Any]],
    system_message_length: int,
    tools=None,
) -> list[int]:
    prefix_message = {"role": "user", "content": "FOR CALCULATING LOSS MASK ONLY"}
    prefix_token_ids = tokenizer.apply_chat_template([prefix_message], tokenize=True, return_dict=False)
    all_token_ids: list[int] = []

    for i, message in enumerate(messages):
        if i == 0:
            tailed_message_ids = tokenizer.apply_chat_template(
                [message, prefix_message],
                tokenize=True,
                tools=tools,
                return_dict=False,
            )
            message_ids = tailed_message_ids[: -len(prefix_token_ids)]
        else:
            prefixed_message_ids = tokenizer.apply_chat_template(
                [prefix_message, message],
                tokenize=True,
                return_dict=False,
            )
            message_ids = prefixed_message_ids[len(prefix_token_ids) :]

        if message["role"] != "system" and i > 0:
            message_ids = message_ids[system_message_length:]

        all_token_ids.extend(message_ids)

    return all_token_ids


def main() -> None:
    args = parse_args()
    if args.loss_mask_type != "qwen3":
        raise ValueError("filter_sft_by_tokens.py currently supports --loss-mask-type qwen3 only")

    tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer), trust_remote_code=True)
    system_message_length = get_qwen_system_message_length(tokenizer)
    rows: list[dict[str, Any]] = []
    stats: Counter[str] = Counter()
    max_kept = 0
    max_dropped = 0

    for row in read_jsonl(args.input):
        stats["source_rows"] += 1
        try:
            token_ids = get_qwen3_token_ids(
                tokenizer,
                row["messages"],
                system_message_length=system_message_length,
                tools=None,
            )
        except Exception:
            stats["bad_rows"] += 1
            continue

        total_tokens = len(token_ids)
        if total_tokens <= args.max_total_tokens:
            rows.append(row)
            stats["kept_rows"] += 1
            max_kept = max(max_kept, total_tokens)
        else:
            stats["dropped_long_rows"] += 1
            max_dropped = max(max_dropped, total_tokens)

    write_jsonl(args.output, rows)
    payload = {
        "input": str(args.input),
        "output": str(args.output),
        "max_total_tokens": args.max_total_tokens,
        "stats": dict(sorted(stats.items())),
        "max_kept_total_tokens": max_kept,
        "max_dropped_total_tokens": max_dropped,
    }
    stats_output = args.stats_output or args.output.with_suffix(args.output.suffix + ".stats.json")
    stats_output.parent.mkdir(parents=True, exist_ok=True)
    stats_output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
