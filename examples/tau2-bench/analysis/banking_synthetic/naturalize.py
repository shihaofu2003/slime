#!/usr/bin/env python3
"""Use Qwen3.8 only to naturalize synthetic user wording."""

from __future__ import annotations

import argparse
import concurrent.futures
import copy
import json
import re
import urllib.request
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from banking_synthetic.constants import MODEL_PATH  # type: ignore
    from banking_synthetic.documents import read_json, read_jsonl, write_json  # type: ignore
else:
    from .constants import MODEL_PATH
    from .documents import read_json, read_jsonl, write_json


SYSTEM_PROMPT = """You edit the wording of a synthetic bank-customer opening message.
Rewrite only the customer's prose so it sounds like a natural customer request.
Tokens such as <PROTECTED_LITERAL_0> stand for immutable case data; copy every such
token exactly. Do not add policies, actions, facts, or outcomes. Return only the
rewritten opening message, with no analysis or fences."""


def _locked_literals(text: str) -> set[str]:
    patterns = (
        r"banking_syn_v1_[A-Za-z0-9_]+",
        r"syn_[A-Za-z0-9_]+",
        r"Synthetic Customer \d{6}",
        r"[A-Za-z0-9._+-]+@example\.test",
        r"\+1-555-\d{3}-\d{4}",
        r"\$\d+(?:\.\d+)?",
        r"\b\d{2}/\d{2}/\d{2,4}\b",
        r"\d+ Synthesis Avenue, Testville, OR 97000",
        r"\[Policy title: [^\]]+\]",
        r"<<Policy claim: .*?>>",
        r"<<Evidence locator: .*?>>",
        r"<<Requested outcome: .*?>>",
        r"<<Response requirement: .*?>>",
        r"<<Action clue: .*?>>",
        r"<<Case precondition: .*?>>",
        r"<<User action protocol: .*?>>",
        r"###STOP###",
    )
    return {match for pattern in patterns for match in re.findall(pattern, text)}


def _mask_locked_literals(text: str) -> tuple[str, dict[str, str]]:
    masked = text
    replacements: dict[str, str] = {}
    for literal in sorted(_locked_literals(text), key=lambda value: (-len(value), value)):
        if literal not in masked:
            continue
        placeholder = f"<PROTECTED_LITERAL_{len(replacements)}>"
        masked = masked.replace(literal, placeholder)
        replacements[placeholder] = literal
    return masked, replacements


def _post_json(url: str, payload: dict[str, Any], timeout: int) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": "Bearer synthetic-local"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def naturalize_one(
    task: dict[str, Any],
    *,
    endpoint: str,
    model: str,
    timeout: int,
    max_attempts: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    history = ((task.get("initial_state") or {}).get("message_history") or [])
    if (
        len(history) != 1
        or history[0].get("role") != "user"
        or not str(history[0].get("content") or "").strip()
    ):
        raise ValueError("naturalization requires one initial User opening message")
    original = str(history[0]["content"])
    original_literals = _locked_literals(original)
    masked_original, replacements = _mask_locked_literals(original)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": masked_original},
    ]
    responses = []
    rewritten = ""
    used_fallback = False
    for attempt in range(1, max_attempts + 1):
        response = _post_json(
            endpoint.rstrip("/") + "/chat/completions",
            {
                "model": model,
                "messages": messages,
                "temperature": 0.0,
                "top_p": 1.0,
                "max_tokens": 1024,
                "chat_template_kwargs": {"enable_thinking": False},
            },
            timeout,
        )
        responses.append(response)
        candidate = str(
            response["choices"][0]["message"].get("content") or ""
        ).strip()
        rewritten = candidate
        missing_placeholders = []
        for placeholder, literal in replacements.items():
            if placeholder in rewritten:
                rewritten = rewritten.replace(placeholder, literal)
            elif literal not in rewritten:
                missing_placeholders.append(placeholder)
        rewritten_literals = _locked_literals(rewritten)
        missing = sorted(original_literals - rewritten_literals)
        added = sorted(rewritten_literals - original_literals)
        malformed_placeholder = "PROTECTED_LITERAL_" in rewritten
        if (
            rewritten
            and not missing_placeholders
            and not missing
            and not added
            and not malformed_placeholder
        ):
            break
        if attempt < max_attempts:
            messages.extend(
                [
                    {"role": "assistant", "content": rewritten},
                    {
                        "role": "user",
                        "content": (
                            "Rewrite again from the original. Copy every protected literal "
                            "placeholder exactly. Missing placeholders: "
                            f"{json.dumps(missing_placeholders)}. Missing literals: "
                            f"{json.dumps(missing, ensure_ascii=False)}. Added literals: "
                            f"{json.dumps(added, ensure_ascii=False)}."
                        ),
                    },
                ]
            )
    else:
        rewritten = original
        used_fallback = True
    updated = copy.deepcopy(task)
    updated["initial_state"]["message_history"][0]["content"] = rewritten
    usage = responses[-1].get("usage") or {}
    return updated, {
        "task_id": task["id"],
        "attempts": len(responses),
        "used_fallback": used_fallback,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--endpoint", default="http://127.0.0.1:30000/v1")
    parser.add_argument("--model", default="Qwen3.8-27B-banking-naturalizer")
    parser.add_argument("--model-path", default=MODEL_PATH)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--max-attempts", type=int, default=4)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.model_path != MODEL_PATH:
        raise ValueError(f"naturalization model path must be {MODEL_PATH}")
    if not 1 <= args.max_attempts <= 4:
        raise ValueError("--max-attempts must be between 1 and 4")
    tasks = read_json(args.tasks)
    updated_by_id: dict[str, dict[str, Any]] = {}
    rows = []
    failures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        future_to_task = {
            executor.submit(
                naturalize_one,
                task,
                endpoint=args.endpoint,
                model=args.model,
                timeout=args.timeout,
                max_attempts=args.max_attempts,
            ): task
            for task in tasks
        }
        for future in concurrent.futures.as_completed(future_to_task):
            task = future_to_task[future]
            try:
                updated, row = future.result()
                updated_by_id[task["id"]] = updated
                rows.append(row)
            except Exception as exc:
                failures.append({"task_id": task["id"], "error": f"{type(exc).__name__}: {exc}"})
    ordered = [updated_by_id[task["id"]] for task in tasks if task["id"] in updated_by_id]
    report = {
        "model_path": args.model_path,
        "served_model": args.model,
        "input_task_count": len(tasks),
        "naturalized_task_count": len(ordered),
        "fallback_count": sum(row["used_fallback"] for row in rows),
        "failure_count": len(failures),
        "usage": rows,
        "failures": failures,
    }
    write_json(args.output, ordered)
    write_json(args.report, report)
    print(json.dumps({key: value for key, value in report.items() if key not in {"usage", "failures"}}, sort_keys=True))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
