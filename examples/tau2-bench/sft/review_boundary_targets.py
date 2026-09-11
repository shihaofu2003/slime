#!/usr/bin/env python3
"""Re-review/repair telecom SFT targets with the local Qwen3.6-27B only."""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Mapping


SFT_DIR = Path(__file__).resolve().parent
if str(SFT_DIR) not in sys.path:
    sys.path.insert(0, str(SFT_DIR))

from build_agent_boundary_v2 import (  # noqa: E402
    DEFAULT_OUTPUT,
    DEFAULT_TOKENIZER,
    MAX_TOTAL_TOKENS,
    _mask_only_final_assistant,
    _runtime,
    read_jsonl,
    sha256_text,
    stable_json,
    write_jsonl,
)


CORE_DIMENSIONS = (
    "business_policy_compliance",
    "task_completion",
    "tool_choice",
    "argument_grounding",
    "dependency_safety",
    "authorization_and_confirmation",
)
SUPPORT_DIMENSIONS = ("call_necessity", "recovery", "communication")
REVIEW_SYSTEM = """You are the strict local quality gate for tau2 Agent SFT targets.
Audit the final Assistant target and every dependency in its prefix against the supplied
policy, Agent-only schemas, and hard Agent/User ownership boundary. Do not infer that a
User/device tool is available to the Agent merely because the manual names it. A User
action must be requested in plain text and followed by a wait for natural-language
feedback. Preserve all facts and never lower a quality bar to retain a sample.

Return one JSON object only. Use quality_verdict keep, repair, or drop. Score every
dimension from 1 to 4. Set repaired_target only for repair; it must be one Assistant
message and may contain either plain text or structured Agent tool_calls, never both."""


def _schema() -> dict[str, Any]:
    dimensions = {
        name: {"type": "integer", "minimum": 1, "maximum": 4}
        for name in (*CORE_DIMENSIONS, *SUPPORT_DIMENSIONS)
    }
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "tau2_boundary_target_review",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "quality_verdict": {"enum": ["keep", "repair", "drop"]},
                    "dimensions": {
                        "type": "object",
                        "properties": dimensions,
                        "required": list(dimensions),
                        "additionalProperties": False,
                    },
                    "issues": {"type": "array", "items": {"type": "string"}},
                    "repaired_target": {
                        "anyOf": [
                            {"type": "null"},
                            {
                                "type": "object",
                                "properties": {
                                    "role": {"const": "assistant"},
                                    "content": {"type": "string"},
                                    "tool_calls": {"type": "array"},
                                },
                                "required": ["role", "content"],
                                "additionalProperties": True,
                            },
                        ]
                    },
                },
                "required": [
                    "quality_verdict",
                    "dimensions",
                    "issues",
                    "repaired_target",
                ],
                "additionalProperties": False,
            },
        },
    }


def _completion_url(base_url: str) -> str:
    parsed = urllib.parse.urlparse(base_url)
    if parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("boundary review is restricted to a local Qwen endpoint")
    return base_url.rstrip("/") + (
        "/chat/completions" if base_url.rstrip("/").endswith("/v1") else "/v1/chat/completions"
    )


def _extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("review response is not a JSON object")
    return value


def _request_review(
    *,
    url: str,
    api_key: str,
    model: str,
    payload: Mapping[str, Any],
    timeout: int,
    retries: int,
) -> dict[str, Any]:
    body = {
        "model": model,
        "temperature": 0,
        "max_tokens": 4096,
        "chat_template_kwargs": {"enable_thinking": False},
        "messages": [
            {"role": "system", "content": REVIEW_SYSTEM},
            {
                "role": "user",
                "content": "Audit this target:\n" + stable_json(payload),
            },
        ],
        "response_format": _schema(),
    }
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        request = urllib.request.Request(
            url,
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                response_body = json.loads(response.read().decode("utf-8"))
            content = response_body["choices"][0]["message"].get("content")
            return _extract_json(content)
        except (urllib.error.URLError, TimeoutError, KeyError, ValueError, json.JSONDecodeError) as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(min(8, 2**attempt))
    raise RuntimeError(f"local review failed after {retries + 1} attempts: {last_error}")


def _valid_scores(review: Mapping[str, Any]) -> bool:
    dimensions = review.get("dimensions") or {}
    return all(
        isinstance(dimensions.get(name), int)
        and not isinstance(dimensions.get(name), bool)
        and dimensions[name] >= 3
        for name in CORE_DIMENSIONS
    ) and all(
        isinstance(dimensions.get(name), int)
        and not isinstance(dimensions.get(name), bool)
        and dimensions[name] >= 2
        for name in SUPPORT_DIMENSIONS
    )


def _mechanically_valid(messages, contract, generator) -> tuple[bool, str | None]:
    try:
        view = contract.build_source_view(messages)
        _mask_only_final_assistant(view.messages)
        token_ids, loss_mask = generator.get_loss_mask(view.messages, tools=contract.tools)
        expected_ids = generator.tokenizer.apply_chat_template(
            view.messages,
            tools=contract.tools,
            tokenize=True,
            return_dict=False,
        )
        if token_ids != expected_ids:
            return False, "qwen3_full_token_identity"
        if len(token_ids) > MAX_TOTAL_TOKENS:
            return False, f"over_cap:{len(token_ids)}"
        if not any(loss_mask):
            return False, "empty_loss_mask"
    except (KeyError, TypeError, ValueError) as exc:
        return False, str(exc)
    return True, None


def review_candidate(
    candidate: Mapping[str, Any],
    *,
    contract,
    generator,
    url: str,
    api_key: str,
    model: str,
    timeout: int,
    retries: int,
) -> dict[str, Any]:
    original_messages = candidate.get("source_messages") or []
    mechanically_valid, mechanical_error = _mechanically_valid(
        original_messages,
        contract,
        generator,
    )
    if not mechanically_valid:
        review = {
            "quality_verdict": "drop",
            "dimensions": {
                name: 1 for name in (*CORE_DIMENSIONS, *SUPPORT_DIMENSIONS)
            },
            "issues": [f"mechanical_rejection:{mechanical_error}"],
            "repaired_target": None,
            "model": model,
            "temperature": 0,
            "all_quality_gates_passed": False,
            "mechanical_error": mechanical_error,
            "repaired": False,
            "agent_contract_signature": contract.agent_contract_signature,
            "original_source_sha256": sha256_text(stable_json(original_messages)),
            "reviewed_source_sha256": sha256_text(stable_json(original_messages)),
        }
        result = {
            "source_row": candidate.get("source_row"),
            "target_id": candidate.get("target_id"),
            "source_dialog_id": candidate.get("source_dialog_id"),
            "approved": False,
            "review": review,
            "repaired_source_messages": None,
        }
        result["review_sha256"] = sha256_text(stable_json(review))
        return result

    def evidence(messages, *, repair_allowed):
        agent_view = contract.build_source_view(messages)
        return {
            "agent_contract_signature": contract.agent_contract_signature,
            "policy": contract.policy,
            "agent_tools": contract.tools,
            "user_tools_ownership_only": contract.user_tools,
            "agent_view_messages_without_system": agent_view.messages[1:],
            "repair_allowed": repair_allowed,
        }

    first_evidence = evidence(original_messages, repair_allowed=True)
    first = _request_review(
        url=url,
        api_key=api_key,
        model=model,
        payload=first_evidence,
        timeout=timeout,
        retries=retries,
    )
    final_messages = list(original_messages)
    repaired = False
    if first.get("quality_verdict") == "repair":
        repaired_target = first.get("repaired_target")
        if not isinstance(repaired_target, dict) or not final_messages:
            first["quality_verdict"] = "drop"
        else:
            final_messages = [*final_messages[:-1], repaired_target]
            repaired = True
            repaired_valid, repaired_error = _mechanically_valid(
                final_messages,
                contract,
                generator,
            )
            if not repaired_valid:
                first["quality_verdict"] = "drop"
                first.setdefault("issues", []).append(
                    f"invalid_repair:{repaired_error}"
                )
                final_review = first
            else:
                second_evidence = evidence(final_messages, repair_allowed=False)
                final_review = _request_review(
                    url=url,
                    api_key=api_key,
                    model=model,
                    payload=second_evidence,
                    timeout=timeout,
                    retries=retries,
                )
    else:
        final_review = first

    if first.get("quality_verdict") == "drop":
        final_review = first
    mechanically_valid, mechanical_error = _mechanically_valid(
        final_messages,
        contract,
        generator,
    )
    review = {
        **final_review,
        "model": model,
        "temperature": 0,
        "all_quality_gates_passed": bool(
            final_review.get("quality_verdict") == "keep"
            and _valid_scores(final_review)
            and mechanically_valid
        ),
        "mechanical_error": mechanical_error,
        "repaired": repaired,
        "agent_contract_signature": contract.agent_contract_signature,
        "original_source_sha256": sha256_text(stable_json(original_messages)),
        "reviewed_source_sha256": sha256_text(stable_json(final_messages)),
    }
    result = {
        "source_row": candidate.get("source_row"),
        "target_id": candidate.get("target_id"),
        "source_dialog_id": candidate.get("source_dialog_id"),
        "approved": bool(review["all_quality_gates_passed"]),
        "review": review,
        "repaired_source_messages": final_messages if repaired else None,
    }
    result["review_sha256"] = sha256_text(stable_json(review))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_OUTPUT / "ordinary_telecom_review_candidates.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT / "ordinary_telecom_reviewed.jsonl",
    )
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--base-url", default="http://127.0.0.1:30000/v1")
    parser.add_argument("--api-key", default="local-qwen")
    parser.add_argument("--model", default="Qwen3.6-27B")
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    if "Qwen3.6-27B" not in args.model:
        parser.error("--model must be the local Qwen3.6-27B reviewer")
    url = _completion_url(args.base_url)
    _, generator, contracts = _runtime(args.tokenizer)
    candidates = list(read_jsonl(args.input))
    if args.limit is not None:
        candidates = candidates[: args.limit]
    existing = {
        row.get("target_id"): row
        for row in read_jsonl(args.output)
    } if args.output.exists() else {}
    pending = [
        candidate
        for candidate in candidates
        if candidate.get("target_id") not in existing
    ]

    def run(candidate):
        return review_candidate(
            candidate,
            contract=contracts["telecom"],
            generator=generator,
            url=url,
            api_key=args.api_key,
            model=args.model,
            timeout=args.timeout,
            retries=args.retries,
        )

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run, candidate): candidate for candidate in pending}
        for future in as_completed(futures):
            result = future.result()
            existing[result["target_id"]] = result
            write_jsonl(
                args.output,
                (existing[key] for key in sorted(existing, key=str)),
            )


if __name__ == "__main__":
    main()
