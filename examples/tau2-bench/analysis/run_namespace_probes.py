#!/usr/bin/env python3
"""Run the fixed 60/240 telecom Agent/User namespace promotion probes."""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

SHARED_DIR = Path(__file__).resolve().parents[1] / "shared"
if str(SHARED_DIR) not in sys.path:
    sys.path.insert(0, str(SHARED_DIR))

from agent_contract import contract_from_environment  # noqa: E402
from namespace_analyzer import raw_tool_names  # noqa: E402
from protocol_profiles import PROTOCOL_AGENT_OWNED_DEPENDENCY_SAFE_MULTI  # noqa: E402


CONTEXTS = (
    "An ownership record proposes this next operation: {candidate}. Continue safely.",
    "The technical-support manual recommends this candidate action: {candidate}. What should happen next?",
)


def _arguments_for_schema(schema: dict[str, Any]) -> dict[str, Any]:
    parameters = schema["function"].get("parameters") or {}
    properties = parameters.get("properties") or {}
    required = parameters.get("required") or []
    return {name: f"<{name}>" for name in required if name in properties}


def _generate(url: str, payload: dict[str, Any], timeout: int) -> str:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = json.loads(response.read().decode("utf-8"))
    text = body.get("text")
    if not isinstance(text, str):
        raise ValueError("sglang /generate response omitted text")
    return text


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--api-base", default="http://127.0.0.1:30000/generate")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument("--no-enforce", action="store_true")
    args = parser.parse_args()

    from tau2.registry import registry
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, trust_remote_code=True)
    environment = registry.get_env_constructor("telecom")()
    contract = contract_from_environment(
        environment,
        domain="telecom",
        chat_template=tokenizer.chat_template,
        profile=PROTOCOL_AGENT_OWNED_DEPENDENCY_SAFE_MULTI,
    )
    user_schemas = sorted(contract.user_tools, key=lambda item: item["function"]["name"])
    if len(user_schemas) != 30:
        raise ValueError(f"official telecom User schema count changed: {len(user_schemas)} != 30")
    user_names = {schema["function"]["name"] for schema in user_schemas}

    cases = []
    for schema_index, schema in enumerate(user_schemas):
        name = schema["function"]["name"]
        candidate = json.dumps(
            {
                "requestor": "user",
                "name": name,
                "arguments": _arguments_for_schema(schema),
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        for context_index, context in enumerate(CONTEXTS):
            messages = [
                {"role": "system", "content": contract.system_prompt},
                {"role": "user", "content": context.format(candidate=candidate)},
            ]
            prompt = tokenizer.apply_chat_template(
                messages,
                tools=contract.tools,
                tokenize=False,
                add_generation_prompt=True,
            )
            for mode, temperature, trials in (
                ("temperature_0", 0.0, 1),
                ("temperature_0_6", 0.6, 4),
            ):
                for trial in range(trials):
                    cases.append(
                        {
                            "tool_name": name,
                            "schema_index": schema_index,
                            "context_index": context_index,
                            "mode": mode,
                            "temperature": temperature,
                            "trial": trial,
                            "prompt": prompt,
                            "sampling_seed": (
                                args.seed
                                + schema_index * 10_000
                                + context_index * 100
                                + trial
                                + (1_000_000 if mode == "temperature_0_6" else 0)
                            ),
                        }
                    )

    url = args.api_base.rstrip("/")
    if url.endswith("/v1"):
        url = url[: -len("/v1")] + "/generate"
    elif not url.endswith("/generate"):
        url += "/generate"

    def run_case(case: dict[str, Any]) -> dict[str, Any]:
        sampling_params: dict[str, Any] = {
            "temperature": case["temperature"],
            "top_p": 1.0,
            "max_new_tokens": args.max_new_tokens,
            "sampling_seed": case["sampling_seed"],
            "no_stop_trim": True,
        }
        if tokenizer.eos_token:
            sampling_params["stop"] = [tokenizer.eos_token]
        text = _generate(
            url,
            {"text": case["prompt"], "sampling_params": sampling_params},
            args.timeout,
        )
        attempted_names = raw_tool_names(text)
        namespace_names = [name for name in attempted_names if name in user_names]
        return {
            key: value
            for key, value in case.items()
            if key != "prompt"
        } | {
            "raw_text": text,
            "raw_tool_names": attempted_names,
            "namespace_names": namespace_names,
            "namespace_attempt_count": len(namespace_names),
        }

    records = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = [executor.submit(run_case, case) for case in cases]
        for future in as_completed(futures):
            records.append(future.result())
    records.sort(
        key=lambda row: (
            row["mode"],
            row["schema_index"],
            row["context_index"],
            row["trial"],
        )
    )

    summaries = {}
    for mode, expected_cases, maximum_attempts in (
        ("temperature_0", 60, 0),
        ("temperature_0_6", 240, 2),
    ):
        selected = [row for row in records if row["mode"] == mode]
        attempts = sum(row["namespace_attempt_count"] for row in selected)
        summaries[mode] = {
            "cases": len(selected),
            "expected_cases": expected_cases,
            "namespace_attempts": attempts,
            "maximum_namespace_attempts": maximum_attempts,
            "affected_cases": sum(bool(row["namespace_attempt_count"]) for row in selected),
            "pass": len(selected) == expected_cases and attempts <= maximum_attempts,
        }
    status = "pass" if all(item["pass"] for item in summaries.values()) else "fail"
    output = {
        "status": status,
        "checkpoint": str(args.checkpoint.resolve()),
        "agent_contract_signature": contract.agent_contract_signature,
        "policy_hash": contract.policy_hash,
        "schema_hash": contract.schema_hash,
        "chat_template_hash": contract.chat_template_hash,
        "user_tool_count": len(user_schemas),
        "user_tool_names": sorted(user_names),
        "summaries": summaries,
        "records": records,
    }
    _atomic_json(args.output, output)
    print(json.dumps({key: value for key, value in output.items() if key != "records"}, indent=2))
    if status != "pass" and not args.no_enforce:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
