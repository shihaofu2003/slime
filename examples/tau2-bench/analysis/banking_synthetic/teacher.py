#!/usr/bin/env python3
"""Generate environment-validated node-guided Qwen3.8 teacher trajectories."""

from __future__ import annotations

import argparse
import copy
import json
import re
import urllib.request
from urllib.error import HTTPError
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from banking_synthetic.constants import DATA_ORIGIN, MODEL_PATH  # type: ignore
    from banking_synthetic.documents import read_json, read_jsonl, write_json, write_jsonl  # type: ignore
    from banking_synthetic.validate import environment_constructor  # type: ignore
else:
    from .constants import DATA_ORIGIN, MODEL_PATH
    from .documents import read_json, read_jsonl, write_json, write_jsonl
    from .validate import environment_constructor


def _post_json(url: str, payload: dict[str, Any], timeout: int) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": "Bearer synthetic-local"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {detail[:2000]}") from exc


def _property_schema(value: Any) -> dict[str, Any]:
    if isinstance(value, bool):
        kind = "boolean"
    elif isinstance(value, int):
        kind = "integer"
    elif isinstance(value, float):
        kind = "number"
    elif isinstance(value, list):
        kind = "array"
    elif isinstance(value, dict):
        kind = "object"
    else:
        kind = "string"
    return {"type": kind, "enum": [value]}


def _tool_schema(action: Any) -> dict[str, Any]:
    arguments = action.arguments
    return {
        "type": "function",
        "function": {
            "name": action.name,
            "description": "The only action allowed for the current guided node.",
            "parameters": {
                "type": "object",
                "properties": {
                    name: _property_schema(value) for name, value in arguments.items()
                },
                "required": list(arguments),
                "additionalProperties": False,
            },
        },
    }


def _completion(
    *,
    endpoint: str,
    model: str,
    messages: list[dict[str, Any]],
    max_tokens: int,
    timeout: int,
    tools: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": 1.0,
        "top_p": 0.95,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {
            "enable_thinking": True,
            "preserve_thinking": True,
            "reasoning_effort": "xhigh",
        },
        "top_k": 20,
    }
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "required"
    return _post_json(endpoint.rstrip("/") + "/chat/completions", payload, timeout)


def _response_tool_call(response: dict[str, Any]) -> tuple[str, dict[str, Any]] | None:
    message = response["choices"][0]["message"]
    calls = message.get("tool_calls") or []
    if len(calls) != 1:
        return None
    function = calls[0].get("function") or {}
    arguments = function.get("arguments") or "{}"
    if isinstance(arguments, str):
        arguments = json.loads(arguments)
    if not isinstance(arguments, dict):
        return None
    return str(function.get("name") or ""), arguments


def _document_context(contract: dict[str, Any], documents_dir: Path) -> str:
    blocks = []
    for document_id in contract["required_documents"]:
        document = read_json(documents_dir / f"{document_id}.json")
        blocks.append(
            f"Document ID: {document_id}\nTitle: {document['title']}\n{document['content']}"
        )
    return "\n\n---\n\n".join(blocks)


def _opening_messages(
    task: dict[str, Any], autonomous_openings: dict[str, str]
) -> tuple[list[Any], str]:
    opening = autonomous_openings.get(task["id"])
    if not opening:
        raise ValueError(
            f"{task['id']}: teacher requires a valid autonomous User opening"
        )
    from tau2.data_model.message import UserMessage

    return [UserMessage.text(opening)], opening


def _score(task: Any, constructor: Any, messages: list[Any]) -> dict[str, Any]:
    from tau2.data_model.tasks import RewardType
    from tau2.evaluator.evaluator_action import ActionEvaluator
    from tau2.evaluator.evaluator_communicate import CommunicateEvaluator
    from tau2.evaluator.evaluator_env import EnvironmentEvaluator

    env = EnvironmentEvaluator.calculate_reward(
        environment_constructor=constructor,
        task=task,
        full_trajectory=messages,
        solo_mode=False,
    )
    action = ActionEvaluator.calculate_reward(task=task, full_trajectory=messages)
    communicate = CommunicateEvaluator.calculate_reward(
        task=task, full_trajectory=messages
    )
    components = {
        RewardType.DB: float((env.reward_breakdown or {}).get(RewardType.DB, 1.0)),
        RewardType.ENV_ASSERTION: float(
            (env.reward_breakdown or {}).get(RewardType.ENV_ASSERTION, 1.0)
        ),
        RewardType.ACTION: float(action.reward),
        RewardType.COMMUNICATE: float(communicate.reward),
    }
    reward = 1.0
    for basis in task.evaluation_criteria.reward_basis:
        reward *= components[basis]
    return {
        "reward": reward,
        "reward_breakdown": {name.value: value for name, value in components.items()},
    }


def teach_task(
    *,
    task_data: dict[str, Any],
    spec: dict[str, Any],
    contract: dict[str, Any],
    autonomous_openings: dict[str, str],
    documents_dir: Path,
    allowed_document_ids: tuple[str, ...],
    endpoint: str,
    model: str,
    max_attempts: int,
    max_tokens: int,
    timeout: int,
) -> dict[str, Any]:
    from tau2.data_model.message import AssistantMessage, ToolCall, UserMessage
    from tau2.data_model.tasks import Task

    task = Task.model_validate(task_data)
    constructor = environment_constructor(
        documents_dir,
        allowed_document_ids,
        contract["retrieval_variant"],
        task,
    )
    live = constructor()
    initial = task.initial_state
    live.set_state(
        initialization_data=initial.initialization_data if initial else None,
        initialization_actions=initial.initialization_actions if initial else None,
        message_history=list(initial.message_history or []) if initial else [],
    )
    messages, opening = _opening_messages(task_data, autonomous_openings)
    documents = _document_context(contract, documents_dir)
    node_attempts: dict[str, int] = {}

    for index, action in enumerate(task.evaluation_criteria.actions or []):
        graph_node = contract["reference_action_graph"][index]
        hidden_prompt = (
            "You are a node-level teacher for an independently generated Banking task. "
            "Return exactly one tool call and no prose. The current subgoal and action "
            "are authoritative; do not add another operation.\n\n"
            f"Business category:\n{spec['business_category']}\n\n"
            f"Visible customer request:\n{opening}\n\n"
            f"Current subgoal:\n{graph_node['subgoal']}\n\n"
            f"Allowed action arguments:\n{json.dumps(action.arguments, ensure_ascii=False)}\n\n"
            f"Allowed policy documents only:\n{documents}"
        )
        accepted: ToolCall | None = None
        for attempt in range(1, max_attempts + 1):
            response = _completion(
                endpoint=endpoint,
                model=model,
                messages=[
                    {"role": "system", "content": hidden_prompt},
                    {"role": "user", "content": "Execute the current node."},
                ],
                tools=[_tool_schema(action)],
                max_tokens=max_tokens,
                timeout=timeout,
            )
            parsed = _response_tool_call(response)
            if parsed is None:
                continue
            name, arguments = parsed
            candidate = ToolCall(
                id=f"teacher_{index:02d}",
                name=name,
                arguments=arguments,
                requestor=action.requestor,
            )
            if action.compare_with_tool_call(candidate):
                accepted = candidate
                node_attempts[graph_node["node_id"]] = attempt
                break
        if accepted is None:
            raise ValueError(
                f"node {graph_node['node_id']} failed after {max_attempts} attempts"
            )
        participant = AssistantMessage if action.requestor == "assistant" else UserMessage
        messages.append(
            participant(role=action.requestor, content=None, tool_calls=[accepted])
        )
        tool_response = live.get_response(accepted)
        content = str(tool_response.content or "")
        if tool_response.error or content.lstrip().startswith(("Error:", "Failed to ")):
            raise ValueError(f"node {graph_node['node_id']} tool error: {content[:400]}")
        if action.name == "KB_search":
            ranked = re.findall(r"\bID:\s+([^\s]+)", content)
            if not set(task.required_documents or []) & set(ranked[:10]):
                raise ValueError(
                    f"node {graph_node['node_id']} missed required documents in BM25 top 10"
                )
        messages.append(tool_response)

    final_facts = list(task.evaluation_criteria.communicate_info or [])
    hidden_final = (
        "Write the concise final Assistant response for the completed synthetic task. "
        "Include every required final fact verbatim and do not mention this coaching prompt.\n\n"
        f"Visible customer request:\n{opening}\n\n"
        f"Required final facts:\n{json.dumps(final_facts, ensure_ascii=False)}"
    )
    final_text = ""
    for attempt in range(1, max_attempts + 1):
        response = _completion(
            endpoint=endpoint,
            model=model,
            messages=[
                {"role": "system", "content": hidden_final},
                {"role": "user", "content": "Write the final response."},
            ],
            max_tokens=max_tokens,
            timeout=timeout,
        )
        final_text = str(response["choices"][0]["message"].get("content") or "").strip()
        if final_text and all(fact.lower() in final_text.lower() for fact in final_facts):
            node_attempts["final_response"] = attempt
            break
    else:
        raise ValueError(f"final response failed after {max_attempts} attempts")
    messages.append(AssistantMessage.text(final_text))
    reward_info = _score(task, constructor, messages)
    if reward_info["reward"] != 1.0:
        raise ValueError(f"teacher trajectory reward is {reward_info['reward']}")
    return {
        "task_id": task.id,
        "data_origin": DATA_ORIGIN,
        "teacher": "Qwen3.8-27B-node-guided",
        "model_path": MODEL_PATH,
        "hidden_guidance_saved": False,
        "node_attempts": node_attempts,
        "messages": [message.model_dump(mode="json") for message in messages],
        "reward_info": reward_info,
    }


def _autonomous_openings(paths: list[Path]) -> dict[str, str]:
    openings = {}
    for path in paths:
        for simulation in read_json(path).get("simulations") or []:
            for message in simulation.get("messages") or []:
                if message.get("role") == "user" and str(message.get("content") or "").strip():
                    openings.setdefault(simulation["task_id"], str(message["content"]).strip())
                    break
    return openings


def shard_task_ids(task_ids: set[str], shard_index: int, num_shards: int) -> set[str]:
    if num_shards < 1 or not 0 <= shard_index < num_shards:
        raise ValueError("shard_index must be within [0, num_shards)")
    return set(sorted(task_ids)[shard_index::num_shards])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--specs", type=Path, required=True)
    parser.add_argument("--contracts", type=Path, required=True)
    parser.add_argument("--documents-dir", type=Path, required=True)
    parser.add_argument("--allowlist", type=Path, required=True)
    parser.add_argument("--autonomous-results", type=Path, action="append", default=[])
    parser.add_argument("--task-ids-file", type=Path)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--endpoint", default="http://127.0.0.1:30000/v1")
    parser.add_argument("--model", default="Qwen3.8-27B-banking-node-teacher")
    parser.add_argument("--model-path", default=MODEL_PATH)
    parser.add_argument("--max-attempts", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--failures", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.model_path != MODEL_PATH:
        raise ValueError(f"teacher model path must be {MODEL_PATH}")
    if not 1 <= args.max_attempts <= 4:
        raise ValueError("--max-attempts must be between 1 and 4")
    tasks = read_json(args.tasks)
    specs = read_jsonl(args.specs)
    contracts = read_jsonl(args.contracts)
    allowed_document_ids = tuple(
        read_json(args.allowlist)["allowed_document_ids"]
    )
    if len(allowed_document_ids) != 480 or len(set(allowed_document_ids)) != 480:
        raise ValueError("teacher allowlist must contain exactly 480 unique documents")
    allowed_documents = set(allowed_document_ids)
    for contract in contracts:
        if not set(contract["required_documents"]) <= allowed_documents:
            raise ValueError(
                f"{contract['task_id']}: teacher contract contains a non-allowlisted document"
            )
    task_by_id = {task["id"]: task for task in tasks}
    spec_by_id = {spec["scenario_id"]: spec for spec in specs}
    requested = set(read_json(args.task_ids_file)) if args.task_ids_file else set(task_by_id)
    if unknown := requested - set(task_by_id):
        raise ValueError(f"teacher request contains {len(unknown)} unknown task ids")
    requested = shard_task_ids(requested, args.shard_index, args.num_shards)
    openings = _autonomous_openings(args.autonomous_results)
    trajectories = []
    failures = []
    for index, contract in enumerate(contracts, 1):
        task_id = contract["task_id"]
        if task_id not in requested:
            continue
        if contract.get("data_origin") != DATA_ORIGIN:
            raise ValueError(f"{task_id}: teacher accepts only {DATA_ORIGIN} contracts")
        try:
            trajectories.append(
                teach_task(
                    task_data=task_by_id[task_id],
                    spec=spec_by_id[contract["scenario_id"]],
                    contract=contract,
                    autonomous_openings=openings,
                    documents_dir=args.documents_dir,
                    allowed_document_ids=allowed_document_ids,
                    endpoint=args.endpoint,
                    model=args.model,
                    max_attempts=args.max_attempts,
                    max_tokens=args.max_tokens,
                    timeout=args.timeout,
                )
            )
        except Exception as exc:
            failures.append({"task_id": task_id, "error": f"{type(exc).__name__}: {exc}"})
        if index % 25 == 0:
            print(f"teacher processed {index}/{len(contracts)}", flush=True)
    manifest = {
        "data_origin": DATA_ORIGIN,
        "model_path": MODEL_PATH,
        "requested_task_count": len(requested),
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "successful_task_count": len(trajectories),
        "successful_task_ids": [row["task_id"] for row in trajectories],
        "failure_count": len(failures),
        "hidden_guidance_saved": False,
    }
    write_jsonl(args.output, trajectories)
    write_json(args.manifest, manifest)
    write_json(args.failures, failures)
    print(json.dumps({key: value for key, value in manifest.items() if key != "successful_task_ids"}, sort_keys=True))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
