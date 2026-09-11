#!/usr/bin/env python3
"""Validate and summarize a sharded VITA-Bench evaluation."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import statistics
import sys
import tempfile
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from json_repair import repair_json as _repair_json
except ImportError:  # Keep strict-JSON validation usable outside the Vita venv.
    _repair_json = None


SUITE_NAMES = ("delivery", "instore", "ota", "cross_domain")
EXPECTED_TASKS_PER_SUITE = 100
EXPECTED_SHARDS_PER_SUITE = 10
EXPECTED_TRIALS = 4
EXPECTED_BASE_SEED = 300
EXPECTED_SHARD_SIZE = 10
SUPPORTED_MANIFEST_SCHEMAS = frozenset({2, 3, 4, 5, 6})
PARALLEL_MANIFEST_SCHEMAS = frozenset({3, 4, 5})
VALID_TERMINATIONS = {
    "user_stop",
    "agent_stop",
    "max_steps",
    "too_many_errors",
    "invalid_agent_message",
}
PREMATURE_TERMINATIONS = {
    "max_steps",
    "too_many_errors",
    "invalid_agent_message",
}
TOKEN_FIELDS = {
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "input_tokens",
    "output_tokens",
    "cached_tokens",
    "reasoning_tokens",
}


class CorruptResultsError(ValueError):
    """The checkpoint cannot be safely resumed or reported."""


class IncompleteResultsError(ValueError):
    """The checkpoint is valid but does not contain every expected run."""


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _finite_number(value: Any, field: str, *, nonnegative: bool = False) -> float:
    if not _is_number(value) or not math.isfinite(float(value)):
        raise CorruptResultsError(f"{field} must be a finite number; got {value!r}")
    number = float(value)
    if nonnegative and number < 0:
        raise CorruptResultsError(f"{field} must be nonnegative; got {value!r}")
    return number


def _required_int(value: Any, field: str, *, minimum: int | None = None) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise CorruptResultsError(f"{field} must be an integer; got {value!r}")
    if minimum is not None and value < minimum:
        raise CorruptResultsError(f"{field} must be at least {minimum}; got {value}")
    return value


def _required_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CorruptResultsError(f"{field} must be a nonempty string")
    return value


def _load_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except FileNotFoundError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CorruptResultsError(f"cannot read JSON from {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise CorruptResultsError(f"{path} must contain a JSON object")
    return data


def _unique_task_ids(values: Any, field: str) -> list[str]:
    if not isinstance(values, list):
        raise CorruptResultsError(f"{field} must be a list")
    task_ids = [_required_string(value, f"{field}[{index}]") for index, value in enumerate(values)]
    duplicates = sorted(task_id for task_id, count in Counter(task_ids).items() if count > 1)
    if duplicates:
        raise CorruptResultsError(f"{field} contains duplicate task IDs: {duplicates[:10]}")
    return task_ids


def _task_ids_from_results(tasks: Any, field: str) -> list[str]:
    if not isinstance(tasks, list):
        raise CorruptResultsError(f"{field} must be a list")
    task_ids: list[str] = []
    for index, task in enumerate(tasks):
        if not isinstance(task, dict):
            raise CorruptResultsError(f"{field}[{index}] must be an object")
        task_ids.append(_required_string(task.get("id"), f"{field}[{index}].id"))
    duplicates = sorted(task_id for task_id, count in Counter(task_ids).items() if count > 1)
    if duplicates:
        raise CorruptResultsError(f"{field} contains duplicate task IDs: {duplicates[:10]}")
    return task_ids


def _trial_seeds(base_seed: int, num_trials: int) -> list[int]:
    generator = random.Random(base_seed)
    return [generator.randint(0, 1_000_000) for _ in range(num_trials)]


def _validate_usage(usage: Any, field: str) -> None:
    if usage is None:
        return
    if not isinstance(usage, dict):
        raise CorruptResultsError(f"{field} must be an object or null")

    def visit(value: Any, path: str) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                visit(child, f"{path}.{key}")
            return
        leaf = path.rsplit(".", 1)[-1]
        if leaf in TOKEN_FIELDS or leaf.endswith("_tokens"):
            _finite_number(value, path, nonnegative=True)

    visit(usage, field)


def _validate_tool_call(tool_call: Any, field: str) -> None:
    if not isinstance(tool_call, dict):
        raise CorruptResultsError(f"{field} must be an object")
    _required_string(tool_call.get("name"), f"{field}.name")
    arguments = tool_call.get("arguments")
    if not isinstance(arguments, dict):
        raise CorruptResultsError(f"{field}.arguments must be an object")
    requestor = tool_call.get("requestor", "assistant")
    if requestor not in {"assistant", "user"}:
        raise CorruptResultsError(f"{field}.requestor is invalid: {requestor!r}")


def _validate_tool_response(message: dict[str, Any], field: str) -> None:
    _required_string(message.get("name"), f"{field}.name")
    requestor = message.get("requestor", "assistant")
    if requestor not in {"assistant", "user"}:
        raise CorruptResultsError(f"{field}.requestor is invalid: {requestor!r}")
    error = message.get("error", False)
    if not isinstance(error, bool):
        raise CorruptResultsError(f"{field}.error must be boolean")


def _validate_message(message: Any, field: str) -> None:
    if not isinstance(message, dict):
        raise CorruptResultsError(f"{field} must be an object")
    role = message.get("role")
    if role not in {"system", "assistant", "user", "tool"}:
        raise CorruptResultsError(f"{field}.role is invalid: {role!r}")
    _validate_usage(message.get("usage"), f"{field}.usage")
    if message.get("cost") is not None:
        _finite_number(message["cost"], f"{field}.cost", nonnegative=True)

    tool_calls = message.get("tool_calls")
    if tool_calls is not None:
        if role not in {"assistant", "user"} or not isinstance(tool_calls, list):
            raise CorruptResultsError(f"{field}.tool_calls is invalid for role {role!r}")
        for index, tool_call in enumerate(tool_calls):
            _validate_tool_call(tool_call, f"{field}.tool_calls[{index}]")

    tool_messages = message.get("tool_messages")
    if tool_messages is not None:
        if role != "tool" or not isinstance(tool_messages, list):
            raise CorruptResultsError(f"{field}.tool_messages is invalid for role {role!r}")
        for index, tool_message in enumerate(tool_messages):
            if not isinstance(tool_message, dict):
                raise CorruptResultsError(f"{field}.tool_messages[{index}] must be an object")
            _validate_tool_response(tool_message, f"{field}.tool_messages[{index}]")
    elif role == "tool":
        _validate_tool_response(message, field)


def _validate_reward_info(reward_info: Any, field: str, termination: str) -> float:
    if not isinstance(reward_info, dict):
        raise CorruptResultsError(f"{field} must be an object")
    reward = _finite_number(reward_info.get("reward"), f"{field}.reward")
    if reward not in {0.0, 1.0}:
        raise CorruptResultsError(f"{field}.reward must be binary 0 or 1; got {reward!r}")

    breakdown = reward_info.get("reward_breakdown")
    if breakdown is not None:
        if not isinstance(breakdown, dict):
            raise CorruptResultsError(f"{field}.reward_breakdown must be an object or null")
        for key, value in breakdown.items():
            _required_string(key, f"{field}.reward_breakdown key")
            _finite_number(value, f"{field}.reward_breakdown.{key}")

    rubrics = reward_info.get("nl_rubrics")
    if rubrics is not None:
        if not isinstance(rubrics, list):
            raise CorruptResultsError(f"{field}.nl_rubrics must be a list or null")
        for index, rubric in enumerate(rubrics):
            if not isinstance(rubric, dict) or not isinstance(rubric.get("met"), bool):
                raise CorruptResultsError(f"{field}.nl_rubrics[{index}].met must be boolean")

    windows = reward_info.get("window_evaluations")
    expected_window_rubric_ids: set[str] | None = None
    last_current_rubrics: dict[str, dict[str, Any]] | None = None
    last_judge_decisions: dict[str, bool] | None = None
    if windows is not None:
        if not isinstance(windows, list):
            raise CorruptResultsError(f"{field}.window_evaluations must be a list or null")
        for index, window in enumerate(windows):
            window_field = f"{field}.window_evaluations[{index}]"
            if not isinstance(window, dict):
                raise CorruptResultsError(f"{window_field} must be an object")
            expected_rubrics = _required_int(
                window.get("expected_rubric_count"),
                f"{window_field}.expected_rubric_count",
                minimum=1,
            )
            validated_rubrics = _required_int(
                window.get("validated_rubric_count"),
                f"{window_field}.validated_rubric_count",
                minimum=1,
            )
            if validated_rubrics != expected_rubrics:
                raise CorruptResultsError(f"{window_field} has incomplete persisted rubric coverage")

            user_prompt = _required_string(window.get("user_prompt"), f"{window_field}.user_prompt")
            closing_tag = "</current_rubrics>"
            opening_tag = "<current_rubrics>"
            closing_index = user_prompt.rfind(closing_tag)
            opening_index = user_prompt.rfind(opening_tag, 0, closing_index)
            if opening_index < 0 or closing_index < 0:
                raise CorruptResultsError(f"{window_field}.user_prompt omits the current_rubrics block")
            current_rubrics_text = user_prompt[opening_index + len(opening_tag) : closing_index]
            try:
                current_rubrics = json.loads(current_rubrics_text)
            except json.JSONDecodeError as exc:
                raise CorruptResultsError(f"{window_field}.current_rubrics is not valid JSON: {exc}") from exc
            if not isinstance(current_rubrics, list) or not current_rubrics:
                raise CorruptResultsError(f"{window_field}.current_rubrics must be a nonempty list")

            current_rubric_ids: list[str] = []
            for rubric_index, rubric in enumerate(current_rubrics):
                rubric_field = f"{window_field}.current_rubrics[{rubric_index}]"
                if not isinstance(rubric, dict):
                    raise CorruptResultsError(f"{rubric_field} must be an object")
                current_rubric_ids.append(_required_string(rubric.get("rubric_idx"), f"{rubric_field}.rubric_idx"))
                if not isinstance(rubric.get("meetExpectation"), bool):
                    raise CorruptResultsError(f"{rubric_field}.meetExpectation must be boolean")
            duplicate_current_ids = sorted(
                rubric_id for rubric_id, count in Counter(current_rubric_ids).items() if count > 1
            )
            if duplicate_current_ids:
                raise CorruptResultsError(
                    f"{window_field}.current_rubrics has duplicate rubric_idx values: " f"{duplicate_current_ids}"
                )
            current_rubric_id_set = set(current_rubric_ids)
            if len(current_rubric_id_set) != expected_rubrics:
                raise CorruptResultsError(f"{window_field}.expected_rubric_count does not match current_rubrics")
            if expected_window_rubric_ids is not None and current_rubric_id_set != expected_window_rubric_ids:
                raise CorruptResultsError(f"{window_field}.current_rubrics differs from the other judge windows")
            expected_window_rubric_ids = current_rubric_id_set
            last_current_rubrics = {rubric["rubric_idx"]: rubric for rubric in current_rubrics}

            assistant_content = _required_string(
                window.get("assistant_message_content"),
                f"{window_field}.assistant_message_content",
            )
            try:
                repaired_content = _repair_json(assistant_content) if _repair_json is not None else assistant_content
                judge_rubrics = json.loads(repaired_content)
            except (TypeError, ValueError) as exc:
                raise CorruptResultsError(
                    f"{window_field}.assistant_message_content is not parseable rubric JSON: {exc}"
                ) from exc
            if not isinstance(judge_rubrics, list) or not judge_rubrics:
                raise CorruptResultsError(
                    f"{window_field}.assistant_message_content must yield a nonempty rubric list"
                )

            judge_rubric_ids: list[str] = []
            for rubric_index, rubric in enumerate(judge_rubrics):
                rubric_field = f"{window_field}.judge_rubrics[{rubric_index}]"
                if not isinstance(rubric, dict):
                    raise CorruptResultsError(f"{rubric_field} must be an object")
                judge_rubric_ids.append(_required_string(rubric.get("rubric_idx"), f"{rubric_field}.rubric_idx"))
                if not isinstance(rubric.get("meetExpectation"), bool):
                    raise CorruptResultsError(f"{rubric_field}.meetExpectation must be boolean")
            judge_rubric_counts = Counter(judge_rubric_ids)
            duplicate_judge_ids = sorted(rubric_id for rubric_id, count in judge_rubric_counts.items() if count > 1)
            missing_judge_ids = sorted(current_rubric_id_set - set(judge_rubric_ids))
            unexpected_judge_ids = sorted(set(judge_rubric_ids) - current_rubric_id_set)
            if duplicate_judge_ids or missing_judge_ids or unexpected_judge_ids:
                raise CorruptResultsError(
                    f"{window_field}.judge_rubrics does not exactly cover current_rubrics; "
                    f"missing={missing_judge_ids}, unexpected={unexpected_judge_ids}, "
                    f"duplicates={duplicate_judge_ids}"
                )
            if len(judge_rubrics) != validated_rubrics:
                raise CorruptResultsError(f"{window_field}.validated_rubric_count does not match judge output")
            last_judge_decisions = {rubric["rubric_idx"]: rubric["meetExpectation"] for rubric in judge_rubrics}

            usage = window.get("assistent_message_usage", window.get("assistant_message_usage"))
            _validate_usage(usage, f"{window_field}.usage")

    if termination in PREMATURE_TERMINATIONS:
        if windows is not None:
            raise CorruptResultsError(
                f"{field}.window_evaluations must be null for premature termination {termination!r}"
            )
    else:
        if not isinstance(rubrics, list) or not rubrics:
            raise CorruptResultsError(f"{field}.nl_rubrics must be nonempty for termination {termination!r}")
        if not isinstance(windows, list) or not windows:
            raise CorruptResultsError(f"{field}.window_evaluations must be nonempty for termination {termination!r}")
        if expected_window_rubric_ids is None or len(rubrics) != len(expected_window_rubric_ids):
            raise CorruptResultsError(f"{field}.nl_rubrics count does not match judge windows")
        if last_current_rubrics is None or last_judge_decisions is None:
            raise CorruptResultsError(f"{field} omits final judge decisions")
        ordered_ids = list(last_current_rubrics)
        for index, (rubric_id, final_rubric) in enumerate(
            zip(ordered_ids, rubrics)  # noqa: B905 - cluster runtime includes Python 3.8.
        ):
            current_rubric = last_current_rubrics[rubric_id]
            if final_rubric.get("nl_rubric") != current_rubric.get("rubric"):
                raise CorruptResultsError(f"{field}.nl_rubrics[{index}] does not match {rubric_id} text")
            if final_rubric["met"] is not last_judge_decisions[rubric_id]:
                raise CorruptResultsError(f"{field}.nl_rubrics[{index}].met does not match the final judge decision")
        expected_reward = 1.0 if all(rubric["met"] for rubric in rubrics) else 0.0
        if reward != expected_reward:
            raise CorruptResultsError(f"{field}.reward does not match the final rubric decisions")
    return reward


def validate_shard_data(
    data: dict[str, Any],
    expected_task_ids: Sequence[str],
    num_trials: int,
    base_seed: int,
    *,
    expected_domain: str | None = None,
    expected_max_steps: int | None = None,
    expected_max_errors: int | None = None,
    expected_agent_model: str | None = None,
    expected_user_model: str | None = None,
) -> dict[str, Any]:
    """Validate a complete or resumable Vita result checkpoint."""
    task_ids = list(expected_task_ids)
    if not task_ids:
        raise CorruptResultsError("expected task IDs must not be empty")
    if len(set(task_ids)) != len(task_ids):
        raise CorruptResultsError("expected task IDs contain duplicates")
    if num_trials <= 0:
        raise CorruptResultsError("num_trials must be positive")

    embedded_task_ids = _task_ids_from_results(data.get("tasks"), "tasks")
    if set(embedded_task_ids) != set(task_ids) or len(embedded_task_ids) != len(task_ids):
        missing = sorted(set(task_ids) - set(embedded_task_ids))
        unexpected = sorted(set(embedded_task_ids) - set(task_ids))
        raise CorruptResultsError(
            f"embedded tasks do not match shard; missing={missing[:10]}, unexpected={unexpected[:10]}"
        )

    info = data.get("info")
    if not isinstance(info, dict):
        raise CorruptResultsError("info must be an object")
    stored_trials = _required_int(info.get("num_trials"), "info.num_trials", minimum=1)
    if stored_trials != num_trials:
        raise CorruptResultsError(f"info.num_trials={stored_trials} does not match expected {num_trials}")
    stored_base_seed = info.get("seed")
    if stored_base_seed is not None:
        stored_base_seed = _required_int(stored_base_seed, "info.seed")
        if stored_base_seed != base_seed:
            raise CorruptResultsError(f"info.seed={stored_base_seed} does not match expected {base_seed}")
    if expected_max_steps is not None:
        stored_max_steps = _required_int(info.get("max_steps"), "info.max_steps", minimum=1)
        if stored_max_steps != expected_max_steps:
            raise CorruptResultsError(
                f"info.max_steps={stored_max_steps} does not match expected {expected_max_steps}"
            )
    if expected_max_errors is not None:
        stored_max_errors = _required_int(info.get("max_errors"), "info.max_errors", minimum=1)
        if stored_max_errors != expected_max_errors:
            raise CorruptResultsError(
                f"info.max_errors={stored_max_errors} does not match expected {expected_max_errors}"
            )
    if expected_domain is not None:
        environment_info = info.get("environment_info")
        if not isinstance(environment_info, dict):
            raise CorruptResultsError("info.environment_info must be an object")
        stored_domain = _required_string(environment_info.get("domain_name"), "info.environment_info.domain_name")
        if stored_domain != expected_domain:
            raise CorruptResultsError(f"result domain {stored_domain!r} does not match expected {expected_domain!r}")
    for participant, expected_model in (
        ("agent", expected_agent_model),
        ("user", expected_user_model),
    ):
        if expected_model is None:
            continue
        participant_info = info.get(f"{participant}_info")
        if not isinstance(participant_info, dict):
            raise CorruptResultsError(f"info.{participant}_info must be an object")
        stored_model = _required_string(participant_info.get("llm"), f"info.{participant}_info.llm")
        if stored_model != expected_model:
            raise CorruptResultsError(
                f"result {participant} model {stored_model!r} does not match expected {expected_model!r}"
            )

    simulations = data.get("simulations")
    if not isinstance(simulations, list):
        raise CorruptResultsError("simulations must be a list")
    expected_pairs = {(task_id, trial) for task_id in task_ids for trial in range(num_trials)}
    expected_seeds = _trial_seeds(base_seed, num_trials)
    seen_pairs: set[tuple[str, int]] = set()
    simulation_ids: set[str] = set()
    represented_seed_count = 0

    for index, simulation in enumerate(simulations):
        field = f"simulations[{index}]"
        if not isinstance(simulation, dict):
            raise CorruptResultsError(f"{field} must be an object")
        simulation_id = _required_string(simulation.get("id"), f"{field}.id")
        if simulation_id in simulation_ids:
            raise CorruptResultsError(f"duplicate simulation id: {simulation_id}")
        simulation_ids.add(simulation_id)

        task_id = _required_string(simulation.get("task_id"), f"{field}.task_id")
        trial = _required_int(simulation.get("trial"), f"{field}.trial", minimum=0)
        pair = (task_id, trial)
        if pair not in expected_pairs:
            raise CorruptResultsError(f"unexpected task/trial pair: {pair}")
        if pair in seen_pairs:
            raise CorruptResultsError(f"duplicate task/trial pair: {pair}")
        seen_pairs.add(pair)

        seed = simulation.get("seed")
        if seed is not None:
            represented_seed_count += 1
            seed = _required_int(seed, f"{field}.seed")
            if seed != expected_seeds[trial]:
                raise CorruptResultsError(
                    f"{field}.seed={seed} does not match trial {trial} seed {expected_seeds[trial]}"
                )

        duration = _finite_number(simulation.get("duration"), f"{field}.duration", nonnegative=True)
        if duration < 0:
            raise CorruptResultsError(f"{field}.duration must be nonnegative")
        for cost_field in ("agent_cost", "user_cost"):
            if simulation.get(cost_field) is not None:
                _finite_number(simulation[cost_field], f"{field}.{cost_field}", nonnegative=True)

        termination = simulation.get("termination_reason")
        if termination not in VALID_TERMINATIONS:
            raise CorruptResultsError(f"{field}.termination_reason is invalid: {termination!r}")
        reward = _validate_reward_info(simulation.get("reward_info"), f"{field}.reward_info", termination)
        if termination in PREMATURE_TERMINATIONS and reward != 0.0:
            raise CorruptResultsError(f"{field} has premature termination {termination!r} but reward {reward}")

        messages = simulation.get("messages")
        if not isinstance(messages, list):
            raise CorruptResultsError(f"{field}.messages must be a list")
        for message_index, message in enumerate(messages):
            _validate_message(message, f"{field}.messages[{message_index}]")

    if represented_seed_count not in {0, len(simulations)}:
        raise CorruptResultsError("simulation seed is present for only part of the checkpoint")

    missing_pairs = sorted(expected_pairs - seen_pairs)
    return {
        "complete": not missing_pairs,
        "expected_tasks": len(task_ids),
        "expected_simulations": len(expected_pairs),
        "completed_simulations": len(seen_pairs),
        "missing_simulations": len(missing_pairs),
        "missing_pairs": [{"task_id": task_id, "trial": trial} for task_id, trial in missing_pairs],
        "trial_seeds": expected_seeds,
        "seed_validation": "validated" if represented_seed_count else "not_represented",
    }


def _canonical_arguments(arguments: Any) -> str:
    try:
        return json.dumps(arguments, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError):
        return repr(arguments)


def _tool_responses(message: dict[str, Any]) -> Iterable[dict[str, Any]]:
    if message.get("role") != "tool":
        return ()
    nested = message.get("tool_messages")
    if isinstance(nested, list):
        return nested
    return (message,)


def _empty_tool_accumulator() -> dict[str, Any]:
    return {
        "calls": 0,
        "calls_by_name": Counter(),
        "tool_call_messages": 0,
        "multitool_messages": 0,
        "responses": 0,
        "responses_by_name": Counter(),
        "errors": 0,
        "errors_by_name": Counter(),
        "unanswered": 0,
        "orphan_responses": 0,
        "repeated_calls": 0,
        "repeated_signature_groups": 0,
        "repeated_calls_by_name": Counter(),
        "simulations_with_calls": 0,
        "simulations_with_multitool_calls": 0,
        "simulations_with_errors": 0,
        "simulations_with_unanswered_calls": 0,
        "simulations_with_repeated_calls": 0,
    }


def _tool_metrics(simulations: Sequence[dict[str, Any]]) -> dict[str, Any]:
    totals = {"assistant": _empty_tool_accumulator(), "user": _empty_tool_accumulator()}
    for simulation in simulations:
        calls = {"assistant": [], "user": []}
        responses = {"assistant": [], "user": []}
        simulation_has_multitool = {"assistant": False, "user": False}
        for message in simulation["messages"]:
            message_calls = {"assistant": [], "user": []}
            for tool_call in message.get("tool_calls") or []:
                requestor = tool_call.get("requestor", "assistant")
                calls[requestor].append(tool_call)
                message_calls[requestor].append(tool_call)
            for requestor in ("assistant", "user"):
                if message_calls[requestor]:
                    totals[requestor]["tool_call_messages"] += 1
                if len(message_calls[requestor]) > 1:
                    totals[requestor]["multitool_messages"] += 1
                    simulation_has_multitool[requestor] = True
            for response in _tool_responses(message):
                requestor = response.get("requestor", "assistant")
                responses[requestor].append(response)

        for requestor in ("assistant", "user"):
            accumulator = totals[requestor]
            request_calls = calls[requestor]
            tool_responses = responses[requestor]
            accumulator["calls"] += len(request_calls)
            accumulator["responses"] += len(tool_responses)
            if request_calls:
                accumulator["simulations_with_calls"] += 1
            if simulation_has_multitool[requestor]:
                accumulator["simulations_with_multitool_calls"] += 1

            for tool_call in request_calls:
                accumulator["calls_by_name"][tool_call["name"]] += 1
            for response in tool_responses:
                accumulator["responses_by_name"][response["name"]] += 1
                if response.get("error", False):
                    accumulator["errors"] += 1
                    accumulator["errors_by_name"][response["name"]] += 1
            if any(response.get("error", False) for response in tool_responses):
                accumulator["simulations_with_errors"] += 1

            call_ids = Counter(str(tool_call.get("id", "")) for tool_call in request_calls)
            response_ids = Counter(str(response.get("id", "")) for response in tool_responses)
            unanswered = sum((call_ids - response_ids).values())
            orphan_responses = sum((response_ids - call_ids).values())
            accumulator["unanswered"] += unanswered
            accumulator["orphan_responses"] += orphan_responses
            if unanswered:
                accumulator["simulations_with_unanswered_calls"] += 1

            signatures = Counter(
                (tool_call["name"], _canonical_arguments(tool_call["arguments"])) for tool_call in request_calls
            )
            repeated_calls = sum(count - 1 for count in signatures.values() if count > 1)
            repeated_groups = sum(1 for count in signatures.values() if count > 1)
            accumulator["repeated_calls"] += repeated_calls
            accumulator["repeated_signature_groups"] += repeated_groups
            if repeated_calls:
                accumulator["simulations_with_repeated_calls"] += 1
            for (name, _), count in signatures.items():
                if count > 1:
                    accumulator["repeated_calls_by_name"][name] += count - 1

    output: dict[str, Any] = {}
    for requestor, accumulator in totals.items():
        output[requestor] = {
            key: dict(sorted(value.items())) if isinstance(value, Counter) else value
            for key, value in accumulator.items()
        }
        output[requestor].update(
            {
                "calls_per_simulation": accumulator["calls"] / len(simulations),
                "response_error_rate": (
                    accumulator["errors"] / accumulator["responses"] if accumulator["responses"] else None
                ),
                "unanswered_call_rate": (
                    accumulator["unanswered"] / accumulator["calls"] if accumulator["calls"] else None
                ),
                "multitool_message_rate": (
                    accumulator["multitool_messages"] / accumulator["tool_call_messages"]
                    if accumulator["tool_call_messages"]
                    else None
                ),
                "simulations_with_multitool_calls_rate": (
                    accumulator["simulations_with_multitool_calls"] / len(simulations)
                ),
            }
        )
    return output


def _flatten_usage(usage: dict[str, Any], prefix: str = "") -> dict[str, float]:
    flattened: dict[str, float] = {}
    for key, value in usage.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flattened.update(_flatten_usage(value, path))
        elif _is_number(value) and (key in TOKEN_FIELDS or key.endswith("_tokens")):
            flattened[path] = float(value)
    return flattened


def _usage_summary(usages: Sequence[dict[str, Any] | None], unit_name: str) -> dict[str, Any]:
    covered = [usage for usage in usages if isinstance(usage, dict)]
    totals: defaultdict[str, float] = defaultdict(float)
    for usage in covered:
        for key, value in _flatten_usage(usage).items():
            totals[key] += value
    return {
        f"{unit_name}_total": len(usages),
        f"{unit_name}_with_usage": len(covered),
        "coverage_rate": len(covered) / len(usages) if usages else None,
        "totals": {key: int(value) if value.is_integer() else value for key, value in sorted(totals.items())},
    }


def _token_metrics(simulations: Sequence[dict[str, Any]]) -> dict[str, Any]:
    agent_usages: list[dict[str, Any] | None] = []
    user_usages: list[dict[str, Any] | None] = []
    evaluator_usages: list[dict[str, Any] | None] = []
    for simulation in simulations:
        for message in simulation["messages"]:
            if message["role"] == "assistant":
                agent_usages.append(message.get("usage"))
            elif message["role"] == "user":
                user_usages.append(message.get("usage"))
        windows = simulation["reward_info"].get("window_evaluations") or []
        for window in windows:
            evaluator_usages.append(window.get("assistent_message_usage", window.get("assistant_message_usage")))
    return {
        "agent": _usage_summary(agent_usages, "messages"),
        "user": _usage_summary(user_usages, "messages"),
        "evaluator": _usage_summary(evaluator_usages, "windows"),
    }


def _cost_summary(values: Sequence[Any]) -> dict[str, Any]:
    present = [float(value) for value in values if value is not None]
    return {
        "simulations_total": len(values),
        "simulations_with_cost": len(present),
        "coverage_rate": len(present) / len(values) if values else None,
        "total": sum(present) if present else None,
        "mean_when_present": statistics.fmean(present) if present else None,
    }


def _cost_metrics(simulations: Sequence[dict[str, Any]]) -> dict[str, Any]:
    agent = [simulation.get("agent_cost") for simulation in simulations]
    user = [simulation.get("user_cost") for simulation in simulations]
    both_present = [
        simulation
        for simulation in simulations
        if simulation.get("agent_cost") is not None and simulation.get("user_cost") is not None
    ]
    return {
        "agent": _cost_summary(agent),
        "user": _cost_summary(user),
        "combined": {
            "simulations_total": len(simulations),
            "simulations_with_both_costs": len(both_present),
            "coverage_rate": len(both_present) / len(simulations) if simulations else None,
            "total": (
                sum(float(simulation["agent_cost"]) + float(simulation["user_cost"]) for simulation in both_present)
                if both_present
                else None
            ),
        },
    }


def _percentile(values: Sequence[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _parse_time(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    for format_string in ("%Y%m%d_%H%M%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(value, format_string).replace(tzinfo=timezone.utc)
        except ValueError:
            pass
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _duration_metrics(simulations: Sequence[dict[str, Any]]) -> dict[str, Any]:
    durations = [float(simulation["duration"]) for simulation in simulations]
    starts = [_parse_time(simulation.get("start_time")) for simulation in simulations]
    ends = [_parse_time(simulation.get("end_time")) for simulation in simulations]
    valid_starts = [value for value in starts if value is not None]
    valid_ends = [value for value in ends if value is not None]
    wall_span = None
    if len(valid_starts) == len(simulations) and len(valid_ends) == len(simulations):
        wall_span = (max(valid_ends) - min(valid_starts)).total_seconds()
    return {
        "mean_seconds": statistics.fmean(durations),
        "p50_seconds": _percentile(durations, 0.50),
        "p95_seconds": _percentile(durations, 0.95),
        "max_seconds": max(durations),
        "sum_seconds": sum(durations),
        "wall_span_seconds": wall_span,
        "timestamp_coverage_rate": min(len(valid_starts), len(valid_ends)) / len(simulations),
    }


def _message_metrics(simulations: Sequence[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(message["role"] for simulation in simulations for message in simulation["messages"])
    per_simulation = [len(simulation["messages"]) for simulation in simulations]
    return {
        "total": sum(counts.values()),
        "by_role": dict(sorted(counts.items())),
        "mean_per_simulation": statistics.fmean(per_simulation),
        "p50_per_simulation": _percentile([float(value) for value in per_simulation], 0.50),
        "p95_per_simulation": _percentile([float(value) for value in per_simulation], 0.95),
        "max_per_simulation": max(per_simulation),
    }


def _reward_detail_metrics(simulations: Sequence[dict[str, Any]]) -> dict[str, Any]:
    breakdown_values: defaultdict[str, list[float]] = defaultdict(list)
    simulations_with_rubrics = 0
    rubric_checks = 0
    rubrics_met = 0
    simulations_with_windows = 0
    evaluator_windows = 0
    for simulation in simulations:
        reward_info = simulation["reward_info"]
        for key, value in (reward_info.get("reward_breakdown") or {}).items():
            breakdown_values[key].append(float(value))
        rubrics = reward_info.get("nl_rubrics")
        if rubrics is not None:
            simulations_with_rubrics += 1
            rubric_checks += len(rubrics)
            rubrics_met += sum(1 for rubric in rubrics if rubric["met"])
        windows = reward_info.get("window_evaluations")
        if windows is not None:
            simulations_with_windows += 1
            evaluator_windows += len(windows)

    breakdown = {}
    for key, values in sorted(breakdown_values.items()):
        breakdown[key] = {
            "simulations_with_value": len(values),
            "coverage_rate": len(values) / len(simulations),
            "mean_when_present": statistics.fmean(values),
            "sum": sum(values),
            "min": min(values),
            "max": max(values),
        }
    return {
        "reward_breakdown": breakdown,
        "rubrics": {
            "simulations_with_rubrics": simulations_with_rubrics,
            "simulation_coverage_rate": simulations_with_rubrics / len(simulations),
            "checks": rubric_checks,
            "met": rubrics_met,
            "met_rate": rubrics_met / rubric_checks if rubric_checks else None,
        },
        "evaluator_windows": {
            "simulations_with_windows": simulations_with_windows,
            "simulation_coverage_rate": simulations_with_windows / len(simulations),
            "windows": evaluator_windows,
        },
    }


def _score_metrics(simulations: Sequence[dict[str, Any]], num_trials: int) -> dict[str, Any]:
    rewards_by_task: defaultdict[str, list[float]] = defaultdict(list)
    for simulation in simulations:
        rewards_by_task[simulation["task_id"]].append(float(simulation["reward_info"]["reward"]))
    if not rewards_by_task:
        raise CorruptResultsError("cannot score an empty simulation set")
    if any(len(rewards) != num_trials for rewards in rewards_by_task.values()):
        raise CorruptResultsError("score input is not rectangular by task and trial")

    success_counts = [int(sum(rewards)) for rewards in rewards_by_task.values()]
    average = statistics.fmean(reward for rewards in rewards_by_task.values() for reward in rewards)
    pass_at_1 = statistics.fmean(count / num_trials for count in success_counts)
    pass_at_all = statistics.fmean(1.0 if count > 0 else 0.0 for count in success_counts)
    pass_pow_all = statistics.fmean(1.0 if count == num_trials else 0.0 for count in success_counts)
    return {
        f"avg_at_{num_trials}": average,
        f"avg_at_{num_trials}_percent": average * 100.0,
        "pass_at_1": pass_at_1,
        "pass_at_1_percent": pass_at_1 * 100.0,
        f"pass_at_{num_trials}_any": pass_at_all,
        f"pass_at_{num_trials}_any_percent": pass_at_all * 100.0,
        f"pass_pow_{num_trials}_all": pass_pow_all,
        f"pass_pow_{num_trials}_all_percent": pass_pow_all * 100.0,
        "successful_simulations": sum(success_counts),
        "success_rate": average,
        "tasks_with_any_success": sum(1 for count in success_counts if count > 0),
        "tasks_with_all_successes": sum(1 for count in success_counts if count == num_trials),
        "success_count_histogram": {str(count): success_counts.count(count) for count in range(num_trials + 1)},
    }


def _metrics(simulations: Sequence[dict[str, Any]], num_trials: int) -> dict[str, Any]:
    task_count = len({simulation["task_id"] for simulation in simulations})
    terminations = Counter(simulation["termination_reason"] for simulation in simulations)
    termination_counts = {reason: terminations.get(reason, 0) for reason in sorted(VALID_TERMINATIONS)}
    return {
        "task_count": task_count,
        "simulation_count": len(simulations),
        "scores": _score_metrics(simulations, num_trials),
        "terminations": {
            "counts": termination_counts,
            "rates": {reason: count / len(simulations) for reason, count in termination_counts.items()},
        },
        "tools": _tool_metrics(simulations),
        "messages": _message_metrics(simulations),
        "tokens": _token_metrics(simulations),
        "costs": _cost_metrics(simulations),
        "duration": _duration_metrics(simulations),
        "reward_details": _reward_detail_metrics(simulations),
    }


def _manifest_protocol(manifest: dict[str, Any]) -> dict[str, Any]:
    protocol = manifest.get("protocol")
    if not isinstance(protocol, dict):
        raise CorruptResultsError("manifest.protocol must be an object")
    num_trials = _required_int(protocol.get("num_trials"), "manifest.protocol.num_trials", minimum=1)
    if num_trials != EXPECTED_TRIALS:
        raise CorruptResultsError(f"official-like run requires {EXPECTED_TRIALS} trials")
    seed = _required_int(protocol.get("seed"), "manifest.protocol.seed")
    if seed != EXPECTED_BASE_SEED:
        raise CorruptResultsError(f"official-like run requires seed {EXPECTED_BASE_SEED}")
    shard_size = _required_int(protocol.get("shard_size"), "manifest.protocol.shard_size", minimum=1)
    if shard_size != EXPECTED_SHARD_SIZE:
        raise CorruptResultsError(f"official-like run requires shard_size {EXPECTED_SHARD_SIZE}")
    if _finite_number(protocol.get("temperature"), "manifest.protocol.temperature") != 0.0:
        raise CorruptResultsError("official-like run requires temperature 0")
    if protocol.get("evaluation_type") != "trajectory":
        raise CorruptResultsError("official-like run requires trajectory evaluation")
    if protocol.get("enable_think") is not True:
        raise CorruptResultsError("official-like Qwen3.5 run requires enable_think=true")
    if protocol.get("language") != "chinese":
        raise CorruptResultsError("official-like run requires language chinese")
    max_steps = _required_int(protocol.get("max_steps"), "manifest.protocol.max_steps", minimum=1)
    if max_steps != 300:
        raise CorruptResultsError("official-like run requires max_steps 300")
    max_errors = _required_int(protocol.get("max_errors"), "manifest.protocol.max_errors", minimum=1)
    if max_errors != 10:
        raise CorruptResultsError("official-like run requires max_errors 10")
    expected_counts = {
        "suite_count": 4,
        "task_count": 400,
        "trajectory_count": 1600,
        "shard_count": 40,
    }
    for key, expected_value in expected_counts.items():
        if _required_int(protocol.get(key), f"manifest.protocol.{key}", minimum=1) != expected_value:
            raise CorruptResultsError(f"manifest.protocol.{key} must be {expected_value}")
    max_concurrency = _required_int(
        protocol.get("max_concurrency"),
        "manifest.protocol.max_concurrency",
        minimum=1,
    )
    schema_version = _required_int(manifest.get("schema_version"), "manifest.schema_version", minimum=1)
    if schema_version == 2:
        if max_concurrency != 8:
            raise CorruptResultsError("schema-2 official-like run requires max_concurrency 8")
    elif schema_version in PARALLEL_MANIFEST_SCHEMAS:
        execution = manifest.get("execution")
        if not isinstance(execution, dict):
            raise CorruptResultsError("manifest.execution must be an object")
        suite_parallelism = _required_int(
            execution.get("suite_parallelism"),
            "manifest.execution.suite_parallelism",
            minimum=1,
        )
        expected_concurrency = {1: 8, 4: 4}.get(suite_parallelism)
        if expected_concurrency is None:
            raise CorruptResultsError("parallel run requires suite_parallelism 1 or 4")
        if max_concurrency != expected_concurrency:
            raise CorruptResultsError("manifest protocol concurrency does not match suite parallelism")
        per_suite = _required_int(
            execution.get("per_suite_max_concurrency"),
            "manifest.execution.per_suite_max_concurrency",
            minimum=1,
        )
        aggregate = _required_int(
            execution.get("aggregate_max_concurrency"),
            "manifest.execution.aggregate_max_concurrency",
            minimum=1,
        )
        remote_limit = _required_int(
            execution.get("remote_api_concurrency_limit"),
            "manifest.execution.remote_api_concurrency_limit",
            minimum=1,
        )
        if per_suite != max_concurrency or aggregate != suite_parallelism * per_suite:
            raise CorruptResultsError("manifest execution concurrency is inconsistent")
        if remote_limit > aggregate:
            raise CorruptResultsError("remote API concurrency limit cannot exceed aggregate concurrency")
        if execution.get("api_response_journal_schema") != ("vita-llm-response-journal/v1"):
            raise CorruptResultsError("manifest has an unsupported API journal schema")
        journaled_models = execution.get("journaled_models")
        expected_models = [
            manifest.get("agent", {}).get("requested"),
            manifest.get("user", {}).get("requested"),
            manifest.get("evaluator", {}).get("requested"),
        ]
        if journaled_models != expected_models or any(
            not isinstance(model, str) or not model for model in expected_models
        ):
            raise CorruptResultsError("manifest journaled models do not match agent/user/evaluator")
        role_backend = execution.get("role_backend", "remote")
        expected_gpu_binding = {
            "remote": "one_distinct_visible_device_selector_per_suite",
            "local": "four_agent_plus_four_role_replicas",
        }.get(role_backend)
        if expected_gpu_binding is None:
            raise CorruptResultsError("manifest has an unsupported role backend")
        if execution.get("gpu_binding") != expected_gpu_binding:
            raise CorruptResultsError("manifest has an unsupported GPU binding policy")
        expected_workers = [
            {
                "suite": name,
                "gpu_slot": index if suite_parallelism == 4 else 0,
                "port": 30000 + (index if suite_parallelism == 4 else 0),
            }
            for index, name in enumerate(SUITE_NAMES)
        ]
        if execution.get("suite_workers") != expected_workers:
            raise CorruptResultsError("manifest suite worker mapping is invalid")
        agent = manifest.get("agent")
        if not isinstance(agent, dict) or agent.get("sglang_instance_count") != (suite_parallelism):
            raise CorruptResultsError("manifest SGLang instance count is invalid")
        if role_backend == "local":
            local_role_serving = manifest.get("local_role_serving")
            if not isinstance(local_role_serving, dict):
                raise CorruptResultsError("local manifest omits role serving metadata")
            if local_role_serving.get("user_instance_count") != 2:
                raise CorruptResultsError("local manifest user instance count is invalid")
            if local_role_serving.get("evaluator_instance_count") != 2:
                raise CorruptResultsError("local manifest evaluator instance count is invalid")
    elif schema_version == 6:
        execution = manifest.get("execution")
        if not isinstance(execution, dict):
            raise CorruptResultsError("manifest.execution must be an object")
        role_backend = execution.get("role_backend")
        if role_backend not in {"local", "remote"}:
            raise CorruptResultsError("manifest has an unsupported role backend")
        suite_parallelism = _required_int(
            execution.get("suite_parallelism"),
            "manifest.execution.suite_parallelism",
            minimum=1,
        )
        if suite_parallelism not in {1, 4}:
            raise CorruptResultsError("schema-6 run requires suite_parallelism 1 or 4")
        if role_backend == "local" and suite_parallelism != 4:
            raise CorruptResultsError("schema-6 local run requires suite_parallelism 4")
        expected_scheduler = "dynamic_shard_queue" if role_backend == "local" else "suite_static"
        if execution.get("scheduler") != expected_scheduler:
            raise CorruptResultsError("manifest scheduler does not match role backend")
        run_mode = execution.get("run_mode")
        if run_mode != "full" or execution.get("selected_shard_index") is not None:
            raise CorruptResultsError("only schema-6 full runs can produce an official summary")
        worker_count = _required_int(
            execution.get("worker_count"),
            "manifest.execution.worker_count",
            minimum=1,
        )
        expected_worker_count = 4 if role_backend == "local" else suite_parallelism
        if worker_count != expected_worker_count:
            raise CorruptResultsError("manifest worker count is invalid")
        expected_concurrency = 4 if worker_count == 4 else 8
        if max_concurrency != expected_concurrency:
            raise CorruptResultsError("manifest protocol concurrency does not match worker count")
        per_worker = _required_int(
            execution.get("per_worker_max_concurrency"),
            "manifest.execution.per_worker_max_concurrency",
            minimum=1,
        )
        aggregate = _required_int(
            execution.get("aggregate_max_concurrency"),
            "manifest.execution.aggregate_max_concurrency",
            minimum=1,
        )
        remote_limit = _required_int(
            execution.get("remote_api_concurrency_limit"),
            "manifest.execution.remote_api_concurrency_limit",
            minimum=1,
        )
        if per_worker != max_concurrency or aggregate != worker_count * per_worker:
            raise CorruptResultsError("manifest execution concurrency is inconsistent")
        if role_backend == "local" and aggregate != 16:
            raise CorruptResultsError("schema-6 local trajectory concurrency must be 16")
        if remote_limit > aggregate:
            raise CorruptResultsError("remote API concurrency limit cannot exceed aggregate concurrency")
        if execution.get("api_response_journal_schema") != ("vita-llm-response-journal/v1"):
            raise CorruptResultsError("manifest has an unsupported API journal schema")
        expected_models = [
            manifest.get("agent", {}).get("requested"),
            manifest.get("user", {}).get("requested"),
            manifest.get("evaluator", {}).get("requested"),
        ]
        if execution.get("journaled_models") != expected_models or any(
            not isinstance(model, str) or not model for model in expected_models
        ):
            raise CorruptResultsError("manifest journaled models do not match agent/user/evaluator")

        agent = manifest.get("agent")
        if not isinstance(agent, dict):
            raise CorruptResultsError("manifest.agent must be an object")
        if role_backend == "local":
            if execution.get("gpu_binding") != "eight_single_gpu_role_instances":
                raise CorruptResultsError("local manifest has an unsupported GPU binding policy")
            expected_queue_lock = {
                "primitive": "fcntl.flock",
                "retry_errno_names": ["EACCES", "EAGAIN"],
                "retry_interval_seconds": 0.05,
                "retry_timeout_seconds": 120.0,
            }
            if execution.get("queue_lock") != expected_queue_lock:
                raise CorruptResultsError("local manifest has an unsupported queue lock policy")
            expected_workers = [
                {
                    "worker_id": worker_id,
                    "agent_instance": worker_id % 2,
                    "agent_gpu_slot": worker_id % 2,
                    "agent_port": 30000 + worker_id % 2,
                    "user_instance": worker_id % 2,
                    "user_gpu_slot": 2 + worker_id % 2,
                    "user_port": 31000 + worker_id % 2,
                    "evaluator_instance": worker_id,
                    "evaluator_gpu_slot": 4 + worker_id,
                    "evaluator_port": 31002 + worker_id,
                }
                for worker_id in range(4)
            ]
            if execution.get("worker_endpoints") != expected_workers:
                raise CorruptResultsError("manifest worker endpoint mapping is invalid")
            if "suite_workers" in execution:
                raise CorruptResultsError("dynamic local manifests must not bind suites to workers")
            if (
                agent.get("sglang_instance_count") != 2
                or agent.get("sglang_max_running_requests") != 16
                or agent.get("dtype") != "bfloat16"
            ):
                raise CorruptResultsError("manifest local agent serving is invalid")
            local_role_serving = manifest.get("local_role_serving")
            if not isinstance(local_role_serving, dict):
                raise CorruptResultsError("local manifest omits role serving metadata")
            local_expected = {
                "agent_instance_count": 2,
                "user_instance_count": 2,
                "evaluator_instance_count": 4,
                "total_instance_count": 8,
                "context_length": 32768,
                "dtype": "bfloat16",
                "user_sglang_max_running_requests": 8,
                "evaluator_sglang_max_running_requests": 4,
                "agent_ports": [30000, 30001],
                "user_ports": [31000, 31001],
                "evaluator_ports": [31002, 31003, 31004, 31005],
            }
            for key, expected_value in local_expected.items():
                if local_role_serving.get(key) != expected_value:
                    raise CorruptResultsError(f"local manifest serving metadata is invalid for {key}")
            gates = manifest.get("gate")
            expected_gate_workers = {name: index for index, name in enumerate(SUITE_NAMES)}
            if (
                not isinstance(gates, list)
                or {gate.get("name"): gate.get("worker_id") for gate in gates if isinstance(gate, dict)}
                != expected_gate_workers
            ):
                raise CorruptResultsError("local manifest gate worker mapping is invalid")
        else:
            if execution.get("gpu_binding") != ("one_distinct_visible_device_selector_per_suite"):
                raise CorruptResultsError("remote manifest has an unsupported GPU binding policy")
            expected_workers = [
                {
                    "suite": name,
                    "gpu_slot": index if suite_parallelism == 4 else 0,
                    "port": 30000 + (index if suite_parallelism == 4 else 0),
                }
                for index, name in enumerate(SUITE_NAMES)
            ]
            if execution.get("suite_workers") != expected_workers:
                raise CorruptResultsError("manifest suite worker mapping is invalid")
            if agent.get("sglang_instance_count") != suite_parallelism:
                raise CorruptResultsError("manifest SGLang instance count is invalid")
    else:
        raise CorruptResultsError("unsupported manifest schema version")
    return protocol


def _validate_manifest_metadata(manifest: dict[str, Any]) -> None:
    schema_version = _required_int(manifest.get("schema_version"), "manifest.schema_version", minimum=1)
    if schema_version not in SUPPORTED_MANIFEST_SCHEMAS:
        raise CorruptResultsError("unsupported manifest schema version")
    if _required_string(manifest.get("benchmark"), "manifest.benchmark") != "VITA-Bench":
        raise CorruptResultsError("manifest benchmark must be VITA-Bench")
    model_name = _required_string(manifest.get("model"), "manifest.model")
    for section in ("agent", "user", "evaluator", "official_reference"):
        if not isinstance(manifest.get(section), dict):
            raise CorruptResultsError(f"manifest.{section} must be an object")
    for field in ("requested", "served", "response_model_family"):
        if _required_string(manifest["agent"].get(field), f"manifest.agent.{field}") != model_name:
            raise CorruptResultsError(f"manifest.agent.{field} must match manifest.model")
    _required_string(manifest["user"].get("model"), "manifest.user.model")
    _required_string(manifest["evaluator"].get("model"), "manifest.evaluator.model")
    _required_string(manifest["evaluator"].get("provider"), "manifest.evaluator.provider")
    if not isinstance(manifest["evaluator"].get("official_comparable"), bool):
        raise CorruptResultsError("manifest.evaluator.official_comparable must be boolean")
    _required_string(
        manifest["evaluator"].get("selection_reason"),
        "manifest.evaluator.selection_reason",
    )
    reference = manifest["official_reference"]
    if _required_string(reference.get("model"), "manifest.official_reference.model") != model_name:
        raise CorruptResultsError("official reference model must match manifest.model")
    score_value = reference.get("score")
    available = reference.get("available", score_value is not None)
    if not isinstance(available, bool):
        raise CorruptResultsError("manifest.official_reference.available must be boolean")
    if available:
        official_score = _finite_number(
            score_value,
            "manifest.official_reference.score",
        )
    elif score_value is not None:
        raise CorruptResultsError("an unavailable official reference must have a null score")
    else:
        official_score = None
    if model_name == "Qwen3.5-4B" and (not available or official_score != 22.0):
        raise CorruptResultsError("Qwen3.5-4B official VITA-Bench reference must be 22")
    if not isinstance(reference.get("strictly_comparable"), bool):
        raise CorruptResultsError("manifest.official_reference.strictly_comparable must be boolean")
    if not available and manifest["evaluator"]["official_comparable"]:
        raise CorruptResultsError("a run without an official reference cannot be comparable")
    official_metric = _required_string(
        reference.get("metric"),
        "manifest.official_reference.metric",
    )
    if official_metric != "four-suite macro Avg@4":
        raise CorruptResultsError("official reference metric is invalid")
    _required_string(
        reference.get("source_url"),
        "manifest.official_reference.source_url",
    )


def _manifest_suites(manifest: dict[str, Any], artifact_dir: Path) -> tuple[list[dict[str, Any]], set[str]]:
    suites = manifest.get("suites")
    if not isinstance(suites, list):
        raise CorruptResultsError("manifest.suites must be a list")
    names = [suite.get("name") if isinstance(suite, dict) else None for suite in suites]
    if (
        len(suites) != len(SUITE_NAMES)
        or any(not isinstance(name, str) for name in names)
        or set(names) != set(SUITE_NAMES)
    ):
        raise CorruptResultsError(f"manifest must contain exactly these suites: {SUITE_NAMES}")

    normalized_suites: list[dict[str, Any]] = []
    all_task_ids: set[str] = set()
    all_result_files: set[Path] = set()
    for expected_name in SUITE_NAMES:
        suite = next(value for value in suites if value["name"] == expected_name)
        domain = _required_string(suite.get("domain"), f"manifest.suites.{expected_name}.domain")
        task_set = _required_string(suite.get("task_set"), f"manifest.suites.{expected_name}.task_set")
        if task_set != expected_name:
            raise CorruptResultsError(f"suite {expected_name} must use task_set {expected_name!r}; got {task_set!r}")
        if expected_name == "cross_domain":
            if set(part.strip() for part in domain.split(",")) != {"delivery", "instore", "ota"}:
                raise CorruptResultsError("cross_domain suite domain must contain delivery, instore, and ota")
        elif domain != expected_name:
            raise CorruptResultsError(f"suite {expected_name} must use domain {expected_name!r}; got {domain!r}")

        task_ids = _unique_task_ids(suite.get("task_ids"), f"manifest.suites.{expected_name}.task_ids")
        if len(task_ids) != EXPECTED_TASKS_PER_SUITE:
            raise CorruptResultsError(f"suite {expected_name} must contain {EXPECTED_TASKS_PER_SUITE} task IDs")
        overlap = all_task_ids.intersection(task_ids)
        if overlap:
            raise CorruptResultsError(f"task IDs overlap across suites: {sorted(overlap)[:10]}")
        all_task_ids.update(task_ids)

        shards = suite.get("shards")
        if not isinstance(shards, list) or len(shards) != EXPECTED_SHARDS_PER_SUITE:
            raise CorruptResultsError(f"suite {expected_name} must contain {EXPECTED_SHARDS_PER_SUITE} shards")
        shard_indices: list[int] = []
        shard_task_ids: list[str] = []
        normalized_shards: list[dict[str, Any]] = []
        for position, shard in enumerate(shards):
            if not isinstance(shard, dict):
                raise CorruptResultsError(f"manifest.suites.{expected_name}.shards[{position}] must be an object")
            shard_index = _required_int(
                shard.get("index"),
                f"manifest.suites.{expected_name}.shards[{position}].index",
                minimum=0,
            )
            shard_indices.append(shard_index)
            ids = _unique_task_ids(
                shard.get("task_ids"),
                f"manifest.suites.{expected_name}.shards[{position}].task_ids",
            )
            if len(ids) != EXPECTED_SHARD_SIZE:
                raise CorruptResultsError(
                    f"suite {expected_name} shard {shard_index} must contain {EXPECTED_SHARD_SIZE} task IDs"
                )
            shard_task_ids.extend(ids)
            result_file = Path(
                _required_string(
                    shard.get("result_file"),
                    f"manifest.suites.{expected_name}.shards[{position}].result_file",
                )
            )
            if not result_file.is_absolute():
                result_file = artifact_dir / result_file
            result_file = result_file.resolve()
            try:
                result_file.relative_to(artifact_dir.resolve())
            except ValueError as exc:
                raise CorruptResultsError(f"result file escapes the artifact directory: {result_file}") from exc
            if result_file in all_result_files:
                raise CorruptResultsError(f"result file is reused by multiple shards: {result_file}")
            all_result_files.add(result_file)
            normalized_shards.append({"index": shard_index, "task_ids": ids, "result_file": result_file})
        if len(set(shard_indices)) != len(shard_indices):
            raise CorruptResultsError(f"suite {expected_name} has duplicate shard indices")
        if sorted(shard_indices) != list(range(EXPECTED_SHARDS_PER_SUITE)):
            raise CorruptResultsError(
                f"suite {expected_name} shard indices must be 0..{EXPECTED_SHARDS_PER_SUITE - 1}"
            )
        if Counter(shard_task_ids) != Counter(task_ids):
            raise CorruptResultsError(f"suite {expected_name} shards do not form an exact partition of suite task IDs")
        normalized_suites.append(
            {
                "name": expected_name,
                "domain": domain,
                "task_set": task_set,
                "task_ids": task_ids,
                "shards": sorted(normalized_shards, key=lambda value: value["index"]),
            }
        )
    if len(all_task_ids) != len(SUITE_NAMES) * EXPECTED_TASKS_PER_SUITE:
        raise CorruptResultsError("full benchmark task count is not 400")
    return normalized_suites, all_task_ids


def _macro_scores(suite_metrics: dict[str, dict[str, Any]], num_trials: int) -> dict[str, Any]:
    score_keys = (
        f"avg_at_{num_trials}",
        f"avg_at_{num_trials}_percent",
        "pass_at_1",
        "pass_at_1_percent",
        f"pass_at_{num_trials}_any",
        f"pass_at_{num_trials}_any_percent",
        f"pass_pow_{num_trials}_all",
        f"pass_pow_{num_trials}_all_percent",
    )
    return {key: statistics.fmean(suite_metrics[name]["scores"][key] for name in SUITE_NAMES) for key in score_keys}


def build_summary(manifest: dict[str, Any], manifest_path: Path, artifact_dir: Path) -> dict[str, Any]:
    _validate_manifest_metadata(manifest)
    protocol = _manifest_protocol(manifest)
    suites, all_task_ids = _manifest_suites(manifest, artifact_dir)
    num_trials = int(protocol["num_trials"])
    base_seed = int(protocol["seed"])
    all_simulations: list[dict[str, Any]] = []
    suite_metrics: dict[str, dict[str, Any]] = {}
    shard_validation: list[dict[str, Any]] = []

    for suite in suites:
        suite_simulations: list[dict[str, Any]] = []
        for shard in suite["shards"]:
            result_file = shard["result_file"]
            if not result_file.exists():
                raise IncompleteResultsError(f"missing shard result: {result_file}")
            result = _load_json(result_file)
            validation = validate_shard_data(
                result,
                shard["task_ids"],
                num_trials,
                base_seed,
                expected_domain=suite["domain"],
                expected_max_steps=int(protocol["max_steps"]),
                expected_max_errors=int(protocol["max_errors"]),
                expected_agent_model=manifest["model"],
                expected_user_model=manifest["user"]["model"],
            )
            if not validation["complete"]:
                raise IncompleteResultsError(
                    f"shard {result_file} is missing {validation['missing_simulations']} simulations"
                )
            if validation["seed_validation"] != "validated":
                raise CorruptResultsError(f"complete shard {result_file} does not persist all simulation seeds")
            suite_simulations.extend(result["simulations"])
            shard_validation.append(
                {
                    "suite": suite["name"],
                    "index": shard["index"],
                    "result_file": str(result_file),
                    "task_count": validation["expected_tasks"],
                    "simulation_count": validation["completed_simulations"],
                    "seed_validation": validation["seed_validation"],
                }
            )
        if len(suite_simulations) != EXPECTED_TASKS_PER_SUITE * num_trials:
            raise CorruptResultsError(f"suite {suite['name']} simulation count is not rectangular")
        if {simulation["task_id"] for simulation in suite_simulations} != set(suite["task_ids"]):
            raise CorruptResultsError(f"suite {suite['name']} simulation task IDs do not match manifest")
        suite_metrics[suite["name"]] = _metrics(suite_simulations, num_trials)
        all_simulations.extend(suite_simulations)

    expected_total = len(all_task_ids) * num_trials
    if len(all_simulations) != expected_total:
        raise CorruptResultsError(f"full run contains {len(all_simulations)} simulations; expected {expected_total}")
    simulation_ids = [simulation["id"] for simulation in all_simulations]
    duplicate_ids = sorted(simulation_id for simulation_id, count in Counter(simulation_ids).items() if count > 1)
    if duplicate_ids:
        raise CorruptResultsError(f"simulation IDs are duplicated across shards: {duplicate_ids[:10]}")
    global_pairs = [(simulation["task_id"], int(simulation["trial"])) for simulation in all_simulations]
    if len(set(global_pairs)) != len(global_pairs):
        raise CorruptResultsError("task/trial pairs are duplicated across shards")
    overall = _metrics(all_simulations, num_trials)
    macro = {
        "suite_count": len(SUITE_NAMES),
        "task_count": len(all_task_ids),
        "simulation_count": len(all_simulations),
        "scores": _macro_scores(suite_metrics, num_trials),
    }
    official_score_value = manifest["official_reference"]["score"]
    official_score = float(official_score_value) if official_score_value is not None else None
    avg_percent = macro["scores"][f"avg_at_{num_trials}_percent"]
    comparison = {
        "evaluated_metric": f"four-suite macro Avg@{num_trials}",
        "evaluated_score_percent": avg_percent,
        "official_reference_available": official_score is not None,
        "official_score": official_score,
        "delta_points": (avg_percent - official_score if official_score is not None else None),
        "official_comparable": manifest["evaluator"]["official_comparable"],
    }

    metadata_keys = (
        "schema_version",
        "benchmark",
        "created_at",
        "model",
        "agent",
        "model_config",
        "official_reference",
        "protocol",
        "execution",
        "user",
        "evaluator",
        "software",
    )
    return {
        "schema_version": 1,
        "status": "complete",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "manifest_path": str(manifest_path.resolve()),
        "artifact_dir": str(artifact_dir.resolve()),
        "run": {key: manifest[key] for key in metadata_keys if key in manifest},
        "validation": {
            "suite_count": len(SUITE_NAMES),
            "task_count": len(all_task_ids),
            "trials_per_task": num_trials,
            "expected_simulations": expected_total,
            "completed_simulations": len(all_simulations),
            "missing_simulations": 0,
            "duplicate_simulations": 0,
            "invalid_simulations": 0,
            "trial_seeds": _trial_seeds(base_seed, num_trials),
            "shards": shard_validation,
        },
        "suites": suite_metrics,
        "macro": macro,
        "overall": overall,
        "official_comparison": comparison,
    }


def _json_cell(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _summary_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    num_trials = int(summary["run"]["protocol"]["num_trials"])
    scopes: list[tuple[str, str, dict[str, Any]]] = [("suite", name, summary["suites"][name]) for name in SUITE_NAMES]
    scopes.append(("macro", "macro", summary["macro"]))
    scopes.append(("overall", "overall", summary["overall"]))
    for scope, name, metrics in scopes:
        scores = metrics["scores"]
        tools = metrics.get("tools", {})
        agent_tools = tools.get("assistant", {})
        user_tools = tools.get("user", {})
        messages = metrics.get("messages", {})
        tokens = metrics.get("tokens", {})
        costs = metrics.get("costs", {})
        duration = metrics.get("duration", {})
        reward_details = metrics.get("reward_details", {})
        row = {
            "scope": scope,
            "name": name,
            "task_count": metrics["task_count"],
            "simulation_count": metrics["simulation_count"],
            f"avg_at_{num_trials}": scores[f"avg_at_{num_trials}"],
            f"avg_at_{num_trials}_percent": scores[f"avg_at_{num_trials}_percent"],
            "pass_at_1": scores["pass_at_1"],
            "pass_at_1_percent": scores["pass_at_1_percent"],
            f"pass_at_{num_trials}_any": scores[f"pass_at_{num_trials}_any"],
            f"pass_at_{num_trials}_any_percent": scores[f"pass_at_{num_trials}_any_percent"],
            f"pass_pow_{num_trials}_all": scores[f"pass_pow_{num_trials}_all"],
            f"pass_pow_{num_trials}_all_percent": scores[f"pass_pow_{num_trials}_all_percent"],
            "termination_counts": _json_cell(metrics.get("terminations", {}).get("counts", {})),
            "assistant_tool_calls": agent_tools.get("calls"),
            "assistant_tool_calls_per_simulation": agent_tools.get("calls_per_simulation"),
            "assistant_tool_responses": agent_tools.get("responses"),
            "assistant_tool_errors": agent_tools.get("errors"),
            "assistant_tool_error_rate": agent_tools.get("response_error_rate"),
            "assistant_unanswered_calls": agent_tools.get("unanswered"),
            "assistant_unanswered_call_rate": agent_tools.get("unanswered_call_rate"),
            "assistant_repeated_calls": agent_tools.get("repeated_calls"),
            "assistant_multitool_messages": agent_tools.get("multitool_messages"),
            "assistant_multitool_simulation_rate": agent_tools.get("simulations_with_multitool_calls_rate"),
            "assistant_tool_calls_by_name": _json_cell(agent_tools.get("calls_by_name", {})),
            "user_tool_calls": user_tools.get("calls"),
            "user_tool_calls_per_simulation": user_tools.get("calls_per_simulation"),
            "user_tool_responses": user_tools.get("responses"),
            "user_tool_errors": user_tools.get("errors"),
            "user_tool_error_rate": user_tools.get("response_error_rate"),
            "user_unanswered_calls": user_tools.get("unanswered"),
            "user_unanswered_call_rate": user_tools.get("unanswered_call_rate"),
            "user_repeated_calls": user_tools.get("repeated_calls"),
            "user_multitool_messages": user_tools.get("multitool_messages"),
            "user_multitool_simulation_rate": user_tools.get("simulations_with_multitool_calls_rate"),
            "message_count": messages.get("total"),
            "message_counts_by_role": _json_cell(messages.get("by_role", {})),
            "agent_token_coverage": tokens.get("agent", {}).get("coverage_rate"),
            "agent_token_totals": _json_cell(tokens.get("agent", {}).get("totals", {})),
            "user_token_coverage": tokens.get("user", {}).get("coverage_rate"),
            "user_token_totals": _json_cell(tokens.get("user", {}).get("totals", {})),
            "evaluator_token_coverage": tokens.get("evaluator", {}).get("coverage_rate"),
            "evaluator_token_totals": _json_cell(tokens.get("evaluator", {}).get("totals", {})),
            "agent_cost_coverage": costs.get("agent", {}).get("coverage_rate"),
            "agent_cost_total": costs.get("agent", {}).get("total"),
            "user_cost_coverage": costs.get("user", {}).get("coverage_rate"),
            "user_cost_total": costs.get("user", {}).get("total"),
            "duration_mean_seconds": duration.get("mean_seconds"),
            "duration_p50_seconds": duration.get("p50_seconds"),
            "duration_p95_seconds": duration.get("p95_seconds"),
            "duration_max_seconds": duration.get("max_seconds"),
            "duration_sum_seconds": duration.get("sum_seconds"),
            "reward_breakdown": _json_cell(reward_details.get("reward_breakdown", {})),
            "rubric_coverage": reward_details.get("rubrics", {}).get("simulation_coverage_rate"),
            "rubric_met_rate": reward_details.get("rubrics", {}).get("met_rate"),
        }
        rows.append(row)
    return rows


def _atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _atomic_write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        raise CorruptResultsError("cannot write an empty summary CSV")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _print_status(status: dict[str, Any]) -> None:
    print(json.dumps(status, ensure_ascii=False, sort_keys=True, separators=(",", ":")))


def check_shard(args: argparse.Namespace) -> int:
    path = args.file.resolve()
    try:
        task_ids = _unique_task_ids(args.task_ids, "--task-ids")
        if args.num_trials <= 0:
            raise CorruptResultsError("--num-trials must be positive")
        if args.expected_domain is not None:
            _required_string(args.expected_domain, "--expected-domain")
        if args.expected_max_steps is not None and args.expected_max_steps <= 0:
            raise CorruptResultsError("--expected-max-steps must be positive")
        if args.expected_max_errors is not None and args.expected_max_errors <= 0:
            raise CorruptResultsError("--expected-max-errors must be positive")
        if args.expected_agent_model is not None:
            _required_string(args.expected_agent_model, "--expected-agent-model")
        if args.expected_user_model is not None:
            _required_string(args.expected_user_model, "--expected-user-model")
    except CorruptResultsError as exc:
        _print_status({"status": "corrupt", "file": str(path), "error": str(exc)})
        return 2
    if not path.exists():
        _print_status(
            {
                "status": "partial",
                "file": str(path),
                "expected_tasks": len(task_ids),
                "expected_simulations": len(task_ids) * args.num_trials,
                "completed_simulations": 0,
                "missing_simulations": len(task_ids) * args.num_trials,
                "reason": "file_missing",
            }
        )
        return 1
    try:
        data = _load_json(path)
        validation = validate_shard_data(
            data,
            task_ids,
            args.num_trials,
            args.base_seed,
            expected_domain=args.expected_domain,
            expected_max_steps=args.expected_max_steps,
            expected_max_errors=args.expected_max_errors,
            expected_agent_model=args.expected_agent_model,
            expected_user_model=args.expected_user_model,
        )
    except CorruptResultsError as exc:
        _print_status({"status": "corrupt", "file": str(path), "error": str(exc)})
        return 2
    status = "complete" if validation["complete"] else "partial"
    _print_status({"status": status, "file": str(path), **validation})
    return 0 if validation["complete"] else 1


def summarize(args: argparse.Namespace) -> int:
    try:
        manifest_path = args.manifest.resolve()
        artifact_dir = args.artifact_dir.resolve()
        manifest = _load_json(manifest_path)
        summary = build_summary(manifest, manifest_path, artifact_dir)
        _atomic_write_json(args.output_json.resolve(), summary)
        _atomic_write_csv(args.output_csv.resolve(), _summary_rows(summary))
    except (CorruptResultsError, IncompleteResultsError, FileNotFoundError) as exc:
        _print_status({"status": "invalid", "error": str(exc)})
        return 2
    _print_status(
        {
            "status": "complete",
            "output_json": str(args.output_json.resolve()),
            "output_csv": str(args.output_csv.resolve()),
            "tasks": summary["overall"]["task_count"],
            "simulations": summary["overall"]["simulation_count"],
            "avg_at_4_percent": summary["overall"]["scores"]["avg_at_4_percent"],
        }
    )
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    check_parser = subparsers.add_parser(
        "check-shard", help="classify a result shard as complete, partial, or corrupt"
    )
    check_parser.add_argument("--file", type=Path, required=True)
    check_parser.add_argument("--task-ids", nargs="+", required=True)
    check_parser.add_argument("--num-trials", type=int, default=EXPECTED_TRIALS)
    check_parser.add_argument("--base-seed", type=int, default=EXPECTED_BASE_SEED)
    check_parser.add_argument("--expected-domain")
    check_parser.add_argument("--expected-max-steps", type=int)
    check_parser.add_argument("--expected-max-errors", type=int)
    check_parser.add_argument("--expected-agent-model")
    check_parser.add_argument("--expected-user-model")
    check_parser.set_defaults(handler=check_shard)

    summary_parser = subparsers.add_parser("summarize", help="validate all manifest shards and write final metrics")
    summary_parser.add_argument("--artifact-dir", type=Path, required=True)
    summary_parser.add_argument("--manifest", type=Path, required=True)
    summary_parser.add_argument("--output-json", type=Path, required=True)
    summary_parser.add_argument("--output-csv", type=Path, required=True)
    summary_parser.set_defaults(handler=summarize)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    sys.exit(main())
