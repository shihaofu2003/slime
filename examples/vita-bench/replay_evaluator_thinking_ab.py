#!/usr/bin/env python3
"""Replay persisted VitaBench evaluator windows with thinking disabled.

The frozen mode reuses every persisted evaluator prompt byte-for-byte.  The
chained mode keeps the persisted system prompt and conversation window but
feeds each non-thinking decision into the next window of the same trajectory.
Together they separate prompt-matched judge agreement from the end-to-end
counterfactual score.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import hashlib
import json
import math
import os
import queue
import random
import statistics
import threading
import time
from collections import Counter, defaultdict
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests


REPLAY_SCHEMA = "vitabench-evaluator-thinking-replay/v1"
MANIFEST_SCHEMA = "vitabench-evaluator-thinking-ab-manifest/v1"
AUDIT_SCHEMA = "vitabench-evaluator-source-audit/v1"
REPORT_SCHEMA = "vitabench-evaluator-thinking-ab-report/v1"
SESSION_SCHEMA = "vitabench-evaluator-thinking-replay-session/v1"
FAILURE_SCHEMA = "vitabench-evaluator-thinking-replay-failure/v1"
SUITES = ("delivery", "instore", "ota", "cross_domain")
CURRENT_RUBRICS_OPEN = "<current_rubrics>"
CURRENT_RUBRICS_CLOSE = "</current_rubrics>"
DEFAULT_MODEL = "Qwen3.6-27B-evaluator"
DEFAULT_MAX_TOKENS = 8192
DEFAULT_SEMANTIC_ATTEMPTS = 3
DEFAULT_HTTP_RETRIES = 5
DEFAULT_BOOTSTRAP_SAMPLES = 10_000
DEFAULT_BOOTSTRAP_SEED = 20260803


class ReplayError(ValueError):
    """The source artifact or replay result violates an experiment contract."""


@dataclass(frozen=True)
class WindowRecord:
    suite: str
    task_id: str
    trial: int
    seed: int
    simulation_id: str
    window_idx: int
    system_prompt: str
    user_prompt: str
    source_states: tuple[dict[str, Any], ...]
    baseline_results: tuple[dict[str, Any], ...]
    baseline_usage: dict[str, Any]
    baseline_content: str
    baseline_journal_id: str | None
    baseline_validation_attempts: int
    baseline_validation_failures: tuple[str, ...]

    @property
    def trajectory_key(self) -> str:
        return f"{self.suite}:{self.task_id}:{self.trial}"

    @property
    def key(self) -> str:
        return f"{self.trajectory_key}:{self.window_idx}"

    @property
    def source_prompt_sha256(self) -> str:
        return prompt_sha256(self.system_prompt, self.user_prompt)


@dataclass(frozen=True)
class TrajectoryRecord:
    suite: str
    task_id: str
    trial: int
    seed: int
    simulation_id: str
    termination_reason: str
    baseline_success: bool
    baseline_final_rubrics: tuple[bool, ...]
    windows: tuple[WindowRecord, ...]

    @property
    def key(self) -> str:
        return f"{self.suite}:{self.task_id}:{self.trial}"

    @property
    def task_key(self) -> str:
        return f"{self.suite}:{self.task_id}"


@dataclass(frozen=True)
class SourceDataset:
    artifact_dir: Path
    manifest: dict[str, Any]
    summary: dict[str, Any]
    shard_paths: tuple[Path, ...]
    trajectories: tuple[TrajectoryRecord, ...]

    @property
    def windows(self) -> tuple[WindowRecord, ...]:
        return tuple(window for trajectory in self.trajectories for window in trajectory.windows)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def prompt_sha256(system_prompt: str, user_prompt: str) -> str:
    return sha256_bytes(canonical_json([system_prompt, user_prompt]))


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}.{threading.get_ident()}")
    encoded = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    with temporary.open("x", encoding="utf-8") as stream:
        os.fchmod(stream.fileno(), 0o600)
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def read_json(path: Path, field: str | None = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReplayError(f"cannot read {field or path}: {exc}") from exc


def _extract_json_array(content: str) -> Any:
    if not isinstance(content, str) or not content.strip():
        raise ReplayError("evaluator content must be a nonempty string")
    try:
        from json_repair import repair_json
    except ImportError:
        repair_json = None

    if repair_json is not None:
        try:
            return json.loads(repair_json(content))
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ReplayError(f"evaluator content is not repairable JSON: {exc}") from exc

    candidate = content.strip()
    if candidate.startswith("```") and candidate.endswith("```"):
        first_newline = candidate.find("\n")
        candidate = candidate[first_newline + 1 : -3].strip()
    try:
        return json.loads(candidate)
    except json.JSONDecodeError as exc:
        start = candidate.find("[")
        end = candidate.rfind("]")
        if start < 0 or end <= start:
            raise ReplayError(f"evaluator content is not JSON: {exc}") from exc
        try:
            return json.loads(candidate[start : end + 1])
        except json.JSONDecodeError as nested_exc:
            raise ReplayError(f"evaluator content is not JSON: {nested_exc}") from nested_exc


def validate_rubric_states(value: Any, field: str) -> tuple[dict[str, Any], ...]:
    if not isinstance(value, list) or not value:
        raise ReplayError(f"{field} must be a nonempty list")
    states: list[dict[str, Any]] = []
    identifiers: list[str] = []
    for index, item in enumerate(value):
        item_field = f"{field}[{index}]"
        if not isinstance(item, dict):
            raise ReplayError(f"{item_field} must be an object")
        rubric_idx = item.get("rubric_idx")
        rubric = item.get("rubric")
        justification = item.get("justification")
        decision = item.get("meetExpectation")
        if not isinstance(rubric_idx, str) or not rubric_idx:
            raise ReplayError(f"{item_field}.rubric_idx must be a nonempty string")
        if not isinstance(rubric, str) or not rubric:
            raise ReplayError(f"{item_field}.rubric must be a nonempty string")
        if not isinstance(justification, str):
            raise ReplayError(f"{item_field}.justification must be a string")
        if not isinstance(decision, bool):
            raise ReplayError(f"{item_field}.meetExpectation must be boolean")
        identifiers.append(rubric_idx)
        states.append(
            {
                "rubric_idx": rubric_idx,
                "rubric": rubric,
                "justification": justification,
                "meetExpectation": decision,
            }
        )
    duplicates = sorted(identifier for identifier, count in Counter(identifiers).items() if count > 1)
    if duplicates:
        raise ReplayError(f"{field} has duplicate rubric_idx values: {duplicates}")
    return tuple(states)


def validate_evaluator_results(
    value: Any,
    expected_states: Sequence[dict[str, Any]],
    field: str,
) -> tuple[dict[str, Any], ...]:
    if not isinstance(value, list) or not value:
        raise ReplayError(f"{field} must be a nonempty list")
    expected_ids = [state["rubric_idx"] for state in expected_states]
    results: list[dict[str, Any]] = []
    returned_ids: list[str] = []
    for index, item in enumerate(value):
        item_field = f"{field}[{index}]"
        if not isinstance(item, dict):
            raise ReplayError(f"{item_field} must be an object")
        rubric_idx = item.get("rubric_idx")
        decision = item.get("meetExpectation")
        if not isinstance(rubric_idx, str):
            raise ReplayError(f"{item_field}.rubric_idx must be a string")
        if not isinstance(decision, bool):
            raise ReplayError(f"{item_field}.meetExpectation must be boolean")
        returned_ids.append(rubric_idx)
        results.append(
            {
                "rubric_idx": rubric_idx,
                "rubric": item.get("rubric") if isinstance(item.get("rubric"), str) else None,
                "justification": (
                    item.get("justification")
                    if isinstance(item.get("justification"), str)
                    else "No justification provided"
                ),
                "meetExpectation": decision,
            }
        )
    if returned_ids != expected_ids:
        raise ReplayError(f"{field} rubric order/coverage mismatch: expected={expected_ids}, returned={returned_ids}")
    return tuple(results)


def extract_current_rubrics(user_prompt: str, field: str) -> tuple[dict[str, Any], ...]:
    if not isinstance(user_prompt, str):
        raise ReplayError(f"{field} must be a string")
    if user_prompt.count(CURRENT_RUBRICS_OPEN) != 1 or user_prompt.count(CURRENT_RUBRICS_CLOSE) != 1:
        raise ReplayError(f"{field} must contain exactly one current_rubrics block")
    start = user_prompt.index(CURRENT_RUBRICS_OPEN) + len(CURRENT_RUBRICS_OPEN)
    end = user_prompt.index(CURRENT_RUBRICS_CLOSE, start)
    try:
        value = json.loads(user_prompt[start:end].strip())
    except json.JSONDecodeError as exc:
        raise ReplayError(f"{field}.current_rubrics is invalid JSON: {exc}") from exc
    return validate_rubric_states(value, f"{field}.current_rubrics")


def replace_current_rubrics(user_prompt: str, states: Sequence[dict[str, Any]]) -> str:
    if user_prompt.count(CURRENT_RUBRICS_OPEN) != 1 or user_prompt.count(CURRENT_RUBRICS_CLOSE) != 1:
        raise ReplayError("user prompt must contain exactly one current_rubrics block")
    start = user_prompt.index(CURRENT_RUBRICS_OPEN) + len(CURRENT_RUBRICS_OPEN)
    end = user_prompt.index(CURRENT_RUBRICS_CLOSE, start)
    serialized = "\n" + json.dumps(list(states), ensure_ascii=False, indent=2) + "\n"
    return user_prompt[:start] + serialized + user_prompt[end:]


def update_states(states: Sequence[dict[str, Any]], results: Sequence[dict[str, Any]]) -> tuple[dict[str, Any], ...]:
    if [state["rubric_idx"] for state in states] != [result["rubric_idx"] for result in results]:
        raise ReplayError("cannot update states from mismatched rubric results")
    updated = []
    for index, state in enumerate(states):
        result = results[index]
        updated.append(
            {
                "rubric_idx": state["rubric_idx"],
                "rubric": state["rubric"],
                "justification": result["justification"],
                "meetExpectation": result["meetExpectation"],
            }
        )
    return tuple(updated)


def _required_int(value: Any, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ReplayError(f"{field} must be an integer")
    return value


def _required_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ReplayError(f"{field} must be a nonempty string")
    return value


def _load_window(
    suite: str,
    simulation: dict[str, Any],
    raw_window: Any,
    position: int,
) -> WindowRecord:
    field = f"{suite}:{simulation.get('task_id')}:{simulation.get('trial')}.windows[{position}]"
    if not isinstance(raw_window, dict):
        raise ReplayError(f"{field} must be an object")
    system_prompt = _required_string(raw_window.get("system_prompt"), f"{field}.system_prompt")
    user_prompt = _required_string(raw_window.get("user_prompt"), f"{field}.user_prompt")
    source_states = extract_current_rubrics(user_prompt, f"{field}.user_prompt")
    content = _required_string(raw_window.get("assistant_message_content"), f"{field}.assistant_message_content")
    baseline_results = validate_evaluator_results(
        _extract_json_array(content), source_states, f"{field}.assistant_message_content"
    )
    expected_count = _required_int(raw_window.get("expected_rubric_count"), f"{field}.expected_rubric_count")
    validated_count = _required_int(raw_window.get("validated_rubric_count"), f"{field}.validated_rubric_count")
    if expected_count != len(source_states) or validated_count != len(baseline_results):
        raise ReplayError(f"{field} persisted rubric counts are inconsistent")
    usage = raw_window.get("assistent_message_usage", raw_window.get("assistant_message_usage"))
    if not isinstance(usage, dict):
        raise ReplayError(f"{field}.usage must be an object")
    raw_data = raw_window.get("assistant_message_raw_data")
    journal_id = None
    if isinstance(raw_data, dict):
        candidate = raw_data.get("_vita_journal_id")
        if candidate is not None:
            journal_id = _required_string(candidate, f"{field}.journal_id")
    validation_attempts = _required_int(raw_window.get("validation_attempts"), f"{field}.validation_attempts")
    validation_failures = raw_window.get("validation_failures")
    if (
        not isinstance(validation_failures, list)
        or not all(isinstance(item, str) for item in validation_failures)
        or validation_attempts != len(validation_failures) + 1
    ):
        raise ReplayError(f"{field}.validation attempt history is inconsistent")
    return WindowRecord(
        suite=suite,
        task_id=_required_string(simulation.get("task_id"), f"{field}.task_id"),
        trial=_required_int(simulation.get("trial"), f"{field}.trial"),
        seed=_required_int(simulation.get("seed"), f"{field}.seed"),
        simulation_id=_required_string(simulation.get("id"), f"{field}.simulation_id"),
        window_idx=_required_int(raw_window.get("window_idx"), f"{field}.window_idx"),
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        source_states=source_states,
        baseline_results=baseline_results,
        baseline_usage=dict(usage),
        baseline_content=content,
        baseline_journal_id=journal_id,
        baseline_validation_attempts=validation_attempts,
        baseline_validation_failures=tuple(validation_failures),
    )


def _load_trajectory(suite: str, simulation: Any, field: str) -> TrajectoryRecord:
    if not isinstance(simulation, dict):
        raise ReplayError(f"{field} must be an object")
    reward_info = simulation.get("reward_info")
    if not isinstance(reward_info, dict):
        raise ReplayError(f"{field}.reward_info must be an object")
    reward = reward_info.get("reward")
    if not isinstance(reward, (int, float)) or isinstance(reward, bool) or not math.isfinite(float(reward)):
        raise ReplayError(f"{field}.reward_info.reward must be finite")
    if float(reward) not in {0.0, 1.0}:
        raise ReplayError(f"{field}.reward_info.reward must be binary")
    raw_windows = reward_info.get("window_evaluations")
    windows: tuple[WindowRecord, ...]
    if raw_windows is None:
        windows = ()
    else:
        if not isinstance(raw_windows, list) or not raw_windows:
            raise ReplayError(f"{field}.window_evaluations must be null or a nonempty list")
        windows = tuple(_load_window(suite, simulation, value, index) for index, value in enumerate(raw_windows))
        if [window.window_idx for window in windows] != list(range(1, len(windows) + 1)):
            raise ReplayError(f"{field}.window_evaluations indices are not contiguous")
        rubric_ids = [state["rubric_idx"] for state in windows[0].source_states]
        for window in windows:
            if [state["rubric_idx"] for state in window.source_states] != rubric_ids:
                raise ReplayError(f"{field} changes rubric identifiers between windows")

    raw_final_rubrics = reward_info.get("nl_rubrics")
    final_rubrics: tuple[bool, ...]
    if windows:
        if not isinstance(raw_final_rubrics, list) or not raw_final_rubrics:
            raise ReplayError(f"{field}.nl_rubrics must be nonempty for a judged trajectory")
        final_values: list[bool] = []
        for index, item in enumerate(raw_final_rubrics):
            if not isinstance(item, dict) or not isinstance(item.get("met"), bool):
                raise ReplayError(f"{field}.nl_rubrics[{index}].met must be boolean")
            final_values.append(item["met"])
        final_rubrics = tuple(final_values)
        last_decisions = tuple(item["meetExpectation"] for item in windows[-1].baseline_results)
        if final_rubrics != last_decisions:
            raise ReplayError(f"{field}.nl_rubrics differs from the final persisted evaluator window")
        if bool(all(final_rubrics)) != bool(reward):
            raise ReplayError(f"{field}.reward differs from its final rubric decisions")
    else:
        if raw_final_rubrics not in (None, []):
            raise ReplayError(f"{field}.nl_rubrics must be null/empty without evaluator windows")
        if float(reward) != 0.0:
            raise ReplayError(f"{field} has positive reward without evaluator windows")
        final_rubrics = ()

    return TrajectoryRecord(
        suite=suite,
        task_id=_required_string(simulation.get("task_id"), f"{field}.task_id"),
        trial=_required_int(simulation.get("trial"), f"{field}.trial"),
        seed=_required_int(simulation.get("seed"), f"{field}.seed"),
        simulation_id=_required_string(simulation.get("id"), f"{field}.id"),
        termination_reason=_required_string(simulation.get("termination_reason"), f"{field}.termination_reason"),
        baseline_success=bool(reward),
        baseline_final_rubrics=final_rubrics,
        windows=windows,
    )


def load_source_dataset(artifact_dir: Path) -> SourceDataset:
    artifact_dir = artifact_dir.resolve()
    manifest_path = artifact_dir / "manifest.json"
    summary_path = artifact_dir / "summary.json"
    manifest = read_json(manifest_path, "source manifest")
    summary = read_json(summary_path, "source summary")
    if not isinstance(manifest, dict) or not isinstance(summary, dict):
        raise ReplayError("source manifest and summary must be objects")
    protocol = manifest.get("protocol")
    if not isinstance(protocol, dict):
        raise ReplayError("source manifest protocol is missing")

    shard_paths = tuple(sorted((artifact_dir / "shards").glob("*/shard_*.json")))
    expected_shards = _required_int(protocol.get("shard_count"), "manifest.protocol.shard_count")
    if len(shard_paths) != expected_shards:
        raise ReplayError(f"source has {len(shard_paths)} shard files; expected {expected_shards}")

    trajectories: list[TrajectoryRecord] = []
    seen_simulation_ids: set[str] = set()
    seen_pairs: set[tuple[str, str, int]] = set()
    for shard_path in shard_paths:
        suite = shard_path.parent.name
        if suite not in SUITES:
            raise ReplayError(f"unexpected source suite {suite!r}")
        shard = read_json(shard_path, str(shard_path))
        simulations = shard.get("simulations") if isinstance(shard, dict) else None
        if not isinstance(simulations, list) or not simulations:
            raise ReplayError(f"{shard_path}.simulations must be a nonempty list")
        for index, simulation in enumerate(simulations):
            trajectory = _load_trajectory(suite, simulation, f"{shard_path}.simulations[{index}]")
            if trajectory.simulation_id in seen_simulation_ids:
                raise ReplayError(f"duplicate simulation id {trajectory.simulation_id}")
            pair = (suite, trajectory.task_id, trajectory.trial)
            if pair in seen_pairs:
                raise ReplayError(f"duplicate task/trial pair {pair}")
            seen_simulation_ids.add(trajectory.simulation_id)
            seen_pairs.add(pair)
            trajectories.append(trajectory)

    expected_trajectories = _required_int(protocol.get("trajectory_count"), "manifest.protocol.trajectory_count")
    if len(trajectories) != expected_trajectories:
        raise ReplayError(f"source has {len(trajectories)} trajectories; expected {expected_trajectories}")
    expected_trials = _required_int(protocol.get("num_trials"), "manifest.protocol.num_trials")
    trials_by_task: dict[tuple[str, str], set[int]] = defaultdict(set)
    for trajectory in trajectories:
        trials_by_task[(trajectory.suite, trajectory.task_id)].add(trajectory.trial)
    expected_trial_ids = set(range(expected_trials))
    incomplete = sorted(key for key, trials in trials_by_task.items() if trials != expected_trial_ids)
    if incomplete:
        raise ReplayError(f"source has incomplete task/trial rectangles: {incomplete[:5]}")
    expected_tasks = _required_int(protocol.get("task_count"), "manifest.protocol.task_count")
    if len(trials_by_task) != expected_tasks:
        raise ReplayError(f"source has {len(trials_by_task)} tasks; expected {expected_tasks}")

    trajectories.sort(key=lambda item: (SUITES.index(item.suite), item.task_id, item.trial))
    return SourceDataset(
        artifact_dir=artifact_dir,
        manifest=manifest,
        summary=summary,
        shard_paths=shard_paths,
        trajectories=tuple(trajectories),
    )


def _journal_path(dataset: SourceDataset, window: WindowRecord) -> Path:
    if window.baseline_journal_id is None:
        raise ReplayError(f"{window.key} has no persisted evaluator journal id")
    return (
        dataset.artifact_dir
        / "api_journal"
        / window.suite
        / DEFAULT_MODEL
        / window.baseline_journal_id[:2]
        / f"{window.baseline_journal_id}.json"
    )


def audit_source_journals(dataset: SourceDataset) -> dict[str, Any]:
    final_prompt_tokens = 0
    final_completion_tokens = 0
    final_reasoning_tokens = 0
    final_journal_ids: set[str] = set()
    for window in dataset.windows:
        path = _journal_path(dataset, window)
        journal = read_json(path, f"journal for {window.key}")
        if not isinstance(journal, dict) or journal.get("schema") != "vita-llm-response-journal/v1":
            raise ReplayError(f"{path} has an unexpected journal schema")
        if journal.get("journal_id") != window.baseline_journal_id:
            raise ReplayError(f"{path} journal id does not match its path")
        final_journal_ids.add(window.baseline_journal_id)
        request = journal.get("request")
        response = journal.get("response")
        if not isinstance(request, dict) or not isinstance(response, dict):
            raise ReplayError(f"{path} is missing request/response objects")
        expected_messages = [
            {"role": "system", "content": window.system_prompt},
            {"role": "user", "content": window.user_prompt},
        ]
        request_messages = request.get("messages")
        if not isinstance(request_messages, list) or request_messages[:2] != expected_messages:
            raise ReplayError(f"{path} request prefix differs from the persisted window prompts")
        expected_message_count = 2 + 2 * (window.baseline_validation_attempts - 1)
        if len(request_messages) != expected_message_count:
            raise ReplayError(
                f"{path} has {len(request_messages)} request messages; expected {expected_message_count} "
                "from the persisted semantic retry history"
            )
        template_kwargs = request.get("chat_template_kwargs")
        if not isinstance(template_kwargs, dict) or template_kwargs.get("enable_thinking") is not True:
            raise ReplayError(f"{path} did not enable thinking")
        choices = response.get("choices")
        try:
            response_content = choices[0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ReplayError(f"{path} has malformed response choices") from exc
        if response_content != window.baseline_content:
            raise ReplayError(f"{path} response content differs from the persisted window")
        usage = response.get("usage")
        if not isinstance(usage, dict):
            raise ReplayError(f"{path} response usage is missing")
        for key in ("prompt_tokens", "completion_tokens", "reasoning_tokens"):
            value = usage.get(key, 0)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ReplayError(f"{path} usage.{key} must be a nonnegative integer")
        final_prompt_tokens += usage["prompt_tokens"]
        final_completion_tokens += usage["completion_tokens"]
        final_reasoning_tokens += usage.get("reasoning_tokens", 0)

    formal_prompt_tokens = 0
    formal_completion_tokens = 0
    formal_reasoning_tokens = 0
    formal_journal_ids: set[str] = set()
    formal_counts_by_trajectory: Counter[tuple[str, str, int]] = Counter()
    total_journals = 0
    for suite in SUITES:
        root = dataset.artifact_dir / "api_journal" / suite / DEFAULT_MODEL
        for path in root.glob("*/*.json"):
            total_journals += 1
            journal = read_json(path, f"evaluator journal {path}")
            if not isinstance(journal, dict) or journal.get("schema") != "vita-llm-response-journal/v1":
                raise ReplayError(f"{path} has an unexpected journal schema")
            context = journal.get("context")
            if not isinstance(context, dict):
                raise ReplayError(f"{path}.context must be an object")
            runner_context = context.get("runner_context")
            if not isinstance(runner_context, str) or not runner_context.startswith("shard:"):
                continue
            journal_id = _required_string(journal.get("journal_id"), f"{path}.journal_id")
            if journal_id in formal_journal_ids:
                raise ReplayError(f"duplicate formal evaluator journal id {journal_id}")
            formal_journal_ids.add(journal_id)
            task_id = _required_string(context.get("task_id"), f"{path}.context.task_id")
            trial = _required_int(context.get("trial"), f"{path}.context.trial")
            formal_counts_by_trajectory[(suite, task_id, trial)] += 1
            request = journal.get("request")
            response = journal.get("response")
            if not isinstance(request, dict) or not isinstance(response, dict):
                raise ReplayError(f"{path} is missing request/response objects")
            template_kwargs = request.get("chat_template_kwargs")
            if not isinstance(template_kwargs, dict) or template_kwargs.get("enable_thinking") is not True:
                raise ReplayError(f"{path} formal evaluator request did not enable thinking")
            if journal.get("requested_model") != DEFAULT_MODEL or response.get("model") != DEFAULT_MODEL:
                raise ReplayError(f"{path} does not use the expected baseline evaluator model")
            usage = response.get("usage")
            if not isinstance(usage, dict):
                raise ReplayError(f"{path} response usage is missing")
            for key in ("prompt_tokens", "completion_tokens", "reasoning_tokens"):
                value = usage.get(key, 0)
                if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                    raise ReplayError(f"{path} usage.{key} must be a nonnegative integer")
            formal_prompt_tokens += usage["prompt_tokens"]
            formal_completion_tokens += usage["completion_tokens"]
            formal_reasoning_tokens += usage.get("reasoning_tokens", 0)

    expected_formal_counts = {
        (trajectory.suite, trajectory.task_id, trajectory.trial): (
            len(trajectory.windows) + sum(len(window.baseline_validation_failures) for window in trajectory.windows)
        )
        for trajectory in dataset.trajectories
        if trajectory.windows
    }
    if dict(formal_counts_by_trajectory) != expected_formal_counts:
        missing_or_extra = sorted(set(formal_counts_by_trajectory) ^ set(expected_formal_counts))
        count_mismatches = sorted(
            key
            for key in set(formal_counts_by_trajectory) & set(expected_formal_counts)
            if formal_counts_by_trajectory[key] != expected_formal_counts[key]
        )
        raise ReplayError(
            "formal evaluator journal coverage differs from persisted validation history: "
            f"missing_or_extra={missing_or_extra[:5]}, count_mismatches={count_mismatches[:5]}"
        )
    if not final_journal_ids <= formal_journal_ids:
        raise ReplayError("one or more final evaluator journals are outside the formal shard set")
    return {
        "evaluator_journals_total_including_gate": total_journals,
        "formal_request_count": len(formal_journal_ids),
        "formal_final_window_count": len(final_journal_ids),
        "formal_semantic_retry_count": len(formal_journal_ids) - len(final_journal_ids),
        "formal_prompt_tokens": formal_prompt_tokens,
        "formal_completion_tokens": formal_completion_tokens,
        "formal_reasoning_tokens": formal_reasoning_tokens,
        "formal_reasoning_share_of_completion": formal_reasoning_tokens / formal_completion_tokens,
        "final_prompt_tokens": final_prompt_tokens,
        "final_completion_tokens": final_completion_tokens,
        "final_reasoning_tokens": final_reasoning_tokens,
        "final_reasoning_share_of_completion": final_reasoning_tokens / final_completion_tokens,
    }


def source_counts(dataset: SourceDataset) -> dict[str, Any]:
    windows = dataset.windows
    judged = [trajectory for trajectory in dataset.trajectories if trajectory.windows]
    final_rubrics = sum(len(trajectory.baseline_final_rubrics) for trajectory in judged)
    baseline_attempts = Counter(window.baseline_validation_attempts for window in windows)
    return {
        "shards": len(dataset.shard_paths),
        "tasks": len({trajectory.task_key for trajectory in dataset.trajectories}),
        "trajectories": len(dataset.trajectories),
        "judged_trajectories": len(judged),
        "windows": len(windows),
        "final_rubric_checks": final_rubrics,
        "successful_trajectories": sum(trajectory.baseline_success for trajectory in dataset.trajectories),
        "baseline_validation_attempt_histogram": {
            str(attempts): count for attempts, count in sorted(baseline_attempts.items())
        },
        "baseline_validation_failures": sum(len(window.baseline_validation_failures) for window in windows),
        "per_suite": {
            suite: {
                "trajectories": sum(trajectory.suite == suite for trajectory in dataset.trajectories),
                "judged_trajectories": sum(
                    trajectory.suite == suite and bool(trajectory.windows) for trajectory in dataset.trajectories
                ),
                "windows": sum(window.suite == suite for window in windows),
            }
            for suite in SUITES
        },
    }


def build_source_inventory(dataset: SourceDataset) -> dict[str, Any]:
    files = []
    for path in dataset.shard_paths:
        stat = path.stat()
        files.append(
            {
                "relative_path": str(path.relative_to(dataset.artifact_dir)),
                "size_bytes": stat.st_size,
                "sha256": sha256_file(path),
            }
        )
    identity = {
        "manifest_sha256": sha256_file(dataset.artifact_dir / "manifest.json"),
        "summary_sha256": sha256_file(dataset.artifact_dir / "summary.json"),
        "shards": files,
    }
    identity["inventory_sha256"] = sha256_bytes(canonical_json(identity))
    return identity


def select_pilot_tasks(trajectories: Sequence[TrajectoryRecord], per_suite: int) -> tuple[TrajectoryRecord, ...]:
    if per_suite <= 0:
        raise ReplayError("pilot tasks per suite must be positive")
    selected: list[TrajectoryRecord] = []
    for suite in SUITES:
        by_task: dict[str, list[TrajectoryRecord]] = defaultdict(list)
        for trajectory in trajectories:
            if trajectory.suite == suite:
                by_task[trajectory.task_key].append(trajectory)
        candidates = [
            task_key
            for task_key, task_trajectories in by_task.items()
            if any(trajectory.windows for trajectory in task_trajectories)
        ]
        candidates.sort(key=lambda task_key: sha256_bytes(task_key.encode("utf-8")))
        if len(candidates) < per_suite:
            raise ReplayError(f"suite {suite} has only {len(candidates)} judged pilot task candidates")
        for task_key in candidates[:per_suite]:
            selected.extend(sorted(by_task[task_key], key=lambda trajectory: trajectory.trial))
    return tuple(selected)


def result_path(output_dir: Path, mode: str, window: WindowRecord) -> Path:
    return (
        output_dir
        / "results"
        / mode
        / window.suite
        / window.task_id
        / f"trial_{window.trial:02d}"
        / f"window_{window.window_idx:04d}.json"
    )


def failure_path(output_dir: Path, mode: str, window: WindowRecord) -> Path:
    return (
        output_dir
        / "failures"
        / mode
        / window.suite
        / window.task_id
        / f"trial_{window.trial:02d}"
        / f"window_{window.window_idx:04d}.json"
    )


def _validate_usage(value: Any, field: str) -> dict[str, int]:
    if not isinstance(value, dict):
        raise ReplayError(f"{field} must be an object")
    normalized: dict[str, int] = {}
    for key in ("prompt_tokens", "completion_tokens", "reasoning_tokens", "total_tokens"):
        raw = value.get(key, 0)
        if raw is None:
            raw = 0
        if not isinstance(raw, int) or isinstance(raw, bool) or raw < 0:
            raise ReplayError(f"{field}.{key} must be a nonnegative integer")
        normalized[key] = raw
    return normalized


def validate_replay_result(
    value: Any,
    *,
    mode: str,
    window: WindowRecord,
    user_prompt: str,
    expected_states: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    field = f"replay result {window.key}"
    if not isinstance(value, dict) or value.get("schema") != REPLAY_SCHEMA:
        raise ReplayError(f"{field} has an unexpected schema")
    expected = {
        "mode": mode,
        "window_key": window.key,
        "source_prompt_sha256": window.source_prompt_sha256,
        "request_prompt_sha256": prompt_sha256(window.system_prompt, user_prompt),
        "enable_thinking": False,
    }
    for key, expected_value in expected.items():
        if value.get(key) != expected_value:
            raise ReplayError(f"{field}.{key} does not match the current experiment")
    attempts = value.get("semantic_attempts")
    if not isinstance(attempts, list) or not attempts:
        raise ReplayError(f"{field}.semantic_attempts must be nonempty")
    for index, attempt in enumerate(attempts):
        if not isinstance(attempt, dict):
            raise ReplayError(f"{field}.semantic_attempts[{index}] must be an object")
        _validate_usage(attempt.get("usage"), f"{field}.semantic_attempts[{index}].usage")
        duration = attempt.get("request_seconds")
        if not isinstance(duration, (int, float)) or isinstance(duration, bool) or duration < 0:
            raise ReplayError(f"{field}.semantic_attempts[{index}].request_seconds must be nonnegative")
    results = validate_evaluator_results(value.get("results"), expected_states, f"{field}.results")
    normalized = dict(value)
    normalized["results"] = list(results)
    return normalized


class EndpointPool:
    """Lease at most N concurrent requests to every evaluator endpoint."""

    def __init__(self, endpoints: Sequence[str], slots_per_endpoint: int):
        if not endpoints or slots_per_endpoint <= 0:
            raise ReplayError("at least one endpoint and one slot per endpoint are required")
        self._queue: queue.Queue[str] = queue.Queue()
        for endpoint in endpoints:
            normalized = endpoint.rstrip("/")
            if not normalized.endswith("/v1/chat/completions"):
                normalized += "/v1/chat/completions"
            for _ in range(slots_per_endpoint):
                self._queue.put(normalized)

    @property
    def slot_count(self) -> int:
        return self._queue.qsize()

    @contextlib.contextmanager
    def lease(self) -> Iterator[tuple[str, float]]:
        started = time.monotonic()
        endpoint = self._queue.get()
        waited = time.monotonic() - started
        try:
            yield endpoint, waited
        finally:
            self._queue.put(endpoint)


_THREAD_LOCAL = threading.local()


def _session() -> requests.Session:
    session = getattr(_THREAD_LOCAL, "session", None)
    if session is None:
        session = requests.Session()
        _THREAD_LOCAL.session = session
    return session


def post_with_retries(
    endpoint: str,
    request_data: dict[str, Any],
    *,
    max_retries: int,
    timeout_seconds: float,
) -> tuple[dict[str, Any], list[dict[str, Any]], float]:
    failures: list[dict[str, Any]] = []
    total_started = time.monotonic()
    delay = 2.0
    for attempt in range(max_retries + 1):
        try:
            response = _session().post(
                endpoint,
                json=request_data,
                headers={"Content-Type": "application/json"},
                timeout=(10, timeout_seconds),
            )
        except requests.RequestException as exc:
            failures.append({"attempt": attempt + 1, "kind": type(exc).__name__})
            if attempt == max_retries:
                raise ReplayError(
                    f"evaluator request failed after {max_retries + 1} HTTP attempts: {type(exc).__name__}"
                ) from exc
        else:
            status = response.status_code
            if 200 <= status < 300:
                try:
                    payload = response.json()
                except ValueError as exc:
                    failures.append({"attempt": attempt + 1, "kind": "invalid_json_response"})
                    if attempt == max_retries:
                        raise ReplayError("endpoint repeatedly returned invalid JSON") from exc
                else:
                    if not isinstance(payload, dict):
                        raise ReplayError("endpoint returned a non-object JSON response")
                    return payload, failures, time.monotonic() - total_started
            else:
                retryable = status in {408, 429} or status >= 500
                failures.append({"attempt": attempt + 1, "kind": "http_status", "status": status})
                if not retryable:
                    raise ReplayError(f"evaluator endpoint returned non-retryable HTTP {status}")
                if attempt == max_retries:
                    raise ReplayError(
                        f"evaluator endpoint still returned HTTP {status} after {max_retries + 1} attempts"
                    )
        time.sleep(delay)
        delay = min(delay * 2, 30.0)
    raise AssertionError("HTTP retry loop exited unexpectedly")


def _response_fields(payload: dict[str, Any], expected_model: str) -> dict[str, Any]:
    actual_model = payload.get("model")
    if actual_model != expected_model:
        raise ReplayError(f"evaluator response model mismatch: expected={expected_model!r}, actual={actual_model!r}")
    try:
        choice = payload["choices"][0]
        message = choice["message"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ReplayError("evaluator response has malformed choices") from exc
    if not isinstance(message, dict) or message.get("role") != "assistant":
        raise ReplayError("evaluator response does not contain an assistant message")
    content = message.get("content")
    if not isinstance(content, str) or not content:
        raise ReplayError("evaluator response content is empty")
    reasoning_content = message.get("reasoning_content") or message.get("reasoning") or ""
    if not isinstance(reasoning_content, str):
        raise ReplayError("evaluator reasoning content has an unexpected type")
    return {
        "content": content,
        "reasoning_content": reasoning_content,
        "finish_reason": choice.get("finish_reason"),
        "usage": _validate_usage(payload.get("usage"), "evaluator response usage"),
        "response_id": payload.get("id"),
        "created": payload.get("created"),
    }


def evaluate_prompt(
    *,
    pool: EndpointPool,
    model: str,
    system_prompt: str,
    user_prompt: str,
    expected_states: Sequence[dict[str, Any]],
    max_tokens: int,
    max_semantic_attempts: int,
    max_http_retries: int,
    timeout_seconds: float,
    failure_path: Path | None = None,
    failure_record: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], tuple[dict[str, Any], ...]]:
    base_messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    messages = base_messages
    attempts: list[dict[str, Any]] = []
    expected_ids = [state["rubric_idx"] for state in expected_states]
    for semantic_attempt in range(1, max_semantic_attempts + 1):
        if semantic_attempt == 1:
            requested_max_tokens = max_tokens
        elif semantic_attempt == 2:
            requested_max_tokens = min(max_tokens, max(1024, len(expected_ids) * 128))
        else:
            requested_max_tokens = min(max_tokens, max(768, len(expected_ids) * 80))
        request_data = {
            "model": model,
            "messages": messages,
            "temperature": 0,
            "max_tokens": requested_max_tokens,
            "stream": False,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        started_at_ns = time.time_ns()
        with pool.lease() as (endpoint, pool_wait_seconds):
            payload, http_failures, request_seconds = post_with_retries(
                endpoint,
                request_data,
                max_retries=max_http_retries,
                timeout_seconds=timeout_seconds,
            )
        ended_at_ns = time.time_ns()
        response = _response_fields(payload, model)
        validation_error = None
        try:
            results = validate_evaluator_results(
                _extract_json_array(response["content"]),
                expected_states,
                f"semantic attempt {semantic_attempt}",
            )
        except ReplayError as exc:
            validation_error = str(exc)
            results = ()
        attempts.append(
            {
                "attempt": semantic_attempt,
                "endpoint": endpoint,
                "started_at_unix_ns": started_at_ns,
                "ended_at_unix_ns": ended_at_ns,
                "pool_wait_seconds": pool_wait_seconds,
                "request_seconds": request_seconds,
                "requested_max_tokens": requested_max_tokens,
                "http_failures": http_failures,
                "content": response["content"],
                "reasoning_content": response["reasoning_content"],
                "finish_reason": response["finish_reason"],
                "usage": response["usage"],
                "response_id": response["response_id"],
                "created": response["created"],
                "validation_error": validation_error,
            }
        )
        if validation_error is not None and failure_path is not None:
            atomic_write_json(
                failure_path,
                {
                    "schema": FAILURE_SCHEMA,
                    **(failure_record or {}),
                    "status": "failed" if semantic_attempt == max_semantic_attempts else "retrying",
                    "semantic_attempts": attempts,
                },
            )
        if validation_error is None:
            if failure_path is not None and len(attempts) > 1:
                atomic_write_json(
                    failure_path,
                    {
                        "schema": FAILURE_SCHEMA,
                        **(failure_record or {}),
                        "status": "resolved",
                        "semantic_attempts": attempts,
                    },
                )
            return attempts, results
        if semantic_attempt == max_semantic_attempts:
            raise ReplayError(
                f"evaluator output remained invalid after {max_semantic_attempts} semantic attempts: "
                f"{validation_error}"
            )
        justification_limit = 40 if semantic_attempt == 1 else 12
        correction = (
            "\n\n# Mandatory compact-format correction\n"
            f"The previous response was invalid: {validation_error}. "
            f"Its finish_reason was {response['finish_reason']!r}. Re-evaluate the same input, "
            "but return only one compact JSON array with exactly one object for every "
            "rubric_idx in this list, in this order: "
            f"{json.dumps(expected_ids, ensure_ascii=False)}. Each object must contain only "
            '"rubric_idx", "justification", and "meetExpectation". Keep each justification '
            f"within {justification_limit} Chinese characters. Every meetExpectation must be "
            "the unquoted JSON boolean true or false. Do not repeat rubric text. Do not use "
            "Markdown. Do not output analysis before, after, or inside the array."
        )
        messages = [base_messages[0], {"role": "user", "content": user_prompt + correction}]
    raise AssertionError("semantic retry loop exited unexpectedly")


def run_one_window(
    *,
    output_dir: Path,
    mode: str,
    window: WindowRecord,
    user_prompt: str,
    expected_states: Sequence[dict[str, Any]],
    pool: EndpointPool,
    model: str,
    max_tokens: int,
    max_semantic_attempts: int,
    max_http_retries: int,
    timeout_seconds: float,
    session_id: str,
    base_output_dir: Path | None = None,
) -> tuple[dict[str, Any], bool]:
    path = result_path(output_dir, mode, window)
    existing_path = path
    if not existing_path.is_file() and base_output_dir is not None:
        existing_path = result_path(base_output_dir, mode, window)
    if existing_path.is_file():
        return (
            validate_replay_result(
                read_json(existing_path),
                mode=mode,
                window=window,
                user_prompt=user_prompt,
                expected_states=expected_states,
            ),
            True,
        )
    journal_path = failure_path(output_dir, mode, window)
    attempts, results = evaluate_prompt(
        pool=pool,
        model=model,
        system_prompt=window.system_prompt,
        user_prompt=user_prompt,
        expected_states=expected_states,
        max_tokens=max_tokens,
        max_semantic_attempts=max_semantic_attempts,
        max_http_retries=max_http_retries,
        timeout_seconds=timeout_seconds,
        failure_path=journal_path,
        failure_record={
            "mode": mode,
            "session_id": session_id,
            "window_key": window.key,
            "trajectory_key": window.trajectory_key,
            "source_prompt_sha256": window.source_prompt_sha256,
            "request_prompt_sha256": prompt_sha256(window.system_prompt, user_prompt),
        },
    )
    value = {
        "schema": REPLAY_SCHEMA,
        "mode": mode,
        "session_id": session_id,
        "window_key": window.key,
        "trajectory_key": window.trajectory_key,
        "suite": window.suite,
        "task_id": window.task_id,
        "trial": window.trial,
        "seed": window.seed,
        "simulation_id": window.simulation_id,
        "window_idx": window.window_idx,
        "source_prompt_sha256": window.source_prompt_sha256,
        "request_prompt_sha256": prompt_sha256(window.system_prompt, user_prompt),
        "system_prompt": window.system_prompt,
        "user_prompt": user_prompt,
        "model": model,
        "temperature": 0,
        "max_tokens": max_tokens,
        "enable_thinking": False,
        "semantic_attempts": attempts,
        "results": list(results),
    }
    atomic_write_json(path, value)
    return (
        validate_replay_result(
            value,
            mode=mode,
            window=window,
            user_prompt=user_prompt,
            expected_states=expected_states,
        ),
        False,
    )


def run_frozen_replay(
    trajectories: Sequence[TrajectoryRecord],
    **kwargs: Any,
) -> tuple[int, int]:
    windows = [window for trajectory in trajectories for window in trajectory.windows]
    if not windows:
        raise ReplayError("frozen replay selection has no evaluator windows")
    pool: EndpointPool = kwargs["pool"]
    completed = 0
    reused = 0

    def run(window: WindowRecord) -> bool:
        _, existed = run_one_window(
            mode="frozen",
            window=window,
            user_prompt=window.user_prompt,
            expected_states=window.source_states,
            **kwargs,
        )
        return existed

    with concurrent.futures.ThreadPoolExecutor(max_workers=pool.slot_count) as executor:
        futures = {executor.submit(run, window): window for window in windows}
        for future in concurrent.futures.as_completed(futures):
            reused += int(future.result())
            completed += 1
            if completed % 50 == 0 or completed == len(windows):
                print(f"[evaluator-replay] frozen progress={completed}/{len(windows)} reused={reused}", flush=True)
    return completed, reused


def run_chained_replay(
    trajectories: Sequence[TrajectoryRecord],
    **kwargs: Any,
) -> tuple[int, int]:
    judged = [trajectory for trajectory in trajectories if trajectory.windows]
    if not judged:
        raise ReplayError("chained replay selection has no judged trajectories")
    pool: EndpointPool = kwargs["pool"]
    completed = 0
    reused = 0
    progress_lock = threading.Lock()

    def run(trajectory: TrajectoryRecord) -> tuple[int, int]:
        current_states = trajectory.windows[0].source_states
        trajectory_completed = 0
        trajectory_reused = 0
        for position, window in enumerate(trajectory.windows):
            user_prompt = (
                window.user_prompt if position == 0 else replace_current_rubrics(window.user_prompt, current_states)
            )
            result, existed = run_one_window(
                mode="chained",
                window=window,
                user_prompt=user_prompt,
                expected_states=current_states,
                **kwargs,
            )
            current_states = update_states(current_states, result["results"])
            trajectory_completed += 1
            trajectory_reused += int(existed)
        return trajectory_completed, trajectory_reused

    with concurrent.futures.ThreadPoolExecutor(max_workers=pool.slot_count) as executor:
        futures = {executor.submit(run, trajectory): trajectory for trajectory in judged}
        for future in concurrent.futures.as_completed(futures):
            trajectory_completed, trajectory_reused = future.result()
            with progress_lock:
                completed += trajectory_completed
                reused += trajectory_reused
                if completed % 50 < trajectory_completed or len(futures) == 1:
                    print(
                        f"[evaluator-replay] chained windows={completed} "
                        f"trajectories_done={sum(item.done() for item in futures)}/{len(futures)} reused={reused}",
                        flush=True,
                    )
    return completed, reused


def _percentile(values: Sequence[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    return float(ordered[lower] * (upper - position) + ordered[upper] * (position - lower))


def score_metrics(trajectories: Sequence[TrajectoryRecord], successes: dict[str, bool]) -> dict[str, Any]:
    by_task: dict[str, list[bool]] = defaultdict(list)
    for trajectory in trajectories:
        if trajectory.key not in successes:
            raise ReplayError(f"score map omits trajectory {trajectory.key}")
        by_task[trajectory.task_key].append(successes[trajectory.key])
    trial_counts = {len(values) for values in by_task.values()}
    if len(trial_counts) != 1:
        raise ReplayError(f"score inputs have inconsistent trials per task: {trial_counts}")
    trial_count = next(iter(trial_counts))
    if trial_count <= 0:
        raise ReplayError("score inputs have no trials")
    total = len(trajectories)
    successful = sum(successes[trajectory.key] for trajectory in trajectories)
    any_success = sum(any(values) for values in by_task.values())
    all_success = sum(all(values) for values in by_task.values())
    histogram = Counter(sum(values) for values in by_task.values())
    return {
        "tasks": len(by_task),
        "trajectories": total,
        "successful_trajectories": successful,
        "avg_at_k": successful / total,
        "pass_at_k_any": any_success / len(by_task),
        "pass_pow_k_all": all_success / len(by_task),
        "success_count_histogram": {str(index): histogram[index] for index in range(trial_count + 1)},
    }


def _bootstrap_metric_differences(
    trajectories: Sequence[TrajectoryRecord],
    baseline: dict[str, bool],
    candidate: dict[str, bool],
    *,
    samples: int,
    seed: int,
) -> dict[str, list[float]]:
    by_task: dict[str, tuple[list[bool], list[bool]]] = {}
    grouped_baseline: dict[str, list[bool]] = defaultdict(list)
    grouped_candidate: dict[str, list[bool]] = defaultdict(list)
    for trajectory in trajectories:
        grouped_baseline[trajectory.task_key].append(baseline[trajectory.key])
        grouped_candidate[trajectory.task_key].append(candidate[trajectory.key])
    for key in grouped_baseline:
        by_task[key] = (grouped_baseline[key], grouped_candidate[key])
    values = list(by_task.values())
    rng = random.Random(seed)
    differences = {"avg_at_k": [], "pass_at_k_any": [], "pass_pow_k_all": []}
    for _ in range(samples):
        sampled = [values[rng.randrange(len(values))] for _ in values]
        trial_count = sum(len(item[0]) for item in sampled)
        differences["avg_at_k"].append(
            (sum(sum(item[1]) for item in sampled) - sum(sum(item[0]) for item in sampled)) / trial_count
        )
        differences["pass_at_k_any"].append(
            sum(int(any(item[1])) - int(any(item[0])) for item in sampled) / len(sampled)
        )
        differences["pass_pow_k_all"].append(
            sum(int(all(item[1])) - int(all(item[0])) for item in sampled) / len(sampled)
        )
    return {key: [_percentile(values, 0.025), _percentile(values, 0.975)] for key, values in differences.items()}


def _agreement(baseline: Sequence[bool], candidate: Sequence[bool]) -> dict[str, Any]:
    if len(baseline) != len(candidate) or not baseline:
        raise ReplayError("agreement inputs must be nonempty and equal length")
    pairs = [(baseline[index], value) for index, value in enumerate(candidate)]
    true_true = sum(left and right for left, right in pairs)
    false_false = sum(not left and not right for left, right in pairs)
    losses = sum(left and not right for left, right in pairs)
    gains = sum(not left and right for left, right in pairs)
    return {
        "count": len(baseline),
        "agreement": (true_true + false_false) / len(baseline),
        "true_true": true_true,
        "false_false": false_false,
        "baseline_true_candidate_false": losses,
        "baseline_false_candidate_true": gains,
    }


def load_mode_results(
    dataset: SourceDataset,
    output_dir: Path,
    mode: str,
    trajectories: Sequence[TrajectoryRecord],
    base_output_dir: Path | None = None,
) -> tuple[dict[str, dict[str, Any]], dict[str, bool], dict[str, tuple[bool, ...]]]:
    results_by_window: dict[str, dict[str, Any]] = {}
    successes: dict[str, bool] = {}
    final_rubrics: dict[str, tuple[bool, ...]] = {}
    for trajectory in trajectories:
        if not trajectory.windows:
            successes[trajectory.key] = trajectory.baseline_success
            final_rubrics[trajectory.key] = trajectory.baseline_final_rubrics
            continue
        current_states = trajectory.windows[0].source_states
        for position, window in enumerate(trajectory.windows):
            user_prompt = (
                window.user_prompt
                if mode == "frozen" or position == 0
                else replace_current_rubrics(window.user_prompt, current_states)
            )
            expected_states = window.source_states if mode == "frozen" else current_states
            path = result_path(output_dir, mode, window)
            if not path.is_file() and base_output_dir is not None:
                path = result_path(base_output_dir, mode, window)
            if not path.is_file():
                raise ReplayError(f"missing {mode} replay result for {window.key}: {path}")
            result = validate_replay_result(
                read_json(path),
                mode=mode,
                window=window,
                user_prompt=user_prompt,
                expected_states=expected_states,
            )
            results_by_window[window.key] = result
            if mode == "chained":
                current_states = update_states(current_states, result["results"])
        last_results = results_by_window[trajectory.windows[-1].key]["results"]
        decisions = tuple(item["meetExpectation"] for item in last_results)
        final_rubrics[trajectory.key] = decisions
        successes[trajectory.key] = bool(decisions) and all(decisions)
    return results_by_window, successes, final_rubrics


def analyze_mode(
    dataset: SourceDataset,
    output_dir: Path,
    mode: str,
    trajectories: Sequence[TrajectoryRecord],
    *,
    baseline_evaluator_mean_seconds: float,
    baseline_total_critical_path_hours: float,
    bootstrap_samples: int,
    base_output_dir: Path | None = None,
) -> dict[str, Any]:
    results, candidate_success, candidate_rubrics = load_mode_results(
        dataset,
        output_dir,
        mode,
        trajectories,
        base_output_dir,
    )
    selected_windows = [window for trajectory in trajectories for window in trajectory.windows]
    overlay_windows = sum(result_path(output_dir, mode, window).is_file() for window in selected_windows)
    semantic_attempt_counts: list[int] = []
    request_seconds: list[float] = []
    window_seconds: list[float] = []
    pool_wait_seconds = 0.0
    completion_tokens = 0
    prompt_tokens = 0
    reasoning_tokens = 0
    reasoning_responses = 0
    http_failures = 0
    non_stop_finishes = 0
    for window in selected_windows:
        result = results[window.key]
        attempts = result["semantic_attempts"]
        semantic_attempt_counts.append(len(attempts))
        one_window_seconds = 0.0
        for attempt in attempts:
            duration = float(attempt["request_seconds"])
            request_seconds.append(duration)
            one_window_seconds += duration
            pool_wait_seconds += float(attempt.get("pool_wait_seconds", 0.0))
            usage = _validate_usage(attempt["usage"], f"{window.key}.usage")
            prompt_tokens += usage["prompt_tokens"]
            completion_tokens += usage["completion_tokens"]
            reasoning_tokens += usage["reasoning_tokens"]
            reasoning_responses += int(bool(attempt.get("reasoning_content")))
            http_failures += len(attempt.get("http_failures") or [])
            non_stop_finishes += int(attempt.get("finish_reason") != "stop")
        window_seconds.append(one_window_seconds)

    baseline_window_decisions: list[bool] = []
    candidate_window_decisions: list[bool] = []
    for window in selected_windows:
        baseline_window_decisions.extend(item["meetExpectation"] for item in window.baseline_results)
        candidate_window_decisions.extend(item["meetExpectation"] for item in results[window.key]["results"])

    baseline_success = {trajectory.key: trajectory.baseline_success for trajectory in trajectories}
    score_baseline = score_metrics(trajectories, baseline_success)
    score_candidate = score_metrics(trajectories, candidate_success)
    score_delta = {
        key: score_candidate[key] - score_baseline[key] for key in ("avg_at_k", "pass_at_k_any", "pass_pow_k_all")
    }
    success_agreement = _agreement(
        [baseline_success[trajectory.key] for trajectory in trajectories],
        [candidate_success[trajectory.key] for trajectory in trajectories],
    )
    baseline_final_decisions: list[bool] = []
    candidate_final_decisions: list[bool] = []
    for trajectory in trajectories:
        if not trajectory.windows:
            continue
        baseline_final_decisions.extend(trajectory.baseline_final_rubrics)
        candidate_final_decisions.extend(candidate_rubrics[trajectory.key])

    per_suite: dict[str, Any] = {}
    for suite in SUITES:
        suite_trajectories = [trajectory for trajectory in trajectories if trajectory.suite == suite]
        if not suite_trajectories:
            continue
        suite_baseline = score_metrics(suite_trajectories, baseline_success)
        suite_candidate = score_metrics(suite_trajectories, candidate_success)
        per_suite[suite] = {
            "baseline": suite_baseline,
            "candidate": suite_candidate,
            "delta": {
                key: suite_candidate[key] - suite_baseline[key]
                for key in ("avg_at_k", "pass_at_k_any", "pass_pow_k_all")
            },
        }

    baseline_evaluator_seconds = baseline_evaluator_mean_seconds * len(selected_windows)
    candidate_evaluator_seconds = sum(window_seconds)
    baseline_evaluator_hours = baseline_evaluator_seconds / 3600
    candidate_evaluator_hours = candidate_evaluator_seconds / 3600
    fixed_other_hours = max(0.0, baseline_total_critical_path_hours - baseline_evaluator_hours)
    estimated_total_candidate_hours = fixed_other_hours + candidate_evaluator_hours
    baseline_completion_tokens = sum(
        _required_int(window.baseline_usage.get("completion_tokens"), f"{window.key}.baseline_completion_tokens")
        for window in selected_windows
    )

    complete_scope = len(trajectories) == len(dataset.trajectories)
    confidence_intervals = (
        _bootstrap_metric_differences(
            trajectories,
            baseline_success,
            candidate_success,
            samples=bootstrap_samples,
            seed=DEFAULT_BOOTSTRAP_SEED,
        )
        if complete_scope
        else None
    )
    operational_gates = {
        "all_windows_present_and_valid": len(results) == len(selected_windows),
        "first_attempt_valid_rate_gte_99_5pct": (
            sum(count == 1 for count in semantic_attempt_counts) / len(semantic_attempt_counts) >= 0.995
        ),
        "zero_reasoning_tokens": reasoning_tokens == 0,
        "zero_reasoning_content": reasoning_responses == 0,
        "all_finish_reason_stop": non_stop_finishes == 0,
    }
    report: dict[str, Any] = {
        "mode": mode,
        "scope": "full" if complete_scope else "pilot",
        "counts": {
            "trajectories": len(trajectories),
            "judged_trajectories": sum(bool(item.windows) for item in trajectories),
            "windows": len(selected_windows),
            "semantic_requests": len(request_seconds),
            "first_attempt_valid": sum(count == 1 for count in semantic_attempt_counts),
            "http_failures_retried": http_failures,
            "reasoning_responses": reasoning_responses,
            "non_stop_finishes": non_stop_finishes,
            "overlay_windows": overlay_windows,
            "base_windows": len(selected_windows) - overlay_windows if base_output_dir is not None else 0,
        },
        "operational_gates": operational_gates,
        "operational_pass": all(operational_gates.values()),
        "latency": {
            "request_mean_seconds": statistics.fmean(request_seconds),
            "request_p50_seconds": _percentile(request_seconds, 0.50),
            "request_p95_seconds": _percentile(request_seconds, 0.95),
            "window_mean_seconds": statistics.fmean(window_seconds),
            "window_p50_seconds": _percentile(window_seconds, 0.50),
            "window_p95_seconds": _percentile(window_seconds, 0.95),
            "pool_wait_seconds": pool_wait_seconds,
            "baseline_evaluator_critical_path_hours": baseline_evaluator_hours,
            "candidate_evaluator_critical_path_hours": candidate_evaluator_hours,
            "evaluator_speedup": (
                baseline_evaluator_seconds / candidate_evaluator_seconds if candidate_evaluator_seconds else None
            ),
            "baseline_total_critical_path_hours": baseline_total_critical_path_hours,
            "estimated_candidate_total_critical_path_hours": estimated_total_candidate_hours,
            "estimated_total_speedup": (
                baseline_total_critical_path_hours / estimated_total_candidate_hours
                if estimated_total_candidate_hours
                else None
            ),
        },
        "tokens": {
            "baseline_completion_tokens": baseline_completion_tokens,
            "candidate_prompt_tokens": prompt_tokens,
            "candidate_completion_tokens": completion_tokens,
            "candidate_reasoning_tokens": reasoning_tokens,
            "completion_token_reduction": (
                1 - completion_tokens / baseline_completion_tokens if baseline_completion_tokens else None
            ),
        },
        "agreement": {
            "all_windows_rubrics": _agreement(baseline_window_decisions, candidate_window_decisions),
            "final_rubrics": _agreement(baseline_final_decisions, candidate_final_decisions),
            "trajectory_success": success_agreement,
        },
        "scores": {
            "baseline": score_baseline,
            "candidate": score_candidate,
            "delta": score_delta,
            "task_cluster_bootstrap_95pct_ci": confidence_intervals,
            "per_suite": per_suite,
        },
    }
    if complete_scope:
        assert confidence_intervals is not None
        quality_gates = {
            "abs_avg_at_k_delta_lte_1pp": abs(score_delta["avg_at_k"]) <= 0.01,
            "avg_at_k_95pct_ci_within_2pp_equivalence_margin": (
                confidence_intervals["avg_at_k"][0] >= -0.02 and confidence_intervals["avg_at_k"][1] <= 0.02
            ),
            "trajectory_success_agreement_gte_97pct": success_agreement["agreement"] >= 0.97,
            "final_rubric_agreement_gte_97pct": report["agreement"]["final_rubrics"]["agreement"] >= 0.97,
            "each_suite_abs_avg_at_k_delta_lte_3pp": all(
                abs(value["delta"]["avg_at_k"]) <= 0.03 for value in per_suite.values()
            ),
        }
        latency_gates = {
            "evaluator_speedup_gte_3x": report["latency"]["evaluator_speedup"] >= 3.0,
            "estimated_total_speedup_gte_2x": report["latency"]["estimated_total_speedup"] >= 2.0,
            "completion_token_reduction_gte_60pct": report["tokens"]["completion_token_reduction"] >= 0.60,
        }
        report["quality_gates"] = quality_gates
        report["quality_pass"] = all(quality_gates.values())
        report["latency_gates"] = latency_gates
        report["latency_pass"] = all(latency_gates.values())
    return report


def verify_fast_source_identity(dataset: SourceDataset, experiment_manifest: dict[str, Any]) -> None:
    source = experiment_manifest.get("source")
    if not isinstance(source, dict):
        raise ReplayError("experiment manifest source identity is missing")
    if source.get("artifact_dir") != str(dataset.artifact_dir):
        raise ReplayError("experiment manifest points at a different source artifact")
    if source.get("manifest_sha256") != sha256_file(dataset.artifact_dir / "manifest.json"):
        raise ReplayError("source manifest changed after experiment creation")
    expected_files = source.get("shards")
    if not isinstance(expected_files, list) or len(expected_files) != len(dataset.shard_paths):
        raise ReplayError("experiment manifest shard inventory is incomplete")
    expected_by_path = {item.get("relative_path"): item for item in expected_files if isinstance(item, dict)}
    for path in dataset.shard_paths:
        relative = str(path.relative_to(dataset.artifact_dir))
        expected = expected_by_path.get(relative)
        if expected is None or expected.get("size_bytes") != path.stat().st_size:
            raise ReplayError(f"source shard identity changed: {relative}")


def verify_base_results_compatibility(
    experiment_manifest: dict[str, Any],
    base_output_dir: Path,
) -> None:
    base_manifest = read_json(base_output_dir / "manifest.json", "base replay manifest")
    if not isinstance(base_manifest, dict) or base_manifest.get("schema") != MANIFEST_SCHEMA:
        raise ReplayError("base replay manifest is missing or invalid")
    if base_manifest.get("source") != experiment_manifest.get("source"):
        raise ReplayError("base replay results use a different immutable source")
    model_fields = ("served_name", "base_model", "model_files_sha256", "dtype", "context_length")
    model = experiment_manifest.get("model") or {}
    base_model = base_manifest.get("model") or {}
    if any(model.get(field) != base_model.get(field) for field in model_fields):
        raise ReplayError("base replay results use a different evaluator model configuration")
    protocol_fields = ("temperature", "max_tokens", "enable_thinking", "modes")
    protocol = experiment_manifest.get("protocol") or {}
    base_protocol = base_manifest.get("protocol") or {}
    if any(protocol.get(field) != base_protocol.get(field) for field in protocol_fields):
        raise ReplayError("base replay results use an incompatible evaluator protocol")


def command_audit(args: argparse.Namespace) -> int:
    dataset = load_source_dataset(args.source_artifact_dir)
    counts = source_counts(dataset)
    journal_usage = audit_source_journals(dataset) if args.verify_journals else None
    inventory = build_source_inventory(dataset)
    audit = {
        "schema": AUDIT_SCHEMA,
        "generated_at": utc_now(),
        "source_artifact_dir": str(dataset.artifact_dir),
        "counts": counts,
        "journal_usage": journal_usage,
        "source_inventory": inventory,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    audit_path = args.output_dir / "source_audit.json"
    atomic_write_json(audit_path, audit)

    manifest_path = args.output_dir / "manifest.json"
    immutable = {
        "schema": MANIFEST_SCHEMA,
        "source": {
            "artifact_dir": str(dataset.artifact_dir),
            **inventory,
        },
        "model": {
            "served_name": args.model,
            "base_model": args.base_model,
            "model_files_sha256": args.model_files_sha256,
            "dtype": "bfloat16",
            "context_length": args.context_length,
        },
        "protocol": {
            "temperature": 0,
            "max_tokens": args.max_tokens,
            "enable_thinking": False,
            "modes": ["frozen", "chained"],
            "semantic_attempts": args.max_semantic_attempts,
            "http_retries": args.max_http_retries,
            "replayer_sha256": args.replayer_sha256,
            "runner_sha256": args.runner_sha256,
        },
    }
    if manifest_path.is_file():
        existing = read_json(manifest_path)
        comparable = dict(existing) if isinstance(existing, dict) else existing
        if isinstance(comparable, dict):
            comparable.pop("created_at", None)
        if comparable != immutable:
            raise ReplayError("existing replay manifest differs from the requested immutable protocol")
    else:
        atomic_write_json(manifest_path, {**immutable, "created_at": utc_now()})
    print("[evaluator-replay] SOURCE_AUDIT_OK", json.dumps(counts, sort_keys=True), flush=True)
    return 0


def command_replay(args: argparse.Namespace) -> int:
    dataset = load_source_dataset(args.source_artifact_dir)
    experiment_manifest = read_json(args.output_dir / "manifest.json", "replay manifest")
    if not isinstance(experiment_manifest, dict) or experiment_manifest.get("schema") != MANIFEST_SCHEMA:
        raise ReplayError("run audit before replay to create the immutable experiment manifest")
    verify_fast_source_identity(dataset, experiment_manifest)
    protocol = experiment_manifest.get("protocol") or {}
    requested_protocol = {
        "temperature": 0,
        "max_tokens": args.max_tokens,
        "enable_thinking": False,
        "semantic_attempts": args.max_semantic_attempts,
        "http_retries": args.max_http_retries,
    }
    if any(protocol.get(key) != value for key, value in requested_protocol.items()):
        raise ReplayError("replay CLI settings differ from the immutable experiment manifest")
    model = experiment_manifest.get("model") or {}
    if model.get("served_name") != args.model:
        raise ReplayError("replay model differs from the immutable experiment manifest")
    if args.base_output_dir is not None:
        verify_base_results_compatibility(experiment_manifest, args.base_output_dir)

    trajectories = dataset.trajectories
    scope = "full"
    if args.pilot_tasks_per_suite is not None:
        trajectories = select_pilot_tasks(trajectories, args.pilot_tasks_per_suite)
        scope = "pilot"
    pool = EndpointPool(args.endpoints, args.slots_per_endpoint)
    session_id = f"{args.mode}-{scope}-{time.time_ns()}"
    started_at_ns = time.time_ns()
    common = {
        "output_dir": args.output_dir,
        "pool": pool,
        "model": args.model,
        "max_tokens": args.max_tokens,
        "max_semantic_attempts": args.max_semantic_attempts,
        "max_http_retries": args.max_http_retries,
        "timeout_seconds": args.timeout_seconds,
        "session_id": session_id,
        "base_output_dir": args.base_output_dir,
    }
    if args.mode == "frozen":
        completed, reused = run_frozen_replay(trajectories, **common)
    else:
        completed, reused = run_chained_replay(trajectories, **common)
    ended_at_ns = time.time_ns()
    session = {
        "schema": SESSION_SCHEMA,
        "session_id": session_id,
        "mode": args.mode,
        "scope": scope,
        "started_at_unix_ns": started_at_ns,
        "ended_at_unix_ns": ended_at_ns,
        "wall_seconds": (ended_at_ns - started_at_ns) / 1e9,
        "windows_checked": completed,
        "windows_reused": reused,
        "windows_requested": completed - reused,
        "endpoint_count": len(args.endpoints),
        "slots_per_endpoint": args.slots_per_endpoint,
        "base_output_dir": str(args.base_output_dir.resolve()) if args.base_output_dir is not None else None,
    }
    atomic_write_json(args.output_dir / "sessions" / f"{session_id}.json", session)
    print("[evaluator-replay] REPLAY_OK", json.dumps(session, sort_keys=True), flush=True)
    return 0


def command_analyze(args: argparse.Namespace) -> int:
    dataset = load_source_dataset(args.source_artifact_dir)
    experiment_manifest = read_json(args.output_dir / "manifest.json", "replay manifest")
    if not isinstance(experiment_manifest, dict) or experiment_manifest.get("schema") != MANIFEST_SCHEMA:
        raise ReplayError("replay manifest is missing or invalid")
    verify_fast_source_identity(dataset, experiment_manifest)
    if args.base_output_dir is not None:
        verify_base_results_compatibility(experiment_manifest, args.base_output_dir)
    trajectories = dataset.trajectories
    scope = "full"
    if args.pilot_tasks_per_suite is not None:
        trajectories = select_pilot_tasks(trajectories, args.pilot_tasks_per_suite)
        scope = "pilot"
    reports = {
        mode: analyze_mode(
            dataset,
            args.output_dir,
            mode,
            trajectories,
            baseline_evaluator_mean_seconds=args.baseline_evaluator_mean_seconds,
            baseline_total_critical_path_hours=args.baseline_total_critical_path_hours,
            bootstrap_samples=args.bootstrap_samples,
            base_output_dir=args.base_output_dir,
        )
        for mode in args.modes
    }
    recommendation = None
    if scope == "full" and "chained" in reports:
        chained = reports["chained"]
        clean_primary_protocol = args.base_output_dir is None
        recommendation = {
            "switch_evaluator_to_nonthinking": bool(
                clean_primary_protocol
                and chained["operational_pass"]
                and chained["quality_pass"]
                and chained["latency_pass"]
            ),
            "requires_all_gates": [
                "clean_primary_protocol",
                "operational_pass",
                "quality_pass",
                "latency_pass",
            ],
            "clean_primary_protocol": clean_primary_protocol,
        }
    output = {
        "schema": REPORT_SCHEMA,
        "generated_at": utc_now(),
        "scope": scope,
        "source_artifact_dir": str(dataset.artifact_dir),
        "experiment_manifest": str((args.output_dir / "manifest.json").resolve()),
        "baseline_latency_assumptions": {
            "evaluator_mean_seconds_per_window": args.baseline_evaluator_mean_seconds,
            "total_critical_path_hours": args.baseline_total_critical_path_hours,
        },
        "modes": reports,
        "recommendation": recommendation,
    }
    if args.base_output_dir is not None:
        output["recovery_supplement"] = {
            "base_output_dir": str(args.base_output_dir.resolve()),
            "reason": args.recovery_reason,
            "primary_protocol_complete": False,
        }
    report_path = args.report or args.output_dir / ("pilot_report.json" if scope == "pilot" else "report.json")
    atomic_write_json(report_path, output)
    print("[evaluator-replay] ANALYSIS_OK", str(report_path), flush=True)
    if args.fail_on_operational_gate and not all(report["operational_pass"] for report in reports.values()):
        raise ReplayError("one or more replay modes failed the operational gate")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    audit = subparsers.add_parser("audit", help="strictly audit and fingerprint the persisted source")
    audit.add_argument("--source-artifact-dir", type=Path, required=True)
    audit.add_argument("--output-dir", type=Path, required=True)
    audit.add_argument("--verify-journals", action="store_true")
    audit.add_argument("--model", default=DEFAULT_MODEL)
    audit.add_argument("--base-model", default="Qwen3.6-27B")
    audit.add_argument("--model-files-sha256", required=True)
    audit.add_argument("--replayer-sha256", required=True)
    audit.add_argument("--runner-sha256", required=True)
    audit.add_argument("--context-length", type=int, default=32768)
    audit.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    audit.add_argument("--max-semantic-attempts", type=int, default=DEFAULT_SEMANTIC_ATTEMPTS)
    audit.add_argument("--max-http-retries", type=int, default=DEFAULT_HTTP_RETRIES)
    audit.set_defaults(handler=command_audit)

    replay = subparsers.add_parser("replay", help="run or resume one replay mode")
    replay.add_argument("--source-artifact-dir", type=Path, required=True)
    replay.add_argument("--output-dir", type=Path, required=True)
    replay.add_argument("--mode", choices=("frozen", "chained"), required=True)
    replay.add_argument("--endpoints", nargs="+", required=True)
    replay.add_argument("--slots-per-endpoint", type=int, default=4)
    replay.add_argument("--base-output-dir", type=Path)
    replay.add_argument("--pilot-tasks-per-suite", type=int)
    replay.add_argument("--model", default=DEFAULT_MODEL)
    replay.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    replay.add_argument("--max-semantic-attempts", type=int, default=DEFAULT_SEMANTIC_ATTEMPTS)
    replay.add_argument("--max-http-retries", type=int, default=DEFAULT_HTTP_RETRIES)
    replay.add_argument("--timeout-seconds", type=float, default=600.0)
    replay.set_defaults(handler=command_replay)

    analyze = subparsers.add_parser("analyze", help="validate and compare completed replay results")
    analyze.add_argument("--source-artifact-dir", type=Path, required=True)
    analyze.add_argument("--output-dir", type=Path, required=True)
    analyze.add_argument("--modes", nargs="+", choices=("frozen", "chained"), default=["frozen", "chained"])
    analyze.add_argument("--pilot-tasks-per-suite", type=int)
    analyze.add_argument("--base-output-dir", type=Path)
    analyze.add_argument("--recovery-reason")
    analyze.add_argument("--baseline-evaluator-mean-seconds", type=float, default=125.5)
    analyze.add_argument("--baseline-total-critical-path-hours", type=float, default=276.0)
    analyze.add_argument("--bootstrap-samples", type=int, default=DEFAULT_BOOTSTRAP_SAMPLES)
    analyze.add_argument("--report", type=Path)
    analyze.add_argument("--fail-on-operational-gate", action="store_true")
    analyze.set_defaults(handler=command_analyze)
    return parser


def validate_args(args: argparse.Namespace) -> None:
    for name in ("max_tokens", "max_semantic_attempts", "max_http_retries"):
        if hasattr(args, name) and getattr(args, name) < (0 if name == "max_http_retries" else 1):
            raise ReplayError(f"--{name.replace('_', '-')} is out of range")
    if hasattr(args, "slots_per_endpoint") and args.slots_per_endpoint <= 0:
        raise ReplayError("--slots-per-endpoint must be positive")
    if hasattr(args, "timeout_seconds") and args.timeout_seconds <= 0:
        raise ReplayError("--timeout-seconds must be positive")
    if hasattr(args, "bootstrap_samples") and args.bootstrap_samples <= 0:
        raise ReplayError("--bootstrap-samples must be positive")
    if hasattr(args, "baseline_evaluator_mean_seconds") and args.baseline_evaluator_mean_seconds <= 0:
        raise ReplayError("--baseline-evaluator-mean-seconds must be positive")
    if hasattr(args, "baseline_total_critical_path_hours") and args.baseline_total_critical_path_hours <= 0:
        raise ReplayError("--baseline-total-critical-path-hours must be positive")
    if getattr(args, "base_output_dir", None) is not None:
        if args.base_output_dir.resolve() == args.output_dir.resolve():
            raise ReplayError("--base-output-dir must differ from --output-dir")
        if args.command == "analyze" and not args.recovery_reason:
            raise ReplayError("--recovery-reason is required with --base-output-dir")


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        validate_args(args)
        return args.handler(args)
    except ReplayError as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
