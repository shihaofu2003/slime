"""State potentials at actual Agent decision boundaries for progress-db-count-v1."""

from __future__ import annotations

import math
import time
from functools import lru_cache
from collections import defaultdict

from tau2.data_model.tasks import RewardType
from tau2.orchestrator.orchestrator import Orchestrator, Role


def flatten_fields(value, path=()):
    """Lists are atomic; empty dictionaries and null remain distinct leaves."""
    fields = {}
    pending = [(path, value)]
    while pending:
        prefix, child = pending.pop()
        if isinstance(child, dict) and child:
            pending.extend((prefix + (name,), item) for name, item in child.items())
        else:
            fields[prefix] = child
    return fields


def database_snapshot(environment):
    databases = {"assistant": environment.tools.db.model_dump(mode="json")}
    if environment.user_tools is not None:
        databases["user"] = environment.user_tools.db.model_dump(mode="json")
    return databases


def database_fields(environment):
    return flatten_fields(database_snapshot(environment))


TELECOM_RECORD_IDS = {
    "plans": "plan_id", "customers": "customer_id", "lines": "line_id",
    "bills": "bill_id", "devices": "device_id",
}


def database_records(snapshot):
    """Agent table rows and User state leaves, in separate namespaces.

    Retail/airline tables are ID-keyed dictionaries; telecom tables are lists
    indexed here by their record IDs. User device/surroundings are structured
    state, so each leaf is an entry (lists remain atomic).
    """
    records = {}
    for table, rows in snapshot["assistant"].items():
        if isinstance(rows, list):
            rows = {row[TELECOM_RECORD_IDS[table]]: row for row in rows}
        records.update({("assistant", table, key): value for key, value in rows.items()})
    if "user" in snapshot:
        records.update(flatten_fields(snapshot["user"], ("user",)))
    return records


def field_distance(current, target):
    missing = object()
    return sum(current.get(key, missing) != target.get(key, missing) for key in current.keys() | target.keys())


def tree_field_distance(current, target):
    """Same leaf distance as flatten_fields, skipping equal subtrees in C.

    Take fresh JSON snapshots on every boundary: tool calls and User sync can
    mutate nested objects in place, so object identity is not a dirty marker.
    Lists remain atomic, and empty dicts remain leaves.
    """
    if current == target:
        return 0
    if isinstance(current, dict) and current and isinstance(target, dict) and target:
        distance = 0
        for key, value in current.items():
            if key in target:
                distance += tree_field_distance(value, target[key])
            else:
                distance += len(flatten_fields(value))
        for key, value in target.items():
            if key not in current:
                distance += len(flatten_fields(value))
        return distance
    # A leaf becoming a nonempty dict removes one path and adds its leaves.
    return field_distance(flatten_fields(current), flatten_fields(target))


def reference_database(task, environment_constructor):
    target = environment_constructor()
    initial = task.initial_state
    target.set_state(
        initialization_data=initial.initialization_data if initial else None,
        initialization_actions=initial.initialization_actions if initial else None,
        message_history=(initial.message_history or []) if initial else [],
    )
    for action in task.evaluation_criteria.actions or []:
        target.make_tool_call(tool_name=action.name, requestor=action.requestor, **action.arguments)
    return database_snapshot(target)


@lru_cache(maxsize=32)
def cached_reference_database(domain, db_path, task_json):
    # Training inputs are read-only. Include the full task, not just its ID;
    # different databases/initial states must never share a reference.
    from pathlib import Path

    from envs import make_environment_constructor
    from tau2.data_model.tasks import Task

    return reference_database(
        Task.model_validate_json(task_json),
        make_environment_constructor(domain=domain, db_path=Path(db_path)),
    )


class StatePotential:
    def __init__(self, task, environment_constructor, *, reference_key=None):
        self.reason = None
        self.seconds = 0.0
        self.assertions = None
        self.target = None
        started = time.perf_counter()
        try:
            criteria = task.evaluation_criteria
            if RewardType.ENV_ASSERTION in criteria.reward_basis:
                self.assertions = criteria.env_assertions
                if not self.assertions:
                    raise ValueError("ENV_ASSERTION task has no assertions")
            elif RewardType.DB in criteria.reward_basis:
                if reference_key is None:
                    self.target = reference_database(task, environment_constructor)
                else:
                    self.target = cached_reference_database(*reference_key, task.model_dump_json())
            else:
                raise ValueError("Unsupported progress reward basis")
        except Exception as exc:
            self.reason = f"target: {type(exc).__name__}: {exc}"
        self.seconds += time.perf_counter() - started

    def score(self, environment):
        if self.reason:
            return None
        started = time.perf_counter()
        try:
            if self.assertions is not None:
                values = [environment.run_env_assertion(a, raise_assertion_error=False) for a in self.assertions]
                return sum(bool(value) for value in values) / len(values)
            return -float(tree_field_distance(database_snapshot(environment), self.target))
        except Exception as exc:
            self.reason = f"boundary: {type(exc).__name__}: {exc}"
            return None
        finally:
            self.seconds += time.perf_counter() - started


class ProgressOrchestrator(Orchestrator):
    def __init__(self, *, progress_potential, **kwargs):
        super().__init__(**kwargs)
        self.progress_potential = progress_potential
        self.progress_scores = []
        self.progress_messages = []

    def step(self):
        decision = self.from_role in (Role.USER, Role.ENV) and self.to_role == Role.AGENT
        if decision:
            score = self.progress_potential.score(self.environment)
        super().step()
        if decision:
            self.progress_scores.append(score)
            self.progress_messages.append(self.message)

    def progress_payload(self, simulation):
        # get_trajectory sorts/deep-copies messages; locate by message identity fields,
        # rather than treating an append-order offset as a simulation index.
        indices = []
        for generated in self.progress_messages:
            matches = [i for i, message in enumerate(simulation.messages)
                       if message.role == "assistant" and message.timestamp == generated.timestamp
                       and message.raw_data == generated.raw_data]
            if len(matches) != 1:
                from reward_postprocess import TurnCreditAlignmentError
                raise TurnCreditAlignmentError("Progress boundary has no unique simulation Assistant message")
            indices.append(matches[0])
        scores = self.progress_scores + [self.progress_potential.score(self.environment)]
        return {"source_indices": indices, "scores": scores,
                "unavailable_reason": self.progress_potential.reason,
                "scoring_seconds": self.progress_potential.seconds}


class DBCountProgressOrchestrator(ProgressOrchestrator):
    """Capture initialized DB baseline and counts before each Agent action."""

    def initialize(self):
        super().initialize()
        self.initial_records = database_records(database_snapshot(self.environment))
        self.db_diff_counts = []
        self.db_diff_side_counts = []

    def step(self):
        decision = self.from_role in (Role.USER, Role.ENV) and self.to_role == Role.AGENT
        if decision:
            current = database_records(database_snapshot(self.environment))
            missing = object()
            counts = {"assistant": 0, "user": 0}
            for key in current.keys() | self.initial_records.keys():
                if current.get(key, missing) != self.initial_records.get(key, missing):
                    counts[key[0]] += 1
        super().step()
        if decision:
            self.db_diff_counts.append(sum(counts.values()))
            self.db_diff_side_counts.append(counts)

    def progress_payload(self, simulation):
        payload = super().progress_payload(simulation)
        payload.update(db_diff_counts=self.db_diff_counts, db_diff_side_counts=self.db_diff_side_counts)
        return payload


def group_normalize(values):
    if len(values) < 2:
        return [0.0] * len(values)
    mean = sum(values) / len(values)
    std = math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))
    return [(value - mean) / (std + 1e-6) for value in values] if std > 1e-6 else [0.0] * len(values)


def progress_group_advantages(score_sequences, gamma=1.0):
    """Discounted progress RTG, normalized only over present trajectories.

    ``scores`` contains one potential at each decision boundary, so the
    immediate progress reward is ``scores[t + 1] - scores[t]``.  Discounting
    must be applied to these increments, rather than to the final potential
    difference, because the latter is only equivalent when ``gamma == 1``.
    """
    gamma = float(gamma)
    if not math.isfinite(gamma) or not 0.0 <= gamma <= 1.0:
        raise ValueError(f"gamma must be finite and in [0, 1], got {gamma}")
    rtgs = []
    for scores in score_sequences:
        values = [0.0] * max(0, len(scores) - 1)
        running = 0.0
        for turn in range(len(values) - 1, -1, -1):
            running = (scores[turn + 1] - scores[turn]) + gamma * running
            values[turn] = running
        rtgs.append(values)
    advantages = [[0.0] * len(values) for values in rtgs]
    for turn in range(max(map(len, rtgs), default=0)):
        present = [i for i, values in enumerate(rtgs) if turn < len(values)]
        normalized = group_normalize([rtgs[i][turn] for i in present])
        for i, value in zip(present, normalized, strict=True):
            advantages[i][turn] = value
    return rtgs, advantages


def db_count_group_advantages(score_sequences, count_sequences, gamma=0.98):
    """Normalize progress RTG across turns with the same pre-action DB count.

    Call once per task rollout group. Every occurrence participates, including
    repeated visits within a trajectory; singleton/zero-variance buckets yield 0.
    """
    gamma = float(gamma)
    if not math.isfinite(gamma) or not 0.0 <= gamma <= 1.0:
        raise ValueError(f"gamma must be finite and in [0, 1], got {gamma}")
    rtgs = []
    buckets = defaultdict(list)
    for i, (scores, counts) in enumerate(zip(score_sequences, count_sequences, strict=True)):
        if len(scores) != len(counts) + 1:
            raise ValueError("DB counts must align with pre-action decision boundaries")
        values = [0.0] * len(counts)
        running = 0.0
        for t in range(len(counts) - 1, -1, -1):
            running = (scores[t + 1] - scores[t]) + gamma * running
            values[t] = running
            buckets[counts[t]].append((i, t))
        rtgs.append(values)
    advantages = [[0.0] * len(values) for values in rtgs]
    for members in buckets.values():
        normalized = group_normalize([rtgs[i][t] for i, t in members])
        for (i, t), value in zip(members, normalized, strict=True):
            advantages[i][t] = value
    return rtgs, advantages
