"""State potentials and DB-count turn credit for tau2 rollouts."""

from __future__ import annotations

import math
import time
import threading
from collections import defaultdict
from functools import lru_cache

from tau2.data_model.tasks import RewardType
from tau2.environment.environment import Environment
from tau2.orchestrator.orchestrator import Orchestrator, Role


def flatten_fields(value, path=()):
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
    records = {}
    for side, database in snapshot.items():
        # Banking tables wrap ID-keyed records in data on both sides.
        banking = all(isinstance(table, dict) and isinstance(table.get("data"), dict)
                      for table in database.values())
        if banking:
            for table, rows in database.items():
                records.update({(side, table, key): value for key, value in rows["data"].items()})
        elif side == "assistant":
            for table, rows in database.items():
                if isinstance(rows, list):
                    rows = {row[TELECOM_RECORD_IDS[table]]: row for row in rows}
                records.update({(side, table, key): value for key, value in rows.items()})
        else:
            records.update(flatten_fields(database, (side,)))
    return records


def field_distance(current, target):
    missing = object()
    return sum(current.get(key, missing) != target.get(key, missing)
               for key in current.keys() | target.keys())


def tree_field_distance(current, target):
    if current == target:
        return 0
    if isinstance(current, dict) and current and isinstance(target, dict) and target:
        distance = 0
        for key, value in current.items():
            distance += tree_field_distance(value, target[key]) if key in target else len(flatten_fields(value))
        for key, value in target.items():
            if key not in current:
                distance += len(flatten_fields(value))
        return distance
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


# lru_cache alone permits concurrent misses to build the same large target K
# times. Serialize the first construction; all members then share read-only state.
REFERENCE_DATABASE_LOCK = threading.Lock()


@lru_cache(maxsize=64)
def cached_reference_database(domain, db_path, task_json):
    from pathlib import Path
    from envs import make_environment_constructor
    from tau2.data_model.tasks import Task
    task = Task.model_validate_json(task_json)
    return reference_database(task, make_environment_constructor(domain=domain, db_path=Path(db_path), task=task))


class StatePotential:
    def __init__(self, task, environment_constructor, *, reference_key=None):
        self.reason = None
        self.seconds = 0.0
        self.assertions = None
        self.target = None
        self._score_snapshot = None
        self._snapshot_score = None
        started = time.perf_counter()
        try:
            criteria = task.evaluation_criteria
            if RewardType.ENV_ASSERTION in criteria.reward_basis:
                self.assertions = criteria.env_assertions
                if not self.assertions:
                    raise ValueError("ENV_ASSERTION task has no assertions")
            elif RewardType.DB in criteria.reward_basis:
                if reference_key is not None:
                    with REFERENCE_DATABASE_LOCK:
                        self.target = cached_reference_database(*reference_key, task.model_dump_json())
                else:
                    self.target = reference_database(task, environment_constructor)
            else:
                raise ValueError("Unsupported progress reward basis")
        except Exception as exc:
            self.reason = f"target: {type(exc).__name__}: {exc}"
        self.seconds += time.perf_counter() - started

    def score(self, environment, *, snapshot=None):
        if self.reason:
            return None
        started = time.perf_counter()
        try:
            if self.assertions is not None:
                values = [environment.run_env_assertion(a, raise_assertion_error=False) for a in self.assertions]
                return sum(bool(value) for value in values) / len(values)
            if snapshot is None:
                snapshot = database_snapshot(environment)
            if snapshot is not self._score_snapshot:
                self._snapshot_score = -float(tree_field_distance(snapshot, self.target))
                self._score_snapshot = snapshot
            return self._snapshot_score
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

    def step(self, *, snapshot=None):
        decision = self.from_role in (Role.USER, Role.ENV) and self.to_role == Role.AGENT
        score = self.progress_potential.score(self.environment, snapshot=snapshot) if decision else None
        super().step()
        if decision:
            self.progress_scores.append(score)
            self.progress_messages.append(self.message)

    def progress_payload(self, simulation):
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
    def initialize(self):
        super().initialize()
        started = time.perf_counter()
        self.initial_records = database_records(database_snapshot(self.environment))
        self.progress_potential.seconds += time.perf_counter() - started
        self.db_diff_counts = []
        self.db_diff_side_counts = []
        # Only environments with a no-op sync can retain state between writes.
        self._reuse_snapshot = type(self.environment).sync_tools is Environment.sync_tools
        self._current_snapshot = None

    def step(self):
        decision = self.from_role in (Role.USER, Role.ENV) and self.to_role == Role.AGENT
        snapshot = None
        reuse = getattr(self, "_reuse_snapshot", False)
        if reuse and self.to_role == Role.ENV:
            if any(self.environment._is_mutating_tool(call.name) for call in self.message.tool_calls):
                # Invalidate before execution: a failing write can still mutate state.
                self._current_snapshot = None
        if decision:
            started = time.perf_counter()
            snapshot = self._current_snapshot if reuse else None
            if snapshot is None:
                snapshot = database_snapshot(self.environment)
                current = database_records(snapshot)
                missing = object()
                counts = {"assistant": 0, "user": 0}
                for key in current.keys() | self.initial_records.keys():
                    if current.get(key, missing) != self.initial_records.get(key, missing):
                        counts[key[0]] += 1
                if reuse:
                    self._current_snapshot = snapshot
                    self._current_counts = counts
            else:
                counts = self._current_counts
            self.progress_potential.seconds += time.perf_counter() - started
        super().step(snapshot=snapshot)
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
    gamma = float(gamma)
    if not math.isfinite(gamma) or not 0.0 <= gamma <= 1.0:
        raise ValueError(f"gamma must be finite and in [0, 1], got {gamma}")
    rtgs = []
    for scores in score_sequences:
        values = [0.0] * max(0, len(scores) - 1)
        running = 0.0
        for turn in range(len(values) - 1, -1, -1):
            running = scores[turn + 1] - scores[turn] + gamma * running
            values[turn] = running
        rtgs.append(values)
    advantages = [[0.0] * len(values) for values in rtgs]
    for turn in range(max(map(len, rtgs), default=0)):
        present = [i for i, values in enumerate(rtgs) if turn < len(values)]
        for i, value in zip(present, group_normalize([rtgs[j][turn] for j in present]), strict=True):
            advantages[i][turn] = value
    return rtgs, advantages


def db_count_group_advantages(score_sequences, count_sequences, gamma=0.98):
    gamma = float(gamma)
    if not math.isfinite(gamma) or not 0.0 <= gamma <= 1.0:
        raise ValueError(f"gamma must be finite and in [0, 1], got {gamma}")
    rtgs, buckets = [], defaultdict(list)
    for i, (scores, counts) in enumerate(zip(score_sequences, count_sequences, strict=True)):
        if len(scores) != len(counts) + 1:
            raise ValueError("DB counts must align with pre-action decision boundaries")
        values, running = [0.0] * len(counts), 0.0
        for t in range(len(counts) - 1, -1, -1):
            running = scores[t + 1] - scores[t] + gamma * running
            values[t] = running
            buckets[counts[t]].append((i, t))
        rtgs.append(values)
    advantages = [[0.0] * len(values) for values in rtgs]
    for members in buckets.values():
        for (i, t), value in zip(members, group_normalize([rtgs[i][j] for i, j in members]), strict=True):
            advantages[i][t] = value
    return rtgs, advantages
