"""Task-defined evidence access and progress for state, actions and communication.

Installed only through the independent v4 worker hook. Terminal DB/ENV and
communication scoring stay at v3; action completion additionally requires a
successful call with the declared arguments and owner.
"""

from __future__ import annotations

import json
import sys
import time
import re
from pathlib import Path

import envs
import progress
import reward_v3
from tau2.data_model.tasks import Action, RewardType, Task
from tau2.evaluator.evaluator_action import ActionEvaluator as _OriginalActionEvaluator

_make_environment_constructor = envs.make_environment_constructor
_DBCountProgressOrchestrator = progress.DBCountProgressOrchestrator


class TrainingTask(Task):
    retrieval_variant: str | None = None


def task_from_metadata(metadata):
    return TrainingTask.model_validate({**metadata["task"], "retrieval_variant": metadata.get("retrieval_variant")})


def make_environment_constructor(*, domain, db_path, task=None, policy_type="manual"):
    if domain not in {"banking", "banking_knowledge"}:
        return _make_environment_constructor(domain=domain, db_path=db_path, task=task, policy_type=policy_type)
    analysis = str(Path(__file__).resolve().parents[1] / "analysis")
    if analysis not in sys.path:
        sys.path.insert(0, analysis)
    from banking_synthetic.validate import environment_constructor
    return environment_constructor(
        documents_dir=envs.BANKING_DOCUMENTS,
        allowed_document_ids=tuple(envs._banking_document_ids()),
        retrieval_variant=getattr(task, "retrieval_variant", None) or "bm25",
        task=task,
    )


def flatten_business_fields(value, path=()):
    if isinstance(value, dict) and value:
        return {key: leaf for name, child in value.items() for key, leaf in flatten_business_fields(child, path + (name,)).items()}
    if isinstance(value, list) and value:
        return {key: leaf for index, child in enumerate(value) for key, leaf in flatten_business_fields(child, path + (index,)).items()}
    return {path: value}


def business_field_distance(current, target):
    """Compare list contents as fields too; retain order, multiplicity and writes."""
    if current == target:
        return 0
    if isinstance(current, dict) and current and isinstance(target, dict) and target:
        return sum(business_field_distance(value, target[key]) if key in target else len(flatten_business_fields(value))
                   for key, value in current.items()) + sum(len(flatten_business_fields(value)) for key, value in target.items() if key not in current)
    if isinstance(current, list) and current and isinstance(target, list) and target:
        common = min(len(current), len(target))
        return (sum(business_field_distance(current[i], target[i]) for i in range(common))
                + sum(len(flatten_business_fields(value)) for value in current[common:])
                + sum(len(flatten_business_fields(value)) for value in target[common:]))
    return progress.field_distance(flatten_business_fields(current), flatten_business_fields(target))


def compare_completed_action(self, tool_call):
    if self.requestor != tool_call.requestor:
        return False
    keys = self.compare_args if self.compare_args is not None else list(self.arguments)
    if any(key not in tool_call.arguments for key in keys):
        return False
    selected = self.model_copy(update={"compare_args": keys})
    return reward_v3.compare_with_tool_call(selected, tool_call)


class ActionEvaluator(_OriginalActionEvaluator):
    @classmethod
    def extract_tool_calls(cls, full_trajectory):
        successful = {
            message.id for message in full_trajectory
            if message.role == "tool" and not message.error
            and not (message.content or "").lstrip().startswith(("Error:", "Failed to "))
        }
        return [call for call in super().extract_tool_calls(full_trajectory) if call.id in successful]


def normalize_fact(text):
    # Same literal fact semantics as CommunicateEvaluator.
    return text.lower().replace(",", "")


def retrieved_document_blocks(content):
    """Parse the existing ranked KB_search format, including full document text."""
    records = list(re.finditer(r"(?m)^\d+\. [^\n]+\n\s+ID:\s*(\S+)\n\s+Score:[^\n]*\n\s+Content:\s*", content))
    return [(match[1], content[match.end():records[i + 1].start() if i + 1 < len(records) else len(content)])
            for i, match in enumerate(records)]


def communication_progress(messages, facts, required_documents, evidence_weight=0.25):
    spoken = [normalize_fact(message.content or "") for message in messages if message.role == "assistant"]
    expected = set(required_documents)
    source_texts = []
    if expected:
        calls = {call.id: call for message in messages if message.role == "assistant" for call in message.tool_calls or []}
        for message in messages:
            if message.role != "tool" or message.error or message.id not in calls:
                continue
            if calls[message.id].name != "KB_search":
                continue
            source_texts.extend((doc_id, normalize_fact(text)) for doc_id, text in retrieved_document_blocks(message.content or "") if doc_id in expected)
    score = 0.0
    for fact in facts:
        normalized = normalize_fact(fact)
        if any(normalized in content for content in spoken):
            continue
        retrieved = any(normalized in content or fact == doc_id for doc_id, content in source_texts)
        score -= 1.0 - evidence_weight * retrieved
    return score


class StatePotential:
    """One potential over every component the task actually requires."""
    def __init__(self, task, environment_constructor, *, reference_key=None):
        self.reason, self.seconds, self.target = None, 0.0, None
        self.task = task
        self.messages = []
        self.criteria = task.evaluation_criteria
        self.basis = set(self.criteria.reward_basis) if self.criteria else set()
        self._score_snapshot, self._snapshot_score = None, None
        started = time.perf_counter()
        try:
            if RewardType.DB in self.basis:
                if reference_key is None:
                    self.target = progress.reference_database(task, environment_constructor)
                else:
                    with progress.REFERENCE_DATABASE_LOCK:
                        self.target = progress.cached_reference_database(*reference_key, task.model_dump_json())
        except Exception as error:
            self.reason = f"target: {type(error).__name__}: {error}"
        self.seconds += time.perf_counter() - started

    def score(self, environment, *, snapshot=None):
        if self.reason:
            return None
        started = time.perf_counter()
        try:
            value = 0.0
            if RewardType.DB in self.basis:
                snapshot = progress.database_snapshot(environment) if snapshot is None else snapshot
                if snapshot is not self._score_snapshot:
                    self._snapshot_score = -float(business_field_distance(snapshot, self.target))
                    self._score_snapshot = snapshot
                value += self._snapshot_score
            if RewardType.ENV_ASSERTION in self.basis:
                checks = self.criteria.env_assertions or []
                if checks:
                    value -= sum(not environment.run_env_assertion(check, raise_assertion_error=False) for check in checks) / len(checks)
            if RewardType.ACTION in self.basis:
                checks = ActionEvaluator.evaluate_actions(self.messages, self.criteria.actions or [])
                value -= sum(not check.action_match for check in checks)
            if RewardType.COMMUNICATE in self.basis:
                value += communication_progress(self.messages, self.criteria.communicate_info or [], self.task.required_documents or [])
            return value
        except Exception as error:
            self.reason = f"boundary: {type(error).__name__}: {error}"
            return None
        finally:
            self.seconds += time.perf_counter() - started


class GoalProgressOrchestrator(_DBCountProgressOrchestrator):
    def initialize(self):
        super().initialize()
        self.progress_potential.messages = self.trajectory

    def step(self):
        self.progress_potential.messages = self.trajectory
        super().step()


def setup_data_worker():
    reward_v3.setup_worker()
    import reward
    envs.task_from_metadata = task_from_metadata
    envs.make_environment_constructor = make_environment_constructor
    Action.compare_with_tool_call = compare_completed_action
    reward.ActionEvaluator = ActionEvaluator
    if "rollout" in sys.modules:
        sys.modules["rollout"].task_from_metadata = task_from_metadata
        sys.modules["rollout"].make_environment_constructor = make_environment_constructor


def setup_worker():
    setup_data_worker()
    import reward_postprocess
    from goal_credit_v4 import attach_advantages
    progress.StatePotential = StatePotential
    progress.DBCountProgressOrchestrator = GoalProgressOrchestrator
    reward_postprocess.attach_progress_group_advantages = attach_advantages
