#!/usr/bin/env python3
"""CPU tests for progress RTG and DB-count grouping."""

import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from progress import db_count_group_advantages, field_distance, flatten_fields, group_normalize


def test_flatten_and_field_distance():
    target = flatten_fields({"a": None, "b": [1, 2], "empty": {}})
    assert field_distance(target, target) == 0
    assert field_distance(flatten_fields({"b": [1, 2], "empty": {}}), target) == 1
    assert field_distance(flatten_fields({"a": None, "b": [2, 1], "extra": 1}), target) == 3


def test_db_count_uses_discounted_step_rewards_and_repeated_buckets():
    rtgs, advantages = db_count_group_advantages(
        [[0, 0, 0, 1], [0, 0, 1]], [[0, 0, 0], [0, 0]], gamma=0.98
    )
    assert rtgs[0] == pytest.approx([0.98 ** 2, 0.98, 1.0])
    assert advantages[0][0] < advantages[0][1] < advantages[0][2]
    assert all(math.isfinite(value) for row in advantages for value in row)


def test_singleton_and_zero_variance_buckets_are_zero():
    assert group_normalize([1.0]) == [0.0]
    assert group_normalize([1.0, 1.0]) == [0.0, 0.0]
    with pytest.raises(ValueError, match="align"):
        db_count_group_advantages([[0, 1]], [[]])

def test_tree_distance_matches_flattened_reference():
    import random
    from progress import tree_field_distance

    rng = random.Random(42)

    def value(depth):
        if depth and rng.random() < 0.6:
            return {key: value(depth - 1) for key in ("a", "b", "c") if rng.random() < 0.6}
        return rng.choice([None, {}, [], [1, 2, 2], [2, 1, 2], 0, 1, "x", "y"])

    cases = [({}, None), ({"a": {}}, {"a": {"b": 1}}),
             ({"a": [1, 2]}, {"a": [2, 1]}),
             ({"a": {"b": 1}}, {"c": {"b": 1}})]
    cases.extend((value(4), value(4)) for _ in range(1000))
    for current, target in cases:
        expected = field_distance(flatten_fields(current), flatten_fields(target))
        assert tree_field_distance(current, target) == expected
        assert tree_field_distance(target, current) == expected

def test_snapshot_detects_in_place_user_and_assistant_changes():
    from copy import deepcopy
    from progress import database_snapshot, tree_field_distance

    class DB:
        def __init__(self):
            self.data = {"entities": {"one": {"status": "ok", "items": [1, 2]}}}

        def model_dump(self, **_):
            return deepcopy(self.data)

    env = SimpleNamespace(tools=SimpleNamespace(db=DB()), user_tools=SimpleNamespace(db=DB()))
    target = database_snapshot(env)
    for tools in (env.tools, env.user_tools):
        entity = tools.db.data["entities"]["one"]
        entity["items"].append(3)
        assert tree_field_distance(database_snapshot(env), target) == 1
        entity["items"].pop()
        entity["extra"] = {"x": 1, "y": 2}
        assert tree_field_distance(database_snapshot(env), target) == 2
        del entity["extra"]
        assert tree_field_distance(database_snapshot(env), target) == 0


def test_concurrent_group_builds_reference_database_once(monkeypatch):
    import threading
    import time
    from concurrent.futures import ThreadPoolExecutor
    import progress
    from tau2.data_model.tasks import RewardType
    from types import SimpleNamespace

    calls = []
    ready = threading.Barrier(8)

    def build(*args):
        calls.append(1)
        time.sleep(0.02)  # Release the GIL while the other seven requests arrive.
        return {"assistant": {"records": {"a": 1}}}

    monkeypatch.setattr(progress, "cached_reference_database", progress.lru_cache(maxsize=64)(build))
    task = SimpleNamespace(evaluation_criteria=SimpleNamespace(reward_basis=[RewardType.DB]),
                           model_dump_json=lambda: "task")

    def create(_):
        ready.wait()
        return progress.StatePotential(task, None, reference_key=("airline", "db"))

    with ThreadPoolExecutor(8) as pool:
        results = list(pool.map(create, range(8)))
    assert len(calls) == 1
    assert all(p.reason is None and p.target is results[0].target for p in results)

@pytest.mark.parametrize("domain,basis", [("airline", "DB"), ("retail", "DB"),
                                        ("telecom", "DB"), ("telecom", "ENV_ASSERTION")])
def test_real_environment(domain, basis):
    from envs import make_environment_constructor, task_from_metadata
    from progress import StatePotential, cached_reference_database, database_fields, database_snapshot

    data = Path(__file__).resolve().parents[3].parent / "datasets/tau2/rl/areal_tasks_slime.jsonl"
    with data.open() as stream:
        for line in stream:
            metadata = json.loads(line)["metadata"]
            task = task_from_metadata(metadata)
            if metadata["domain"] == domain and basis in task.evaluation_criteria.reward_basis:
                break
        else:
            raise AssertionError(f"Missing fixture {domain}/{basis}")
    constructor = make_environment_constructor(domain=domain, db_path=Path(metadata["db_path"]))
    potential = StatePotential(task, constructor)
    assert potential.reason is None
    if basis == "DB":
        cached_reference_database.cache_clear()
        key = (domain, str(Path(metadata["db_path"]).resolve()))
        cached = StatePotential(task, constructor, reference_key=key)
        repeated = StatePotential(task, constructor, reference_key=key)
        assert cached.target == potential.target
        assert repeated.target is cached.target
        assert cached_reference_database.cache_info().hits == 1
        changed = task.model_copy(deep=True)
        changed.evaluation_criteria.actions = []
        StatePotential(changed, constructor, reference_key=key)
        assert cached_reference_database.cache_info().misses == 2
    env = constructor()
    initial = task.initial_state
    env.set_state(initialization_data=initial.initialization_data if initial else None,
                  initialization_actions=initial.initialization_actions if initial else None,
                  message_history=(initial.message_history or []) if initial else [])
    before = database_fields(env)
    first = potential.score(env)
    assert potential.score(env, snapshot=database_snapshot(env)) == first
    assert potential.score(env) == first
    assert database_fields(env) == before  # Repeated assertion checks must be read-only.
    for action in task.evaluation_criteria.actions or []:
        env.make_tool_call(tool_name=action.name, requestor=action.requestor, **action.arguments)
        assert potential.score(env, snapshot=database_snapshot(env)) == potential.score(env)
    if basis == "DB":
        assert potential.score(env) == 0
        assert cached.score(env) == repeated.score(env) == 0
        # Corrupt an unrelated field (Telecom stores entities in atomic lists), then restore it.
        assert first == -field_distance(before, flatten_fields(potential.target))
        path = next(path for path, value in flatten_fields(potential.target).items()
                    if path[0] == "assistant" and (isinstance(value, str) or isinstance(value, list) and value))
        parent = env.tools.db
        for key in path[1:-1]:
            parent = parent[key] if isinstance(parent, dict) else getattr(parent, key)
        key = path[-1]
        original = parent[key] if isinstance(parent, dict) else getattr(parent, key)
        damaged = original + "_damaged" if isinstance(original, str) else original + original[:1]
        if isinstance(parent, dict):
            parent[key] = damaged
        else:
            setattr(parent, key, damaged)
        assert potential.score(env) == -1
        assert cached.score(env) == repeated.score(env) == -1
        if isinstance(parent, dict):
            parent[key] = original
        else:
            setattr(parent, key, original)
        assert potential.score(env) == 0
        assert cached.score(env) == repeated.score(env) == 0
    else:
        expected = sum(env.run_env_assertion(a, raise_assertion_error=False)
                       for a in task.evaluation_criteria.env_assertions) / len(task.evaluation_criteria.env_assertions)
        assert potential.score(env) == expected

def test_cp_and_policy_loss_equivalence(monkeypatch):
    import torch
    import reward_postprocess as rp
    from slime.backends.megatron_utils import cp_utils
    from slime.utils.ppo_utils import compute_policy_loss

    mask = torch.tensor([1, 1, 0, 0, 1, 1, 0, 1], dtype=torch.float)
    scalar = 0.7071058
    full = mask * scalar
    for size in (1, 2):
        monkeypatch.setattr(cp_utils.mpu, "get_context_parallel_world_size", lambda size=size: size)
        for rank in range(size):
            monkeypatch.setattr(cp_utils.mpu, "get_context_parallel_rank", lambda rank=rank: rank)
            local = cp_utils.slice_log_prob_with_cp(full, 12, 8)
            data = {"metadata": [{"turn_credit_version": rp.PROGRESS_DB_COUNT_V1, "token_advantages": full.tolist()}],
                    "rewards": [scalar], "kl": [torch.zeros_like(local)], "response_lengths": [8],
                    "total_lengths": [12], "loss_masks": [mask]}
            rp.turn_aware_grpo_advantage(SimpleNamespace(advantage_estimator="grpo"), data)
            assert torch.equal(data["advantages"][0], local)
            local_mask = cp_utils.slice_log_prob_with_cp(mask, 12, 8)
            delta = torch.linspace(-0.5, 0.5, local.numel(), requires_grad=True)
            new_loss = (compute_policy_loss(delta, local, 0.2, 0.2)[0] * local_mask).sum() / mask.sum()
            old_loss = (compute_policy_loss(delta, torch.full_like(local, scalar), 0.2, 0.2)[0] * local_mask).sum() / mask.sum()
            assert torch.equal(new_loss, old_loss)
            assert torch.equal(torch.autograd.grad(new_loss, delta, retain_graph=True)[0],
                               torch.autograd.grad(old_loss, delta)[0])

def test_auxiliary_failure_disables_signal():
    from progress import StatePotential
    from tau2.data_model.tasks import RewardType

    task = SimpleNamespace(evaluation_criteria=SimpleNamespace(reward_basis=[RewardType.DB], actions=[
        SimpleNamespace(name="bad", requestor="assistant", arguments={})]), initial_state=None)
    def fail(**_):
        raise RuntimeError("reference failure")
    potential = StatePotential(task, lambda: SimpleNamespace(set_state=lambda **_: None, make_tool_call=fail))
    assert "reference failure" in potential.reason
    assert potential.score(None) is None


@pytest.mark.parametrize("domain", ["airline", "retail", "telecom"])
def test_each_task_database_path_is_loaded_and_isolated(domain):
    from envs import make_environment_constructor, FlightDB, RetailDB, TelecomDB
    data = Path(__file__).resolve().parents[4] / "datasets/tau2/rl/areal_tasks_slime.jsonl"
    paths = {json.loads(line)["metadata"]["db_path"] for line in data.open()
             if json.loads(line)["metadata"]["domain"] == domain}
    model = {"airline": FlightDB, "retail": RetailDB, "telecom": TelecomDB}[domain]
    for path in paths:
        constructor = make_environment_constructor(domain=domain, db_path=Path(path))
        first, second = constructor(), constructor()
        assert first.tools.db.model_dump(mode="json") == model.load(path).model_dump(mode="json")
        assert first.tools.db is not second.tools.db
        assert first.tools.db.model_dump(mode="json") == second.tools.db.model_dump(mode="json")




def test_raw_rollout_maps_real_tokens_and_format_error_to_source_turn(monkeypatch):
    import os
    from copy import deepcopy
    from transformers import AutoTokenizer
    from tau2.data_model.message import AssistantMessage, UserMessage
    from slime.utils.mask_utils import MultiTurnLossMaskGenerator
    from slime.utils.types import Sample
    from slime.rollout.agent_tokens import recorded_turn, expand_training_segments
    import rollout
    import reward_postprocess as rp
    monkeypatch.setenv("TAU2_RAW_TOKENS", "1")
    monkeypatch.setenv("TAU2_TURN_CREDIT_VERSION", rp.PROGRESS_DB_COUNT_V1)
    monkeypatch.setenv("TAU2_PROGRESS_WEIGHT", "0")
    monkeypatch.setenv("TAU2_FORMAT_WEIGHT", "1")
    tokenizer = AutoTokenizer.from_pretrained(os.environ["HF_CHECKPOINT"])
    generator = MultiTurnLossMaskGenerator(tokenizer, tokenizer_type="qwen3_full")
    def message(body, prompt):
        tokens = tokenizer.encode(body + "<|im_end|>", add_special_tokens=False)
        behavior = recorded_turn(tokenizer.encode(prompt, add_special_tokens=False),
            {"output_token_logprobs": [(-0.5, token) for token in tokens], "completion_tokens": len(tokens)}, 0)
        return AssistantMessage(role="assistant", content=body, raw_data={"tau2_rl_agent": True,
            "text": body + "<|im_end|>", "meta_info": {"behavior_turn": behavior}})
    simulation = SimpleNamespace(messages=[UserMessage(role="user", content="start"), message("hello", "context one"),
        UserMessage(role="user", content="continue"), message("done", "context two")], termination_reason="user_stop")
    sample = Sample(index=0, group_index=0, metadata={"tau2_progress": {
        "source_indices": [1, 3], "scores": [0, 1, 2], "db_diff_counts": [0, 1],
        "db_diff_side_counts": [{}, {}], "unavailable_reason": None}, "tau2_domain": "retail", "tau2_task_id": "raw"})
    rollout._fill_sample_from_simulation(generator=generator, sample=sample, simulation=simulation,
        reward_value=0, system_prompt="test", protocol_profile="official-native", native_tools=[],
        field_reward_signals={"turn_errors": [{"type": "malformed_json", "turn_index": 3}]})
    assert len(sample.training_segments) == 2
    assert [d["simulation_message_index"] for d in sample.metadata["tau2_turn_credits"]] == [1, 3]
    args = SimpleNamespace(n_samples_per_prompt=2, advantage_estimator="grpo", rewards_normalization=True,
                           grpo_std_normalization=True, reward_key=None)
    other = deepcopy(sample)
    other.index = 1
    raw, advantages = rp.tau2_reward_post_process(args, [sample, other])
    expanded, _, _ = expand_training_segments([sample, other], raw, advantages)
    assert set(expanded[0].train_metadata["token_advantages"]) == {0}
    assert set(expanded[1].train_metadata["token_advantages"]) == {-0.1}
    assert expanded[1].tokens[-1] == tokenizer.convert_tokens_to_ids("<|im_end|>")
    # A scoring failure disables progress across K, retaining outcome and format.
    monkeypatch.setenv("TAU2_PROGRESS_WEIGHT", "1")
    other.reward = 1.0
    other.metadata["tau2_progress"]["unavailable_reason"] = "boundary: simulated scoring failure"
    other.metadata["tau2_progress"]["scores"] = [None, None, None]
    raw, advantages = rp.tau2_reward_post_process(args, [sample, other])
    for item, outcome in zip([sample, other], advantages, strict=True):
        turns = item.metadata["tau2_turn_credits"]
        assert [turn["progress_advantage"] for turn in turns] == [0, 0]
        assert turns[0]["advantage"] == outcome
        assert turns[1]["advantage"] == outcome - 0.1
        assert item.metadata["tau2_progress_group_unavailable_reasons"]


@pytest.mark.parametrize("reuse", [False, True])
def test_decision_boundaries_include_tools_user_and_sync(monkeypatch, reuse):
    from copy import deepcopy
    from tau2.data_model.message import AssistantMessage, UserMessage, ToolMessage, ToolCall
    from tau2.orchestrator.orchestrator import Role
    from progress import DBCountProgressOrchestrator as ProgressOrchestrator, database_records, database_snapshot

    state = {"value": 0, "sync_count": 0, "assistant": 0, "user": 0}
    def response(call):
        state["value"] += call.arguments["change"]
        state[call.requestor] += call.arguments["change"]
        return ToolMessage(role="tool", id=call.id, name=call.name, requestor=call.requestor,
                           content="failed after mutation" if call.name == "failed" else "ok",
                           error=call.name == "failed")
    def sync():
        state["sync_count"] += 1

    messages = iter([
        AssistantMessage(role="assistant", tool_calls=[
            ToolCall(id="a", name="write", arguments={"change": 1}, requestor="assistant"),
            ToolCall(id="b", name="failed", arguments={"change": -2}, requestor="assistant")],
            raw_data={"tau2_rl_agent": True}),
        AssistantMessage(role="assistant", content="Please repair it", raw_data={"tau2_rl_agent": True, "text": "Please repair it<|im_end|>"}),
        AssistantMessage(role="assistant", content="done", raw_data={"tau2_rl_agent": True, "text": "done<|im_end|>"}),
    ])
    users = iter([
        UserMessage(role="user", tool_calls=[ToolCall(id="c", name="repair", arguments={"change": 3}, requestor="user")]),
        UserMessage(role="user", content="repaired"),
    ])
    orchestrator = ProgressOrchestrator.__new__(ProgressOrchestrator)
    orchestrator._reuse_snapshot = reuse
    orchestrator._current_snapshot = None
    orchestrator.environment = SimpleNamespace(
        get_response=response, sync_tools=sync, _is_mutating_tool=lambda name: True,
        tools=SimpleNamespace(db=SimpleNamespace(model_dump=lambda **_: {"orders": {"one": {"value": state["assistant"]}}})),
        user_tools=SimpleNamespace(db=SimpleNamespace(model_dump=lambda **_: {"device": {"value": state["user"]}})))
    orchestrator.initial_records = database_records(database_snapshot(orchestrator.environment))
    orchestrator.db_diff_counts, orchestrator.db_diff_side_counts = [], []
    orchestrator.agent = SimpleNamespace(generate_next_message=lambda *_: (next(messages), None), is_stop=lambda m: m.content == "done")
    orchestrator.user = SimpleNamespace(generate_next_message=lambda *_: (next(users), None))
    orchestrator.agent_state = orchestrator.user_state = None
    orchestrator.from_role, orchestrator.to_role = Role.USER, Role.AGENT
    orchestrator.message = UserMessage(role="user", content="start")
    orchestrator.trajectory = [orchestrator.message]
    orchestrator.done = False
    orchestrator.solo_mode = orchestrator.validate_communication = False
    orchestrator.step_count = orchestrator.num_errors = 0
    orchestrator.progress_scores, orchestrator.progress_messages = [], []
    snapshots = []
    def score(_, *, snapshot=None):
        if snapshot is not None:
            snapshots.append(snapshot)
            assert snapshot["assistant"]["orders"]["one"]["value"] == state["assistant"]
            assert snapshot["user"]["device"]["value"] == state["user"]
        return state["value"]
    orchestrator.progress_potential = SimpleNamespace(score=score, reason=None, seconds=0)
    import progress
    snapshot_calls = []
    def counted_snapshot(env):
        snapshot_calls.append(None)
        return database_snapshot(env)
    monkeypatch.setattr(progress, "database_snapshot", counted_snapshot)
    # Voice metadata is unrelated to these text-only test messages.
    orchestrator._update_voice_metadata = lambda _: None
    while not orchestrator.done:
        orchestrator.step()
    payload = orchestrator.progress_payload(SimpleNamespace(messages=deepcopy(orchestrator.trajectory)))
    assert payload["scores"] == [0, -1, 2, 2]
    assert payload["db_diff_counts"] == [0, 1, 2]
    assert payload["db_diff_side_counts"] == [
        {"assistant": 0, "user": 0}, {"assistant": 1, "user": 0}, {"assistant": 1, "user": 1}]
    assert len(payload["source_indices"]) == 3  # A tool batch remains one Agent turn.
    assert len(snapshot_calls) == len(snapshots) == 3
    assert orchestrator.num_errors == 1
    assert state["sync_count"] == orchestrator.step_count


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
