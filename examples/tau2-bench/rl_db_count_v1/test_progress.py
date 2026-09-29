"""Experiment preflight; run in the tau2 job environment, alongside rollout preflight."""

import math
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from progress import field_distance, flatten_fields, group_normalize, progress_group_advantages


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


def test_field_distance():
    target = flatten_fields({"a": None, "b": [1, 2, 2], "empty": {}})
    assert field_distance(target, target) == 0
    assert field_distance(flatten_fields({"b": [1, 2, 2], "empty": {}}), target) == 1
    assert field_distance(flatten_fields({"a": None, "b": [2, 1, 2], "empty": {}, "extra": 1}), target) == 2
    assert field_distance(flatten_fields({"a": None, "b": [1, 2], "empty": None}), target) == 2


@pytest.mark.parametrize("rewards,expected", [
    ([0, 0, 1], [1, 1, 1]), ([0, 0, 0], [0, 0, 0]),
    ([1, -1], [0, -1]), ([1, -1, 1], [1, 0, 1]),
])
def test_rtg(rewards, expected):
    scores = [0]
    for reward in rewards:
        scores.append(scores[-1] + reward)
    rtg, advantages = progress_group_advantages([scores])
    assert rtg == [expected]
    assert advantages == [[0] * len(rewards)]


def test_discounted_rtg_uses_step_rewards():
    rtg, _ = progress_group_advantages([[0, 0, 0, 1]], gamma=0.98)
    assert rtg[0] == pytest.approx([0.98**2, 0.98, 1.0])
    rtg_one, _ = progress_group_advantages([[0, 0, 0, 1]], gamma=1.0)
    assert rtg_one[0] == pytest.approx([1.0, 1.0, 1.0])


def test_discounted_rtg_rejects_invalid_gamma():
    with pytest.raises(ValueError, match="gamma"):
        progress_group_advantages([[0, 1]], gamma=1.01)


def test_sample_std_and_variable_lengths():
    expected = 0.5 / (math.sqrt(0.5) + 1e-6)
    assert group_normalize([0, 1]) == pytest.approx([-expected, expected])
    assert group_normalize([1, 1]) == [0, 0]
    assert group_normalize([0, 1e-9]) == [0, 0]
    _, values = progress_group_advantages([[0, 1, 1], [0, 0, 1, 2]])
    assert values[0][1] < 0 < values[1][1]
    assert values[1][2] == 0


@pytest.mark.parametrize("domain,basis", [("airline", "DB"), ("retail", "DB"),
                                        ("telecom", "DB"), ("telecom", "ENV_ASSERTION")])
def test_real_environment(domain, basis):
    from envs import make_environment_constructor, task_from_metadata
    from progress import StatePotential, database_fields, cached_reference_database

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
    assert potential.score(env) == first
    assert database_fields(env) == before  # Repeated assertion checks must be read-only.
    for action in task.evaluation_criteria.actions or []:
        env.make_tool_call(tool_name=action.name, requestor=action.requestor, **action.arguments)
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


def test_group_token_advantages(monkeypatch):
    import reward_postprocess as rp
    from slime.utils.types import Sample

    monkeypatch.setenv("TAU2_TURN_CREDIT_VERSION", "progress-db-count-v1")
    args = SimpleNamespace(n_samples_per_prompt=2, advantage_estimator="grpo", rewards_normalization=True,
                           grpo_std_normalization=True, reward_key=None)
    samples = []
    for i, scores in enumerate(([0, 1, 1], [0, 0, 1])):
        samples.append(Sample(group_index=3, reward=float(i), response_length=5, loss_mask=[1, 1, 0, 1, 1],
                              metadata={"tau2_domain": "retail", "tau2_task_id": "test",
                                        "tau2_progress": {"scores": scores, "source_indices": [1, 4], "unavailable_reason": None,
                                                          "db_diff_counts": [0, i], "db_diff_side_counts": [{"assistant": 0, "user": 0}, {"assistant": i, "user": 0}]},
                                        "tau2_turn_credits": [
                                            {"simulation_message_index": 1, "response_span": [0, 2], "errors": []},
                                            {"simulation_message_index": 4, "response_span": [3, 5], "errors": [
                                                {"type": "malformed_json"}, {"type": "nonexistent_tool"}]}]}))
    raw, outcome = rp.tau2_reward_post_process(args, samples)
    assert raw == [0, 1]
    for sample in samples:
        assert sample.train_metadata["token_advantages"][2] == 0
        assert sample.metadata["tau2_turn_credits"][1]["format_penalty"] == -0.1
    samples[0].metadata["tau2_progress"]["unavailable_reason"] = "target failed"
    rp.tau2_reward_post_process(args, samples)
    assert all(d["progress_advantage"] == 0 for s in samples for d in s.metadata["tau2_turn_credits"])
    monkeypatch.setenv("TAU2_PROGRESS_WEIGHT", "0")
    monkeypatch.setenv("TAU2_FORMAT_WEIGHT", "0")
    rp.tau2_reward_post_process(args, samples)
    for i, sample in enumerate(samples):
        assert sample.train_metadata["token_advantages"] == [outcome[i], outcome[i], 0, outcome[i], outcome[i]]


@pytest.mark.parametrize("reward", [0.0, 1.0])
def test_zero_signal_groups(monkeypatch, reward):
    import reward_postprocess as rp
    import filters
    from slime.utils.types import Sample

    monkeypatch.setenv("TAU2_TURN_CREDIT_VERSION", "progress-db-count-v1")
    monkeypatch.setenv("TAU2_REPLACE_ZERO_SIGNAL_GROUPS", "0")
    args = SimpleNamespace(n_samples_per_prompt=2, advantage_estimator="grpo", rewards_normalization=True,
                           grpo_std_normalization=True, reward_key=None)
    samples = [Sample(group_index=7, reward=reward, response_length=1, loss_mask=[1],
                      train_metadata={"turn_credit_version": rp.PROGRESS_DB_COUNT_V1},
                      metadata={"domain": "airline", "tau2_domain": "airline", "tau2_task_id": "zero",
                                "tau2_turn_credit_version": rp.PROGRESS_DB_COUNT_V1,
                                "tau2_progress": {"source_indices": [1], "scores": [0, 0], "unavailable_reason": None,
                                                  "db_diff_counts": [0], "db_diff_side_counts": [{"assistant": 0, "user": 0}]},
                                "tau2_turn_credits": [{"simulation_message_index": 1, "response_span": [0, 1], "errors": []}]})
               for _ in range(2)]
    assert filters.drop_zero_std_or_unsampleable(args, samples).keep
    rp.tau2_reward_post_process(args, samples)
    assert all(s.train_metadata["token_advantages"] == [0] for s in samples)
    assert all(not s.metadata["tau2_turn_credit_group_has_signal"] for s in samples)
    samples[0].group_index = 8
    with pytest.raises(rp.TurnCreditAlignmentError, match="mixed"):
        rp.tau2_reward_post_process(args, samples)


def test_cp_and_policy_loss_equivalence(monkeypatch):
    import torch
    import reward_postprocess as rp
    from slime.backends.megatron_utils import cp_utils
    from slime.utils.ppo_utils import compute_policy_loss

    mask = torch.tensor([1, 1, 0, 0, 1, 1, 0, 1], dtype=torch.float)
    scalar = 0.7071058
    full = mask * scalar
    for size in (1, 2):
        monkeypatch.setattr(cp_utils.mpu, "get_context_parallel_world_size", lambda: size)
        for rank in range(size):
            monkeypatch.setattr(cp_utils.mpu, "get_context_parallel_rank", lambda: rank)
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


def test_decision_boundaries_include_tools_user_and_sync(monkeypatch):
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
    orchestrator.environment = SimpleNamespace(
        get_response=response, sync_tools=sync,
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
    orchestrator.progress_potential = SimpleNamespace(score=lambda _: state["value"], reason=None, seconds=0)
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
    assert orchestrator.num_errors == 1
    assert state["sync_count"] == orchestrator.step_count

    import os
    from transformers import AutoTokenizer
    from slime.utils.mask_utils import MultiTurnLossMaskGenerator
    from slime.utils.types import Sample
    import rollout
    import reward_postprocess as rp

    monkeypatch.setenv("TAU2_TURN_CREDIT_VERSION", rp.PROGRESS_DB_COUNT_V1)
    tokenizer = AutoTokenizer.from_pretrained(os.environ["HF_CHECKPOINT"])
    generator = MultiTurnLossMaskGenerator(tokenizer, tokenizer_type="qwen3_full")
    simulation = SimpleNamespace(messages=deepcopy(orchestrator.trajectory), termination_reason="agent_stop")
    sample = Sample(group_index=1, metadata={"tau2_progress": payload, "tau2_domain": "retail", "tau2_task_id": "boundary"})
    rollout._fill_sample_from_simulation(generator=generator, sample=sample, simulation=simulation,
                                        reward_value=1.0, system_prompt="Test", protocol_profile="official-native",
                                        native_tools=[], field_reward_signals={"turn_errors": []})
    args = SimpleNamespace(n_samples_per_prompt=2, advantage_estimator="grpo", rewards_normalization=True,
                           grpo_std_normalization=True, reward_key=None)
    rp.tau2_reward_post_process(args, [sample, deepcopy(sample)])
    assert [d["simulation_message_index"] for d in sample.metadata["tau2_turn_credits"]] == payload["source_indices"]
    assert len(sample.metadata["tau2_turn_credits"]) == 3
    assert all(turn["raw_rerender"]["body_equal"] for turn in sample.metadata["tau2_turn_credits"][1:])
    assert all(a == 0 for a, m in zip(sample.train_metadata["token_advantages"], sample.loss_mask) if not m)


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


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
