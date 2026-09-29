"""Manual experiment preflight; requires the tau2 job environment."""

from copy import deepcopy
from types import SimpleNamespace

import pytest

from progress import (
    DBCountProgressOrchestrator, ProgressOrchestrator, database_records,
    database_snapshot, db_count_group_advantages, field_distance, group_normalize,
)


def test_agent_rows_user_fields_and_restoration():
    initial = {
        "assistant": {"orders": {"one": {"status": "pending", "items": [1, 2]},
                                 "two": {"status": "pending"}}},
        "user": {"device": {"wifi": False, "airplane": True}, "surroundings": {"signal": 0}},
    }
    current = deepcopy(initial)
    reference = database_records(initial)
    current["assistant"]["orders"]["one"].update(status="done", items=[1, 2, 3])
    assert field_distance(database_records(current), reference) == 1
    current["user"]["device"].update(wifi=True, airplane=False)
    assert field_distance(database_records(current), reference) == 3
    del current["assistant"]["orders"]["two"]
    current["assistant"]["orders"]["three"] = {"status": "new"}
    assert field_distance(database_records(current), reference) == 5
    assert field_distance(database_records(deepcopy(initial)), reference) == 0


def test_telecom_lists_use_record_ids_not_order():
    initial = {"assistant": {"lines": [{"line_id": "a", "status": "active"},
                                         {"line_id": "b", "status": "active"}]}}
    current = deepcopy(initial)
    current["assistant"]["lines"].reverse()
    reference = database_records(initial)
    assert field_distance(database_records(current), reference) == 0
    current["assistant"]["lines"][0]["status"] = "closed"
    current["assistant"]["lines"][1]["status"] = "closed"
    assert field_distance(database_records(current), reference) == 2
    current["assistant"]["lines"].pop()
    assert field_distance(database_records(current), reference) == 2


def test_initial_baseline_is_after_task_initialization_and_is_fresh(monkeypatch):
    state = {"assistant": {"orders": {}}, "user": {"device": {"wifi": False}}}
    env = SimpleNamespace(
        tools=SimpleNamespace(db=SimpleNamespace(model_dump=lambda **_: deepcopy(state["assistant"]))),
        user_tools=SimpleNamespace(db=SimpleNamespace(model_dump=lambda **_: deepcopy(state["user"]))))
    def initialize(self):
        state["assistant"]["orders"] = {"initialized": {"status": "pending"}}
        state["user"]["device"]["wifi"] = True
    monkeypatch.setattr(ProgressOrchestrator, "initialize", initialize)
    orch = DBCountProgressOrchestrator.__new__(DBCountProgressOrchestrator)
    orch.environment = env
    orch.initialize()
    assert field_distance(database_records(database_snapshot(env)), orch.initial_records) == 0
    state["assistant"]["orders"]["initialized"]["status"] = "done"
    state["user"]["device"]["wifi"] = False
    assert field_distance(database_records(database_snapshot(env)), orch.initial_records) == 2
    orch.db_diff_counts.append(2)
    orch.initialize()
    assert orch.db_diff_counts == []
    assert field_distance(database_records(database_snapshot(env)), orch.initial_records) == 0


def test_cross_turn_grouping_and_singletons():
    # Same count (1) occurs at t=1 and t=0, with different remaining returns.
    rtgs, advantages = db_count_group_advantages(
        [[0, 1, 3], [0, 4, 4]], [[0, 1], [1, 2]], gamma=1)
    assert rtgs == [[3, 2], [4, 0]]
    expected = group_normalize([2, 4])
    assert advantages[0] == pytest.approx([0, expected[0]])
    assert advantages[1] == pytest.approx([expected[1], 0])


def test_discounted_progress_and_repeated_visits():
    rtgs, advantages = db_count_group_advantages([[0, 0, 0, 1]], [[0, 0, 0]], gamma=0.98)
    assert rtgs[0] == pytest.approx([0.98**2, 0.98, 1])
    assert advantages[0] == pytest.approx(group_normalize(rtgs[0]))
    assert advantages[0][0] < advantages[0][1] < advantages[0][2]
    _, zeros = db_count_group_advantages([[0, 1], [0, 1]], [[0], [0]])
    assert zeros == [[0], [0]]


def test_count_boundary_alignment():
    with pytest.raises(ValueError, match="align"):
        db_count_group_advantages([[0, 1, 2]], [[0]])


@pytest.mark.parametrize("domain", ["retail", "airline", "telecom"])
def test_real_database_shapes(domain):
    from tau2.registry import registry
    env = registry.get_env_constructor(domain)()
    records = database_records(database_snapshot(env))
    assert records
    assert field_distance(database_records(database_snapshot(env)), records) == 0
    if domain == "telecom":
        assert any(key[0] == "user" for key in records)
        assert any(key[:2] == ("assistant", "lines") for key in records)
    else:
        assert all(key[0] == "assistant" for key in records)


def test_postprocess_keeps_task_groups_separate_and_outcome_on_singletons(monkeypatch):
    import reward_postprocess as rp
    from slime.utils.types import Sample
    monkeypatch.setenv("TAU2_TURN_CREDIT_VERSION", rp.PROGRESS_DB_COUNT_V1)
    monkeypatch.setenv("TAU2_PROGRESS_WEIGHT", "1")
    monkeypatch.setenv("TAU2_FORMAT_WEIGHT", "1")
    args = SimpleNamespace(n_samples_per_prompt=2, advantage_estimator="grpo",
                           rewards_normalization=True, grpo_std_normalization=True, reward_key=None)
    samples = []
    for group, rewards, scores in [(1, [0, 1], [[0, 1], [0, 1]]),
                                    (2, [0, 0], [[0, 2], [0, 3]])]:
        for i in range(2):
            samples.append(Sample(group_index=group, reward=rewards[i], response_length=2, loss_mask=[1, 0],
                metadata={"tau2_domain": "retail", "tau2_task_id": str(group),
                    "tau2_progress": {"scores": scores[i], "source_indices": [1], "unavailable_reason": None,
                        "db_diff_counts": [i], "db_diff_side_counts": [{"assistant": i, "user": 0}]},
                    "tau2_turn_credits": [{"simulation_message_index": 1, "response_span": [0, 1], "errors": []}]}))
    rp.tau2_reward_post_process(args, samples)
    for sample in samples:
        detail = sample.metadata["tau2_turn_credits"][0]
        assert detail["progress_advantage"] == 0
        assert detail["db_bucket_turns"] == detail["db_bucket_trajectories"] == 1
        assert sample.train_metadata["token_advantages"] == [detail["outcome_advantage"], 0]
    assert samples[0].train_metadata["token_advantages"][0] < 0
    assert samples[1].train_metadata["token_advantages"][0] > 0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
