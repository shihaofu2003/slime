"""Independent CPU preflight for v4 task grounding and goal credit."""

import copy
import json
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

NUM_GPUS = 0
ROOT = Path(__file__).resolve().parents[3]
RL = Path(__file__).resolve().parent
SOURCE = ROOT / "output/experiments/tau2-areal-async-rl/dependencies/tau2/src"
sys.path[:0] = [str(ROOT), str(RL), str(SOURCE), str(RL.parent / "analysis")]
os.environ.setdefault("TAU2_DATA_DIR", str(ROOT.parent / "tau2-bench/data"))
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

import envs
import progress
import reward
import reward_postprocess
import reward_v3
import task_goals_v4
import goal_progress_v4 as goal
from goal_credit_v4 import attach_advantages, goal_stage_advantages
from tau2.data_model.tasks import Action, Task
from tau2.data_model.message import AssistantMessage, ToolCall, ToolMessage, UserMessage
from tau2.domains.retail.tools import RetailTools


@pytest.fixture(scope="module")
def rows():
    path = ROOT / "output/experiments/tau2-mix-rl-sft4505-reward-v3-20260927/data/train.jsonl"
    return [json.loads(line) for line in path.open()]


@pytest.fixture
def installed(monkeypatch):
    for toolkit, name, method, _ in reward_v3.numeric_tool_methods():
        monkeypatch.setattr(toolkit, name, method)
    for owner, names in ((Action, ["compare_with_tool_call"]), (RetailTools, ["modify_pending_order_items"]),
                         (envs, ["task_from_metadata", "make_environment_constructor"]),
                         (progress, ["StatePotential", "DBCountProgressOrchestrator", "reference_database", "database_snapshot"]),
                         (reward, ["EnvironmentEvaluator", "ActionEvaluator"]),
                         (reward_postprocess, ["attach_progress_group_advantages"])):
        for name in names:
            monkeypatch.setattr(owner, name, getattr(owner, name))
    monkeypatch.delenv("TAU2_PROGRESS_DIAGNOSTICS_PATH", raising=False)
    goal.setup_worker()
    progress.cached_reference_database.cache_clear()
    yield
    progress.cached_reference_database.cache_clear()


def selected(rows, task_id):
    return next(row for row in rows if row["metadata"]["task"]["id"] == task_id)


def test_retail_preferences_and_product_facts_follow_the_database(rows):
    for task_id in ("retail_265", "retail_347", "retail_541"):
        original = selected(rows, task_id)
        prepared, detail = task_goals_v4.prepare_row(original)
        assert detail["status"] == "retail_grounded"
        instructions = prepared["metadata"]["task"]["user_scenario"]["instructions"]
        known = json.loads(instructions["known_info"])
        assert prepared["metadata"]["task"]["evaluation_criteria"] == original["metadata"]["task"]["evaluation_criteria"]
        if task_id == "retail_265":
            refunds = {change["order"]: change["refund_preference"] for change in known["requested_changes"] if "refund_preference" in change}
            assert "visa credit card" in refunds["#W8367851"]
            assert "gift card" in refunds["#W5712077"]
        if task_id == "retail_541":
            assert "smartwatch" not in json.dumps(instructions).lower()
            assert "Tablet" in json.dumps(known)
        if task_id == "retail_347":
            changes = [change for change in known["requested_changes"] if "items" in change]
            assert {change["order"] for change in changes} == {"#W6788631", "#W4593656"}
            assert all(len(change["items"]) == 1 for change in changes)
        assert "new_item_ids" not in json.dumps(instructions)
        assert "modify_pending_order_items" not in json.dumps(instructions)


def test_duplicate_products_keep_their_quantity(rows):
    row = next(row for row in rows if row["metadata"]["domain"] == "retail" and any(
        a["name"] == "exchange_delivered_order_items" and len(a["arguments"]["item_ids"]) != len(set(a["arguments"]["item_ids"]))
        for a in row["metadata"]["task"]["evaluation_criteria"].get("actions") or []))
    prepared, _ = task_goals_v4.prepare_row(row)
    known = json.loads(prepared["metadata"]["task"]["user_scenario"]["instructions"]["known_info"])
    assert any(item["quantity"] > 1 for change in known["requested_changes"] for item in change.get("items", []))


def test_selection_tasks_do_not_reveal_the_reference_choice(rows):
    original = selected(rows, "retail_106")
    prepared, detail = task_goals_v4.prepare_row(original)
    assert detail["status"] == "selection_rule_kept"
    assert prepared == original


def test_preparation_preserves_pool_and_reward_conditions(rows):
    counts = {}
    for original in rows:
        prepared, detail = task_goals_v4.prepare_row(original)
        old, new = original["metadata"], prepared["metadata"]
        assert new["domain"] == old["domain"] and new["db_path"] == old["db_path"]
        assert new["task"]["id"] == old["task"]["id"]
        assert new["task"]["evaluation_criteria"] == old["task"]["evaluation_criteria"]
        counts[detail["status"]] = counts.get(detail["status"], 0) + 1
    assert counts["retail_grounded"] + counts["selection_rule_kept"] == 527
    assert counts["banking_golden_retrieval"] == 244
    assert counts["banking_bm25"] == 298


@pytest.mark.parametrize("variant", ["golden_retrieval", "bm25"])
def test_environment_consumes_declared_evidence_mode(installed, rows, variant):
    settings = task_goals_v4.banking_task_settings()
    task_id = next(key for key, item in settings.items() if item["retrieval_variant"] == variant)
    row, _ = task_goals_v4.prepare_row(selected(rows, task_id))
    metadata = row["metadata"]
    task = envs.task_from_metadata(metadata)
    environment = envs.make_environment_constructor(domain="banking", db_path=Path(metadata["db_path"]), task=task)()
    assert task.retrieval_variant == variant
    names = {tool.name for tool in environment.get_tools()}
    assert ("KB_search" in names) == (variant == "bm25")
    if variant == "golden_retrieval":
        doc = json.loads((envs.BANKING_DOCUMENTS / (task.required_documents[0] + ".json")).read_text())
        assert doc["content"] in environment.get_policy()


def test_legacy_metadata_retains_its_bm25_default(rows):
    metadata = next(row["metadata"] for row in rows if row["metadata"]["domain"] == "banking")
    task = goal.task_from_metadata(metadata)
    assert task.retrieval_variant is None
    environment = goal.make_environment_constructor(domain="banking", db_path=Path(metadata["db_path"]), task=task)()
    assert "KB_search" in {tool.name for tool in environment.get_tools()}


def test_list_progress_distinguishes_partial_product_completion():
    initial = {"items": [{"item_id": "old_a"}, {"item_id": "old_b"}, {"item_id": "old_c"}]}
    target = {"items": [{"item_id": "new_a"}, {"item_id": "new_b"}, {"item_id": "new_c"}]}
    partial = copy.deepcopy(target)
    partial["items"][-1] = initial["items"][-1]
    assert goal.business_field_distance(initial, target) == 3
    assert goal.business_field_distance(partial, target) == 1
    assert goal.business_field_distance(target, target) == 0
    assert goal.business_field_distance({"amount": 2}, {"amount": 2.0}) == 0
    assert goal.business_field_distance({"items": [1, 2]}, {"items": [2, 1]}) > 0
    assert goal.business_field_distance({"items": [1]}, {"items": [1, 1]}) > 0
    assert goal.business_field_distance({**target, "extra_write": 1}, target) > 0


def communication_task():
    return Task.model_validate(dict(id="quote", user_scenario={"instructions": "Quote the limits"}, required_documents=["policy"],
                                    evaluation_criteria={"reward_basis": ["COMMUNICATE"], "communicate_info": ["37,500", "375,000"]}))


def search_messages(document="policy", content="The range is $37,500 to $375,000.", call_id="lookup"):
    call = ToolCall(id=call_id, name="KB_search", requestor="assistant", arguments={"query": "credit limits"})
    return [AssistantMessage(role="assistant", tool_calls=[call]),
            ToolMessage(role="tool", id=call_id, requestor="assistant", error=False,
                        content=f"1. Policy\n   ID: {document}\n   Score: 1.0\n   Content: {content}")]


def test_communication_progress_requires_correct_facts_or_relevant_evidence():
    potential = goal.StatePotential(communication_task(), lambda: None)
    assert potential.score(None) == -2
    potential.messages = search_messages(document="unrelated")
    assert potential.score(None) == -2
    potential.messages = search_messages(content="Unrelated policy content.")
    assert potential.score(None) == -2
    potential.messages = search_messages()
    assert potential.score(None) == -1.5
    potential.messages += search_messages(call_id="repeat")
    assert potential.score(None) == -1.5
    potential.messages.append(AssistantMessage(role="assistant", content="The lower limit is $37,500."))
    assert potential.score(None) == -.75
    potential.messages.append(AssistantMessage(role="assistant", content="The upper limit is $375,000."))
    assert potential.score(None) == 0


def test_action_progress_and_reward_share_successful_call_semantics(installed):
    action = Action(action_id="required", name="change", arguments={"order_id": "002"}, requestor="assistant")
    task = communication_task().model_copy(update={"evaluation_criteria": communication_task().evaluation_criteria.model_copy(
        update={"reward_basis": ["ACTION"], "actions": [action], "communicate_info": []})})
    potential = goal.StatePotential(task, lambda: None)
    call = ToolCall(id="write", name="change", requestor="assistant", arguments={"order_id": "002"})
    potential.messages = [AssistantMessage(role="assistant", tool_calls=[call])]
    assert potential.score(None) == -1
    failed = ToolMessage(role="tool", id="write", requestor="assistant", content="Error: not found", error=True)
    potential.messages.append(failed)
    assert potential.score(None) == -1
    assert reward.ActionEvaluator.calculate_reward(task, potential.messages).reward == 0
    potential.messages[-1] = failed.model_copy(update={"content": "Done", "error": False})
    assert potential.score(None) == 0
    assert reward.ActionEvaluator.calculate_reward(task, potential.messages).reward == 1
    assert not action.compare_with_tool_call(call.model_copy(update={"arguments": {}}))
    assert not action.compare_with_tool_call(call.model_copy(update={"requestor": "user"}))


def test_identical_multistage_successes_do_not_create_fake_group_signal():
    _, advantages, _ = goal_stage_advantages([[-2, -1.5, -.75, 0]] * 8)
    assert not any(value for row in advantages for value in row)


def test_all_failed_groups_can_distinguish_verified_partial_progress():
    _, advantages, _ = goal_stage_advantages([[-2, -1]] * 4 + [[-2, -2]] * 4)
    assert all(row[0] > 0 for row in advantages[:4])
    assert all(row[0] < 0 for row in advantages[4:])


def test_one_trajectory_cannot_supply_its_own_group_variance():
    _, advantages, _ = goal_stage_advantages([[-2, -2, -1], [-3, -3, -3]])
    assert not any(value for row in advantages for value in row)
    _, advantages, _ = goal_stage_advantages([[-2, -2, -1], [-2, -1]], gamma=1)
    assert not any(value for row in advantages for value in row)


def test_token_credit_alignment_and_uniform_group_filter(installed, monkeypatch):
    from test_credit_signal_v1 import make_group
    import credit_signal_v1
    args = SimpleNamespace(n_samples_per_prompt=8, advantage_estimator="grpo", reward_key=None,
                           rewards_normalization=True, grpo_std_normalization=True)
    monkeypatch.setenv("TAU2_TURN_CREDIT_VERSION", "progress-db-count-v1")
    monkeypatch.setenv("TAU2_PROGRESS_WEIGHT", "1")
    monkeypatch.setenv("TAU2_FORMAT_WEIGHT", "1")
    monkeypatch.setenv("TAU2_DROP_UNIFORM_OUTCOME_GROUPS", "1")
    samples = make_group(reward=0.0)
    assert credit_signal_v1.filter_group(args, samples).keep
    assert all(sample.train_metadata["credit_recipe"] == "goal-progress-v4" for sample in samples)
    for sample in samples:
        for segment in sample.training_segments:
            assert len(segment["train_metadata"]["token_advantages"]) == segment["response_length"]
            assert all(value == 0 for value, mask in zip(segment["train_metadata"]["token_advantages"], segment["loss_mask"]) if not mask)
    samples = make_group(reward=1.0)
    for sample in samples:
        sample.metadata["tau2_progress"]["scores"] = [-2, -1.5, -.75, 0]
    assert not credit_signal_v1.filter_group(args, samples).keep


def test_live_orchestrator_observes_communication_at_the_correct_boundary(installed, rows):
    metadata = next(row["metadata"] for row in rows if row["metadata"]["domain"] == "retail")
    task = communication_task()
    constructor = envs.make_environment_constructor(domain="retail", db_path=Path(metadata["db_path"]), task=task)

    class Agent:
        def set_seed(self, seed):
            pass

        def get_init_state(self, message_history=None):
            return SimpleNamespace(messages=list(message_history or []))

        def generate_next_message(self, message, state):
            reply = AssistantMessage(role="assistant", content="The range is $37,500 to $375,000.")
            state.messages.extend([message, reply])
            return reply, state

        def is_stop(self, message):
            return False

    class User(Agent):
        voice_settings = None

        def generate_next_message(self, message, state):
            content = "###STOP###" if "375,000" in (message.content or "") else "Please quote the limits."
            reply = UserMessage(role="user", content=content)
            state.messages.extend([message, reply])
            return reply, state

    potential = goal.StatePotential(task, constructor)
    orchestrator = goal.GoalProgressOrchestrator(progress_potential=potential, domain="retail", agent=Agent(), user=User(),
                                               environment=constructor(), task=task, max_steps=8, validate_communication=False)
    simulation = orchestrator.run()
    payload = orchestrator.progress_payload(simulation)
    assert payload["scores"] == [-2, 0]
    assert len(payload["source_indices"]) == 1
    assert simulation.messages[payload["source_indices"][0]].content == "The range is $37,500 to $375,000."


def test_ray_hook_is_inherited_by_environment_workers():
    import ray
    paths = os.pathsep.join([str(ROOT), str(RL), str(SOURCE), str(RL.parent / "analysis")])
    with tempfile.TemporaryDirectory(prefix="gv4-") as temporary:
        try:
            ray.init(address="local", num_cpus=2, num_gpus=0, include_dashboard=False, object_store_memory=128 * 1024 * 1024,
                     _temp_dir=temporary, runtime_env={"env_vars": {"PYTHONPATH": paths, "TAU2_DATA_DIR": os.environ["TAU2_DATA_DIR"],
                                                                  "LITELLM_LOCAL_MODEL_COST_MAP": "True"},
                                                       "worker_process_setup_hook": "goal_progress_v4.setup_worker"})
            @ray.remote(num_cpus=0)
            class Leaf:
                def probe(self):
                    import envs, progress, reward, reward_postprocess
                    return (envs.task_from_metadata.__module__, envs.make_environment_constructor.__module__,
                            progress.StatePotential.__module__, progress.DBCountProgressOrchestrator.__module__,
                            reward.ActionEvaluator.__module__, reward_postprocess.attach_progress_group_advantages.__module__)

            @ray.remote(num_cpus=0)
            class Parent:
                def probe(self):
                    return ray.get(Leaf.options(runtime_env={"env_vars": {"GOAL_TEST_CHILD": "1"}}).remote().probe.remote())
            assert ray.get(Parent.remote().probe.remote(), timeout=90) == ("goal_progress_v4",) * 5 + ("goal_credit_v4",)
        finally:
            ray.shutdown()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
