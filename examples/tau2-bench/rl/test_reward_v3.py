"""CPU launch preflight for numeric equivalence, embedded JSON and worker hooks."""

import copy
import json
import os
import tempfile
from types import MethodType

import pytest

from test_reward_v2 import (
    ROOT, RL, SOURCE, constructor_and_task, initialized, tasks,
    reference_simulation,
    test_each_domain_reference_replay_matches_reward_and_progress,
    test_real_numeric_equivalent_calls_receive_identical_reward,
    test_retail_item_variants_duplicates_and_price_match_catalog,
    test_retail_invalid_item_does_not_charge_or_mutate_order,
    test_action_and_communication_requirements_still_gate_reward,
    test_premature_termination_remains_zero,
)

import progress
import reward
import reward_v2
import reward_v3
from tau2.data_model.message import ToolCall, ToolMessage
from tau2.data_model.simulation import SimulationRun
from tau2.data_model.tasks import Action
from tau2.domains.retail.tools import RetailTools
from tau2.environment.tool import as_tool

NUM_GPUS = 0


@pytest.fixture
def corrected(monkeypatch):
    # Record pre-test methods for restoration; the production hook is the same
    # one used by Ray workers and the all-task replay below.
    for toolkit, name, method, _ in reward_v3.numeric_tool_methods():
        monkeypatch.setattr(toolkit, name, method)
    monkeypatch.setattr(Action, "compare_with_tool_call", Action.compare_with_tool_call)
    monkeypatch.setattr(RetailTools, "modify_pending_order_items", RetailTools.modify_pending_order_items)
    monkeypatch.setattr(reward, "EnvironmentEvaluator", reward.EnvironmentEvaluator)
    monkeypatch.setattr(progress, "reference_database", progress.reference_database)
    monkeypatch.setattr(progress, "database_snapshot", progress.database_snapshot)
    reward_v3.setup_worker()
    progress.cached_reference_database.cache_clear()
    yield
    progress.cached_reference_database.cache_clear()


@pytest.fixture
def recorded_telecom(corrected):
    data = json.loads((RL / "fixtures/reward_v2_telecom_118.json").read_text())
    constructor, task = constructor_and_task(data)
    historical = SimulationRun.model_validate(data["simulation"])
    env = initialized(constructor, task)
    messages, responses = [], {}
    # Replay historical actions in the new environment to obtain its actual
    # responses. v3 formats numeric tool outputs consistently, so old response
    # text is intentionally not presented as a v3-generated trajectory.
    for message in historical.messages:
        if isinstance(message, ToolMessage):
            messages.append(responses.pop(message.id))
        else:
            messages.append(message)
            for call in getattr(message, "tool_calls", None) or []:
                responses[call.id] = env.get_response(call)
    simulation = historical.model_copy(update={"messages": messages, "reward_info": None})
    return constructor, task, simulation


def test_historical_successful_actions_replayed_in_v3_match_reward_and_progress(recorded_telecom):
    constructor, task, simulation = recorded_telecom
    result = reward.evaluate_simulation_with_constructor(
        simulation=simulation, task=task, domain="telecom", environment_constructor=constructor,
    )
    assert result.reward == 1 and result.db_check.db_match
    assert all(check.met for check in result.env_assertions)
    actual = initialized(constructor, task, simulation.messages)
    assert progress.StatePotential(task, constructor).score(actual) == 0


def test_numeric_method_wrappers_keep_the_tool_schemas():
    methods = list(reward_v3.numeric_tool_methods())
    assert len(methods) == 21
    for _, _, method, numeric in methods:
        changed = reward_v3.normalize_numeric_arguments(method, numeric)
        assert as_tool(MethodType(method, object())).openai_schema == as_tool(MethodType(changed, object())).openai_schema


@pytest.mark.parametrize("name", sorted(reward_v3.JSON_ARGUMENT_TOOLS))
def test_encoded_json_is_compared_as_values_without_rewriting_calls(corrected, name):
    expected = {"account_id": "002", "amount": 6.0, "items": [1, 2]}
    arguments = {"arguments": json.dumps(expected)}
    gold = Action(action_id="reference", name=name, arguments=arguments)
    actual = {"arguments": '{ "items": [1.0, 2.0], "amount": 6, "account_id": "002" }'}
    call = ToolCall(id="call", name=name, arguments=actual, requestor="assistant")
    original = copy.deepcopy(call.arguments)
    assert gold.compare_with_tool_call(call)
    assert call.arguments == original and gold.arguments == arguments
    for update in ({"amount": 7}, {"account_id": "2"}, {"items": [2, 1]}, {"items": [1]}, {"amount": "6"}):
        call.arguments["arguments"] = json.dumps({**expected, **update})
        assert not gold.compare_with_tool_call(call)
    call.arguments["arguments"] = '{"amount":'
    assert not gold.compare_with_tool_call(call)


def test_json_decoding_is_limited_to_documented_fields(corrected):
    gold = Action(action_id="reference", name="send_message", arguments={"message": '{"amount": 6.0}'})
    call = ToolCall(id="call", name="send_message", arguments={"message": '{"amount":6}'}, requestor="assistant")
    assert not gold.compare_with_tool_call(call)
    gold.compare_args = []
    assert gold.compare_with_tool_call(call)


@pytest.mark.parametrize("left,right", [(True, 1), (False, 0.0), ({"flag": True}, {"flag": 1}), ("002", "2"), (2, 2.001)])
def test_real_value_or_type_differences_are_preserved(left, right):
    assert not reward_v3.equivalent_values(left, right)


def test_method_normalization_preserves_nonnumeric_and_fractional_values():
    def capture(self, amount: float, count: int):
        return amount, count
    call = reward_v3.normalize_numeric_arguments(capture, {"amount": float, "count": int})
    amount, count = call(None, 6, 2.0)
    assert type(amount) is float and type(count) is int
    assert call(None, "6", 2.5) == ("6", 2.5)
    amount, count = call(None, True, False)
    assert amount is True and count is False


def test_nested_bank_transaction_identity_and_real_amount_change(corrected, tasks):
    metadata = next(m for m in tasks if m["task"]["id"] == "banking_syn_v1_train_000588_l3_single_action")
    constructor, task = constructor_and_task(metadata)
    states = []
    for amount in (6.0, 6, 7):
        alternate = copy.deepcopy(task)
        action = alternate.evaluation_criteria.actions[0]
        arguments = json.loads(action.arguments["arguments"])
        arguments["amount"] = amount
        action.arguments["arguments"] = json.dumps(arguments, sort_keys=True, indent=2)
        simulation, actual = reference_simulation(constructor, alternate)
        result = reward.evaluate_simulation_with_constructor(
            simulation=simulation, task=task, domain="banking", environment_constructor=constructor,
        )
        assert result.reward == (1 if amount == 6 else 0)
        score = progress.StatePotential(task, constructor).score(actual)
        assert score == 0 if amount == 6 else score < 0
        states.append(reward_v2.database_snapshot(actual))
    assert states[0] == states[1] and states[0] != states[2]


def test_ray_hook_reaches_nested_environment_workers():
    import ray
    paths = os.pathsep.join([str(ROOT), str(RL), str(SOURCE), str(RL.parent / "analysis")])
    with tempfile.TemporaryDirectory(prefix="rw3-") as temporary:
        try:
            ray.init(address="local", num_cpus=2, num_gpus=0, include_dashboard=False,
                     object_store_memory=128 * 1024 * 1024, _temp_dir=temporary,
                     runtime_env={"env_vars": {"PYTHONPATH": paths, "TAU2_DATA_DIR": os.environ["TAU2_DATA_DIR"]},
                                  "worker_process_setup_hook": "reward_v3.setup_worker"})

            @ray.remote(num_cpus=0)
            class Leaf:
                def probe(self):
                    import reward
                    import progress
                    from tau2.data_model.tasks import Action
                    from tau2.domains.banking_knowledge.tools import KnowledgeTools, KnowledgeUserTools
                    return (
                        reward.EnvironmentEvaluator.__module__, progress.reference_database.__module__,
                        Action.compare_with_tool_call.__module__,
                        KnowledgeTools.apply_checking_account_credit_5829.__globals__["__name__"],
                        KnowledgeUserTools.deposit_check_3847.__globals__["__name__"],
                    )

            @ray.remote(num_cpus=0)
            class Parent:
                def probe(self):
                    leaf = Leaf.options(runtime_env={"env_vars": {"REWARD_TEST_CHILD": "1"}}).remote()
                    return ray.get(leaf.probe.remote())

            actor = Parent.options(runtime_env={"env_vars": {"REWARD_TEST_PARENT": "1"}}).remote()
            assert ray.get(actor.probe.remote(), timeout=90) == ("reward_v2", "reward_v2", "reward_v3", "reward_v3", "reward_v3")
        finally:
            ray.shutdown()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
