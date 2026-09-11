"""Serializable contracts owned by the independent synthetic pipeline."""

from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass, field
from typing import Any

from .constants import DATA_ORIGIN, TASK_ID_PREFIX


@dataclass(frozen=True)
class ActionNode:
    node_id: str
    subgoal: str
    action: dict[str, Any]
    depends_on: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["depends_on"] = list(self.depends_on)
        return value


@dataclass(frozen=True)
class ScenarioSpec:
    scenario_id: str
    split: str
    business_category: str
    template_id: str
    template_version: str
    difficulty: str
    task_form: str
    case_kind: str
    atomic_capabilities: tuple[str, ...]
    initialization_data: dict[str, Any]
    required_documents: tuple[str, ...]
    decisive_facts: tuple[str, ...]
    user_known_info: tuple[str, ...]
    information_to_collect: tuple[str, ...]
    user_tools: tuple[str, ...]
    action_nodes: tuple[ActionNode, ...]
    expected_db_changes: tuple[dict[str, Any], ...]
    final_response_facts: tuple[str, ...]
    canonical_user_request: str
    initial_user_message: str
    retrieval_variant: str
    initialization_actions: tuple[dict[str, Any], ...] = ()
    data_origin: str = DATA_ORIGIN

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["atomic_capabilities"] = list(self.atomic_capabilities)
        result["required_documents"] = list(self.required_documents)
        result["decisive_facts"] = list(self.decisive_facts)
        result["user_known_info"] = list(self.user_known_info)
        result["information_to_collect"] = list(self.information_to_collect)
        result["user_tools"] = list(self.user_tools)
        result["action_nodes"] = [node.to_dict() for node in self.action_nodes]
        result["expected_db_changes"] = copy.deepcopy(list(self.expected_db_changes))
        result["final_response_facts"] = list(self.final_response_facts)
        result["initialization_actions"] = copy.deepcopy(
            list(self.initialization_actions)
        )
        return result


@dataclass(frozen=True)
class TaskContract:
    task_id: str
    scenario_id: str
    split: str
    business_category: str
    template_id: str
    template_version: str
    difficulty: str
    task_form: str
    case_kind: str
    atomic_capabilities: tuple[str, ...]
    required_documents: tuple[str, ...]
    decisive_facts: tuple[str, ...]
    user_known_info: tuple[str, ...]
    information_to_collect: tuple[str, ...]
    allowed_user_tools: tuple[str, ...]
    reference_action_graph: tuple[dict[str, Any], ...]
    expected_db_changes: tuple[dict[str, Any], ...]
    final_response_facts: tuple[str, ...]
    retrieval_variant: str
    data_origin: str = DATA_ORIGIN

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        for name in (
            "atomic_capabilities",
            "required_documents",
            "decisive_facts",
            "user_known_info",
            "information_to_collect",
            "allowed_user_tools",
            "reference_action_graph",
            "expected_db_changes",
            "final_response_facts",
        ):
            result[name] = copy.deepcopy(list(result[name]))
        return result


def user_action_protocol(spec: ScenarioSpec) -> str:
    """Describe user-owned actions to the private User simulator."""
    user_action_indexes = [
        index
        for index, node in enumerate(spec.action_nodes)
        if node.action["requestor"] == "user"
    ]
    user_actions = [spec.action_nodes[index].action for index in user_action_indexes]
    if not user_actions:
        return ""

    pre_given = {
        action["arguments"]["discoverable_tool_name"]
        for action in spec.initialization_actions
        if action["func_name"] == "give_discoverable_user_tool"
    }
    instructions = [
        "Your opening message is already present in the conversation; do not call a tool before the Agent responds."
    ]
    for action_index, action in zip(user_action_indexes, user_actions):
        name = action["name"]
        arguments = json.dumps(
            action.get("arguments") or {},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        if name == "call_discoverable_user_tool":
            discoverable_name = action["arguments"]["discoverable_tool_name"]
            timing = (
                "After the Agent's first response"
                if discoverable_name in pre_given
                else f"As soon as the Agent gives you `{discoverable_name}`"
            )
            instructions.append(
                f"{timing}, invoke your `{name}` tool exactly once with arguments {arguments}."
            )
        else:
            timing = (
                "After the Agent's first response"
                if action_index == 0
                else "Once the Agent completes the preceding policy or verification steps and invites the customer-side action"
            )
            instructions.append(
                f"{timing}, invoke your `{name}` tool exactly once with arguments {arguments}."
            )
        if name == "request_human_agent_transfer":
            instructions.append(
                "This records your customer-side request only: explicitly tell the Agent not to transfer you with its own tool."
            )
    instructions.append("Do not ask the Agent to substitute an Agent-owned tool.")
    if user_action_indexes[-1] == len(spec.action_nodes) - 1:
        instructions.append(
            "After every required user tool succeeds and the requested outcome is complete, respond exactly ###STOP### on your next turn."
        )
    else:
        instructions.append(
            "After a user tool succeeds, report that result in ordinary language and continue until the Agent completes the remaining requested outcome; do not stop early."
        )
    return "<<User action protocol: " + " ".join(instructions) + ">>"


def scenario_to_task(spec: ScenarioSpec) -> tuple[dict[str, Any], TaskContract]:
    task_id = f"{TASK_ID_PREFIX}{spec.split}_{spec.scenario_id.rsplit('_', 1)[-1]}_{spec.task_form}"
    actions = [copy.deepcopy(node.action) for node in spec.action_nodes]
    if spec.task_form == "l1_evidence" or spec.task_form == "l3_decision":
        reward_basis = ["COMMUNICATE"]
    elif spec.task_form == "l2_retrieval":
        reward_basis = ["ACTION", "COMMUNICATE"]
    elif not spec.expected_db_changes and actions:
        reward_basis = ["ACTION"]
    else:
        reward_basis = ["DB"]

    initial_state = {
        "initialization_data": {
            "agent_data": copy.deepcopy(spec.initialization_data),
            "user_data": None,
        },
        "initialization_actions": copy.deepcopy(list(spec.initialization_actions)) or None,
        "message_history": [
            {"role": "user", "content": spec.initial_user_message}
        ],
    }
    private_user_protocol = user_action_protocol(spec)
    user_instructions = spec.canonical_user_request
    if private_user_protocol:
        user_instructions += "\n\n" + private_user_protocol
    user_instructions += (
        "\n\nWhen the requested outcome is complete, respond exactly ###STOP###."
    )
    task = {
        "id": task_id,
        "description": {
            "purpose": f"Independent synthetic Banking {spec.task_form} task.",
            "relevant_policies": None,
            "notes": None,
        },
        "user_scenario": {
            "persona": "A concise customer who reveals private details only when asked.",
            "instructions": user_instructions,
        },
        "ticket": None,
        "initial_state": initial_state,
        "evaluation_criteria": {
            "actions": actions or None,
            "env_assertions": None,
            "communicate_info": list(spec.final_response_facts) or None,
            "nl_assertions": None,
            "reward_basis": reward_basis,
        },
        "issues": None,
        "required_documents": list(spec.required_documents),
        "user_tools": list(spec.user_tools),
    }
    contract = TaskContract(
        task_id=task_id,
        scenario_id=spec.scenario_id,
        split=spec.split,
        business_category=spec.business_category,
        template_id=spec.template_id,
        template_version=spec.template_version,
        difficulty=spec.difficulty,
        task_form=spec.task_form,
        case_kind=spec.case_kind,
        atomic_capabilities=spec.atomic_capabilities,
        required_documents=spec.required_documents,
        decisive_facts=spec.decisive_facts,
        user_known_info=spec.user_known_info,
        information_to_collect=spec.information_to_collect,
        allowed_user_tools=spec.user_tools,
        reference_action_graph=tuple(node.to_dict() for node in spec.action_nodes),
        expected_db_changes=spec.expected_db_changes,
        final_response_facts=spec.final_response_facts,
        retrieval_variant=spec.retrieval_variant,
    )
    return task, contract
