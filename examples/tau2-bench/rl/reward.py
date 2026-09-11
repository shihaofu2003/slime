"""Reward helpers that preserve tau2's evaluator semantics with custom DBs."""

from __future__ import annotations

import json

from tau2.data_model.simulation import RewardInfo, SimulationRun, TerminationReason
from tau2.data_model.tasks import RewardType, Task
from tau2.environment.toolkit import get_tool_types
from tau2.evaluator.evaluator_action import ActionEvaluator
from tau2.evaluator.evaluator_communicate import CommunicateEvaluator
from tau2.evaluator.evaluator_env import EnvironmentEvaluator
from tau2.orchestrator.modes import CommunicationMode


def _reward_info_to_dict(info: RewardInfo | None) -> dict | None:
    if info is None:
        return None
    if hasattr(info, "model_dump_json"):
        return json.loads(info.model_dump_json())
    if hasattr(info, "model_dump"):
        return info.model_dump()
    if hasattr(info, "json"):
        return json.loads(info.json())
    return info.dict()


def reward_info_to_dict(info: RewardInfo | None) -> dict | None:
    return _reward_info_to_dict(info)


def evaluate_simulation_with_constructor(
    *,
    simulation: SimulationRun,
    task: Task,
    domain: str,
    environment_constructor,
    solo_mode: bool = False,
) -> RewardInfo:
    """Evaluate a tau2 simulation using the task-specific environment DB."""

    if simulation.termination_reason not in {
        TerminationReason.AGENT_STOP.value,
        TerminationReason.USER_STOP.value,
    }:
        return RewardInfo(
            reward=0.0,
            reward_basis=None,
            info={
                "note": (
                    "Simulation terminated prematurely. "
                    f"Termination reason: {simulation.termination_reason}"
                )
            },
        )

    if task.evaluation_criteria is None:
        return RewardInfo(
            reward=1.0,
            reward_basis=None,
            info={"note": "No evaluation criteria"},
        )

    if getattr(simulation, "mode", None) not in {None, CommunicationMode.HALF_DUPLEX.value}:
        raise ValueError(f"tau2 RL only supports half-duplex simulations, got mode={simulation.mode}")

    trajectory = simulation.messages
    env_reward_info = EnvironmentEvaluator.calculate_reward(
        environment_constructor=environment_constructor,
        task=task,
        full_trajectory=trajectory,
        solo_mode=solo_mode,
        env_kwargs={},
    )

    tool_types = None
    try:
        env = environment_constructor(solo_mode=solo_mode)
        if env.tools is not None:
            tool_types = get_tool_types(env.tools)
        if env.user_tools is not None:
            user_tool_types = get_tool_types(env.user_tools)
            tool_types = user_tool_types if tool_types is None else {**tool_types, **user_tool_types}
    except Exception:
        tool_types = None

    action_reward_info = ActionEvaluator.calculate_reward(
        task=task,
        full_trajectory=trajectory,
        tool_types=tool_types,
    )
    communicate_reward_info = CommunicateEvaluator.calculate_reward(
        task=task,
        full_trajectory=trajectory,
    )

    task_reward_basis = set(task.evaluation_criteria.reward_basis)
    unsupported = task_reward_basis - {
        RewardType.DB,
        RewardType.ENV_ASSERTION,
        RewardType.ACTION,
        RewardType.COMMUNICATE,
    }
    if unsupported:
        raise ValueError(f"Unsupported tau2 RL reward basis: {unsupported}")

    reward = 1.0
    reward_breakdown = {}
    if task_reward_basis & {RewardType.DB, RewardType.ENV_ASSERTION}:
        if env_reward_info.reward_breakdown is not None:
            reward_breakdown.update(env_reward_info.reward_breakdown)
        reward *= env_reward_info.reward
    if RewardType.ACTION in task_reward_basis:
        if action_reward_info.reward_breakdown is not None:
            reward_breakdown.update(action_reward_info.reward_breakdown)
        reward *= action_reward_info.reward
    if RewardType.COMMUNICATE in task_reward_basis:
        if communicate_reward_info.reward_breakdown is not None:
            reward_breakdown.update(communicate_reward_info.reward_breakdown)
        reward *= communicate_reward_info.reward

    return RewardInfo(
        reward=reward,
        db_check=env_reward_info.db_check,
        env_assertions=env_reward_info.env_assertions,
        action_checks=action_reward_info.action_checks,
        communicate_checks=communicate_reward_info.communicate_checks,
        reward_basis=task.evaluation_criteria.reward_basis,
        reward_breakdown=reward_breakdown,
        info={
            "domain": domain,
            "env": env_reward_info.info,
            "action": action_reward_info.info,
            "communicate": communicate_reward_info.info,
        },
    )
