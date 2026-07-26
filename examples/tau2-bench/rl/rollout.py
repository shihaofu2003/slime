"""slime custom-generate entrypoint for tau2-bench RL."""

from __future__ import annotations

import asyncio
import json
import os
import time
import traceback
from pathlib import Path
from typing import Any

from agent import TrainableSGLangAgent, render_tool_call
from envs import (
    db_path_from_metadata,
    domain_from_metadata,
    make_environment_constructor,
    task_from_metadata,
)
from reward import evaluate_simulation_with_constructor, reward_info_to_dict

from slime.rollout.sglang_rollout import GenerateState
from slime.utils.mask_utils import MultiTurnLossMaskGenerator
from slime.utils.types import Sample

from tau2.data_model.message import AssistantMessage, Message, ToolMessage, UserMessage
from tau2.data_model.simulation import SimulationRun
from tau2.orchestrator.orchestrator import DEFAULT_FIRST_AGENT_MESSAGE, Orchestrator
from tau2.runner.build import build_user


MASK_GENERATOR: MultiTurnLossMaskGenerator | None = None
DEFAULT_TRAIN_MAX_TOKENS = 32768


def _get_mask_generator(args) -> MultiTurnLossMaskGenerator:
    global MASK_GENERATOR
    if MASK_GENERATOR is None:
        state = GenerateState(args)
        MASK_GENERATOR = MultiTurnLossMaskGenerator(
            state.tokenizer,
            tokenizer_type=getattr(args, "loss_mask_type", "qwen3"),
        )
    return MASK_GENERATOR


def _with_openai_prefix(model: str) -> str:
    if "/" in model:
        return model
    return f"openai/{model}"


def _agent_api_base(args) -> str:
    return f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"


def _sampling_params_for_agent(sampling_params: dict[str, Any]) -> dict[str, Any]:
    params = dict(sampling_params)
    max_tokens = os.environ.get("TAU2_AGENT_MAX_TOKENS")
    if max_tokens:
        params["max_new_tokens"] = int(max_tokens)
    return params


def _user_llm_args() -> dict[str, Any]:
    args: dict[str, Any] = {
        "api_base": os.environ.get("TAU2_USER_API_BASE", "http://127.0.0.1:30001/v1"),
        "api_key": os.environ.get("TAU2_USER_API_KEY", "dummy-key-for-local-server"),
        "temperature": float(os.environ.get("TAU2_USER_TEMPERATURE", "0.0")),
    }
    if os.environ.get("TAU2_USER_TOP_P"):
        args["top_p"] = float(os.environ["TAU2_USER_TOP_P"])
    if os.environ.get("TAU2_USER_MAX_TOKENS"):
        args["max_tokens"] = int(os.environ["TAU2_USER_MAX_TOKENS"])
    extra_body = os.environ.get("TAU2_USER_EXTRA_BODY_JSON")
    if extra_body:
        args["extra_body"] = json.loads(extra_body)
    return args


def _message_to_training_chat(message: Message) -> dict[str, Any] | None:
    if isinstance(message, UserMessage):
        if message.is_tool_call():
            return None
        return {"role": "user", "content": message.content or ""}

    if isinstance(message, ToolMessage):
        if message.requestor != "assistant":
            return None
        return {"role": "user", "content": f"Tool result:\n{message.content or ''}"}

    if isinstance(message, AssistantMessage):
        raw_data = message.raw_data if isinstance(message.raw_data, dict) else {}
        trainable = bool(raw_data.get("tau2_rl_agent"))
        if message.is_tool_call():
            content = "\n\n".join(render_tool_call(tool_call) for tool_call in message.tool_calls)
        else:
            content = message.content or ""
        chat = {"role": "assistant", "content": content}
        if not trainable or message == DEFAULT_FIRST_AGENT_MESSAGE:
            chat["step_loss_mask"] = 0
        return chat

    return None


def _training_messages(simulation: SimulationRun, system_prompt: str) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt, "step_loss_mask": 0}]
    for message in simulation.messages:
        chat = _message_to_training_chat(message)
        if chat is not None:
            messages.append(chat)
    return messages


def _sample_status(simulation: SimulationRun) -> Sample.Status:
    reason = simulation.termination_reason
    if reason in {"agent_stop", "user_stop"}:
        return Sample.Status.COMPLETED
    if reason in {"max_steps", "timeout", "context_window_exceeded"}:
        return Sample.Status.TRUNCATED
    return Sample.Status.FAILED


def _limit_training_context(
    generator: MultiTurnLossMaskGenerator,
    messages: list[dict[str, Any]],
    max_tokens: int,
) -> tuple[list[int], list[int], int, int]:
    token_ids, loss_mask = generator.get_loss_mask(messages)
    original_length = len(token_ids)
    if original_length <= max_tokens:
        return token_ids, loss_mask, 0, original_length

    system_message = messages[0]
    cache: dict[int, tuple[list[int], list[int]]] = {}

    def tokenize_suffix(start: int) -> tuple[list[int], list[int]]:
        if start not in cache:
            cache[start] = generator.get_loss_mask([system_message, *messages[start:]])
        return cache[start]

    # Find the longest recent, message-aligned suffix that fits beside the system prompt.
    low, high = 1, len(messages)
    while low < high:
        middle = (low + high) // 2
        suffix_ids, _ = tokenize_suffix(middle)
        if len(suffix_ids) <= max_tokens:
            high = middle
        else:
            low = middle + 1

    limited_ids, limited_mask = tokenize_suffix(low)
    if len(limited_ids) <= max_tokens and 1 in limited_mask:
        return limited_ids, limited_mask, low - 1, original_length

    # A single recent message can itself exceed the limit. Preserve the complete
    # system prompt and the most recent supervised token span in that rare case.
    system_ids, system_mask = generator.get_loss_mask([system_message])
    if len(system_ids) >= max_tokens:
        raise ValueError(
            f"TAU2_TRAIN_MAX_TOKENS={max_tokens} must exceed the system prompt length "
            f"of {len(system_ids)} tokens"
        )
    tail_size = max_tokens - len(system_ids)
    return (
        system_ids + token_ids[-tail_size:],
        system_mask + loss_mask[-tail_size:],
        len(messages) - 1,
        original_length,
    )


def _fill_sample_from_simulation(
    *,
    args,
    sample: Sample,
    simulation: SimulationRun,
    reward_value: float,
    system_prompt: str,
) -> Sample:
    generator = _get_mask_generator(args)
    messages = _training_messages(simulation, system_prompt)
    max_tokens = int(os.environ.get("TAU2_TRAIN_MAX_TOKENS", str(DEFAULT_TRAIN_MAX_TOKENS)))
    if max_tokens <= 0:
        raise ValueError(f"TAU2_TRAIN_MAX_TOKENS must be positive, got {max_tokens}")
    token_ids, full_loss_mask, dropped_messages, original_length = _limit_training_context(
        generator, messages, max_tokens
    )
    response_length = generator.get_response_lengths([full_loss_mask])[0]

    if response_length == 0:
        fallback = generator.tokenizer("\n", add_special_tokens=False)["input_ids"]
        token_ids = token_ids + fallback
        response_length = len(fallback)
        loss_mask = [0] * response_length
    else:
        loss_mask = full_loss_mask[-response_length:]

    sample.tokens = token_ids
    sample.response_length = response_length
    sample.loss_mask = loss_mask
    sample.rollout_log_probs = [0.0] * response_length
    sample.response = generator.tokenizer.decode(token_ids[-response_length:])
    sample.reward = reward_value
    sample.status = _sample_status(simulation)
    sample.metadata.update(
        {
            "tau2_original_total_tokens": original_length,
            "tau2_train_total_tokens": len(token_ids),
            "tau2_training_context_truncated": dropped_messages > 0,
            "tau2_training_messages_dropped": dropped_messages,
        }
    )
    return sample


def _dump_trajectory(sample: Sample, simulation: SimulationRun) -> None:
    dump_path = os.environ.get("TAU2_RL_TRAJECTORY_DUMP_PATH")
    if not dump_path:
        return
    try:
        if hasattr(simulation, "model_dump_json"):
            simulation_payload = json.loads(simulation.model_dump_json())
        elif hasattr(simulation, "model_dump"):
            simulation_payload = simulation.model_dump()
        else:
            simulation_payload = simulation.dict()
        record = {
            "sample_index": sample.index,
            "task_id": simulation.task_id,
            "reward": sample.reward,
            "status": sample.status.value,
            "metadata": sample.metadata,
            "simulation": simulation_payload,
        }
        path = Path(dump_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    except Exception:
        return


def _run_tau2_rollout_sync(args, sample: Sample, sampling_params: dict[str, Any]) -> Sample:
    started = time.perf_counter()
    task = task_from_metadata(sample.metadata)
    domain = domain_from_metadata(sample.metadata, task)
    db_path = db_path_from_metadata(sample.metadata)
    environment_constructor = make_environment_constructor(domain=domain, db_path=db_path)
    environment = environment_constructor()

    agent = TrainableSGLangAgent(
        tools=environment.get_tools(),
        domain_policy=environment.get_policy(),
        llm=args.hf_checkpoint,
        llm_args={
            "api_base": _agent_api_base(args),
            "sampling_params": _sampling_params_for_agent(sampling_params),
        },
    )
    user_model = os.environ.get("TAU2_USER_MODEL", "Qwen3-4B-tau2-user-sft-stop-iter0006899")
    user = build_user(
        "user_simulator",
        environment,
        task,
        llm=_with_openai_prefix(user_model),
        llm_args=_user_llm_args(),
    )

    orchestrator = Orchestrator(
        domain=domain,
        agent=agent,
        user=user,
        environment=environment,
        task=task,
        max_steps=int(os.environ.get("TAU2_MAX_STEPS", "80")),
        max_errors=int(os.environ.get("TAU2_MAX_ERRORS", "10")),
        seed=getattr(args, "rollout_seed", None),
        validate_communication=os.environ.get("TAU2_VALIDATE_COMMUNICATION", "0") != "0",
        timeout=float(os.environ["TAU2_SIMULATION_TIMEOUT"]) if os.environ.get("TAU2_SIMULATION_TIMEOUT") else None,
    )
    simulation = orchestrator.run()
    reward_info = evaluate_simulation_with_constructor(
        simulation=simulation,
        task=task,
        domain=domain,
        environment_constructor=environment_constructor,
    )
    simulation.reward_info = reward_info

    sample = _fill_sample_from_simulation(
        args=args,
        sample=sample,
        simulation=simulation,
        reward_value=float(reward_info.reward),
        system_prompt=agent.system_prompt,
    )
    sample.metadata.update(
        {
            "tau2_domain": domain,
            "tau2_task_id": task.id,
            "tau2_db_path": str(db_path),
            "tau2_termination_reason": simulation.termination_reason,
            "tau2_reward_info": reward_info_to_dict(reward_info),
            "tau2_num_messages": len(simulation.messages),
        }
    )
    sample.non_generation_time = time.perf_counter() - started
    _dump_trajectory(sample, simulation)
    return sample


def _failed_sample(args, sample: Sample, error: BaseException) -> Sample:
    generator = _get_mask_generator(args)
    prompt = str(sample.prompt or "")
    prompt_tokens = generator.tokenizer(prompt, add_special_tokens=False)["input_ids"]
    fallback = generator.tokenizer("\n", add_special_tokens=False)["input_ids"]
    sample.tokens = prompt_tokens + fallback
    sample.response = "\n"
    sample.response_length = len(fallback)
    sample.loss_mask = [0] * sample.response_length
    sample.rollout_log_probs = [0.0] * sample.response_length
    sample.reward = 0.0
    sample.status = Sample.Status.FAILED
    sample.remove_sample = True
    sample.metadata.update(
        {
            "tau2_rollout_error": repr(error),
            "tau2_rollout_traceback": traceback.format_exc(limit=20),
        }
    )
    return sample


async def generate(args, sample: Sample, sampling_params: dict[str, Any]) -> Sample:
    assert not args.partial_rollout, "Partial rollout is not supported for tau2-bench RL."
    try:
        return await asyncio.to_thread(_run_tau2_rollout_sync, args, sample, sampling_params)
    except Exception as exc:
        if os.environ.get("TAU2_RL_RAISE_ERRORS", "0") != "0":
            raise
        return _failed_sample(args, sample, exc)
