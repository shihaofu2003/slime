"""slime custom-generate entrypoint for tau2-bench RL."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

SHARED_DIR = Path(__file__).resolve().parents[1] / "shared"
if str(SHARED_DIR) not in sys.path:
    sys.path.insert(0, str(SHARED_DIR))
ANALYSIS_DIR = Path(__file__).resolve().parents[1] / "analysis"
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

from protocol_profiles import (  # noqa: E402
    PROTOCOL_CURRENT_SINGLE,
    protocol_signature,
    validate_protocol_profile,
)
from agent_contract import (  # noqa: E402
    OFFICIAL_AGENT_VIEW,
    AgentContract,
    native_agent_message_to_chat,
)
from namespace_analyzer import raw_tool_names  # noqa: E402

from agent import TrainableSGLangAgent, render_tool_call
from envs import (
    db_path_from_metadata,
    domain_from_metadata,
    make_environment_constructor,
    task_from_metadata,
)
from reward import evaluate_simulation_with_constructor, reward_info_to_dict
from reward_postprocess import (
    TURN_CREDIT_V1,
    TURN_CREDIT_V2,
    TurnCreditAlignmentError,
    build_turn_credit_v2,
    build_token_penalties,
    configured_turn_credit_version,
    compute_field_reward_signals,
    compute_rollout_score,
)

from slime.rollout.sglang_rollout import GenerateState
from slime.utils.mask_utils import MultiTurnLossMaskGenerator
from slime.utils.types import Sample

from tau2.data_model.message import AssistantMessage, Message, ToolMessage, UserMessage
from tau2.data_model.simulation import SimulationRun
from tau2.orchestrator.orchestrator import DEFAULT_FIRST_AGENT_MESSAGE, Orchestrator
from tau2.runner.build import build_user


MASK_GENERATOR: MultiTurnLossMaskGenerator | None = None

# Per-trajectory token cap. A trajectory longer than this is neutralized
# per-sample (see _fill_sample_from_simulation) so its [T, V] logits cannot OOM
# the log-prob forward. Set TAU2_RL_MAX_TRAIN_TOKENS to override; keep
# --max-tokens-per-gpu equal to this so the bin packer and the cap agree.
DEFAULT_MAX_TRAIN_TOKENS = 16384


def max_train_tokens() -> int:
    raw = os.environ.get("TAU2_RL_MAX_TRAIN_TOKENS")
    if raw:
        try:
            value = int(raw)
            if value > 0:
                return value
        except ValueError:
            pass
    return DEFAULT_MAX_TRAIN_TOKENS


def agent_protocol_profile() -> str:
    profile = os.environ.get(
        "TAU2_AGENT_PROTOCOL_PROFILE",
        PROTOCOL_CURRENT_SINGLE,
    )
    return validate_protocol_profile(profile)


def turn_credit_enabled() -> bool:
    return bool(configured_turn_credit_version())


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


def _sampling_params_for_attempt(
    base_params: dict[str, Any],
    *,
    attempt: int,
    retry_seed_stride: int,
) -> dict[str, Any]:
    params = dict(base_params)
    if "sampling_seed" in params:
        params["sampling_seed"] = (
            int(params["sampling_seed"]) + attempt * retry_seed_stride
        )
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


def _message_to_training_chat(
    message: Message,
    contract: AgentContract | None = None,
    native_tools: list[dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    if contract is not None or native_tools is not None:
        chat = (
            contract.message_to_chat(
                message,
                validate_assistant_tools=False,
            )
            if contract is not None
            else native_agent_message_to_chat(
                message,
                validate_assistant_tools=False,
            )
        )
        if chat is None:
            return None
        if isinstance(message, AssistantMessage):
            raw_data = message.raw_data if isinstance(message.raw_data, dict) else {}
            trainable = bool(raw_data.get("tau2_rl_agent"))
            if not trainable or message == DEFAULT_FIRST_AGENT_MESSAGE:
                chat["step_loss_mask"] = 0
        return chat

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


def _training_messages(
    simulation: SimulationRun,
    system_prompt: str,
    contract: AgentContract | None = None,
    native_tools: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], dict[int, int]]:
    messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt, "step_loss_mask": 0}]
    assistant_source_indices: dict[int, int] = {}
    for source_index, message in enumerate(simulation.messages):
        chat = _message_to_training_chat(message, contract, native_tools)
        if chat is not None:
            training_message_index = len(messages)
            messages.append(chat)
            if isinstance(message, AssistantMessage):
                assistant_source_indices[training_message_index] = source_index
    return messages, assistant_source_indices


def _sample_status(simulation: SimulationRun) -> Sample.Status:
    reason = simulation.termination_reason
    if reason in {"agent_stop", "user_stop"}:
        return Sample.Status.COMPLETED
    if reason in {"max_steps", "timeout", "context_window_exceeded"}:
        return Sample.Status.TRUNCATED
    return Sample.Status.FAILED


def _fill_sample_from_simulation(
    *,
    generator: MultiTurnLossMaskGenerator,
    sample: Sample,
    simulation: SimulationRun,
    reward_value: float,
    system_prompt: str,
    protocol_profile: str,
    contract: AgentContract | None = None,
    native_tools: list[dict[str, Any]] | None = None,
    field_reward_signals: dict[str, Any] | None = None,
    permanently_too_long: bool = False,
) -> Sample:
    """Tokenize the FULL rolled-out conversation verbatim (on-policy).

    The training tokens are exactly the conversation the agent generated against
    during rollout — no leading messages are dropped. Truncating the training
    context after the fact would make the policy log-probs (computed under the
    shortened context) disagree with the log-probs the actions were actually
    sampled under, breaking the on-policy assumption GRPO relies on.

    Over-long trajectories are resampled per-sample upstream
    (``_run_tau2_rollout_sync`` retries the rollout). If a trajectory is STILL
    too long after all retries (``permanently_too_long``), it is neutralized
    here as a memory-safe fallback (1-token placeholder, zero loss mask,
    ``remove_sample``) and flagged; the group-level dynamic filter then drops
    the whole group. The neutralization also guards the no-trainable-tokens
    (``response_length == 0``) case.
    """
    messages, assistant_source_indices = _training_messages(
        simulation,
        system_prompt,
        contract,
        native_tools,
    )
    template_tools = contract.tools if contract is not None else native_tools
    assistant_spans: list[dict[str, int | bool]] = []
    if turn_credit_enabled():
        if generator.tokenizer_type != "qwen3_full":
            raise TurnCreditAlignmentError(
                "turn-aware tau2 credit requires --loss-mask-type qwen3_full"
            )
        token_ids, full_loss_mask, assistant_spans = (
            generator.gen_multi_turn_loss_mask_qwen3_full(
                messages,
                tools=template_tools,
                return_assistant_spans=True,
            )
        )
    else:
        token_ids, full_loss_mask = generator.get_loss_mask(
            messages,
            tools=template_tools,
        )
    original_total_tokens = len(token_ids)
    response_length = generator.get_response_lengths([full_loss_mask])[0]
    no_trainable_tokens = response_length == 0
    cap = max_train_tokens()
    too_long = original_total_tokens > cap

    if no_trainable_tokens or too_long:
        # No useful trainable tokens here (none produced, or the full
        # conversation is too long to forward safely). Shrink to a placeholder
        # with a zero loss mask so the sample is memory-cheap and contributes no
        # gradient. For too_long we also flag remove_sample.
        fallback = generator.tokenizer("\n", add_special_tokens=False)["input_ids"]
        token_ids = fallback
        response_length = len(fallback)
        loss_mask = [0] * response_length
        sample.remove_sample = True
    else:
        loss_mask = full_loss_mask[-response_length:]

    sample.tokens = token_ids
    sample.response_length = response_length
    sample.loss_mask = loss_mask
    sample.rollout_log_probs = [0.0] * response_length
    sample.response = generator.tokenizer.decode(token_ids[-response_length:])
    sample.reward = reward_value
    sample.status = Sample.Status.TRUNCATED if too_long else _sample_status(simulation)
    if turn_credit_enabled():
        turn_credit_version = configured_turn_credit_version()
        sample.metadata["tau2_turn_credit_version"] = turn_credit_version
        if no_trainable_tokens or too_long:
            sample.train_metadata = (
                {"token_penalties": [0.0] * response_length}
                if turn_credit_version == TURN_CREDIT_V1
                else {
                    "turn_credit_version": TURN_CREDIT_V2,
                    "token_advantages": [0.0] * response_length,
                }
            )
            sample.metadata["tau2_turn_credit_invalid"] = (
                "no_trainable_response" if no_trainable_tokens else "over_train_token_cap"
            )
        else:
            if not isinstance(field_reward_signals, dict):
                raise TurnCreditAlignmentError("turn-aware rollout is missing field reward signals")
            response_start = len(token_ids) - response_length
            if turn_credit_version == TURN_CREDIT_V1:
                token_penalties, turn_details = build_token_penalties(
                    assistant_spans=assistant_spans,
                    assistant_source_indices=assistant_source_indices,
                    response_start=response_start,
                    response_length=response_length,
                    loss_mask=loss_mask,
                    turn_errors=list(field_reward_signals.get("turn_errors") or []),
                )
                sample.train_metadata = {"token_penalties": token_penalties}
            else:
                turn_details = build_turn_credit_v2(
                    assistant_spans=assistant_spans,
                    assistant_source_indices=assistant_source_indices,
                    response_start=response_start,
                    response_length=response_length,
                    loss_mask=loss_mask,
                    field_reward_signals=field_reward_signals,
                )
                sample.train_metadata = {"turn_credit_version": TURN_CREDIT_V2}
            sample.metadata.update(
                {
                    "tau2_turn_credits": turn_details,
                    "tau2_turn_credit_response_start": response_start,
                }
            )
    sample.metadata.update(
        {
            "tau2_original_total_tokens": original_total_tokens,
            "tau2_train_total_tokens": len(token_ids),
            "tau2_response_tokens": sum(loss_mask),
            "tau2_agent_protocol_profile": protocol_profile,
            "tau2_dropped_too_long": too_long,
            "tau2_permanently_too_long": permanently_too_long,
        }
    )
    active_signature = protocol_signature(protocol_profile)
    if active_signature is not None:
        sample.metadata["tau2_agent_protocol_signature"] = active_signature
    if contract is not None:
        sample.metadata.update(contract.metadata(view=OFFICIAL_AGENT_VIEW))
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
            "group_index": sample.group_index,
            "task_id": simulation.task_id,
            "reward": sample.reward,
            "status": sample.status.value,
            "remove_sample": sample.remove_sample,
            "response_length": sample.response_length,
            "loss_mask": sample.loss_mask,
            "train_metadata": sample.train_metadata,
            "metadata": sample.metadata,
            "simulation": simulation_payload,
        }
        path = Path(dump_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    except Exception:
        return


def _conversation_token_len(
    generator: MultiTurnLossMaskGenerator,
    simulation: SimulationRun,
    system_prompt: str,
    contract: AgentContract | None = None,
    native_tools: list[dict[str, Any]] | None = None,
) -> int:
    """Token count of the full rolled-out conversation (the forward-pass length)."""
    messages, _ = _training_messages(
        simulation,
        system_prompt,
        contract,
        native_tools,
    )
    token_ids, _ = generator.get_loss_mask(
        messages,
        tools=contract.tools if contract is not None else native_tools,
    )
    return len(token_ids)


def _field_reward_payload(
    *,
    task,
    simulation: SimulationRun,
    allowed_tool_names: set[str],
    user_tool_names: set[str],
    reward_info: dict[str, Any],
) -> dict[str, Any]:
    """Collect serializable action, outcome, and rule signals for reward credit."""
    golden_actions = []
    criteria = getattr(task, "evaluation_criteria", None)
    for action in (getattr(criteria, "actions", None) or []):
        requestor = getattr(action, "requestor", "assistant")
        requestor = getattr(requestor, "value", requestor)
        if requestor != "assistant":
            continue
        golden_actions.append(
            {
                "name": action.name,
                "arguments": dict(action.arguments),
                "compare_args": action.compare_args,
            }
        )

    predicted_calls = []
    attempted_calls = []
    parse_error_count = 0
    parse_error_turns: list[int] = []
    last_trainable_turn_index: int | None = None
    errored_assistant_call_ids = {
        message.id
        for message in simulation.messages
        if isinstance(message, ToolMessage)
        and message.requestor == "assistant"
        and message.error
    }
    for turn_index, message in enumerate(simulation.messages):
        if not isinstance(message, AssistantMessage):
            continue
        raw_data = message.raw_data if isinstance(message.raw_data, dict) else {}
        if not raw_data.get("tau2_rl_agent"):
            continue
        last_trainable_turn_index = turn_index
        if raw_data.get("tau2_tool_parse_error"):
            turn_parse_error_count = int(
                raw_data.get("tau2_tool_parse_error_count", 1)
            )
            if turn_parse_error_count < 1:
                raise TurnCreditAlignmentError(
                    "tau2_tool_parse_error_count must be positive"
                )
            parse_error_count += turn_parse_error_count
            parse_error_turns.extend([turn_index] * turn_parse_error_count)
        raw_names = raw_tool_names(str(raw_data.get("text") or ""))
        if raw_names:
            attempted_calls.extend(
                {"name": name, "turn_index": turn_index} for name in raw_names
            )
        if message.is_tool_call():
            structured = [
                {
                    "id": call.id,
                    "name": call.name,
                    "arguments": dict(call.arguments),
                    "turn_index": turn_index,
                    "execution_error": call.id in errored_assistant_call_ids,
                }
                for call in message.tool_calls
            ]
            predicted_calls.extend(structured)
            if not raw_names:
                attempted_calls.extend(
                    {"name": call["name"], "turn_index": turn_index}
                    for call in structured
                )

    return compute_field_reward_signals(
        golden_actions=golden_actions,
        predicted_calls=predicted_calls,
        attempted_calls=attempted_calls,
        allowed_tool_names=allowed_tool_names,
        user_tool_names=user_tool_names,
        parse_error_count=parse_error_count,
        parse_error_turns=parse_error_turns,
        last_trainable_turn_index=last_trainable_turn_index,
        termination_reason=simulation.termination_reason,
        reward_info=reward_info,
    )


def _run_tau2_rollout_sync(args, sample: Sample, sampling_params: dict[str, Any]) -> Sample:
    started = time.perf_counter()
    task = task_from_metadata(sample.metadata)
    domain = domain_from_metadata(sample.metadata, task)
    db_path = db_path_from_metadata(sample.metadata)
    environment_constructor = make_environment_constructor(domain=domain, db_path=db_path)

    # One environment for the agent/user schema (tools + policy are domain-level
    # and identical across DB reloads); each rollout attempt below reloads a
    # FRESH environment so tool side-effects do not leak across retries.
    setup_environment = environment_constructor()
    try:
        setup_user_tools = setup_environment.get_user_tools() or []
    except ValueError:
        setup_user_tools = []
    protocol_profile = agent_protocol_profile()
    base_agent_sampling_params = _sampling_params_for_agent(sampling_params)
    agent = TrainableSGLangAgent(
        tools=setup_environment.get_tools(),
        domain_policy=setup_environment.get_policy(),
        llm=args.hf_checkpoint,
        llm_args={
            "api_base": _agent_api_base(args),
            "sampling_params": base_agent_sampling_params,
            "protocol_profile": protocol_profile,
        },
        domain=domain,
        user_tools=setup_user_tools,
    )
    user_model = os.environ.get("TAU2_USER_MODEL", "Qwen3-4B-tau2-user-sft-stop-iter0006899")
    system_prompt = agent.system_prompt

    generator = _get_mask_generator(args)
    cap = max_train_tokens()
    max_retries = max(0, int(os.environ.get("TAU2_RL_MAX_ROLLOUT_RETRIES", "2")))
    max_steps = int(os.environ.get("TAU2_MAX_STEPS", "80"))
    max_errors = int(os.environ.get("TAU2_MAX_ERRORS", "10"))
    seed = getattr(args, "rollout_seed", None)
    validate_communication = os.environ.get("TAU2_VALIDATE_COMMUNICATION", "0") != "0"
    timeout = (
        float(os.environ["TAU2_SIMULATION_TIMEOUT"])
        if os.environ.get("TAU2_SIMULATION_TIMEOUT")
        else None
    )

    # Per-sample resampling for over-long trajectories: re-run the rollout
    # (a fresh stochastic sample under temperature sampling) until it fits the
    # cap, up to max_retries extra attempts. Only if every attempt is too long
    # do we mark the sample "permanently too long" — the group-level dynamic
    # filter then drops the whole group (the user's escalation). This keeps the
    # other trajectories in the group on-policy and untouched.
    simulation = None
    attempts = 0
    permanently_too_long = False
    for attempt in range(max_retries + 1):
        attempts = attempt + 1
        agent.llm_args["sampling_params"] = _sampling_params_for_attempt(
            base_agent_sampling_params,
            attempt=attempt,
            retry_seed_stride=args.n_samples_per_prompt,
        )
        run_environment = environment_constructor()
        # User tools mutate the device DB owned by their Environment instance,
        # and UserSimulator carries conversation state.  Rebuild both together
        # on every retry so an over-cap attempt cannot leak state into the next
        # attempt or execute User actions against the setup-only environment.
        run_user = build_user(
            "user_simulator",
            run_environment,
            task,
            llm=_with_openai_prefix(user_model),
            llm_args=_user_llm_args(),
        )
        orchestrator = Orchestrator(
            domain=domain,
            agent=agent,
            user=run_user,
            environment=run_environment,
            task=task,
            max_steps=max_steps,
            max_errors=max_errors,
            seed=seed,
            validate_communication=validate_communication,
            timeout=timeout,
        )
        candidate = orchestrator.run()
        token_len = _conversation_token_len(
            generator,
            candidate,
            system_prompt,
            agent.contract,
            agent.native_tools,
        )
        simulation = candidate
        if token_len <= cap:
            break
        permanently_too_long = attempt == max_retries

    reward_info = evaluate_simulation_with_constructor(
        simulation=simulation,
        task=task,
        domain=domain,
        environment_constructor=environment_constructor,
    )
    simulation.reward_info = reward_info

    reward_info_dict = reward_info_to_dict(reward_info)
    field_reward_signals = _field_reward_payload(
        task=task,
        simulation=simulation,
        allowed_tool_names=(
            agent.contract.agent_tool_names
            if agent.contract is not None
            else {tool.name for tool in setup_environment.get_tools()}
        ),
        user_tool_names=(
            agent.contract.user_tool_names
            if agent.contract is not None
            else {tool.name for tool in setup_user_tools}
        ),
        reward_info=reward_info_dict,
    )

    sample = _fill_sample_from_simulation(
        generator=generator,
        sample=sample,
        simulation=simulation,
        reward_value=float(reward_info.reward),
        system_prompt=system_prompt,
        protocol_profile=protocol_profile,
        contract=agent.contract,
        native_tools=agent.native_tools,
        field_reward_signals=field_reward_signals,
        permanently_too_long=permanently_too_long,
    )
    sample.metadata.update(
        {
            "tau2_domain": domain,
            "tau2_task_id": task.id,
            "tau2_agent_protocol_profile": protocol_profile,
            "tau2_db_path": str(db_path),
            "tau2_termination_reason": simulation.termination_reason,
            "tau2_reward_info": reward_info_dict,
            "tau2_field_reward_signals": field_reward_signals,
            "tau2_num_messages": len(simulation.messages),
            "tau2_rollout_attempts": attempts,
            "tau2_permanently_too_long": permanently_too_long,
        }
    )
    if agent.protocol_signature is not None:
        sample.metadata["tau2_agent_protocol_signature"] = agent.protocol_signature
    if os.environ.get("TAU2_USE_REWARD_SHAPING", "1") != "0":
        _, global_score_details = compute_rollout_score(args, sample)
        sample.metadata.update(global_score_details)
    else:
        sample.metadata["raw_reward"] = float(sample.reward)
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
    turn_credit_version = configured_turn_credit_version()
    if turn_credit_version == TURN_CREDIT_V1:
        sample.train_metadata = {"token_penalties": [0.0] * sample.response_length}
    elif turn_credit_version == TURN_CREDIT_V2:
        sample.train_metadata = {
            "turn_credit_version": TURN_CREDIT_V2,
            "token_advantages": [0.0] * sample.response_length,
        }
    profile = os.environ.get(
        "TAU2_AGENT_PROTOCOL_PROFILE",
        PROTOCOL_CURRENT_SINGLE,
    )
    sample.metadata.update(
        {
            "tau2_rollout_error": repr(error),
            "tau2_rollout_traceback": traceback.format_exc(limit=20),
            "tau2_agent_protocol_profile": profile,
            "tau2_turn_credit_version": turn_credit_version or None,
            "tau2_turn_credit_invalid": (
                "alignment_error"
                if isinstance(error, TurnCreditAlignmentError)
                else "rollout_error"
            ),
        }
    )
    active_signature = protocol_signature(profile)
    if active_signature is not None:
        sample.metadata["tau2_agent_protocol_signature"] = active_signature
    return sample


async def generate(args, sample: Sample, sampling_params: dict[str, Any]) -> Sample:
    assert not args.partial_rollout, "Partial rollout is not supported for tau2-bench RL."
    try:
        return await asyncio.to_thread(_run_tau2_rollout_sync, args, sample, sampling_params)
    except Exception as exc:
        if os.environ.get("TAU2_RL_RAISE_ERRORS", "0") != "0":
            raise
        return _failed_sample(args, sample, exc)
