"""Preflight checks for tau2-bench RL rollout logic.

Runs in the job container (Python 3.12 + tau2 + slime deps) BEFORE the heavy
Ray/sglang rollout, so logic regressions surface in seconds instead of after a
full rollout+train cycle. It does not need GPUs.

Verifies:
  1. On-policy tokenization: the training tokens produced from a rolled-out
     conversation (``_training_messages`` + ``MultiTurnLossMaskGenerator``)
     match the chat-template rendering the agent actually generated against.
  2. The loss mask marks only trainable assistant tokens (``step_loss_mask=0``
     is honoured; tool / user / system turns are masked off).
  3. ``_fill_sample_from_simulation`` keeps the FULL conversation (no
     leading-message truncation) — the bug that previously broke on-policy.
  4. The dynamic filter drops over-length and signal-free groups, but retains
     zero-global-variance groups with local turn penalties.
  5. The signed Agent contract is identical across official environment
     construction and rollout tokenization, with Agent-only native schemas.
  6. Every behavior error is attributed to exactly one trainable Assistant
     span with the fixed per-turn penalty and cap.
  7. Exact per-domain quotas survive zero-std/over-cap replacement and resume.
  8. Train metadata survives DP partitioning, while CP=1/2 slice the response
     penalty vector exactly like log probabilities.

Usage:
    PYTHONPATH=<slime>:<rl_dir>:<tau2_src> python3 test_rollout_logic.py
Exit code is non-zero on any failure.
"""

from __future__ import annotations

import os
import sys
from collections import Counter, deque
from pathlib import Path
from types import SimpleNamespace

import torch
from transformers import AutoTokenizer

from tau2.data_model.message import AssistantMessage, ToolCall, ToolMessage, UserMessage
from tau2.data_model.simulation import SimulationRun, TerminationReason

from slime.utils.mask_utils import MultiTurnLossMaskGenerator

import filters  # noqa: E402  (on PYTHONPATH alongside this file)
import reward_postprocess  # noqa: E402
import rollout  # noqa: E402
from agent import TrainableSGLangAgent  # noqa: E402
from protocol_profiles import (  # noqa: E402
    DEPENDENCY_SAFE_MULTI_RULE,
    PROTOCOL_AGENT_OWNED_DEPENDENCY_SAFE_MULTI,
    PROTOCOL_CURRENT_SINGLE,
    PROTOCOL_DEPENDENCY_SAFE_MULTI,
    PROTOCOL_STRICT_SINGLE_V1,
    domain_policy_for_profile,
    protocol_block_for_profile,
    protocol_signature,
    strict_single_system_prompt,
)
from agent_contract import AgentContract, contract_from_environment  # noqa: E402

HF_CHECKPOINT = os.environ.get(
    "HF_CHECKPOINT",
    "/mnt/afs/users/fush/projects/ServiceAgent/checkpoints/"
    "Qwen3-4B-Instruct-2507_tau2_agent_sft_multitool_max8192_20260714/iter_0002413_hf",
)

DOMAIN_POLICY_FIXTURES = {
    "airline": (
        "Airline. You should only make one tool call at a time, and if you make a "
        "tool call, you should not respond to the user simultaneously. If you respond "
        "to the user, you should not make a tool call at the same time."
    ),
    "retail": (
        "Retail. You should at most make one tool call at a time, and if you take a "
        "tool call, you should not respond to the user at the same time. If you respond "
        "to the user, you should not make a tool call at the same time."
    ),
    "telecom": (
        "Telecom. You should only make one tool call at a time, and if you make a "
        "tool call, you should not respond to the user simultaneously. If you respond "
        "to the user, you should not make a tool call at the same time."
    ),
}


def _system_prompt_for_profile(policy: str, profile: str) -> str:
    amended_policy, _ = domain_policy_for_profile(policy, profile)
    agent = TrainableSGLangAgent.__new__(TrainableSGLangAgent)
    agent.tools = []
    agent.contract = None
    agent.native_tools = [] if profile == PROTOCOL_STRICT_SINGLE_V1 else None
    agent.domain_policy = amended_policy
    agent.protocol_profile = profile
    return agent.system_prompt


def _legacy_current_single_prompt(policy: str) -> str:
    return (
        "You are a customer service agent. Complete the user's task while "
        "following the policy exactly.\n\n"
        "In each turn, choose exactly one action:\n"
        "- Send one plain-text message to the user, or\n"
        "- Make one or more tool calls in this exact format:\n"
        '<tool_call>{"name":"tool_name","arguments":{"param":"value"}}</tool_call>\n\n'
        "If you make multiple tool calls, emit one <tool_call> block for each call "
        "and no other content. Do not send both text and a tool call in the same turn.\n\n"
        "<policy>\n"
        f"{policy}\n"
        "</policy>\n\n"
        "<tools>\n[]\n</tools>"
    )


def _make_simulation(*, strict_single: bool = False) -> SimulationRun:
    """A short, representative multi-turn tau2 conversation."""
    tool_flow = [
        AssistantMessage(
            role="assistant",
            tool_calls=[
                ToolCall(id="call_1", name="get_order", arguments={"order_id": "123"}),
            ],
            raw_data={"tau2_rl_agent": True},
        ),
        ToolMessage(
            id="call_1",
            role="tool",
            content="order 123 status=shipped",
            requestor="assistant",
        ),
        AssistantMessage(
            role="assistant",
            tool_calls=[
                ToolCall(id="call_2", name="get_order", arguments={"order_id": "456"}),
            ],
            raw_data={"tau2_rl_agent": True},
        ),
        ToolMessage(
            id="call_2",
            role="tool",
            content="order 456 status=processing",
            requestor="assistant",
        ),
    ] if strict_single else [
        AssistantMessage(
            role="assistant",
            tool_calls=[
                ToolCall(id="call_1", name="get_order", arguments={"order_id": "123"}),
                ToolCall(id="call_2", name="get_order", arguments={"order_id": "456"}),
            ],
            raw_data={"tau2_rl_agent": True},
        ),
        ToolMessage(
            id="call_1",
            role="tool",
            content="order 123 status=shipped",
            requestor="assistant",
        ),
        ToolMessage(
            id="call_2",
            role="tool",
            content="order 456 status=processing",
            requestor="assistant",
        ),
    ]
    messages = [
        UserMessage(role="user", content="Hi, I want to cancel my order 123."),
        *tool_flow,
        AssistantMessage(
            role="assistant",
            content="Your order has already shipped, so I cannot cancel it.",
            raw_data={"tau2_rl_agent": True},
        ),
        UserMessage(role="user", content="OK, thanks."),
    ]
    return SimulationRun(
        id="sim_test_0",
        task_id="retail_test_0",
        start_time="2026-07-26T00:00:00Z",
        end_time="2026-07-26T00:00:01Z",
        duration=1.0,
        messages=messages,
        termination_reason=TerminationReason.AGENT_STOP.value,
    )


def _chat_dicts_for_template(
    simulation: SimulationRun,
    system_prompt: str,
    contract: AgentContract | None = None,
    native_tools: list[dict] | None = None,
) -> list[dict]:
    """Mirror TrainableSGLangAgent._message_to_chat (content-only, no masks)."""
    msgs = [{"role": "system", "content": system_prompt}]
    for message in simulation.messages:
        msgs.append(rollout._message_to_training_chat(message, contract, native_tools))
    # strip the step_loss_mask keys the template does not understand
    return [{k: v for k, v in m.items() if k != "step_loss_mask"} for m in msgs if m is not None]


def _tool_schema(name: str) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": name,
            "parameters": {
                "type": "object",
                "properties": {"order_id": {"type": "string"}},
            },
        },
    }


def _fixture_contract(tokenizer) -> AgentContract:
    return AgentContract.create(
        domain="retail",
        domain_policy=DOMAIN_POLICY_FIXTURES["retail"],
        agent_tools=[_tool_schema("get_order")],
        user_tools=[_tool_schema("check_device")],
        chat_template=tokenizer.chat_template,
        profile=PROTOCOL_AGENT_OWNED_DEPENDENCY_SAFE_MULTI,
    )


def _quota_source(quota: dict[str, int]):
    """Construct a pure in-memory DomainQuotaDataSource preflight fixture."""
    source = filters.DomainQuotaDataSource.__new__(filters.DomainQuotaDataSource)
    source.args = SimpleNamespace(
        n_samples_per_prompt=2,
        rollout_shuffle=False,
        rollout_seed=17,
    )
    source.quota = {domain: quota.get(domain, 0) for domain in filters._DOMAINS}
    source.buffer = []
    source.metadata = {}
    source.sample_group_index = 0
    source.sample_index = 0
    source._domain_samples = {
        domain: [
            rollout.Sample(prompt=f"{domain}-{index}", metadata={"domain": domain})
            for index in range(12)
        ]
        for domain in filters._DOMAINS
    }
    source._domain_offsets = {domain: 0 for domain in filters._DOMAINS}
    source._domain_epochs = {domain: 0 for domain in filters._DOMAINS}
    source._domain_orders = {}
    for domain in filters._DOMAINS:
        source._refresh_domain_order(domain)
    source._dataset_fingerprint = (
        None
        if os.environ.get("TAU2_AGENT_PROTOCOL_PROFILE")
        == PROTOCOL_STRICT_SINGLE_V1
        else "preflight-domain-dataset-v1"
    )
    source._active_rollout_id = None
    source._initial_domains = deque()
    source._replacement_domains = deque()
    source._accepted = Counter()
    source._rejected = Counter()
    source._replacement_draws = Counter()
    return source


def main() -> int:
    # `failures` block the run (they verify logic this pipeline changed: the
    # full-conversation tokenization and the length/reward filter). `warnings`
    # are informational — they verify pre-existing slime/tau2 tokenization
    # behaviour that prior smoke runs already exercised, so a surprise there is
    # worth surfacing but should not block the OOM-verification rollout.
    failures: list[str] = []
    warnings: list[str] = []

    seeded_retry_params = rollout._sampling_params_for_attempt(
        {"sampling_seed": 42, "temperature": 1.0},
        attempt=2,
        retry_seed_stride=8,
    )
    if seeded_retry_params != {"sampling_seed": 58, "temperature": 1.0}:
        failures.append("over-cap retry did not advance the deterministic sampling seed")
    unseeded_retry_params = rollout._sampling_params_for_attempt(
        {"temperature": 1.0},
        attempt=2,
        retry_seed_stride=8,
    )
    if unseeded_retry_params != {"temperature": 1.0}:
        failures.append("over-cap retry changed unseeded sampling parameters")

    tokenizer = AutoTokenizer.from_pretrained(HF_CHECKPOINT, trust_remote_code=True)

    # ---- check 0: shared profile selection and exact prompt wiring -------
    for domain, policy in DOMAIN_POLICY_FIXTURES.items():
        unchanged, replaced = domain_policy_for_profile(
            policy,
            PROTOCOL_CURRENT_SINGLE,
        )
        if unchanged != policy or replaced:
            failures.append(f"{domain}: current-single changed the domain policy")
        amended, replaced = domain_policy_for_profile(
            policy,
            PROTOCOL_DEPENDENCY_SAFE_MULTI,
        )
        if not replaced or DEPENDENCY_SAFE_MULTI_RULE not in amended:
            failures.append(f"{domain}: dependency-safe clause was not substituted")
        if "one tool call at a time" in amended:
            failures.append(f"{domain}: single-call clause remains")
        strict_policy, strict_replaced = domain_policy_for_profile(
            policy,
            PROTOCOL_STRICT_SINGLE_V1,
        )
        if not strict_replaced or "one tool call at a time" in strict_policy:
            failures.append(f"{domain}: strict-single policy was not normalized")

    try:
        domain_policy_for_profile("policy", "not-a-profile")
    except ValueError:
        pass
    else:
        failures.append("invalid protocol profile was accepted")

    fixture_policy = DOMAIN_POLICY_FIXTURES["airline"]
    current_prompt = _system_prompt_for_profile(
        fixture_policy,
        PROTOCOL_CURRENT_SINGLE,
    )
    if current_prompt != _legacy_current_single_prompt(fixture_policy):
        failures.append("current-single Agent prompt changed from the legacy prompt")
    dependency_safe_prompt = _system_prompt_for_profile(
        fixture_policy,
        PROTOCOL_DEPENDENCY_SAFE_MULTI,
    )
    protocol_block = protocol_block_for_profile(PROTOCOL_DEPENDENCY_SAFE_MULTI)
    if protocol_block not in dependency_safe_prompt:
        failures.append("dependency-safe Agent prompt omitted the shared protocol block")
    if "one tool call at a time" in dependency_safe_prompt:
        failures.append("dependency-safe Agent prompt retained a single-call clause")
    strict_prompt = _system_prompt_for_profile(
        fixture_policy,
        PROTOCOL_STRICT_SINGLE_V1,
    )
    normalized_strict_policy, _ = domain_policy_for_profile(
        fixture_policy,
        PROTOCOL_STRICT_SINGLE_V1,
    )
    if strict_prompt != strict_single_system_prompt(normalized_strict_policy):
        failures.append("strict-single Agent prompt differs from the shared prompt")

    parse_agent = TrainableSGLangAgent.__new__(TrainableSGLangAgent)
    parse_agent.protocol_profile = PROTOCOL_AGENT_OWNED_DEPENDENCY_SAFE_MULTI
    parse_agent.protocol_signature = protocol_signature(
        PROTOCOL_AGENT_OWNED_DEPENDENCY_SAFE_MULTI
    )
    parse_agent.contract = None
    mixed_message = parse_agent._assistant_from_text(
        "prefix "
        '<tool_call>{"name":"get_order","arguments":{}}</tool_call>'
        '<tool_call>{"name":"get_order","arguments":{}}</tool_call>',
        {},
    )
    if (mixed_message.raw_data or {}).get("tau2_tool_parse_error_count") != 2:
        failures.append("multiple mixed-text tool calls were not counted per event")

    strict_parse_agent = TrainableSGLangAgent.__new__(TrainableSGLangAgent)
    strict_parse_agent.protocol_profile = PROTOCOL_STRICT_SINGLE_V1
    strict_parse_agent.protocol_signature = None
    strict_parse_agent.contract = None
    strict_parse_agent.native_tools = []
    strict_multi_message = strict_parse_agent._assistant_from_text(
        '<tool_call>{"name":"get_order","arguments":{}}</tool_call>'
        '<tool_call>{"name":"get_order","arguments":{}}</tool_call>',
        {},
    )
    strict_raw_data = strict_multi_message.raw_data or {}
    if strict_multi_message.is_tool_call():
        failures.append("strict-single parser batch-executed a multi-call output")
    if (
        strict_raw_data.get("tau2_single_call_protocol_error") is not True
        or strict_raw_data.get("tau2_multi_tool_attempt_count") != 2
    ):
        failures.append("strict-single parser did not record its protocol error")
    if any("signature" in key or "hash" in key for key in strict_raw_data):
        failures.append("strict-single parser emitted a hash or signature")

    active_profile = rollout.agent_protocol_profile()
    contract = None
    native_tools = None
    if active_profile == PROTOCOL_AGENT_OWNED_DEPENDENCY_SAFE_MULTI:
        contract = _fixture_contract(tokenizer)
        system_prompt = contract.system_prompt
        generator = MultiTurnLossMaskGenerator(tokenizer, tokenizer_type="qwen3_full")

        # Resolve every official domain through the same constructor used by
        # SFT/eval/RL and require deterministic content signatures.
        from tau2.registry import registry

        for domain in ("telecom", "airline", "retail"):
            environment = registry.get_env_constructor(domain)()
            first = contract_from_environment(
                environment,
                domain=domain,
                chat_template=tokenizer.chat_template,
                profile=active_profile,
            )
            second = contract_from_environment(
                registry.get_env_constructor(domain)(),
                domain=domain,
                chat_template=tokenizer.chat_template,
                profile=active_profile,
            )
            if first.agent_contract_signature != second.agent_contract_signature:
                failures.append(f"{domain}: Agent contract signature is not deterministic")
            agent_names = {schema["function"]["name"] for schema in first.tools}
            user_names = {schema["function"]["name"] for schema in first.user_tools}
            if agent_names & user_names:
                failures.append(f"{domain}: Agent/User schemas overlap")
            if any(name in agent_names for name in user_names):
                failures.append(f"{domain}: native tools include User-owned schemas")
            if "<tools>" in first.system_prompt or any(name == "done" for name in agent_names):
                failures.append(f"{domain}: contract retained handwritten tools or fake done")
            if domain == "telecom" and "<tech_support_policy>" not in first.system_prompt:
                failures.append("telecom: full technical-support manual is missing")
    elif active_profile == PROTOCOL_STRICT_SINGLE_V1:
        native_tools = [_tool_schema("get_order")]
        system_prompt = _system_prompt_for_profile(fixture_policy, active_profile)
        generator = MultiTurnLossMaskGenerator(tokenizer, tokenizer_type="qwen3_full")
    else:
        system_prompt = _system_prompt_for_profile(fixture_policy, active_profile)
        generator = MultiTurnLossMaskGenerator(tokenizer, tokenizer_type="qwen3")

    simulation = _make_simulation(
        strict_single=active_profile == PROTOCOL_STRICT_SINGLE_V1
    )
    messages, _ = rollout._training_messages(
        simulation,
        system_prompt,
        contract,
        native_tools,
    )
    if messages[0]["content"] != system_prompt:
        failures.append("training tokenization did not receive the actual Agent prompt")
    if active_profile == PROTOCOL_STRICT_SINGLE_V1:
        if any(
            len(message.get("tool_calls") or []) > 1
            for message in messages
            if message.get("role") == "assistant"
        ):
            failures.append("strict-single training view retained a multi-call turn")
        if not any(message.get("role") == "tool" for message in messages):
            failures.append("strict-single training view lost native tool results")

    # ---- check 1: on-policy tokenization (informational) -----------------
    template_tools = contract.tools if contract is not None else native_tools
    train_ids, train_mask = generator.get_loss_mask(messages, tools=template_tools)
    # Render the template to text, then tokenize (apply_chat_template(tokenize=True)
    # can return an Encoding object rather than a list of ids, depending on the
    # tokenizer). This mirrors the reference check in mask_utils.gen_multi_turn_loss_mask_qwen3_5.
    template_text = tokenizer.apply_chat_template(
        _chat_dicts_for_template(
            simulation,
            system_prompt,
            contract,
            native_tools,
        ),
        add_generation_prompt=False,
        tokenize=False,
        tools=template_tools,
    )
    template_ids = tokenizer(template_text, add_special_tokens=False)["input_ids"]
    if train_ids != template_ids:
        first_diff = next(
            (i for i in range(min(len(train_ids), len(template_ids))) if train_ids[i] != template_ids[i]),
            min(len(train_ids), len(template_ids)),
        )
        destination = failures if template_tools is not None else warnings
        destination.append(
            f"on-policy mismatch: training tokens ({len(train_ids)}) != "
            f"chat-template tokens ({len(template_ids)}); first diff at {first_diff}: "
            f"train={train_ids[first_diff:first_diff+8]} "
            f"template={template_ids[first_diff:first_diff+8]}"
        )

    # qwen3_full must derive IDs, loss mask, and Assistant spans from this same
    # rendering, including one structured multitool turn followed by two
    # consecutive role=tool results.
    span_simulation = _make_simulation()
    span_contract = _fixture_contract(tokenizer)
    span_generator = MultiTurnLossMaskGenerator(tokenizer, tokenizer_type="qwen3_full")
    span_messages, span_sources = rollout._training_messages(
        span_simulation,
        span_contract.system_prompt,
        span_contract,
    )
    span_ids, span_mask, assistant_spans = (
        span_generator.gen_multi_turn_loss_mask_qwen3_full(
            span_messages,
            tools=span_contract.tools,
            return_assistant_spans=True,
        )
    )
    span_template_text = tokenizer.apply_chat_template(
        _chat_dicts_for_template(
            span_simulation,
            span_contract.system_prompt,
            span_contract,
        ),
        add_generation_prompt=False,
        tokenize=False,
        tools=span_contract.tools,
    )
    span_template_ids = tokenizer(span_template_text, add_special_tokens=False)["input_ids"]
    if span_ids != span_template_ids:
        failures.append("qwen3_full IDs changed when Assistant spans were requested")
    trainable_span_tokens = {
        index
        for span in assistant_spans
        if span["trainable"]
        for index in range(int(span["token_start"]), int(span["token_end"]))
    }
    if trainable_span_tokens != {index for index, mask in enumerate(span_mask) if mask}:
        failures.append("qwen3_full Assistant spans do not exactly cover the loss mask")
    trainable_sources = {
        span_sources[int(span["message_index"])]
        for span in assistant_spans
        if span["trainable"]
    }
    if trainable_sources != {1, 4}:
        failures.append(f"Assistant span/source identity changed: {trainable_sources}")

    # ---- check 2: loss mask marks only trainable assistant tokens --------
    # Decode the masked-on tokens: they must contain the assistant outputs and
    # must NOT contain the user's utterances. Verifies the mask without
    # re-stating the generator's own logic.
    supervised_text = tokenizer.decode(
        [t for t, m in zip(train_ids, train_mask, strict=True) if m]
    )
    if "get_order" not in supervised_text or "cannot cancel" not in supervised_text:
        warnings.append(f"masked-on tokens missing assistant content: {supervised_text!r}")
    if "want to cancel" in supervised_text or "OK, thanks" in supervised_text:
        warnings.append(f"masked-on tokens leaked user content: {supervised_text!r}")

    # ---- check 3: _fill_sample_from_simulation keeps the full convo ------
    sample = rollout.Sample(index=0, prompt="retail_test_0", metadata={})
    rollout._fill_sample_from_simulation(
        generator=generator,
        sample=sample,
        simulation=simulation,
        reward_value=1.0,
        system_prompt=system_prompt,
        protocol_profile=active_profile,
        contract=contract,
        native_tools=native_tools,
        field_reward_signals={"turn_errors": []},
    )
    # Full conversation preserved verbatim (no leading-message truncation).
    if sample.tokens != train_ids:
        failures.append("_fill_sample_from_simulation did not tokenize the full conversation")
    # response_length is slime's prompt/response split: the suffix from the first
    # trainable token to the end (it includes interleaved user/tool turns as
    # mask-0 within the response). loss_mask is that suffix and must preserve
    # every trainable 1.
    if len(sample.loss_mask) != sample.response_length:
        failures.append("loss_mask length != response_length")
    if sum(sample.loss_mask) != sum(train_mask):
        failures.append(
            f"loss_mask sum {sum(sample.loss_mask)} != full-mask sum {sum(train_mask)}"
        )
    if 1 in train_mask:
        expected_response_length = len(train_mask) - train_mask.index(1)
        if sample.response_length != expected_response_length:
            failures.append(
                f"response_length {sample.response_length} != suffix length {expected_response_length}"
            )
    if sample.metadata.get("tau2_agent_protocol_profile") != active_profile:
        failures.append("sample metadata omitted the active protocol profile")
    active_signature = protocol_signature(active_profile)
    if active_signature is None:
        if any("signature" in key or "hash" in key for key in sample.metadata):
            failures.append("unsigned protocol emitted a hash or signature in sample metadata")
    elif sample.metadata.get("tau2_agent_protocol_signature") != active_signature:
        failures.append("sample metadata omitted the active protocol content signature")

    previous_turn_credit_version = os.environ.get("TAU2_TURN_CREDIT_VERSION")
    os.environ["TAU2_TURN_CREDIT_VERSION"] = reward_postprocess.TURN_CREDIT_VERSION
    turn_sample = rollout.Sample(index=20, prompt="retail_test_0", metadata={})
    try:
        rollout._fill_sample_from_simulation(
            generator=span_generator,
            sample=turn_sample,
            simulation=span_simulation,
            reward_value=1.0,
            system_prompt=span_contract.system_prompt,
            protocol_profile=PROTOCOL_AGENT_OWNED_DEPENDENCY_SAFE_MULTI,
            contract=span_contract,
            field_reward_signals={"turn_errors": []},
        )
    finally:
        if previous_turn_credit_version is None:
            os.environ.pop("TAU2_TURN_CREDIT_VERSION", None)
        else:
            os.environ["TAU2_TURN_CREDIT_VERSION"] = previous_turn_credit_version
    if turn_sample.tokens != span_ids or turn_sample.metadata.get(
        "tau2_turn_credit_version"
    ) != reward_postprocess.TURN_CREDIT_VERSION:
        failures.append("turn-aware rollout did not preserve the qwen3_full token identity")
    turn_penalties = (turn_sample.train_metadata or {}).get("token_penalties")
    if turn_penalties is None or len(turn_penalties) != turn_sample.response_length:
        failures.append("turn-aware rollout omitted its response-length token penalties")
    elif any(turn_penalties):
        failures.append("clean turn-aware rollout fabricated a local penalty")

    # The historical signed profile keeps its existing SFT-gate preflight.
    # single-call-v1 intentionally starts from its own checkpoint without a gate.
    if active_profile != PROTOCOL_STRICT_SINGLE_V1:
        promotion_path = (
            Path(__file__).resolve().parents[3]
            / "output/experiments/tau2-sft-agent-user-boundary-v2/SFT_PROMOTION.json"
        )
        try:
            import json

            promotion = json.loads(promotion_path.read_text(encoding="utf-8"))
            promotion_decisions = {
                decision["label"]: decision for decision in promotion["decisions"]
            }
            if (
                promotion.get("gate_version") != "turn-aware-rl-v3"
                or promotion.get("selected") != "contract-boundary"
                or promotion_decisions["contract-boundary"].get("status") != "pass"
                or promotion_decisions["contract-only"].get("status") != "fail"
                or "namespace_probe_failed"
                not in promotion_decisions["contract-only"].get("reasons", [])
            ):
                failures.append(
                    "formal v3 SFT gate did not select only Contract + boundary"
                )
        except (KeyError, OSError, ValueError) as exc:
            failures.append(f"formal v3 SFT gate artifact is invalid: {exc}")

    # ---- check 4: over-long trajectory (retries exhausted) is neutralized
    # and flagged so the group filter can drop the whole group. The per-sample
    # *retry* itself needs a live orchestrator and is exercised by the GPU run;
    # here we verify the downstream contract _fill_sample_from_simulation signs.
    cap = int(os.environ.get("TAU2_RL_MAX_TRAIN_TOKENS", rollout.DEFAULT_MAX_TRAIN_TOKENS))
    os.environ["TAU2_RL_MAX_TRAIN_TOKENS"] = str(max(8, len(train_ids) - 1))
    long_sample = rollout.Sample(index=1, prompt="retail_test_0", metadata={})
    rollout._fill_sample_from_simulation(
        generator=generator,
        sample=long_sample,
        simulation=simulation,
        reward_value=0.0,
        system_prompt=system_prompt,
        protocol_profile=active_profile,
        contract=contract,
        native_tools=native_tools,
        field_reward_signals={"turn_errors": []},
        permanently_too_long=True,
    )
    os.environ["TAU2_RL_MAX_TRAIN_TOKENS"] = str(cap)  # restore
    if not long_sample.remove_sample:
        failures.append("over-long sample was not flagged remove_sample")
    if len(long_sample.tokens) >= len(train_ids):
        failures.append("over-long sample was not shrunk to a placeholder")
    if any(long_sample.loss_mask):
        failures.append("over-long sample has nonzero loss mask")
    if not long_sample.metadata.get("tau2_permanently_too_long"):
        failures.append("over-long sample missing tau2_permanently_too_long flag")

    # The historical checks below pin v1 semantics even when this preflight is
    # launched for a v2 candidate. The dedicated v2 block restores and tests
    # the new recipe explicitly.
    configured_turn_credit_for_preflight = os.environ.get("TAU2_TURN_CREDIT_VERSION")
    configured_replace_for_preflight = os.environ.get(
        "TAU2_REPLACE_ZERO_SIGNAL_GROUPS"
    )
    os.environ["TAU2_TURN_CREDIT_VERSION"] = reward_postprocess.TURN_CREDIT_V1
    os.environ["TAU2_REPLACE_ZERO_SIGNAL_GROUPS"] = "1"

    # ---- check 5: filter drops unsampleable and shaped-zero-std groups ----
    class _Args:
        def __getattr__(self, k):
            return None

    good_a = rollout.Sample(index=0, prompt="x", tokens=[1] * 8, reward=1.0, metadata={})
    good_b = rollout.Sample(index=1, prompt="x", tokens=[1] * 8, reward=0.0, metadata={})
    keep_mixed = filters.drop_zero_std_or_unsampleable(_Args(), [good_a, good_b])
    if not keep_mixed.keep:
        failures.append("filter dropped a valid mixed-reward group")

    zero_b = rollout.Sample(index=1, prompt="x", tokens=[1] * 8, reward=1.0, metadata={})
    keep_zero = filters.drop_zero_std_or_unsampleable(_Args(), [good_a, zero_b])
    if keep_zero.keep:
        failures.append("filter kept a zero-global-std group without local signal")

    local_a = rollout.Sample(
        index=2,
        prompt="x",
        tokens=[1, 2],
        response_length=2,
        loss_mask=[1, 1],
        reward=1.0,
        metadata={"tau2_turn_credit_version": reward_postprocess.TURN_CREDIT_VERSION},
        train_metadata={"token_penalties": [1.0, 1.0]},
    )
    local_b = rollout.Sample(
        index=3,
        prompt="x",
        tokens=[1, 2],
        response_length=2,
        loss_mask=[1, 1],
        reward=1.0,
        metadata={"tau2_turn_credit_version": reward_postprocess.TURN_CREDIT_VERSION},
        train_metadata={"token_penalties": [0.0, 0.0]},
    )
    keep_local = filters.drop_zero_std_or_unsampleable(_Args(), [local_a, local_b])
    if not keep_local.keep:
        failures.append("filter dropped a zero-global-std group with local turn signal")

    unsampleable = rollout.Sample(
        index=0, prompt="x", tokens=[1] * 8, reward=1.0,
        metadata={"tau2_permanently_too_long": True},
    )
    keep_uns = filters.drop_zero_std_or_unsampleable(_Args(), [unsampleable, good_b])
    if keep_uns.keep:
        failures.append("filter kept a group with a permanently-too-long sample")

    # ---- check 6: reward shaping gives partial credit to near-misses -----
    # Two trajectories with the SAME binary reward (0) but different action-match
    # fractions must get different shaped advantages: the near-miss (4/5 golden
    # actions matched) outranks the total failure (0/5). This is the whole point
    # of the shaper — it makes "right tool, wrong arg" distinguishable from
    # "totally wrong", which the binary reward collapses together.
    class _ShapingArgs:
        advantage_estimator = "grpo"
        rewards_normalization = True
        grpo_std_normalization = True
        n_samples_per_prompt = 2

        def __getattr__(self, k):
            return None

    def _action_checks(n_match: int, n_total: int = 5) -> list[dict]:
        return [{"action_match": i < n_match} for i in range(n_total)]

    # partial_score is the action-match fraction when only action_checks are present
    score_4, _ = reward_postprocess.compute_partial_score({"action_checks": _action_checks(4)})
    score_0, _ = reward_postprocess.compute_partial_score({"action_checks": _action_checks(0)})
    if abs(score_4 - 0.8) > 1e-9 or abs(score_0 - 0.0) > 1e-9:
        failures.append(f"partial_score wrong: 4/5 -> {score_4}, 0/5 -> {score_0}")

    near_miss = rollout.Sample(
        index=0, prompt="t", reward=0.0,
        metadata={"tau2_domain": "airline", "tau2_reward_info": {"action_checks": _action_checks(4)}},
    )
    total_fail = rollout.Sample(
        index=1, prompt="t", reward=0.0,
        metadata={"tau2_domain": "airline", "tau2_reward_info": {"action_checks": _action_checks(0)}},
    )
    raw, shaped = reward_postprocess.tau2_reward_post_process(
        _ShapingArgs(), [near_miss, total_fail]
    )
    if raw != [0.0, 0.0]:
        failures.append(f"shaping did not preserve raw rewards: {raw}")
    # same binary reward, but the near-miss must get the higher advantage
    if not (shaped[0] > shaped[1]):
        failures.append(
            f"shaping failed to rank near-miss above total failure: {shaped}"
        )
    expected_global_scores = torch.tensor(
        [
            reward_postprocess.compute_rollout_score(_ShapingArgs(), near_miss)[0],
            reward_postprocess.compute_rollout_score(_ShapingArgs(), total_fail)[0],
        ],
        dtype=torch.float32,
    )
    expected_grpo = (
        expected_global_scores - expected_global_scores.mean()
    ) / (expected_global_scores.std() + 1e-6)
    if not torch.equal(torch.tensor(shaped), expected_grpo):
        failures.append(
            "clean global shaping no longer exactly matches slime's GRPO normalization"
        )

    # ---- check 7: field-level credit plus turn-attributed errors ---------
    field_signals = reward_postprocess.compute_field_reward_signals(
        golden_actions=[
            {
                "name": "book_flight",
                "arguments": {"flight_id": "F1", "cabin": "economy"},
                "compare_args": ["flight_id", "cabin"],
            }
        ],
        predicted_calls=[
            {
                "name": "book_flight",
                "arguments": {"flight_id": "F1", "cabin": "business"},
                "execution_error": True,
                "id": "call_1",
                "turn_index": 1,
            },
            {
                "name": "book_flight",
                "arguments": {"flight_id": "F1", "cabin": "business"},
                "id": "call_2",
                "turn_index": 3,
            },
            {"name": "made_up_tool", "arguments": {}, "turn_index": 5},
        ],
        allowed_tool_names={"book_flight"},
        parse_error_count=1,
        parse_error_turns=[7],
        last_trainable_turn_index=7,
        termination_reason="max_steps",
        reward_info={"db_check": {"db_match": False}},
    )
    expected_components = {"tool_name": 1.0, "argument": 0.5, "db": 0.0}
    if field_signals["components"] != expected_components:
        failures.append(f"field components wrong: {field_signals['components']}")
    expected_counts = {
        "golden_actions": 1,
        "predicted_calls": 3,
        "argument_fields": 2,
        "matched_argument_fields": 2,
        "wrong_argument_fields": 1,
        "tool_execution_error": 1,
        "malformed_json": 1,
        "wrong_namespace_tool": 0,
        "nonexistent_tool": 1,
        "repetition": 1,
        "max_steps": 1,
    }
    if field_signals["counts"] != expected_counts:
        failures.append(f"behavior counts wrong: {field_signals['counts']}")

    field_sample = rollout.Sample(
        index=0,
        prompt="t",
        reward=0.0,
        metadata={"tau2_domain": "airline", "tau2_field_reward_signals": field_signals},
    )
    field_reward, _ = reward_postprocess.compute_global_score(
        _ShapingArgs(), field_sample
    )
    clean_signals = {
        **field_signals,
        "counts": {key: 0 for key in field_signals["counts"]},
    }
    clean_sample = rollout.Sample(
        index=1,
        prompt="t",
        reward=0.0,
        metadata={"tau2_domain": "airline", "tau2_field_reward_signals": clean_signals},
    )
    clean_reward, _ = reward_postprocess.compute_global_score(_ShapingArgs(), clean_sample)
    if field_reward != clean_reward:
        failures.append("behavior errors leaked into the positive global score")

    errors_by_turn: dict[int, Counter[str]] = {}
    for error in field_signals["turn_errors"]:
        errors_by_turn.setdefault(error["turn_index"], Counter()).update([error["type"]])
    expected_errors_by_turn = {
        1: Counter({"wrong_argument_field": 1, "tool_execution_error": 1}),
        3: Counter({"repetition": 1}),
        5: Counter({"nonexistent_tool": 1}),
        7: Counter({"malformed_json": 1, "max_steps": 1}),
    }
    if errors_by_turn != expected_errors_by_turn:
        failures.append(f"field errors were attributed to the wrong turns: {errors_by_turn}")

    error_simulation = _make_simulation()
    error_simulation.messages[2].error = True
    collected_error_signals = rollout._field_reward_payload(
        task=SimpleNamespace(
            evaluation_criteria=SimpleNamespace(
                actions=[
                    SimpleNamespace(
                        requestor="assistant",
                        name="get_order",
                        arguments={"order_id": "123"},
                        compare_args=["order_id"],
                    )
                ]
            )
        ),
        simulation=error_simulation,
        allowed_tool_names={"get_order"},
        user_tool_names={"check_device"},
        reward_info={},
    )
    if collected_error_signals["counts"]["tool_execution_error"] != 1:
        failures.append("rollout did not attach the Agent tool execution error")

    # ---- check 8: namespace classification and fixed per-turn penalties ---
    namespace_signals = reward_postprocess.compute_field_reward_signals(
        golden_actions=[],
        predicted_calls=[{"name": "get_order", "arguments": {}, "turn_index": 1}],
        attempted_calls=[
            {"name": "check_device", "arguments": {}, "turn_index": 3},
            {"name": "made_up_tool", "arguments": {}, "turn_index": 5},
        ],
        allowed_tool_names={"get_order"},
        user_tool_names={"check_device"},
        parse_error_count=0,
        termination_reason="too_many_errors",
        reward_info={},
    )
    namespace_counts = namespace_signals["counts"]
    if namespace_counts["wrong_namespace_tool"] != 1 or namespace_counts["nonexistent_tool"] != 1:
        failures.append(f"namespace/nonexistent signals overlap: {namespace_counts}")
    if namespace_signals["wrong_namespace"] != {
        "count": 1,
        "names": {"check_device": 1},
        "affected_turns": [3],
        "termination_reason": "too_many_errors",
        "namespace_attributed_termination": True,
    }:
        failures.append(f"namespace attribution is incomplete: {namespace_signals['wrong_namespace']}")
    if any(error["type"] == "max_steps" for error in namespace_signals["turn_errors"]):
        failures.append("too_many_errors received a duplicate local termination penalty")

    no_namespace_signals = reward_postprocess.compute_field_reward_signals(
        golden_actions=[],
        predicted_calls=[{"name": "check_device", "arguments": {}}],
        allowed_tool_names={"get_order"},
        user_tool_names=set(),
        parse_error_count=0,
        termination_reason="user_stop",
        reward_info={},
    )
    if (
        no_namespace_signals["counts"]["wrong_namespace_tool"] != 0
        or no_namespace_signals["counts"]["nonexistent_tool"] != 1
    ):
        failures.append("domain without User schemas fabricated a namespace signal")

    source_turns = list(range(10, 17))
    penalty_spans = [
        {
            "message_index": message_index,
            "token_start": 100 + (0 if message_index == 1 else message_index),
            "token_end": 100 + message_index + 1,
            "trainable": True,
        }
        for message_index in range(1, 8)
    ]
    # First turn owns two tokens; every other turn owns one.
    penalty_spans[0]["token_start"] = 100
    penalty_sources = {
        message_index: source_turns[message_index - 1]
        for message_index in range(1, 8)
    }
    capped_events = (
        [{"type": "wrong_namespace_tool", "turn_index": 10}] * 3
        + [{"type": "nonexistent_tool", "turn_index": 11}] * 3
        + [{"type": "malformed_json", "turn_index": 12}] * 3
        + [{"type": "wrong_argument_field", "turn_index": 13}] * 5
        + [{"type": "tool_execution_error", "turn_index": 14}] * 3
        + [{"type": "repetition", "turn_index": 15}] * 5
        + [{"type": "max_steps", "turn_index": 16}]
    )
    capped_penalties, capped_details = reward_postprocess.build_token_penalties(
        assistant_spans=penalty_spans,
        assistant_source_indices=penalty_sources,
        response_start=100,
        response_length=8,
        loss_mask=[1] * 8,
        turn_errors=capped_events,
    )
    expected_penalties = [1.2, 1.2, 0.30, 0.30, 0.40, 0.50, 0.20, 0.20]
    if any(abs(a - b) > 1e-9 for a, b in zip(capped_penalties, expected_penalties, strict=True)):
        failures.append(f"turn penalty weights/caps changed: {capped_penalties}")
    if [detail["response_span"] for detail in capped_details] != [
        [0, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8]
    ]:
        failures.append(f"turn penalty response spans changed: {capped_details}")

    try:
        reward_postprocess.build_token_penalties(
            assistant_spans=penalty_spans,
            assistant_source_indices=penalty_sources,
            response_start=100,
            response_length=8,
            loss_mask=[1] * 8,
            turn_errors=[{"type": "malformed_json", "turn_index": 999}],
        )
    except reward_postprocess.TurnCreditAlignmentError:
        pass
    else:
        failures.append("orphan turn error was silently ignored")
    try:
        reward_postprocess.validate_token_penalties(
            [0.0, 0.1],
            response_length=2,
            loss_mask=[1, 0],
        )
    except reward_postprocess.TurnCreditAlignmentError:
        pass
    else:
        failures.append("penalty outside the loss mask was accepted")
    try:
        reward_postprocess.validate_token_penalties(
            [0.0],
            response_length=2,
            loss_mask=[1, 1],
        )
    except reward_postprocess.TurnCreditAlignmentError:
        pass
    else:
        failures.append("misaligned token penalty length was accepted")

    # ---- check 9: turn-aware advantage and DP/CP metadata transport -----
    class _AdvantageArgs:
        advantage_estimator = "grpo"

    from slime.backends.megatron_utils import cp_utils

    original_cp_size = cp_utils.mpu.get_context_parallel_world_size
    original_cp_rank = cp_utils.mpu.get_context_parallel_rank
    try:
        cp_utils.mpu.get_context_parallel_world_size = lambda: 1
        cp_utils.mpu.get_context_parallel_rank = lambda: 0

        clean_data = {
            "metadata": [{"token_penalties": [0.0, 0.0, 0.0]}],
            "rewards": [0.75],
            "kl": [torch.zeros(3)],
            "response_lengths": [3],
            "total_lengths": [5],
            "loss_masks": [torch.tensor([1, 0, 1])],
        }
        expected_clean = torch.ones_like(clean_data["kl"][0], dtype=torch.float32) * 0.75
        reward_postprocess.turn_aware_grpo_advantage(_AdvantageArgs(), clean_data)
        if not torch.equal(clean_data["advantages"][0], expected_clean):
            failures.append("zero-penalty turn-aware advantage changed normal GRPO")
        if clean_data["returns"] is not clean_data["advantages"]:
            failures.append("turn-aware returns are not the advantages")
        if "metadata" in clean_data:
            failures.append("custom advantage did not consume train metadata")

        # Both members have identical global scores and identical namespace
        # faults. Absolute local credit must remain negative instead of being
        # erased by any second group-centering operation.
        all_namespace_data = {
            "metadata": [
                {"token_penalties": [1.0, 1.0, 0.0, 0.0]},
                {"token_penalties": [1.0, 1.0, 0.0, 0.0]},
            ],
            "rewards": [0.0, 0.0],
            "kl": [torch.zeros(4), torch.zeros(4)],
            "response_lengths": [4, 4],
            "total_lengths": [6, 6],
            "loss_masks": [torch.ones(4), torch.ones(4)],
        }
        reward_postprocess.turn_aware_grpo_advantage(
            _AdvantageArgs(), all_namespace_data
        )
        expected_namespace = torch.tensor([-1.0, -1.0, 0.0, 0.0])
        if any(
            not torch.equal(advantage, expected_namespace)
            for advantage in all_namespace_data["advantages"]
        ):
            failures.append(
                "same-group namespace errors lost their absolute negative signal"
            )

        # The existing CP helper determines the exact response-token layout.
        # Exercise both ranks of CP=2, in addition to CP=1 above.
        for cp_rank, expected in (
            (0, torch.tensor([0.6])),
            (1, torch.tensor([0.9, 0.8, 0.7])),
        ):
            cp_utils.mpu.get_context_parallel_world_size = lambda: 2
            cp_utils.mpu.get_context_parallel_rank = lambda rank=cp_rank: rank
            cp_data = {
                "metadata": [{"token_penalties": [0.1, 0.2, 0.3, 0.4]}],
                "rewards": [1.0],
                "kl": [torch.zeros(len(expected))],
                "response_lengths": [4],
                "total_lengths": [8],
                "loss_masks": [torch.ones(4)],
            }
            reward_postprocess.turn_aware_grpo_advantage(_AdvantageArgs(), cp_data)
            if not torch.allclose(cp_data["advantages"][0], expected):
                failures.append(
                    f"CP=2 rank {cp_rank} penalty slice changed: "
                    f"{cp_data['advantages'][0].tolist()}"
                )

        # V2 keeps official outcome and rule credit in separate, explicit
        # scales. Exact full-argument matches are positive critiques, partial
        # matches are neutral, and concrete rule failures are negative.
        v2_details = reward_postprocess.build_turn_credit_v2(
            assistant_spans=[
                {
                    "message_index": 0,
                    "token_start": 0,
                    "token_end": 2,
                    "trainable": True,
                },
                {
                    "message_index": 1,
                    "token_start": 2,
                    "token_end": 3,
                    "trainable": True,
                },
                {
                    "message_index": 2,
                    "token_start": 3,
                    "token_end": 4,
                    "trainable": True,
                },
            ],
            assistant_source_indices={0: 10, 1: 11, 2: 12},
            response_start=0,
            response_length=4,
            loss_mask=[1, 1, 1, 1],
            field_reward_signals={
                "action_details": [
                    {
                        "tool": "exact_tool",
                        "all_arguments": {"x": True, "y": True},
                        "predicted_turn_index": 10,
                    },
                    {
                        "tool": "partial_tool",
                        "all_arguments": {"x": True, "y": False},
                        "predicted_turn_index": 11,
                    },
                ],
                "turn_errors": [
                    {"type": "max_steps", "turn_index": 11},
                    {"type": "wrong_argument_field", "turn_index": 11},
                    {"type": "malformed_json", "turn_index": 12},
                ],
            },
        )
        if [detail["critique"] for detail in v2_details] != [1, 0, -1]:
            failures.append(f"v2 directional critiques are wrong: {v2_details}")
        if reward_postprocess.turn_credit_v2_modifiers(v2_details) != [0.5, 0.0, -0.5]:
            failures.append("v2 did not allocate a fixed zero-sum turn budget")
        if v2_details[1]["rule_errors"]:
            failures.append(
                "v2 treated reference-argument mismatch or max_steps as a rule violation"
            )

        negative_only_details = [
            {"response_span": [0, 1], "critique": 0},
            {"response_span": [1, 2], "critique": -1},
            {"response_span": [2, 3], "critique": 0},
        ]
        if reward_postprocess.turn_credit_v2_modifiers(negative_only_details) != [
            0.5,
            -0.5,
            0.0,
        ]:
            failures.append(
                "v2 negative-only fallback did not use a non-final neutral counterpart"
            )
        positive_only_details = [
            {"response_span": [0, 1], "critique": 1},
            {"response_span": [1, 2], "critique": 0},
        ]
        if any(reward_postprocess.turn_credit_v2_modifiers(positive_only_details)):
            failures.append("v2 fabricated a negative counterpart for a positive-only path")

        neutral_details = [
            {
                "response_span": [0, 2],
                "critique": 0,
            },
            {
                "response_span": [2, 4],
                "critique": 0,
            },
        ]
        v2_a = rollout.Sample(
            index=30,
            prompt="v2",
            tokens=[1, 2, 3, 4],
            response_length=4,
            loss_mask=[1, 1, 1, 1],
            reward=0.0,
            metadata={
                "tau2_turn_credit_version": reward_postprocess.TURN_CREDIT_V2,
                "tau2_turn_credits": v2_details,
            },
            train_metadata={"turn_credit_version": reward_postprocess.TURN_CREDIT_V2},
        )
        v2_b = rollout.Sample(
            index=31,
            prompt="v2",
            tokens=[1, 2, 3, 4],
            response_length=4,
            loss_mask=[1, 1, 1, 1],
            reward=0.0,
            metadata={
                "tau2_turn_credit_version": reward_postprocess.TURN_CREDIT_V2,
                "tau2_turn_credits": neutral_details,
            },
            train_metadata={"turn_credit_version": reward_postprocess.TURN_CREDIT_V2},
        )
        if configured_turn_credit_for_preflight is None:
            os.environ.pop("TAU2_TURN_CREDIT_VERSION", None)
        else:
            os.environ["TAU2_TURN_CREDIT_VERSION"] = configured_turn_credit_for_preflight
        if configured_replace_for_preflight is None:
            os.environ.pop("TAU2_REPLACE_ZERO_SIGNAL_GROUPS", None)
        else:
            os.environ[
                "TAU2_REPLACE_ZERO_SIGNAL_GROUPS"
            ] = configured_replace_for_preflight
        previous_v2_version = os.environ.get("TAU2_TURN_CREDIT_VERSION")
        previous_v2_weight = os.environ.get("TAU2_TURN_CREDIT_REALLOCATION_WEIGHT")
        previous_v2_replace = os.environ.get("TAU2_REPLACE_ZERO_SIGNAL_GROUPS")
        os.environ["TAU2_TURN_CREDIT_VERSION"] = reward_postprocess.TURN_CREDIT_V2
        os.environ["TAU2_TURN_CREDIT_REALLOCATION_WEIGHT"] = "0.5"
        os.environ["TAU2_REPLACE_ZERO_SIGNAL_GROUPS"] = "0"
        try:
            raw_v2, scalar_v2 = reward_postprocess.tau2_reward_post_process(
                _ShapingArgs(), [v2_a, v2_b]
            )
            if raw_v2 != [0.0, 0.0] or scalar_v2 != [0.0, 0.0]:
                failures.append(
                    f"v2 changed the official all-fail outcome channel: {raw_v2}, {scalar_v2}"
                )
            expected_v2 = [0.5, 0.5, 0.0, -1.0]
            if (v2_a.train_metadata or {}).get("token_advantages") != expected_v2:
                failures.append(
                    "v2 response-token advantages did not follow turn reallocation: "
                    f"{v2_a.train_metadata}"
                )
            if abs(v2_a.metadata["tau2_turn_credit_summary"]["modifier_sum"]) > 1e-9:
                failures.append("v2 turn reallocation is not zero-sum")
            if v2_a.metadata["tau2_turn_credit_summary"]["weighted_modifier_abs_sum"] != 0.5:
                failures.append("v2 did not record the fixed reducer-level local budget")
            if not filters.drop_zero_std_or_unsampleable(
                _ShapingArgs(), [v2_a, v2_b]
            ).keep:
                failures.append("filter dropped an all-fail group with v2 turn signal")

            cp_utils.mpu.get_context_parallel_world_size = lambda: 1
            cp_utils.mpu.get_context_parallel_rank = lambda: 0
            v2_reducer = cp_utils.get_sum_of_sample_mean(
                [4], [4], [torch.ones(4)]
            )
            v2_local = torch.tensor(expected_v2)
            if not torch.isclose(v2_reducer(v2_local), torch.tensor(0.0)):
                failures.append("v2 local channel is not zero-sum after the loss reducer")
            if not torch.isclose(v2_reducer(v2_local.abs()), torch.tensor(0.5)):
                failures.append("v2 local channel does not have lambda L1 budget after reduction")

            long_details = [
                {"response_span": [0, 4], "critique": 1},
                {"response_span": [4, 8], "critique": -1},
            ]
            long_neutral_details = [
                {"response_span": [0, 4], "critique": 0},
                {"response_span": [4, 8], "critique": 0},
            ]
            v2_long = rollout.Sample(
                index=34,
                prompt="v2-long",
                tokens=list(range(8)),
                response_length=8,
                loss_mask=[1] * 8,
                reward=0.0,
                metadata={
                    "tau2_turn_credit_version": reward_postprocess.TURN_CREDIT_V2,
                    "tau2_turn_credits": long_details,
                },
                train_metadata={"turn_credit_version": reward_postprocess.TURN_CREDIT_V2},
            )
            v2_long_neutral = rollout.Sample(
                index=35,
                prompt="v2-long",
                tokens=list(range(8)),
                response_length=8,
                loss_mask=[1] * 8,
                reward=0.0,
                metadata={
                    "tau2_turn_credit_version": reward_postprocess.TURN_CREDIT_V2,
                    "tau2_turn_credits": long_neutral_details,
                },
                train_metadata={"turn_credit_version": reward_postprocess.TURN_CREDIT_V2},
            )
            reward_postprocess.tau2_reward_post_process(
                _ShapingArgs(), [v2_long, v2_long_neutral]
            )
            v2_long_local = torch.tensor(v2_long.train_metadata["token_advantages"])
            long_reducer = cp_utils.get_sum_of_sample_mean(
                [8], [8], [torch.ones(8)]
            )
            if not torch.isclose(long_reducer(v2_long_local), torch.tensor(0.0)):
                failures.append("v2 longer local channel is not zero-sum after reduction")
            if not torch.isclose(long_reducer(v2_long_local.abs()), torch.tensor(0.5)):
                failures.append("v2 local budget changed with trajectory or turn length")

            v2_neutral_a = rollout.Sample(
                index=32,
                prompt="v2-zero",
                tokens=[1, 2, 3, 4],
                response_length=4,
                loss_mask=[1, 1, 1, 1],
                reward=0.0,
                metadata={
                    "tau2_turn_credit_version": reward_postprocess.TURN_CREDIT_V2,
                    "tau2_turn_credits": [dict(detail) for detail in neutral_details],
                },
                train_metadata={"turn_credit_version": reward_postprocess.TURN_CREDIT_V2},
            )
            v2_neutral_b = rollout.Sample(
                index=33,
                prompt="v2-zero",
                tokens=[1, 2, 3, 4],
                response_length=4,
                loss_mask=[1, 1, 1, 1],
                reward=0.0,
                metadata={
                    "tau2_turn_credit_version": reward_postprocess.TURN_CREDIT_V2,
                    "tau2_turn_credits": [dict(detail) for detail in neutral_details],
                },
                train_metadata={"turn_credit_version": reward_postprocess.TURN_CREDIT_V2},
            )
            if not filters.drop_zero_std_or_unsampleable(
                _ShapingArgs(), [v2_neutral_a, v2_neutral_b]
            ).keep:
                failures.append("filter changed the prompt mix when zero-signal replacement was disabled")
            if any(
                sample.metadata.get("tau2_turn_credit_group_has_signal")
                for sample in (v2_neutral_a, v2_neutral_b)
            ):
                failures.append("filter marked a neutral v2 group as having advantage signal")
            os.environ["TAU2_REPLACE_ZERO_SIGNAL_GROUPS"] = "1"
            if filters.drop_zero_std_or_unsampleable(
                _ShapingArgs(), [v2_neutral_a, v2_neutral_b]
            ).keep:
                failures.append("filter kept a neutral v2 group when replacement was enabled")
            os.environ["TAU2_REPLACE_ZERO_SIGNAL_GROUPS"] = "0"

            v2_data = {
                "metadata": [v2_a.train_metadata],
                "rewards": [0.0],
                "kl": [torch.zeros(4)],
                "response_lengths": [4],
                "total_lengths": [4],
                "loss_masks": [torch.ones(4)],
            }
            reward_postprocess.turn_aware_grpo_advantage(_AdvantageArgs(), v2_data)
            if not torch.equal(
                v2_data["advantages"][0], torch.tensor(expected_v2)
            ):
                failures.append("custom advantage did not consume the v2 token vector")
        finally:
            if previous_v2_version is None:
                os.environ.pop("TAU2_TURN_CREDIT_VERSION", None)
            else:
                os.environ["TAU2_TURN_CREDIT_VERSION"] = previous_v2_version
            if previous_v2_weight is None:
                os.environ.pop("TAU2_TURN_CREDIT_REALLOCATION_WEIGHT", None)
            else:
                os.environ["TAU2_TURN_CREDIT_REALLOCATION_WEIGHT"] = previous_v2_weight
            if previous_v2_replace is None:
                os.environ.pop("TAU2_REPLACE_ZERO_SIGNAL_GROUPS", None)
            else:
                os.environ["TAU2_REPLACE_ZERO_SIGNAL_GROUPS"] = previous_v2_replace
    finally:
        cp_utils.mpu.get_context_parallel_world_size = original_cp_size
        cp_utils.mpu.get_context_parallel_rank = original_cp_rank

    # Keep this close to the rollout boundary: metadata must be carried as a
    # per-sample Python object, never tensorized or broadcast as one dictionary.
    import slime.ray.rollout as ray_rollout

    original_build_dp_schedule = ray_rollout.build_dp_schedule
    original_ray_put = ray_rollout.ray.put
    try:
        ray_rollout.ray.put = lambda value, **_kwargs: value

        def _fixture_dp_schedule(
            _args,
            config,
            _total_lengths,
            *,
            global_batch_size,
            rollout_indices,
        ):
            del global_batch_size, rollout_indices
            if config["dp_size"] == 1:
                return [[0, 1]], [[[0, 1]]], [1], [2]
            return [[0], [1]], [[[0]], [[0]]], [1], [2]

        ray_rollout.build_dp_schedule = _fixture_dp_schedule
        base_data = {
            "tokens": [[1, 2], [3, 4]],
            "response_lengths": [1, 1],
            "rewards": [0.0, 0.0],
            "truncated": [False, False],
            "loss_masks": [[1], [1]],
            "sample_indices": [10, 11],
            "rollout_ids": [10, 11],
            "rollout_mask_sums": [1, 1],
            "metadata": [
                {"token_penalties": [0.25]},
                {"token_penalties": [0.50]},
            ],
        }
        # ``RolloutManager`` is a normal class in lightweight local test
        # environments, but a Ray ``ActorClass`` after ``@ray.remote`` is
        # active in the real job image.  Exercise the same implementation in
        # both cases without constructing an actor.
        ray_metadata = getattr(ray_rollout.RolloutManager, "__ray_metadata__", None)
        manager_class = getattr(
            ray_metadata,
            "modified_class",
            ray_rollout.RolloutManager,
        )
        split_train_data_by_dp = manager_class._split_train_data_by_dp
        for dp_size in (1, 2):
            manager = SimpleNamespace()
            manager.args = SimpleNamespace(
                global_batch_size=2,
                rollout_data_transport="object-store",
            )
            manager.train_parallel_config = {"dp_size": dp_size}
            per_rank = split_train_data_by_dp(
                manager,
                {key: list(value) for key, value in base_data.items()}
            )
            observed = [
                metadata
                for box in per_rank
                for metadata in box.inner["metadata"]
            ]
            if observed != base_data["metadata"]:
                failures.append(
                    f"DP={dp_size} changed turn-credit metadata: {observed}"
                )
    finally:
        ray_rollout.build_dp_schedule = original_build_dp_schedule
        ray_rollout.ray.put = original_ray_put

    # ---- check 10: exact domain quota, same-domain refill, reset, resume -
    for raw_quota, expected in (
        ("telecom:3,airline:2,retail:1", {"telecom": 3, "airline": 2, "retail": 1}),
        ("telecom:3,airline:1,retail:2", {"telecom": 3, "airline": 1, "retail": 2}),
        ("telecom:2,airline:2,retail:2", {"telecom": 2, "airline": 2, "retail": 2}),
        ("airline:6,retail:0,telecom:0", {"telecom": 0, "airline": 6, "retail": 0}),
        ("airline:0,retail:6,telecom:0", {"telecom": 0, "airline": 0, "retail": 6}),
        ("airline:0,retail:0,telecom:6", {"telecom": 6, "airline": 0, "retail": 0}),
    ):
        if filters.parse_domain_quota(raw_quota) != expected:
            failures.append(f"fixed quota parser changed for {raw_quota}")

    quota = {"telecom": 3, "airline": 2, "retail": 1}
    quota_source = _quota_source(quota)
    quota_source.begin_rollout(7)
    initial_groups = quota_source.get_samples(6)
    initial_domains = [group[0].metadata["domain"] for group in initial_groups]
    if Counter(initial_domains) != Counter(quota):
        failures.append(f"initial domain quota changed: {initial_domains}")

    initial_groups[0][0].metadata["tau2_permanently_too_long"] = True
    rejected_domain = initial_groups[0][0].metadata["domain"]
    for group_index, group in enumerate(initial_groups):
        for sample_index, group_sample in enumerate(group):
            group_sample.reward = float(sample_index)
        outcome = filters.drop_zero_std_or_unsampleable(
            _Args(),
            group,
            data_source=quota_source,
            rollout_id=7,
        )
        if group_index == 0 and outcome.keep:
            failures.append("quota filter kept an over-cap group")
        if group_index > 0 and not outcome.keep:
            failures.append("quota filter dropped an informative group")

    replacement_groups = quota_source.get_samples(6)
    if len(replacement_groups) != 1:
        failures.append(f"quota drew {len(replacement_groups)} replacements instead of one")
    else:
        replacement = replacement_groups[0]
        if replacement[0].metadata["domain"] != rejected_domain:
            failures.append("quota replacement changed domains")
        if not replacement[0].metadata["tau2_domain_quota_replacement"]:
            failures.append("quota replacement provenance is missing")
        replacement[0].reward = 0.0
        replacement[1].reward = 1.0
        if not filters.drop_zero_std_or_unsampleable(
            _Args(),
            replacement,
            data_source=quota_source,
            rollout_id=7,
        ).keep:
            failures.append("informative same-domain replacement was dropped")
    quota_metrics = quota_source.finish_rollout(7)
    for domain, count in quota.items():
        if quota_metrics[f"rollout/domain_quota/{domain}/accepted"] != float(count):
            failures.append(f"accepted quota metric changed for {domain}")
    if quota_metrics[f"rollout/domain_quota/{rejected_domain}/rejected"] != 1.0:
        failures.append("rejected quota metric was not reported")

    # A completed round can reset, and restoring its state yields the same next
    # per-domain prompt stream as uninterrupted sampling.
    resume_state = quota_source.checkpoint_state()
    quota_source.begin_rollout(8)
    uninterrupted = quota_source.get_samples(6)
    uninterrupted_prompts = [group[0].prompt for group in uninterrupted]
    for group in uninterrupted:
        quota_source.record_filter_result(group[0].metadata["domain"], keep=True)
    quota_source.finish_rollout(8)

    resumed_source = _quota_source(quota)
    resumed_source.restore_checkpoint_state(resume_state)
    resumed_source.begin_rollout(8)
    resumed = resumed_source.get_samples(6)
    if [group[0].prompt for group in resumed] != uninterrupted_prompts:
        failures.append("domain quota resume changed the sampling stream")
    for group in resumed:
        resumed_source.record_filter_result(group[0].metadata["domain"], keep=True)
    resumed_source.finish_rollout(8)

    # A domain expert changes only the quota after restoring mixed iter99. The
    # dataset fingerprint, epoch, and all three cursors remain the saved state;
    # drawing then advances only the selected domain.
    expert_source = _quota_source({"telecom": 0, "airline": 6, "retail": 0})
    expert_source.restore_checkpoint_state(resume_state)
    if expert_source.checkpoint_state() != resume_state:
        failures.append("mixed-to-expert restore reset a domain cursor or epoch")
    expert_source.begin_rollout(8)
    expert_groups = expert_source.get_samples(6)
    if {group[0].metadata["domain"] for group in expert_groups} != {"airline"}:
        failures.append("airline expert quota drew a non-airline group")
    for group in expert_groups:
        expert_source.record_filter_result("airline", keep=True)
    expert_source.finish_rollout(8)
    expert_state = expert_source.checkpoint_state()
    for inactive_domain in ("telecom", "retail"):
        if (
            expert_state["domain_offsets"][inactive_domain]
            != resume_state["domain_offsets"][inactive_domain]
            or expert_state["domain_epochs"][inactive_domain]
            != resume_state["domain_epochs"][inactive_domain]
        ):
            failures.append(f"expert sampling advanced inactive {inactive_domain} cursor")
    if active_profile == PROTOCOL_STRICT_SINGLE_V1:
        if "dataset_fingerprint" in resume_state:
            failures.append("strict-single checkpoint metadata emitted a dataset hash")
    else:
        wrong_fingerprint = dict(resume_state)
        wrong_fingerprint["dataset_fingerprint"] = "wrong"
        try:
            expert_source.restore_checkpoint_state(wrong_fingerprint)
        except ValueError:
            pass
        else:
            failures.append("domain quota resume accepted a wrong dataset fingerprint")

    # ---- report ---------------------------------------------------------
    if warnings:
        print("WARN (informational, non-blocking):")
        for w in warnings:
            print("  -", w)
    if failures:
        print("FAIL:")
        for f in failures:
            print("  -", f)
        return 1
    print(
        f"OK: full-conversation tokens ({len(train_ids)} tok), "
        f"mask sum={sum(train_mask)}, response_length={sample.response_length}, "
        f"cap={cap}; over-long path flags+neutralizes; filter drops shaped-zero-std "
        f"and unsampleable groups; turn penalties, DP/CP metadata, and exact domain "
        f"quotas verified."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
