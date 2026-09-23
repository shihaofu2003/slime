"""Trainable sglang-backed tau2 agent for slime RL rollouts."""

from __future__ import annotations

import json
import logging
import os
import re
import sys
import time
import uuid
from functools import lru_cache

from slime.rollout.agent_tokens import recorded_turn
from pathlib import Path
from typing import Any

import httpx
from tau2.agent.base_agent import ValidAgentInputMessage
from tau2.agent.llm_agent import LLMAgent
from tau2.data_model.message import (
    AssistantMessage,
    Message,
    MultiToolMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from transformers import AutoTokenizer

class RolloutContextOverflow(ValueError):
    """A trajectory exceeds the inference or training context window."""


SHARED_DIR = Path(__file__).resolve().parents[1] / "shared"
if str(SHARED_DIR) not in sys.path:
    sys.path.insert(0, str(SHARED_DIR))

from protocol_profiles import (  # noqa: E402
    PROTOCOL_AGENT_OWNED_DEPENDENCY_SAFE_MULTI,
    PROTOCOL_CURRENT_SINGLE,
    PROTOCOL_OFFICIAL_NATIVE,
    PROTOCOL_STRICT_SINGLE_V1,
    domain_policy_for_profile,
    protocol_block_for_profile,
    protocol_signature,
    strict_single_system_prompt,
)
from agent_contract import (  # noqa: E402
    OFFICIAL_AGENT_VIEW,
    AgentContract,
    native_agent_message_to_chat,
    openai_tool_schemas,
)


_TOOL_CALL_START = "<tool_call>"
_TOOL_CALL_END = "</tool_call>"
_TOOL_CALL_RE = re.compile(
    re.escape(_TOOL_CALL_START) + r"(?P<body>.*?)" + re.escape(_TOOL_CALL_END),
    re.S,
)
_FUNC_RE = re.compile(r"<function=(?P<name>[^>\s]+)>(?P<body>.*?)</function>", re.S)
_PARAM_RE = re.compile(r"<parameter=(?P<key>[^>\s]+)>(?P<val>.*?)</parameter>", re.S)
_CHAT_SPECIAL_TOKENS = ("<|im_end|>", "<|endoftext|>")


def _coerce_param_value(value: str) -> Any:
    value = value.strip()
    try:
        return json.loads(value)
    except Exception:
        return value


def _parse_function_format(content: str) -> tuple[str, dict[str, Any]] | None:
    match = _FUNC_RE.search(content)
    if not match:
        return None
    args: dict[str, Any] = {}
    for param in _PARAM_RE.finditer(match.group("body")):
        args[param.group("key").strip()] = _coerce_param_value(param.group("val"))
    return match.group("name").strip(), args


def _parse_tool_call_content(content: str) -> tuple[str, dict[str, Any]]:
    try:
        parsed = json.loads(content)
    except Exception:
        parsed = None

    if isinstance(parsed, dict):
        name = parsed.get("name")
        args = parsed.get("arguments") or {}
        if isinstance(args, str):
            args = json.loads(args) if args.strip() else {}
        if isinstance(name, str) and name and isinstance(args, dict):
            return name, args

    parsed_function = _parse_function_format(content)
    if parsed_function and parsed_function[0]:
        return parsed_function

    raise ValueError("tool_call content is not a supported Qwen tool-call format")


def _strip_thinking(text: str) -> str:
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    if "<think>" in text and "</think>" not in text:
        return ""
    text = text.strip()
    changed = True
    while changed:
        changed = False
        for token in _CHAT_SPECIAL_TOKENS:
            if text.endswith(token):
                text = text[: -len(token)].strip()
                changed = True
    return text.strip()


def _tool_schema_text(tools) -> str:
    return json.dumps(
        [tool.openai_schema for tool in tools],
        indent=2,
        ensure_ascii=False,
    )


def render_tool_call(tool_call: ToolCall) -> str:
    content = json.dumps(
        {"name": tool_call.name, "arguments": tool_call.arguments},
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return f"{_TOOL_CALL_START}{content}{_TOOL_CALL_END}"


def _with_generate_endpoint(api_base: str) -> str:
    api_base = api_base.rstrip("/")
    if api_base.endswith("/v1"):
        return api_base[: -len("/v1")] + "/generate"
    if api_base.endswith("/generate"):
        return api_base
    return f"{api_base}/generate"


@lru_cache(maxsize=2)
def shared_tokenizer(model):
    return AutoTokenizer.from_pretrained(model, trust_remote_code=True)


@lru_cache(maxsize=4)
def shared_client(timeout):
    capacity = int(os.environ.get("TAU2_AGENT_CONCURRENCY", "32"))
    return httpx.Client(timeout=httpx.Timeout(timeout),
                        limits=httpx.Limits(max_connections=capacity, max_keepalive_connections=capacity))


class TrainableSGLangAgent(LLMAgent):
    """Half-duplex tau2 agent that calls slime's sglang router."""

    def __init__(
        self,
        tools,
        domain_policy: str,
        *,
        llm: str,
        llm_args: dict | None = None,
        domain: str | None = None,
        user_tools=(),
    ):
        llm_args = dict(llm_args or {})
        self.protocol_profile = str(
            llm_args.get("protocol_profile") or PROTOCOL_CURRENT_SINGLE
        )
        self.protocol_signature = protocol_signature(self.protocol_profile)
        supplied_signature = llm_args.get("protocol_signature")
        if supplied_signature and supplied_signature != self.protocol_signature:
            raise ValueError(
                "protocol_signature does not match the selected protocol profile"
            )
        if self.protocol_signature is not None:
            llm_args["protocol_signature"] = self.protocol_signature
        self.tokenizer = shared_tokenizer(llm)
        self.contract: AgentContract | None = None
        self.native_tools: list[dict[str, Any]] | None = None
        if self.protocol_profile == PROTOCOL_AGENT_OWNED_DEPENDENCY_SAFE_MULTI:
            if not domain:
                raise ValueError("agent-owned protocol requires the tau2 domain")
            self.contract = AgentContract.create(
                domain=domain,
                domain_policy=domain_policy,
                agent_tools=tools,
                user_tools=user_tools,
                chat_template=self.tokenizer.chat_template,
                profile=self.protocol_profile,
            )
            supplied_contract_signature = llm_args.get("agent_contract_signature")
            if (
                supplied_contract_signature
                and supplied_contract_signature != self.contract.agent_contract_signature
            ):
                raise ValueError(
                    "agent_contract_signature does not match the resolved contract"
                )
            llm_args["agent_contract_signature"] = (
                self.contract.agent_contract_signature
            )
            domain_policy = self.contract.policy
            self.single_call_clause_replaced = True
        elif self.protocol_profile in {
            PROTOCOL_OFFICIAL_NATIVE,
            PROTOCOL_STRICT_SINGLE_V1,
        }:
            domain_policy, self.single_call_clause_replaced = domain_policy_for_profile(
                domain_policy,
                self.protocol_profile,
            )
            self.native_tools = openai_tool_schemas(tools)
        else:
            domain_policy, self.single_call_clause_replaced = domain_policy_for_profile(
                domain_policy,
                self.protocol_profile,
            )
        super().__init__(
            tools=tools,
            domain_policy=domain_policy,
            llm=llm,
            llm_args=llm_args,
        )
        self.sglang_url = _with_generate_endpoint(self.llm_args["api_base"])
        self.timeout = float(self.llm_args.get("timeout", os.environ.get("TAU2_AGENT_TIMEOUT", 600.0)))

    @property
    def system_prompt(self) -> str:
        if self.contract is not None:
            return self.contract.system_prompt
        if self.protocol_profile == PROTOCOL_OFFICIAL_NATIVE:
            return super().system_prompt
        if self.protocol_profile == PROTOCOL_STRICT_SINGLE_V1:
            return strict_single_system_prompt(self.domain_policy)
        protocol = protocol_block_for_profile(self.protocol_profile)
        return (
            "You are a customer service agent. Complete the user's task while "
            "following the policy exactly.\n\n"
            "In each turn, choose exactly one action:\n"
            "- Send one plain-text message to the user, or\n"
            "- Make one or more tool calls in this exact format:\n"
            '<tool_call>{"name":"tool_name","arguments":{"param":"value"}}</tool_call>\n\n'
            "If you make multiple tool calls, emit one <tool_call> block for each call "
            "and no other content. Do not send both text and a tool call in the same turn."
            f"{protocol}\n\n"
            "<policy>\n"
            f"{self.domain_policy}\n"
            "</policy>\n\n"
            "<tools>\n"
            f"{_tool_schema_text(self.tools)}\n"
            "</tools>"
        )

    def _message_to_chat(self, message: Message) -> dict[str, Any] | None:
        if self.contract is not None:
            return self.contract.message_to_chat(
                message,
                validate_assistant_tools=False,
            )
        if self.native_tools is not None:
            return native_agent_message_to_chat(
                message,
                validate_assistant_tools=False,
            )
        if isinstance(message, UserMessage):
            if message.is_tool_call():
                return None
            return {"role": "user", "content": message.content or ""}
        if isinstance(message, ToolMessage):
            if message.requestor != "assistant":
                return None
            return {"role": "user", "content": f"Tool result:\n{message.content or ''}"}
        if isinstance(message, AssistantMessage):
            if message.is_tool_call():
                return {
                    "role": "assistant",
                    "content": "\n\n".join(render_tool_call(tool_call) for tool_call in message.tool_calls),
                }
            return {"role": "assistant", "content": message.content or ""}
        return None

    def _chat_messages(self, state) -> list[dict[str, Any]]:
        messages = [{"role": "system", "content": self.system_prompt}]
        for message in state.messages:
            chat_message = self._message_to_chat(message)
            if chat_message is not None:
                messages.append(chat_message)
        return messages

    def _sampling_params(self) -> dict[str, Any]:
        params = dict(self.llm_args.get("sampling_params") or {})
        params.setdefault("temperature", self.llm_args.get("temperature", 0.6))
        params.setdefault("top_p", self.llm_args.get("top_p", 1.0))
        params.setdefault("max_new_tokens", self.llm_args.get("max_tokens", 1200))
        params.setdefault("presence_penalty", 0.0)
        if self.tokenizer.eos_token:
            stop = params.get("stop") or []
            if isinstance(stop, str):
                stop = [stop]
            params["stop"] = list(dict.fromkeys([*stop, self.tokenizer.eos_token]))
        return params

    def _generate_raw(self, prompt: str | list[int]) -> tuple[str, dict[str, Any]]:
        started = time.perf_counter()
        raw_tokens = os.environ.get("TAU2_RAW_TOKENS", "0") == "1"
        request = {"sampling_params": self._sampling_params()}
        if raw_tokens:
            from continuous import AGENT_TURNS

            cap = getattr(self, "max_train_tokens", 0)
            if cap and len(prompt) >= cap:
                # Even one output token would exceed the training cap. Let
                # rollout resample this attempt without spending more GPU time.
                raise RolloutContextOverflow(
                    f"Agent prompt has {len(prompt)} tokens; training cap is {cap}"
                )
            request.update(input_ids=prompt, return_logprob=True, logprob_start_len=-1)
            with AGENT_TURNS.request() as version:
                try:
                    response = shared_client(self.timeout).post(self.sglang_url, json=request)
                except (httpx.ReadError, httpx.ReadTimeout):
                    # No response was accepted or executed in the environment.
                    # Keep the turn/version gate held while retrying once on a
                    # fresh connection; do not close other threads' shared pool.
                    logging.getLogger(__name__).warning(
                        "tau2_agent_read_retry version=%s: retrying once on a fresh connection", version,
                        exc_info=True,
                    )
                    with httpx.Client(timeout=httpx.Timeout(self.timeout)) as retry_client:
                        response = retry_client.post(self.sglang_url, json=request)
                if response.status_code == 400 and (
                    "maximum context length" in response.text
                    or "is longer than the model's context length" in response.text
                ):
                    raise RolloutContextOverflow(response.text)
                response.raise_for_status()
                payload = response.json()
                turn = recorded_turn(prompt, payload.get("meta_info") or {}, version)
        else:
            request["text"] = prompt
            response = shared_client(self.timeout).post(self.sglang_url, json=request)
            response.raise_for_status()
            payload = response.json()
        meta_info = dict(payload.get("meta_info") or {})
        if raw_tokens:
            meta_info["behavior_turn"] = turn
        meta_info["generation_time_seconds"] = time.perf_counter() - started
        return (payload.get("text") or "").strip(), meta_info

    def _assistant_from_text(self, text: str, meta_info: dict[str, Any]) -> AssistantMessage:
        content = _strip_thinking(text)
        matches = list(_TOOL_CALL_RE.finditer(content))
        tool_parse_error_count = 0
        multi_tool_attempt_count = (
            len(matches)
            if self.protocol_profile == PROTOCOL_STRICT_SINGLE_V1 and len(matches) > 1
            else 0
        )
        if matches:
            non_tool_content = _TOOL_CALL_RE.sub("", content).strip()
            if multi_tool_attempt_count:
                tool_parse_error_count = len(matches)
            elif not non_tool_content:
                tool_calls = []
                for match in matches:
                    try:
                        name, arguments = _parse_tool_call_content(match.group("body").strip())
                        tool_calls.append(
                            ToolCall(
                                id=f"call_{uuid.uuid4().hex[:12]}",
                                name=name,
                                arguments=arguments,
                            )
                        )
                    except Exception:
                        tool_parse_error_count += 1
                if not tool_parse_error_count:
                    return AssistantMessage(
                        role="assistant",
                        tool_calls=tool_calls,
                        raw_data=self._raw_data(text, meta_info),
                        generation_time_seconds=meta_info.get("generation_time_seconds"),
                    )
            else:
                # Tool calls mixed with prose violate the half-duplex output
                # contract even if each JSON block itself is valid.
                tool_parse_error_count = len(matches)
        elif "<tool_call" in content or "</tool_call>" in content:
            # Includes unclosed tags and malformed blocks that the strict
            # regular expression intentionally did not parse.
            starts = len(re.findall(r"<tool_call(?:\s[^>]*)?>", content, re.IGNORECASE))
            ends = content.lower().count("</tool_call>")
            tool_parse_error_count = max(1, starts, ends)

        if not content:
            content = "I need a bit more information to continue. Could you clarify your request?"
        return AssistantMessage(
            role="assistant",
            content=content,
            raw_data=self._raw_data(
                text,
                meta_info,
                tool_parse_error_count=tool_parse_error_count,
                multi_tool_attempt_count=multi_tool_attempt_count,
            ),
            generation_time_seconds=meta_info.get("generation_time_seconds"),
        )

    def _raw_data(
        self,
        text: str,
        meta_info: dict[str, Any],
        *,
        tool_parse_error_count: int = 0,
        multi_tool_attempt_count: int = 0,
    ) -> dict[str, Any]:
        result: dict[str, Any] = {
            "tau2_rl_agent": True,
            "tau2_agent_protocol_profile": self.protocol_profile,
            "text": text,
            "meta_info": meta_info,
        }
        if self.protocol_signature is not None:
            result["tau2_agent_protocol_signature"] = self.protocol_signature
        if tool_parse_error_count:
            result["tau2_tool_parse_error"] = True
            result["tau2_tool_parse_error_count"] = int(tool_parse_error_count)
        if multi_tool_attempt_count:
            result["tau2_single_call_protocol_error"] = True
            result["tau2_multi_tool_attempt_count"] = int(multi_tool_attempt_count)
        if self.contract is not None:
            result.update(self.contract.metadata(view=OFFICIAL_AGENT_VIEW))
        return result

    def _generate_next_message(
        self,
        message: ValidAgentInputMessage,
        state,
    ) -> AssistantMessage:
        if isinstance(message, UserMessage) and message.is_audio:
            raise ValueError("User message cannot be audio.")
        if isinstance(message, MultiToolMessage):
            state.messages.extend(message.tool_messages)
        else:
            state.messages.append(message)

        prompt = self.tokenizer.apply_chat_template(
            self._chat_messages(state),
            tokenize=os.environ.get("TAU2_RAW_TOKENS", "0") == "1",
            return_dict=False,
            add_generation_prompt=True,
            tools=(
                self.contract.tools
                if self.contract is not None
                else self.native_tools
            ),
        )
        text, meta_info = self._generate_raw(prompt)
        return self._assistant_from_text(text, meta_info)
