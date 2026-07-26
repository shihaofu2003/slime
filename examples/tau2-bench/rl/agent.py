"""Trainable sglang-backed tau2 agent for slime RL rollouts."""

from __future__ import annotations

import json
import os
import re
import time
import uuid
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


class TrainableSGLangAgent(LLMAgent):
    """Half-duplex tau2 agent that calls slime's sglang router."""

    def __init__(
        self,
        tools,
        domain_policy: str,
        *,
        llm: str,
        llm_args: dict | None = None,
    ):
        super().__init__(tools=tools, domain_policy=domain_policy, llm=llm, llm_args=llm_args)
        self.sglang_url = _with_generate_endpoint(self.llm_args["api_base"])
        self.tokenizer = AutoTokenizer.from_pretrained(llm, trust_remote_code=True)
        self.timeout = float(self.llm_args.get("timeout", os.environ.get("TAU2_AGENT_TIMEOUT", 600.0)))

    @property
    def system_prompt(self) -> str:
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
            f"{self.domain_policy}\n"
            "</policy>\n\n"
            "<tools>\n"
            f"{_tool_schema_text(self.tools)}\n"
            "</tools>"
        )

    def _message_to_chat(self, message: Message) -> dict[str, str] | None:
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

    def _chat_messages(self, state) -> list[dict[str, str]]:
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

    def _generate_raw(self, prompt: str) -> tuple[str, dict[str, Any]]:
        started = time.perf_counter()
        with httpx.Client(timeout=httpx.Timeout(self.timeout)) as client:
            response = client.post(
                self.sglang_url,
                json={"text": prompt, "sampling_params": self._sampling_params()},
            )
            response.raise_for_status()
            payload = response.json()
        meta_info = dict(payload.get("meta_info") or {})
        meta_info["generation_time_seconds"] = time.perf_counter() - started
        return (payload.get("text") or "").strip(), meta_info

    def _assistant_from_text(self, text: str, meta_info: dict[str, Any]) -> AssistantMessage:
        content = _strip_thinking(text)
        matches = list(_TOOL_CALL_RE.finditer(content))
        if matches:
            non_tool_content = _TOOL_CALL_RE.sub("", content).strip()
            if not non_tool_content:
                try:
                    tool_calls = []
                    for match in matches:
                        name, arguments = _parse_tool_call_content(match.group("body").strip())
                        tool_calls.append(
                            ToolCall(
                                id=f"call_{uuid.uuid4().hex[:12]}",
                                name=name,
                                arguments=arguments,
                            )
                        )
                    return AssistantMessage(
                        role="assistant",
                        tool_calls=tool_calls,
                        raw_data={
                            "tau2_rl_agent": True,
                            "text": text,
                            "meta_info": meta_info,
                        },
                        generation_time_seconds=meta_info.get("generation_time_seconds"),
                    )
                except Exception:
                    pass

        if not content:
            content = "I need a bit more information to continue. Could you clarify your request?"
        return AssistantMessage(
            role="assistant",
            content=content,
            raw_data={
                "tau2_rl_agent": True,
                "text": text,
                "meta_info": meta_info,
            },
            generation_time_seconds=meta_info.get("generation_time_seconds"),
        )

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
            tokenize=False,
            add_generation_prompt=True,
        )
        text, meta_info = self._generate_raw(prompt)
        return self._assistant_from_text(text, meta_info)
