"""Local sglang-backed tau2 agent registration.

Tau2's official requirement is the `HalfDuplexAgent` interface. This agent keeps
that interface and uses a small adapter internally: it calls sglang's raw
`/generate` endpoint, parses Qwen native tool-call text, and returns tau2
`AssistantMessage` objects for the official orchestrator to execute.
"""

from __future__ import annotations

import json
import re
import uuid
from typing import Any

import httpx
from tau2.agent.base_agent import ValidAgentInputMessage
from tau2.agent.llm_agent import LLMAgent
from tau2.data_model.message import (
    AssistantMessage,
    Message,
    MultiToolMessage,
    SystemMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from tau2.registry import registry
from transformers import AutoTokenizer


AGENT_NAME = "slime_sglang_agent"
_TOOL_CALL_START = "<tool_call>"
_TOOL_CALL_END = "</tool_call>"
_TOOL_CALL_RE = re.compile(
    re.escape(_TOOL_CALL_START) + r"(?P<body>.*?)" + re.escape(_TOOL_CALL_END),
    re.S,
)
_FUNC_RE = re.compile(r"<function=(?P<name>[^>\s]+)>(?P<body>.*?)</function>", re.S)
_PARAM_RE = re.compile(r"<parameter=(?P<key>[^>\s]+)>(?P<val>.*?)</parameter>", re.S)


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
    return text.strip()


def _tool_schema_text(tools) -> str:
    return json.dumps(
        [tool.openai_schema for tool in tools],
        indent=2,
        ensure_ascii=False,
    )


def _render_tool_call(tool_call: ToolCall) -> str:
    content = json.dumps(
        {"name": tool_call.name, "arguments": tool_call.arguments},
        ensure_ascii=False,
    )
    return f"{_TOOL_CALL_START}{content}{_TOOL_CALL_END}"


class SlimeSGLangAgent(LLMAgent):
    """Tau2 text agent for slime-hosted raw sglang generation."""

    def __init__(self, tools, domain_policy: str, llm: str, llm_args: dict | None = None):
        super().__init__(
            tools=tools,
            domain_policy=domain_policy,
            llm=llm,
            llm_args=llm_args,
        )
        self.sglang_url = (self.llm_args.get("api_base") or "").rstrip("/")
        if self.sglang_url.endswith("/v1"):
            self.sglang_url = self.sglang_url[: -len("/v1")] + "/generate"
        if not self.sglang_url.endswith("/generate"):
            self.sglang_url = f"{self.sglang_url}/generate"
        self.tokenizer = AutoTokenizer.from_pretrained(llm, trust_remote_code=True)
        self.timeout = float(self.llm_args.get("timeout", 300.0))

    @property
    def system_prompt(self) -> str:
        return (
            "You are a customer service agent. Complete the user's task while "
            "following the policy exactly.\n\n"
            "In each turn, choose exactly one action:\n"
            "- Send one plain-text message to the user, or\n"
            "- Make one or more tool calls in this exact format:\n"
            '<tool_call>{"name": "tool_name", "arguments": {"param": "value"}}</tool_call>\n\n'
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
            return {"role": "user", "content": message.content or ""}
        if isinstance(message, ToolMessage):
            return {"role": "user", "content": f"Tool result:\n{message.content or ''}"}
        if isinstance(message, AssistantMessage):
            if message.is_tool_call():
                return {
                    "role": "assistant",
                    "content": "\n\n".join(
                        _render_tool_call(tool_call) for tool_call in message.tool_calls
                    ),
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
        extra_body = self.llm_args.get("extra_body") or {}
        params = {
            "temperature": self.llm_args.get("temperature", 0.6),
            "top_p": self.llm_args.get("top_p", 1.0),
            "max_new_tokens": self.llm_args.get("max_tokens", 1200),
            "presence_penalty": 0.0,
        }
        if self.tokenizer.eos_token:
            params["stop"] = [self.tokenizer.eos_token]
        for key in (
            "top_k",
            "presence_penalty",
            "frequency_penalty",
            "repetition_penalty",
        ):
            if key in self.llm_args:
                params[key] = self.llm_args[key]
            if isinstance(extra_body, dict) and key in extra_body:
                params[key] = extra_body[key]
        return params

    def _generate_raw(self, prompt: str) -> str:
        with httpx.Client(timeout=httpx.Timeout(self.timeout)) as client:
            response = client.post(
                self.sglang_url,
                json={"text": prompt, "sampling_params": self._sampling_params()},
            )
            response.raise_for_status()
            return (response.json().get("text") or "").strip()

    def _assistant_from_text(self, text: str) -> AssistantMessage:
        content = _strip_thinking(text)
        matches = list(_TOOL_CALL_RE.finditer(content))
        if matches:
            non_tool_content = _TOOL_CALL_RE.sub("", content).strip()
            if not non_tool_content:
                try:
                    tool_calls = []
                    for match in matches:
                        name, arguments = _parse_tool_call_content(
                            match.group("body").strip()
                        )
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
                        raw_data={"text": text},
                    )
                except Exception:
                    pass

        if not content:
            content = "I need a bit more information to continue. Could you clarify your request?"
        return AssistantMessage(role="assistant", content=content, raw_data={"text": text})

    def _generate_next_message(
        self, message: ValidAgentInputMessage, state
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
        return self._assistant_from_text(self._generate_raw(prompt))


def create_slime_sglang_agent(tools, domain_policy: str, **kwargs: Any) -> SlimeSGLangAgent:
    return SlimeSGLangAgent(
        tools=tools,
        domain_policy=domain_policy,
        llm=kwargs["llm"],
        llm_args=kwargs.get("llm_args"),
    )


def register_slime_sglang_agent() -> None:
    if registry.get_agent_factory(AGENT_NAME) is None:
        registry.register_agent_factory(create_slime_sglang_agent, AGENT_NAME)
