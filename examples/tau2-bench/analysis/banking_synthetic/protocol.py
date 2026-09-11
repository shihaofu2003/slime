"""Shared classification for autonomous rollout protocol failures."""

from __future__ import annotations

from typing import Any


VALID_STOPS = {"agent_stop", "user_stop"}


def protocol_failure_reason(simulation: dict[str, Any]) -> str | None:
    termination = simulation.get("termination_reason")
    if termination not in VALID_STOPS:
        return str(termination or "missing_termination")
    assistant_messages = [
        message
        for message in simulation.get("messages") or []
        if message.get("role") == "assistant"
    ]
    if not assistant_messages:
        return "no_assistant_message"
    finish_reasons = {
        choice.get("finish_reason")
        for message in assistant_messages
        for choice in ((message.get("raw_data") or {}).get("choices") or [])
    }
    if "length" in finish_reasons:
        return "output_length_exhausted"
    if not str(assistant_messages[-1].get("content") or "").strip():
        return "no_final_message"
    return None
