"""Shared request metadata for the official tau2 SGLang adapters."""

from __future__ import annotations

from typing import Any


def build_sglang_sampling_params(
    llm_args: dict[str, Any],
    *,
    eos_token: str | None,
) -> dict[str, Any]:
    extra_body = llm_args.get("extra_body") or {}
    params = {
        "temperature": llm_args.get("temperature", 0.6),
        "top_p": llm_args.get("top_p", 1.0),
        "max_new_tokens": llm_args.get("max_tokens", 1200),
        "presence_penalty": 0.0,
    }
    if eos_token:
        params["stop"] = [eos_token]
    for key in (
        "top_k",
        "presence_penalty",
        "frequency_penalty",
        "repetition_penalty",
    ):
        if key in llm_args:
            params[key] = llm_args[key]
        if isinstance(extra_body, dict) and key in extra_body:
            params[key] = extra_body[key]
    if "seed" in llm_args:
        params["sampling_seed"] = llm_args["seed"]
    return params


def with_request_timing(
    raw_data: dict[str, Any] | None,
    *,
    participant: str,
    elapsed_seconds: float,
) -> dict[str, Any]:
    result = dict(raw_data or {})
    result["tau2_eval_timing"] = {
        "participant": participant,
        "elapsed_seconds": elapsed_seconds,
    }
    return result
