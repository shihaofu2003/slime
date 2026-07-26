#!/usr/bin/env python3
"""Run tau2-bench eval through tau2's official runner.

This replaces the legacy direct-AgentGymEnv eval with the official
HalfDuplexAgent/TextRunConfig/run_domain path. A local sglang policy server is
addressed through raw `/generate`; the user simulator remains the official
LiteLLM-backed tau2 user.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

DEFAULT_DOMAINS = ("airline", "retail", "telecom")
AGENT_NAME = "slime_sglang_agent"


def _parse_csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_json_object(value: str | None, *, name: str) -> dict[str, Any]:
    if not value:
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"{name} must be valid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise SystemExit(f"{name} must be a JSON object")
    return parsed


def _with_openai_prefix(model: str) -> str:
    if "/" in model:
        return model
    return f"openai/{model}"


def _merge_litellm_args(
    *,
    json_args: str | None,
    api_base: str | None,
    api_key: str | None,
    temperature: float | None,
    top_p: float | None,
    max_tokens: int | None,
    extra_body_json: str | None,
    name: str,
) -> dict[str, Any]:
    args = _parse_json_object(json_args, name=f"{name} args")
    if api_base and "api_base" not in args:
        args["api_base"] = api_base
    if api_key and "api_key" not in args:
        args["api_key"] = api_key
    if temperature is not None and "temperature" not in args:
        args["temperature"] = temperature
    if top_p is not None and "top_p" not in args:
        args["top_p"] = top_p
    if max_tokens is not None and "max_tokens" not in args:
        args["max_tokens"] = max_tokens

    extra_body = _parse_json_object(extra_body_json, name=f"{name} extra body")
    if extra_body:
        merged = dict(args.get("extra_body") or {})
        merged.update(extra_body)
        args["extra_body"] = merged
    return args


def _metrics_to_dict(metrics: Any) -> dict[str, Any]:
    if hasattr(metrics, "model_dump"):
        return metrics.model_dump()
    if hasattr(metrics, "dict"):
        return metrics.dict()
    return dict(metrics)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run official tau2-bench eval for local sglang policy/user models")
    parser.add_argument("--domains", default=",".join(DEFAULT_DOMAINS))
    parser.add_argument("--task-split-name", default=os.environ.get("TASK_SPLIT", "test"))
    parser.add_argument("--task-ids", default=None, help="Comma-separated task ids applied to each selected domain")
    parser.add_argument("--num-tasks", type=int, default=None, help="Tasks per domain; unset means all tasks")
    parser.add_argument("--num-trials", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--max-errors", type=int, default=10)
    parser.add_argument("--max-concurrency", type=int, default=1)
    parser.add_argument("--seed", type=int, default=300)
    parser.add_argument("--save-prefix", required=True)
    parser.add_argument("--summary-output", default=None)
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--auto-resume", action="store_true")
    parser.add_argument("--verbose-logs", action="store_true")

    parser.add_argument("--agent", default=AGENT_NAME)
    parser.add_argument("--agent-llm", required=True)
    parser.add_argument("--agent-api-base", default=None)
    parser.add_argument("--agent-api-key", default=None)
    parser.add_argument("--agent-temperature", type=float, default=0.6)
    parser.add_argument("--agent-top-p", type=float, default=1.0)
    parser.add_argument("--agent-max-tokens", type=int, default=1200)
    parser.add_argument("--agent-llm-args-json", default=None)
    parser.add_argument("--agent-extra-body-json", default=None)

    parser.add_argument("--user", default="user_simulator")
    parser.add_argument("--user-llm", required=True)
    parser.add_argument("--user-api-base", default=None)
    parser.add_argument("--user-api-key", default=None)
    parser.add_argument("--user-temperature", type=float, default=0.0)
    parser.add_argument("--user-top-p", type=float, default=None)
    parser.add_argument("--user-max-tokens", type=int, default=None)
    parser.add_argument("--user-llm-args-json", default=None)
    parser.add_argument("--user-extra-body-json", default=None)
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    if args.num_trials < 1:
        parser.error("--num-trials must be >= 1")
    if args.num_tasks is not None and args.num_tasks < 1:
        parser.error("--num-tasks must be >= 1 when set")

    from sglang_agent import register_slime_sglang_agent
    from tau2.data_model.simulation import TextRunConfig
    from tau2.metrics.agent_metrics import compute_metrics
    from tau2.runner import run_domain

    register_slime_sglang_agent()

    domains = _parse_csv(args.domains)
    task_ids = _parse_csv(args.task_ids)
    agent_args = _merge_litellm_args(
        json_args=args.agent_llm_args_json,
        api_base=args.agent_api_base,
        api_key=args.agent_api_key,
        temperature=args.agent_temperature,
        top_p=args.agent_top_p,
        max_tokens=args.agent_max_tokens,
        extra_body_json=args.agent_extra_body_json,
        name="agent",
    )
    user_args = _merge_litellm_args(
        json_args=args.user_llm_args_json,
        api_base=args.user_api_base,
        api_key=args.user_api_key,
        temperature=args.user_temperature,
        top_p=args.user_top_p,
        max_tokens=args.user_max_tokens,
        extra_body_json=args.user_extra_body_json,
        name="user",
    )

    summary: dict[str, Any] = {
        "task_split_name": args.task_split_name,
        "num_trials": args.num_trials,
        "domains": {},
    }

    for domain in domains:
        save_to = f"{args.save_prefix}_{domain}_{args.task_split_name}_{args.num_trials}trials"
        config = TextRunConfig(
            domain=domain,
            agent=args.agent,
            user=args.user,
            llm_agent=_with_openai_prefix(args.agent_llm),
            llm_args_agent=agent_args,
            llm_user=_with_openai_prefix(args.user_llm),
            llm_args_user=user_args,
            task_split_name=args.task_split_name,
            task_ids=task_ids or None,
            num_tasks=args.num_tasks,
            num_trials=args.num_trials,
            max_steps=args.max_steps,
            max_errors=args.max_errors,
            max_concurrency=args.max_concurrency,
            seed=args.seed,
            save_to=save_to,
            log_level=args.log_level,
            auto_resume=args.auto_resume,
            verbose_logs=args.verbose_logs,
        )
        results = run_domain(config)
        metrics = compute_metrics(results)
        summary["domains"][domain] = {
            "save_to": save_to,
            "results_file": f"data/simulations/{save_to}/results.json",
            "metrics": _metrics_to_dict(metrics),
        }

    if args.summary_output:
        output_path = Path(args.summary_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
        tmp_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        tmp_path.replace(output_path)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
