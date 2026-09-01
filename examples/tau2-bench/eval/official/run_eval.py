#!/usr/bin/env python3
"""Run tau2-bench eval through tau2's official runner.

This replaces the legacy direct-AgentGymEnv eval with the official
HalfDuplexAgent/TextRunConfig/run_domain path. A local sglang policy server is
addressed through raw `/generate`; the user simulator remains the official
LiteLLM-backed tau2 user.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import multiprocessing
import os
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

SHARED_DIR = Path(__file__).resolve().parents[2] / "shared"
if str(SHARED_DIR) not in sys.path:
    sys.path.insert(0, str(SHARED_DIR))
ANALYSIS_DIR = Path(__file__).resolve().parents[2] / "analysis"
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

from protocol_profiles import (  # noqa: E402
    PROTOCOL_AGENT_OWNED_DEPENDENCY_SAFE_MULTI,
    protocol_signature,
    validate_protocol_profile,
)
from agent_contract import contract_from_environment  # noqa: E402
from namespace_analyzer import (  # noqa: E402
    analyze_namespace_trajectories,
    analyze_single_call_attempts,
)

DEFAULT_DOMAINS = ("airline", "retail", "telecom")
AGENT_NAME = "slime_sglang_agent"


def _parse_csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_domain_concurrency(
    value: str | None,
    domains: list[str],
    *,
    default_concurrency: int,
) -> dict[str, int]:
    if not value:
        return {domain: default_concurrency for domain in domains}

    parsed: dict[str, int] = {}
    for item in _parse_csv(value):
        domain, separator, raw_concurrency = item.partition(":")
        if not separator or not domain or not raw_concurrency:
            raise ValueError(
                "domain concurrency entries must use domain:positive_integer"
            )
        if domain in parsed:
            raise ValueError(f"duplicate domain concurrency entry: {domain}")
        try:
            concurrency = int(raw_concurrency)
        except ValueError as exc:
            raise ValueError(
                f"domain concurrency for {domain} must be a positive integer"
            ) from exc
        if concurrency <= 0:
            raise ValueError(
                f"domain concurrency for {domain} must be a positive integer"
            )
        parsed[domain] = concurrency

    missing = [domain for domain in domains if domain not in parsed]
    if missing:
        raise ValueError(f"domain concurrency is missing: {','.join(missing)}")
    return {domain: parsed[domain] for domain in domains}


def _allocate_released_slots(
    slots: int,
    remaining_domains: list[str],
    initial_weights: dict[str, int],
) -> dict[str, int]:
    if slots <= 0 or not remaining_domains:
        return {}

    total_weight = sum(initial_weights[domain] for domain in remaining_domains)
    allocations: dict[str, int] = {}
    remainders: dict[str, int] = {}
    for domain in remaining_domains:
        allocation, remainder = divmod(
            slots * initial_weights[domain], total_weight
        )
        allocations[domain] = allocation
        remainders[domain] = remainder

    unallocated = slots - sum(allocations.values())
    order = {domain: index for index, domain in enumerate(remaining_domains)}
    ranked_domains = sorted(
        remaining_domains,
        key=lambda domain: (-remainders[domain], order[domain]),
    )
    for domain in ranked_domains[:unallocated]:
        allocations[domain] += 1
    return allocations


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


def _as_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "dict"):
        return value.dict()
    return dict(value)


def _percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _latency_stats(
    samples: list[float],
    *,
    trajectory_seconds: float,
) -> dict[str, Any]:
    elapsed = sum(samples)
    return {
        "calls": len(samples),
        "seconds": elapsed,
        "mean_seconds": elapsed / len(samples) if samples else None,
        "p50_seconds": _percentile(samples, 0.50),
        "p95_seconds": _percentile(samples, 0.95),
        "trajectory_share": _ratio(elapsed, trajectory_seconds),
    }


def _timing_summary(
    results: Any,
    *,
    wall_seconds: float,
) -> tuple[dict[str, Any], dict[str, list[float]]]:
    simulations = getattr(results, "simulations", None)
    if simulations is None and isinstance(results, dict):
        simulations = results.get("simulations")

    samples: dict[str, list[float]] = {"agent": [], "user": []}
    trajectory_seconds = 0.0
    other_seconds = 0.0
    for simulation in simulations or []:
        record = _as_mapping(simulation)
        duration = float(record.get("duration") or 0.0)
        trajectory_seconds += duration
        timed_seconds = 0.0
        for message in record.get("messages") or []:
            message_record = _as_mapping(message)
            raw_data = _as_mapping(message_record.get("raw_data") or {})
            timing = _as_mapping(raw_data.get("tau2_eval_timing") or {})
            participant = timing.get("participant")
            if participant not in samples:
                continue
            elapsed = float(timing.get("elapsed_seconds") or 0.0)
            samples[participant].append(elapsed)
            timed_seconds += elapsed
        other_seconds += max(0.0, duration - timed_seconds)

    summary = {
        "wall_seconds": wall_seconds,
        "trajectory_seconds": trajectory_seconds,
        "parallelism_factor": _ratio(trajectory_seconds, wall_seconds),
        "agent": _latency_stats(
            samples["agent"], trajectory_seconds=trajectory_seconds
        ),
        "user": _latency_stats(
            samples["user"], trajectory_seconds=trajectory_seconds
        ),
        "other": {
            "seconds": other_seconds,
            "trajectory_share": _ratio(other_seconds, trajectory_seconds),
        },
    }
    return summary, samples


def _combined_timing_summary(
    domain_timings: dict[str, dict[str, Any]],
    domain_samples: dict[str, dict[str, list[float]]],
    *,
    wall_seconds: float,
) -> dict[str, Any]:
    trajectory_seconds = sum(
        float(timing.get("trajectory_seconds") or 0.0)
        for timing in domain_timings.values()
    )
    other_seconds = sum(
        float((timing.get("other") or {}).get("seconds") or 0.0)
        for timing in domain_timings.values()
    )
    samples = {
        participant: [
            elapsed
            for domain in domain_samples.values()
            for elapsed in domain.get(participant, [])
        ]
        for participant in ("agent", "user")
    }
    return {
        "wall_seconds": wall_seconds,
        "trajectory_seconds": trajectory_seconds,
        "parallelism_factor": _ratio(trajectory_seconds, wall_seconds),
        "agent": _latency_stats(
            samples["agent"], trajectory_seconds=trajectory_seconds
        ),
        "user": _latency_stats(
            samples["user"], trajectory_seconds=trajectory_seconds
        ),
        "other": {
            "seconds": other_seconds,
            "trajectory_share": _ratio(other_seconds, trajectory_seconds),
        },
    }


def _overall_summary(
    domains: dict[str, dict[str, Any]],
    *,
    timing: dict[str, Any],
) -> dict[str, Any]:
    total_tasks = sum(domain["pass_metrics"]["tasks"] for domain in domains.values())
    total_simulations = sum(
        domain["pass_metrics"]["simulations"] for domain in domains.values()
    )
    total_action_count = sum(
        int(domain["metrics"].get("total_read_actions", 0))
        + int(domain["metrics"].get("total_write_actions", 0))
        for domain in domains.values()
    )
    correct_action_count = sum(
        int(domain["metrics"].get("correct_read_actions", 0))
        + int(domain["metrics"].get("correct_write_actions", 0))
        for domain in domains.values()
    )
    db_checked_count = sum(
        int(domain["metrics"].get("db_match_count", 0))
        + int(domain["metrics"].get("db_mismatch_count", 0))
        for domain in domains.values()
    )
    db_match_count = sum(
        int(domain["metrics"].get("db_match_count", 0))
        for domain in domains.values()
    )
    single_call_keys = (
        "trajectory_count",
        "trajectories_with_multi_call_output",
        "multi_call_output_turns",
        "attempted_calls_in_multi_call_outputs",
        "parsed_multi_call_turns",
        "single_call_protocol_error_turns",
    )
    return {
        "tasks": total_tasks,
        "simulations": total_simulations,
        **{
            metric: (
                sum(
                    domain["pass_metrics"][metric]
                    * domain["pass_metrics"]["tasks"]
                    for domain in domains.values()
                )
                / total_tasks
                if total_tasks
                else 0.0
            )
            for metric in ("pass_at_1", "pass_at_4_any", "pass_power_4")
        },
        "single_call": {
            key: sum(
                int(domain["single_call"].get(key, 0))
                for domain in domains.values()
            )
            for key in single_call_keys
        },
        "diagnostics": {
            "action_accuracy": _ratio(correct_action_count, total_action_count),
            "db_accuracy": _ratio(db_match_count, db_checked_count),
            "max_steps": sum(
                int(domain["metrics"].get("termination_max_steps", 0))
                for domain in domains.values()
            ),
            "infrastructure_errors": sum(
                int(domain["metrics"].get("infra_error_count", 0))
                for domain in domains.values()
            ),
        },
        "timing": timing,
    }


def _require_no_infrastructure_errors(
    domains: dict[str, dict[str, Any]],
) -> None:
    error_count = sum(
        int(domain["metrics"].get("infra_error_count", 0))
        for domain in domains.values()
    )
    if error_count:
        raise RuntimeError(
            f"official eval is invalid: {error_count} infrastructure errors"
        )


def _pass_metrics(results: Any, expected_trials: int) -> dict[str, Any]:
    simulations = getattr(results, "simulations", None)
    if simulations is None and isinstance(results, dict):
        simulations = results.get("simulations")
    task_rewards: dict[str, list[float]] = {}
    for simulation in simulations or []:
        record = _as_mapping(simulation)
        reward_info = _as_mapping(record.get("reward_info") or {})
        reward = float(reward_info.get("reward") or 0.0)
        task_rewards.setdefault(str(record.get("task_id")), []).append(reward)
    if any(len(rewards) != expected_trials for rewards in task_rewards.values()):
        raise ValueError("official eval returned an incomplete task trial group")
    return {
        "tasks": len(task_rewards),
        "simulations": sum(len(rewards) for rewards in task_rewards.values()),
        "pass_at_1": (
            sum(sum(rewards) for rewards in task_rewards.values())
            / sum(len(rewards) for rewards in task_rewards.values())
            if task_rewards
            else 0.0
        ),
        "pass_at_4_any": (
            sum(any(rewards) for rewards in task_rewards.values()) / len(task_rewards)
            if task_rewards
            else 0.0
        ),
        "pass_power_4": (
            sum(all(rewards) for rewards in task_rewards.values()) / len(task_rewards)
            if task_rewards
            else 0.0
        ),
    }


def _ratio(numerator: int | float, denominator: int | float) -> float | None:
    return numerator / denominator if denominator else None


def _json_finite(value: Any) -> Any:
    """Represent unavailable non-finite evaluator diagnostics as JSON null."""

    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_finite(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_finite(item) for item in value]
    return value


def _run_domain_jobs(
    domains: list[str],
    payload: dict[str, Any],
    *,
    parallel: bool,
    evaluator: Callable[[dict[str, Any], str], Any],
    domain_concurrency: dict[str, int],
    global_concurrency: int,
    borrow_completed_slots: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not domains:
        return {}, []
    if not parallel:
        results = {}
        for domain in domains:
            domain_payload = dict(payload)
            domain_payload["_domain_concurrency"] = domain_concurrency[domain]
            results[domain] = evaluator(domain_payload, domain)
        return results, []

    process_context = multiprocessing.get_context("spawn")
    if not borrow_completed_slots:
        completed_results = {}
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=len(domains),
            mp_context=process_context,
        ) as executor:
            future_domains = {}
            for domain in domains:
                domain_payload = dict(payload)
                domain_payload["_domain_concurrency"] = domain_concurrency[domain]
                future = executor.submit(evaluator, domain_payload, domain)
                future_domains[future] = domain
            for future in concurrent.futures.as_completed(future_domains):
                domain = future_domains[future]
                completed_results[domain] = future.result()
        return {domain: completed_results[domain] for domain in domains}, []

    if sum(domain_concurrency.values()) != global_concurrency:
        raise ValueError(
            "elastic domain concurrency must sum to global concurrency"
        )

    events: list[dict[str, Any]] = []
    completed_results = {}
    current_concurrency = dict(domain_concurrency)
    remaining_domains = list(domains)
    started_at = time.perf_counter()
    with multiprocessing.Manager() as manager:
        slot_semaphores = {
            domain: manager.Semaphore(domain_concurrency[domain])
            for domain in domains
        }
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=len(domains),
            mp_context=process_context,
        ) as executor:
            future_domains = {}
            for domain in domains:
                domain_payload = dict(payload)
                domain_payload["_domain_concurrency"] = domain_concurrency[domain]
                domain_payload["_executor_max_workers"] = global_concurrency
                domain_payload["_slot_semaphore"] = slot_semaphores[domain]
                future = executor.submit(evaluator, domain_payload, domain)
                future_domains[future] = domain

            for future in concurrent.futures.as_completed(future_domains):
                completed_domain = future_domains[future]
                completed_results[completed_domain] = future.result()
                remaining_domains.remove(completed_domain)
                if not remaining_domains:
                    continue

                released_slots = current_concurrency[completed_domain]
                current_concurrency[completed_domain] = 0
                allocations = _allocate_released_slots(
                    released_slots,
                    remaining_domains,
                    domain_concurrency,
                )
                for domain, slots in allocations.items():
                    for _ in range(slots):
                        slot_semaphores[domain].release()
                    current_concurrency[domain] += slots

                event = {
                    "elapsed_seconds": time.perf_counter() - started_at,
                    "completed_domain": completed_domain,
                    "released_slots": released_slots,
                    "domain_concurrency": dict(current_concurrency),
                }
                events.append(event)
                print(
                    "[tau2-official-eval] "
                    "stage=domain_concurrency_reallocated "
                    f"completed_domain={completed_domain} "
                    f"released_slots={released_slots} "
                    f"domain_concurrency={current_concurrency}",
                    flush=True,
                )

    ordered_results = {
        domain: completed_results[domain]
        for domain in domains
    }
    return ordered_results, events


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
    parser.add_argument("--domain-concurrency", default=None)
    parser.add_argument("--global-concurrency", type=int, default=None)
    parser.add_argument("--borrow-completed-domain-slots", action="store_true")
    parser.add_argument("--parallel-domains", action="store_true")
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
    parser.add_argument("--agent-protocol-profile", default=None)
    parser.add_argument("--agent-llm-args-json", default=None)
    parser.add_argument("--agent-extra-body-json", default=None)

    parser.add_argument("--user", default="slime_timed_user_simulator")
    parser.add_argument("--user-llm", required=True)
    parser.add_argument("--user-api-base", default=None)
    parser.add_argument("--user-api-key", default=None)
    parser.add_argument("--user-temperature", type=float, default=0.0)
    parser.add_argument("--user-top-p", type=float, default=None)
    parser.add_argument("--user-max-tokens", type=int, default=None)
    parser.add_argument("--user-llm-args-json", default=None)
    parser.add_argument("--user-extra-body-json", default=None)
    return parser


def _evaluate_domain(payload: dict[str, Any], domain: str) -> dict[str, Any]:
    from sglang_agent import register_slime_sglang_agent
    from timed_user import register_timed_user_simulator
    from tau2.data_model.simulation import TextRunConfig
    from tau2.metrics.agent_metrics import compute_metrics
    from tau2.registry import registry
    from tau2.runner import run_domain
    from transformers import AutoTokenizer

    register_slime_sglang_agent()
    register_timed_user_simulator()

    args = argparse.Namespace(**payload["args"])
    agent_args = dict(payload["agent_args"])
    user_args = dict(payload["user_args"])
    selected_protocol_profile = payload["selected_protocol_profile"]
    domain_agent_args = dict(agent_args)
    contract_metadata = None
    environment = registry.get_env_constructor(domain)()
    if selected_protocol_profile == PROTOCOL_AGENT_OWNED_DEPENDENCY_SAFE_MULTI:
        agent_tokenizer = AutoTokenizer.from_pretrained(
            args.agent_llm,
            trust_remote_code=True,
        )
        contract = contract_from_environment(
            environment,
            domain=domain,
            chat_template=agent_tokenizer.chat_template,
            profile=selected_protocol_profile,
        )
        domain_agent_args["contract_domain"] = domain
        domain_agent_args["agent_contract_signature"] = (
            contract.agent_contract_signature
        )
        contract_metadata = contract.metadata(view="official_agent")

    save_to = (
        f"{args.save_prefix}_{domain}_{args.task_split_name}_"
        f"{args.num_trials}trials"
    )
    config = TextRunConfig(
        domain=domain,
        agent=args.agent,
        user=args.user,
        llm_agent=_with_openai_prefix(args.agent_llm),
        llm_args_agent=domain_agent_args,
        llm_user=_with_openai_prefix(args.user_llm),
        llm_args_user=user_args,
        task_split_name=args.task_split_name,
        task_ids=_parse_csv(args.task_ids) or None,
        num_tasks=args.num_tasks,
        num_trials=args.num_trials,
        max_steps=args.max_steps,
        max_errors=args.max_errors,
        max_concurrency=payload.get("_domain_concurrency", args.max_concurrency),
        seed=args.seed,
        save_to=save_to,
        log_level=args.log_level,
        auto_resume=args.auto_resume,
        verbose_logs=args.verbose_logs,
    )

    print(
        f"[tau2-official-eval] stage=domain_eval_start domain={domain}",
        flush=True,
    )
    domain_started_at = time.perf_counter()
    if "_slot_semaphore" in payload:
        results = run_domain(
            config,
            executor_max_workers=payload["_executor_max_workers"],
            slot_semaphore=payload["_slot_semaphore"],
        )
    else:
        results = run_domain(config)
    domain_wall_seconds = time.perf_counter() - domain_started_at

    postprocess_started_at = time.perf_counter()
    metrics_dict = _metrics_to_dict(compute_metrics(results))
    try:
        user_tool_names = {tool.name for tool in environment.get_user_tools()}
    except ValueError:
        user_tool_names = set()
    timing, timing_samples = _timing_summary(
        results,
        wall_seconds=domain_wall_seconds,
    )
    domain_summary = {
        "save_to": save_to,
        "results_file": f"data/simulations/{save_to}/results.json",
        "metrics": metrics_dict,
        "pass_metrics": _pass_metrics(results, args.num_trials),
        "agent_contract": contract_metadata,
        "namespace": analyze_namespace_trajectories(
            results,
            user_tool_names=user_tool_names,
        ),
        "single_call": analyze_single_call_attempts(results),
        "diagnostics": {
            "action_accuracy": _ratio(
                int(metrics_dict.get("correct_read_actions", 0))
                + int(metrics_dict.get("correct_write_actions", 0)),
                int(metrics_dict.get("total_read_actions", 0))
                + int(metrics_dict.get("total_write_actions", 0)),
            ),
            "db_accuracy": _ratio(
                int(metrics_dict.get("db_match_count", 0)),
                int(metrics_dict.get("db_match_count", 0))
                + int(metrics_dict.get("db_mismatch_count", 0)),
            ),
            "max_steps": int(metrics_dict.get("termination_max_steps", 0)),
        },
        "timing": timing,
    }
    postprocess_seconds = time.perf_counter() - postprocess_started_at
    domain_summary["timing"]["postprocess_seconds"] = postprocess_seconds
    print(
        "[tau2-official-eval] "
        f"stage=domain_eval_end domain={domain} "
        f"elapsed_seconds={domain_wall_seconds:.3f} "
        f"postprocess_seconds={postprocess_seconds:.3f}",
        flush=True,
    )
    return {
        "summary": domain_summary,
        "timing_samples": timing_samples,
    }


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    if args.num_trials < 1:
        parser.error("--num-trials must be >= 1")
    if args.num_tasks is not None and args.num_tasks < 1:
        parser.error("--num-tasks must be >= 1 when set")

    domains = _parse_csv(args.domains)
    if not domains:
        parser.error("--domains must select at least one domain")
    try:
        domain_concurrency = _parse_domain_concurrency(
            args.domain_concurrency,
            domains,
            default_concurrency=args.max_concurrency,
        )
    except ValueError as exc:
        parser.error(str(exc))
    global_concurrency = (
        args.global_concurrency
        if args.global_concurrency is not None
        else sum(domain_concurrency.values())
    )
    if global_concurrency < 1:
        parser.error("--global-concurrency must be >= 1")
    use_parallel_domains = args.parallel_domains and len(domains) > 1
    if args.borrow_completed_domain_slots:
        if not use_parallel_domains:
            parser.error(
                "--borrow-completed-domain-slots requires parallel evaluation "
                "of at least two domains"
            )
        if sum(domain_concurrency.values()) != global_concurrency:
            parser.error(
                "initial domain concurrency must sum to --global-concurrency"
            )
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
    selected_protocol_profile = args.agent_protocol_profile or agent_args.get(
        "protocol_profile"
    )
    if args.agent_protocol_profile:
        existing_profile = agent_args.get("protocol_profile")
        if existing_profile and existing_profile != args.agent_protocol_profile:
            parser.error(
                "--agent-protocol-profile conflicts with protocol_profile in "
                "--agent-llm-args-json"
            )
        agent_args["protocol_profile"] = args.agent_protocol_profile
    selected_protocol_signature = None
    if selected_protocol_profile:
        try:
            validate_protocol_profile(selected_protocol_profile)
        except ValueError as exc:
            parser.error(str(exc))
        selected_protocol_signature = protocol_signature(selected_protocol_profile)
        existing_signature = agent_args.get("protocol_signature")
        if existing_signature and existing_signature != selected_protocol_signature:
            parser.error(
                "protocol_signature in --agent-llm-args-json does not match "
                "the selected protocol profile"
            )
        agent_args["protocol_profile"] = selected_protocol_profile
        if selected_protocol_signature is not None:
            agent_args["protocol_signature"] = selected_protocol_signature
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

    payload = {
        "args": vars(args),
        "agent_args": agent_args,
        "user_args": user_args,
        "selected_protocol_profile": selected_protocol_profile,
    }
    print(
        "[tau2-official-eval] "
        f"stage=eval_start domains={','.join(domains)} "
        f"parallel_domains={int(use_parallel_domains)} "
        f"initial_domain_concurrency={domain_concurrency} "
        f"global_concurrency={global_concurrency} "
        f"borrow_completed_domain_slots={int(args.borrow_completed_domain_slots)}",
        flush=True,
    )
    evaluation_started_at = time.perf_counter()
    jobs, concurrency_events = _run_domain_jobs(
        domains,
        payload,
        parallel=use_parallel_domains,
        evaluator=_evaluate_domain,
        domain_concurrency=domain_concurrency,
        global_concurrency=global_concurrency,
        borrow_completed_slots=args.borrow_completed_domain_slots,
    )
    evaluation_wall_seconds = time.perf_counter() - evaluation_started_at
    domain_summaries = {
        domain: jobs[domain]["summary"]
        for domain in domains
    }
    domain_samples = {
        domain: jobs[domain]["timing_samples"]
        for domain in domains
    }
    domain_timings = {
        domain: domain_summaries[domain]["timing"]
        for domain in domains
    }
    overall_timing = _combined_timing_summary(
        domain_timings,
        domain_samples,
        wall_seconds=evaluation_wall_seconds,
    )
    overall_timing["domain_wall_seconds"] = {
        domain: domain_timings[domain]["wall_seconds"]
        for domain in domains
    }
    overall_timing["postprocess_seconds"] = sum(
        float(domain_timings[domain].get("postprocess_seconds") or 0.0)
        for domain in domains
    )

    summary: dict[str, Any] = {
        "task_split_name": args.task_split_name,
        "num_trials": args.num_trials,
        "num_tasks": args.num_tasks,
        "seed": args.seed,
        "max_steps": args.max_steps,
        "max_errors": args.max_errors,
        "max_concurrency_per_domain": args.max_concurrency,
        "initial_domain_concurrency": domain_concurrency,
        "global_concurrency": global_concurrency,
        "borrow_completed_domain_slots": args.borrow_completed_domain_slots,
        "concurrency_events": concurrency_events,
        "parallel_domains": use_parallel_domains,
        "agent_protocol_profile": selected_protocol_profile,
        "domains": domain_summaries,
    }
    if selected_protocol_signature is not None:
        summary["agent_protocol_signature"] = selected_protocol_signature
    summary["overall"] = _overall_summary(
        domain_summaries,
        timing=overall_timing,
    )

    summary = _json_finite(summary)
    if args.summary_output:
        output_path = Path(args.summary_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
        tmp_path.write_text(
            json.dumps(summary, indent=2, allow_nan=False),
            encoding="utf-8",
        )
        tmp_path.replace(output_path)

    print(
        "[tau2-official-eval] "
        f"stage=eval_end elapsed_seconds={evaluation_wall_seconds:.3f}",
        flush=True,
    )
    print(json.dumps(summary, indent=2, allow_nan=False))
    _require_no_infrastructure_errors(domain_summaries)


if __name__ == "__main__":
    main()
