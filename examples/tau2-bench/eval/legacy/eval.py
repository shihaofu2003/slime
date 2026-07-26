#!/usr/bin/env python3
"""Pass@K evaluation for tau2-bench using an SGLang-served policy.

Adapted from examples/tau-bench/tau-bench-example/tau2/eval.py. The policy model
is served locally by sglang (raw ``/generate`` endpoint, Qwen3 chat template +
native ``<tool_call>`` format); tau2-bench's ``AgentGymEnv`` plays the user
simulator + environment. We report pass@1 / pass@k, official pass^k
(``pass_hat_k``), and a partial score derived from tau2's reward_info.

The ONE material change vs. the reference: the user simulator is configured for
the OpenAI-compatible proxy in tau2-bench/.env (OPENAI_API_BASE + OPENAI_API_KEY)
by default, instead of assuming a keyless local sglang user-sim. See
``_resolve_user_llm``.

Thinking policy: thinking is ON by default. No ``enable_thinking`` kwarg is
passed unless ``--disable-thinking`` is set, so each model's chat template uses
its native default — Qwen3.5-4B thinks (``<think>…</think>`` then the tool call),
while Qwen3-4B-Instruct-2507 / Qwen3-4B-tau2-grpo-v1 have no ``<think>`` in their
template and never think. ``--disable-thinking`` only matters to force a thinking
model into non-thinking mode for comparison. Thinking models need a larger
``--max-new-tokens`` and benefit from ``--presence-penalty`` > 0 (set in their
run wrapper). Default sampling: agent ``temperature=0.6``; user-sim
``--user-temperature=0.0``.

Usage (sglang already serving the policy on :30000):

    python3 examples/tau2-bench/eval/legacy/eval.py \\
      --hf-checkpoint /path/to/model \\
      --sglang-url http://127.0.0.1:30000/generate \\
      --domains airline,retail,telecom --num-samples 4 \\
      --output examples/tau2-bench/eval/legacy/outputs/eval/pass4.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import os
import sys
from dataclasses import asdict, dataclass
from typing import Any

import httpx
from transformers import AutoTokenizer

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from actions import env_action_from_parsed_action, followup_messages_for_observation, parse_action
from env_util import compute_partial_score_from_reward_info, parse_reward_info
from prompting import build_tau2_agent_system_prompt

logger = logging.getLogger(__name__)

DEFAULT_DOMAINS = ("airline", "retail", "telecom")
METRIC_NOTE = (
    "pass@k = any success among k attempts; "
    "pass_hat_k/pass^k = C(successes, k) / C(num_samples, k)"
)
# Retry transient sglang transport errors so a single dropped keep-alive
# connection (stale pooled conn, momentary socket reset) can't abort a
# multi-hour run. 2**attempt backoff capped at SGLANG_MAX_BACKOFF seconds.
SGLANG_MAX_RETRIES = 5
SGLANG_MAX_BACKOFF = 8


def _parse_csv(value: str) -> list[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def _resolve_user_llm(*, user_model: str, temperature: float) -> tuple[str, dict[str, Any]]:
    """Resolve ``(user_llm, user_llm_args)`` for tau2's litellm-backed user simulator.

    Two modes:
      - LOCAL sglang user-sim (``TAU2_USER_API_BASE`` set): keyless local server,
        dummy key, plus the Qwen chat-template kwarg to disable thinking.
      - PROXY (default): the OpenAI-compatible endpoint in tau2-bench/.env
        (``OPENAI_API_BASE`` + ``OPENAI_API_KEY``); ``gpt-4.1-mini`` by default.

    litellm needs the ``openai/`` provider prefix to route a custom
    OpenAI-compatible base, so the model string is prefixed in both modes.
    """
    local_base = os.environ.get("TAU2_USER_API_BASE", "").strip()
    if local_base:
        return f"openai/{user_model}", {
            "temperature": temperature,
            "api_base": local_base,
            "api_key": "dummy-key-for-local-server",
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
        }

    api_base = os.environ.get("OPENAI_API_BASE", "").strip()
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise SystemExit(
            "No user-simulator credentials found: set OPENAI_API_KEY/OPENAI_API_BASE "
            "(proxy) or TAU2_USER_API_BASE (local sglang user-sim) before running eval."
        )
    args: dict[str, Any] = {"temperature": temperature}
    if api_base:
        args["api_base"] = api_base
    args["api_key"] = api_key
    return f"openai/{user_model}", args


@dataclass(frozen=True, slots=True)
class AttemptResult:
    success: bool
    reward: float
    partial_score: float
    partial_components: dict[str, float]
    steps: int
    status: str
    error: str | None = None
    reward_info: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class PassKResult:
    domain: str
    task_split: str
    task_index: int
    task_id: str
    num_samples: int
    best_success: bool
    best_reward: float
    best_partial_score: float
    best_sample_idx: int
    attempts: list[dict[str, Any]]
    pass_at_1: float
    pass_at_k: float
    pass_hat_ks: dict[int, float]


class SGLangClient:
    def __init__(self, url: str) -> None:
        self.url = url
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(300.0))

    async def close(self) -> None:
        await self._client.aclose()

    async def generate(self, *, text: str, sampling_params: dict[str, Any]) -> dict[str, Any]:
        # sglang can transiently drop a pooled keep-alive connection (stale conn
        # reuse, momentary socket reset) while the server itself stays healthy;
        # one such ReadError must not abort a multi-hour run. Retry transport /
        # timeout errors with a short backoff; surface real HTTP status errors
        # (4xx/5xx) immediately.
        last_exc: Exception | None = None
        for attempt in range(SGLANG_MAX_RETRIES):
            try:
                resp = await self._client.post(self.url, json={"text": text, "sampling_params": sampling_params})
                resp.raise_for_status()
                return resp.json()
            except httpx.HTTPStatusError:
                raise
            except (httpx.TransportError, httpx.TimeoutException) as exc:
                last_exc = exc
                logger.warning(f"sglang /generate transient error (attempt {attempt + 1}/{SGLANG_MAX_RETRIES}): {exc!r}")
                await asyncio.sleep(min(2 ** attempt, SGLANG_MAX_BACKOFF))
        assert last_exc is not None
        raise last_exc


def _load_tasks(domain: str, task_split: str) -> list[str]:
    from tau2.registry import registry

    return [t.id for t in registry.get_tasks_loader(domain)(task_split)]


async def _run_one_attempt(
    *,
    client: SGLangClient,
    tokenizer,
    domain: str,
    task_id: str,
    sampling_params: dict[str, Any],
    max_steps: int,
    user_llm: str,
    user_llm_args: dict[str, Any],
    chat_template_kwargs: dict[str, Any],
) -> AttemptResult:
    from tau2.gym.gym_agent import AgentGymEnv

    env = AgentGymEnv(
        domain=domain,
        task_id=task_id,
        max_steps=max_steps,
        solo_mode=False,
        user_llm=user_llm,
        user_llm_args=user_llm_args,
        all_messages_as_observation=False,
    )

    observation, info = env.reset()
    tools = info.get("tools", [])
    tools_openai = [t if isinstance(t, dict) else t.openai_schema for t in tools]
    policy = info.get("policy", "")

    system_prompt = build_tau2_agent_system_prompt(domain=domain, policy=policy, tools_openai=tools_openai)
    messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
    messages.extend(
        followup_messages_for_observation(
            observation=observation,
            last_action_call="(reset)",
            last_action_was_tool=False,
        )
    )

    reward = 0.0
    reward_info: dict[str, Any] = {}

    for step in range(max_steps):
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, **chat_template_kwargs
        )
        out = await client.generate(text=prompt_text, sampling_params=sampling_params)

        if out.get("meta_info", {}).get("finish_reason", {}).get("type") == "abort":
            return AttemptResult(
                success=False,
                reward=0.0,
                partial_score=0.0,
                partial_components={},
                steps=step,
                status="aborted",
                error="sglang_abort",
            )

        assistant_text = (out.get("text") or "").strip()
        if not assistant_text:
            return AttemptResult(
                success=False,
                reward=0.0,
                partial_score=0.0,
                partial_components={},
                steps=step,
                status="empty_generation",
                error="empty_generation",
            )

        try:
            parsed = parse_action(assistant_text, allow_prose_respond=True)
            # Thinking/base models that reply in prose (no <tool_call>) get lifted
            # to a `respond` action above; surface it so the run log is honest.
            if parsed.name == "respond" and "<tool_call>" not in assistant_text:
                logger.info("No <tool_call> in output; using prose as `respond` (thinking/base model).")
        except Exception as exc:
            logger.warning(f"parse_action failed ({exc}); raw assistant output:\n{assistant_text[:2000]}")
            messages.append({"role": "assistant", "content": assistant_text})
            messages.append(
                {
                    "role": "user",
                    "content": "FORMAT ERROR. Re-output EXACTLY in the required <tool_call> format: "
                    '<tool_call>{"name": "...", "arguments": {...}}</tool_call>. One action only.',
                }
            )
            repair_params = {**sampling_params, "temperature": 0.0}
            out = await client.generate(
                text=tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True, **chat_template_kwargs
                ),
                sampling_params=repair_params,
            )
            assistant_text = (out.get("text") or "").strip()
            try:
                parsed = parse_action(assistant_text, allow_prose_respond=True)
            except Exception:
                partial_score, partial_components = compute_partial_score_from_reward_info(reward_info)
                return AttemptResult(
                    success=False,
                    reward=float(reward),
                    partial_score=partial_score,
                    partial_components=partial_components,
                    steps=step + 1,
                    status="parse_error",
                    error=str(exc),
                    reward_info=reward_info,
                )

        messages.append({"role": "assistant", "content": assistant_text})

        env_action = env_action_from_parsed_action(parsed)
        observation, reward, terminated, _truncated, info = env.step(env_action)

        if terminated:
            reward_info = parse_reward_info(info)
            partial_score, partial_components = compute_partial_score_from_reward_info(reward_info)
            return AttemptResult(
                success=float(reward) >= 1.0,
                reward=float(reward),
                partial_score=partial_score,
                partial_components=partial_components,
                steps=step + 1,
                status="completed",
                reward_info=reward_info,
            )

        messages.extend(
            followup_messages_for_observation(
                observation=observation,
                last_action_call=parsed.raw_action_call,
                last_action_was_tool=(parsed.name != "respond"),
            )
        )

    partial_score, partial_components = compute_partial_score_from_reward_info(reward_info)
    return AttemptResult(
        success=False,
        reward=float(reward),
        partial_score=partial_score,
        partial_components=partial_components,
        steps=max_steps,
        status="truncated",
        reward_info=reward_info,
    )


def _pass_hat_k(num_trials: int, success_count: int, k: int) -> float:
    if k < 1:
        raise ValueError("k must be >= 1")
    if num_trials < k:
        raise ValueError(f"num_trials={num_trials} is less than k={k}")
    if success_count < k:
        return 0.0
    return math.comb(success_count, k) / math.comb(num_trials, k)


def _pass_hat_ks(attempts: list[AttemptResult]) -> dict[int, float]:
    num_trials = len(attempts)
    success_count = sum(1 for attempt in attempts if attempt.success)
    return {k: _pass_hat_k(num_trials, success_count, k) for k in range(1, num_trials + 1)}


def _summarize(results: list[PassKResult], *, k: int) -> dict[str, Any]:
    total = len(results)
    pass1 = sum(r.pass_at_1 for r in results) / total if total else 0.0
    passk = sum(r.pass_at_k for r in results) / total if total else 0.0
    pass_hat_by_k = {
        str(i): sum(r.pass_hat_ks.get(i, 0.0) for r in results) / total if total else 0.0
        for i in range(1, k + 1)
    }
    summary: dict[str, Any] = {
        "total": total,
        "pass_at_1": pass1,
        f"pass_at_{k}": passk,
        "pass_hat_by_k": pass_hat_by_k,
    }
    for i, value in pass_hat_by_k.items():
        summary[f"pass_hat_{i}"] = value
    return summary


def _write_report(path: str, args: argparse.Namespace, domains: list[str], all_results: list[PassKResult]) -> dict[str, Any]:
    """Assemble + atomically write the report for the results gathered so far.

    Called after every task so a crash late in a multi-hour run still leaves a
    complete partial report on disk. The temp-file + os.replace makes each write
    atomic — the report file is never half-written.
    """
    by_domain: dict[str, list[PassKResult]] = {}
    for r in all_results:
        by_domain.setdefault(r.domain, []).append(r)
    report = {
        "hf_checkpoint": args.hf_checkpoint,
        "sglang_url": args.sglang_url,
        "task_split": args.task_split,
        "domains": domains,
        "k": args.num_samples,
        "metric_note": METRIC_NOTE,
        "summary": _summarize(all_results, k=args.num_samples),
        "by_domain": {d: _summarize(rs, k=args.num_samples) for d, rs in sorted(by_domain.items())},
        "results": [asdict(r) for r in all_results],
    }
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        json.dump(report, f, indent=2)
    os.replace(tmp, path)
    return report


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate tau2-bench with Pass@K sampling")
    parser.add_argument("--hf-checkpoint", required=True)
    parser.add_argument("--sglang-url", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--domains", default=",".join(DEFAULT_DOMAINS))
    parser.add_argument("--task-split", default="test", choices=("train", "test", "base"))
    parser.add_argument("--max-tasks-per-domain", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=int(os.environ.get("TAU2_MAX_STEPS", "100")))
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    # presence_penalty > 0 suppresses the endless-repetition / non-convergent
    # <think> loops that thinking models (Qwen3.5) are prone to (Qwen3.5 README,
    # "Best Practices"). 0.0 = no-op, so non-thinking runs are unaffected.
    parser.add_argument("--presence-penalty", type=float, default=0.0)
    parser.add_argument("--max-new-tokens", type=int, default=1200)
    parser.add_argument("--user-model", default=os.environ.get("TAU2_USER_MODEL", "gpt-4.1-mini"))
    parser.add_argument("--user-temperature", type=float, default=float(os.environ.get("TAU2_USER_TEMPERATURE", "0.0")))
    parser.add_argument(
        "--disable-thinking",
        action="store_true",
        help="Force enable_thinking=False in the chat template. Thinking is ON by "
        "default: thinking-capable models (Qwen3.5) emit <think> then the tool call; "
        "non-thinking models (Qwen3-4B-Instruct-2507, Qwen3-4B-tau2-grpo-v1) have no "
        "<think> in their template and are unaffected either way. Use this flag only to "
        "force a thinking model into non-thinking mode for an apples-to-apples comparison.",
    )
    return parser


def _validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.num_samples < 1:
        parser.error("--num-samples must be >= 1")


async def main_async() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()
    _validate_args(args, parser)

    domains = _parse_csv(args.domains)
    tokenizer = AutoTokenizer.from_pretrained(args.hf_checkpoint, trust_remote_code=True)

    sampling_params: dict[str, Any] = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "repetition_penalty": args.repetition_penalty,
        "presence_penalty": args.presence_penalty,
        "max_new_tokens": args.max_new_tokens,
        "stop": ["</tool_call>"],
        "no_stop_trim": True,
    }
    if args.top_k > 0:
        sampling_params["top_k"] = args.top_k

    user_llm, user_llm_args = _resolve_user_llm(user_model=args.user_model, temperature=args.user_temperature)
    logger.info(f"User simulator: model={user_llm} args={user_llm_args}")

    chat_template_kwargs: dict[str, Any] = {"enable_thinking": False} if args.disable_thinking else {}
    if args.disable_thinking:
        logger.info("Thinking forced OFF (enable_thinking=False): tool call emitted directly, no <think>.")
    else:
        # No enable_thinking kwarg → each model's chat template uses its native
        # default: Qwen3.5 thinks (<think>…</think> then the tool call); the
        # Qwen3-4B-Instruct / grpo templates have no <think> at all.
        logger.info("Thinking default (no enable_thinking kwarg): thinking-capable models think, non-thinking models don't.")

    client = SGLangClient(args.sglang_url)
    try:
        all_results: list[PassKResult] = []

        for domain in domains:
            task_ids = _load_tasks(domain, args.task_split)
            if args.max_tasks_per_domain is not None:
                task_ids = task_ids[: args.max_tasks_per_domain]

            logger.info(f"Evaluating domain={domain} split={args.task_split} tasks={len(task_ids)} k={args.num_samples}")
            for i, task_id in enumerate(task_ids):
                attempts: list[AttemptResult] = []
                for _ in range(args.num_samples):
                    attempts.append(
                        await _run_one_attempt(
                            client=client,
                            tokenizer=tokenizer,
                            domain=domain,
                            task_id=task_id,
                            sampling_params=sampling_params,
                            max_steps=args.max_steps,
                            user_llm=user_llm,
                            user_llm_args=user_llm_args,
                            chat_template_kwargs=chat_template_kwargs,
                        )
                    )

                best_idx = 0
                for j in range(1, len(attempts)):
                    a = attempts[j]
                    b = attempts[best_idx]
                    if a.success and not b.success:
                        best_idx = j
                    elif a.success == b.success and a.partial_score > b.partial_score:
                        best_idx = j

                pass_at_1 = 1.0 if attempts and attempts[0].success else 0.0
                pass_at_k = 1.0 if any(a.success for a in attempts) else 0.0
                pass_hat_ks = _pass_hat_ks(attempts)
                best = attempts[best_idx]
                all_results.append(
                    PassKResult(
                        domain=domain,
                        task_split=args.task_split,
                        task_index=i,
                        task_id=task_id,
                        num_samples=args.num_samples,
                        best_success=best.success,
                        best_reward=best.reward,
                        best_partial_score=best.partial_score,
                        best_sample_idx=best_idx,
                        attempts=[asdict(a) for a in attempts],
                        pass_at_1=pass_at_1,
                        pass_at_k=pass_at_k,
                        pass_hat_ks=pass_hat_ks,
                    )
                )
                _write_report(args.output, args, domains, all_results)

        report = _write_report(args.output, args, domains, all_results)

        logger.info(f"Wrote {len(all_results)} results to {args.output}")
        logger.info(f"Overall: {report['summary']}")
        for d, s in report["by_domain"].items():
            logger.info(f"{d}: {s}")
    finally:
        await client.close()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
