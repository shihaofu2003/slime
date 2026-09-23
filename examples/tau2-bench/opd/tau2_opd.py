"""Pure OPD hooks for mixed Tau2 domain rollouts."""

from __future__ import annotations

import copy
import math
import os
from collections.abc import Mapping
from typing import Any

import aiohttp

from slime.utils.processing_utils import encode_image_for_rollout_engine
from slime.utils.types import Sample


_DOMAIN_ENV_KEYS = {
    "airline": "TAU2_OPD_AIRLINE_URL",
    "retail": "TAU2_OPD_RETAIL_URL",
    "telecom": "TAU2_OPD_TELECOM_URL",
    "banking": "TAU2_OPD_BANKING_URL",
    "banking_knowledge": "TAU2_OPD_BANKING_URL",
}
_DEFAULT_TEACHER_URLS = {
    "airline": "http://127.0.0.1:31001/generate",
    "retail": "http://127.0.0.1:31002/generate",
    "telecom": "http://127.0.0.1:31003/generate",
    "banking": "http://127.0.0.1:31004/generate",
    "banking_knowledge": "http://127.0.0.1:31004/generate",
}


def _sample_domain(sample: Sample) -> str:
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    domain = metadata.get("domain") or metadata.get("tau2_domain")
    if not isinstance(domain, str) or not domain:
        raise ValueError("OPD sample metadata must contain a domain")
    return domain


def domain_teacher_url(domain: str, environ: Mapping[str, str] | None = None) -> str:
    """Return the expert endpoint for a Tau2 domain."""

    if domain not in _DOMAIN_ENV_KEYS:
        raise ValueError(f"Unsupported OPD teacher domain: {domain!r}")
    env = os.environ if environ is None else environ
    return env.get(_DOMAIN_ENV_KEYS[domain]) or _DEFAULT_TEACHER_URLS[domain]


def build_teacher_payload(sample: Sample) -> dict[str, Any]:
    """Build an SGLang input-logprob request from the complete on-policy sample."""

    tokens = [int(token) for token in sample.tokens]
    payload: dict[str, Any] = {
        "input_ids": tokens,
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": 0,
            "skip_special_tokens": False,
        },
        "return_logprob": True,
        # Tau2 response_length is the suffix from the first trainable
        # Assistant token through the end of the conversation.  It includes
        # intervening tool/User tokens whose loss mask is zero, so request the
        # complete input range and take the aligned suffix below.
        "logprob_start_len": 0,
    }
    if sample.multimodal_inputs and sample.multimodal_inputs.get("images"):
        payload["image_data"] = [
            encode_image_for_rollout_engine(image)
            for image in sample.multimodal_inputs["images"]
        ]
    return payload


async def _request_teacher(url: str, payload: dict[str, Any]) -> dict[str, Any]:
    timeout = aiohttp.ClientTimeout(
        total=float(os.environ.get("TAU2_OPD_TEACHER_TIMEOUT", "120"))
    )
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(url, json=payload) as response:
            response.raise_for_status()
            result = await response.json()
    if not isinstance(result, dict):
        raise ValueError(f"OPD teacher returned {type(result).__name__}, expected an object")
    return result


async def reward_func(args, sample: Sample, **kwargs) -> dict[str, Any]:
    """Request teacher input-token log-probs for one completed Tau2 sample."""

    del args, kwargs
    if sample.remove_sample:
        # The post-processor does not use this response, but keeping a token-
        # aligned placeholder makes the custom RM contract explicit.
        return {
            "tau2_opd_skipped": True,
            "meta_info": {
                "input_token_logprobs": [[0.0, 0] for _ in sample.tokens],
            },
        }

    domain = _sample_domain(sample)
    result = await _request_teacher(domain_teacher_url(domain), build_teacher_payload(sample))
    result["tau2_opd_domain"] = domain
    return result


async def reward_func_for_training_segments(args, sample: Sample) -> list[dict[str, Any]]:
    """Request one complete-context teacher response for every train segment.

    Raw Tau2 rollouts can contain several disjoint conversation prefixes.  The
    downstream conversion expands those prefixes into separate training
    samples, so each prefix must be scored against its own input token list.
    """

    segments = sample.training_segments or []
    if not segments:
        return [await reward_func(args, sample)]

    responses = []
    for segment in segments:
        segment_sample = copy.copy(sample)
        segment_sample.tokens = list(segment["tokens"])
        segment_sample.response_length = int(segment["response_length"])
        segment_sample.training_segments = None
        responses.append(await reward_func(args, segment_sample))
    return responses


def extract_teacher_log_probs(
    reward: Mapping[str, Any], *, tokens_length: int, response_length: int
) -> list[float]:
    """Align SGLang input-token log-probs to the sampled response.

    SGLang versions in the cluster expose the first-token ``None`` inside the
    first entry, while older responses prepend a separate placeholder entry.
    Accept both representations, but keep the token-length check strict.
    """

    try:
        values = reward["meta_info"]["input_token_logprobs"]
    except (KeyError, TypeError) as exc:
        raise ValueError("OPD teacher response is missing meta_info.input_token_logprobs") from exc
    if not isinstance(values, list) or not values:
        raise ValueError("OPD teacher input_token_logprobs must be a non-empty list")
    if len(values) == tokens_length + 1:
        values = values[1:]
    if len(values) != tokens_length:
        raise ValueError(
            "OPD teacher input log-prob length mismatch: "
            f"received {len(values)}, expected {tokens_length}"
        )
    if response_length < 0 or response_length > tokens_length:
        raise ValueError(
            f"OPD response length {response_length} is incompatible with token length {tokens_length}"
        )
    parsed: list[float] = []
    response_values = values[-response_length:] if response_length else []
    for index, item in enumerate(response_values):
        if isinstance(item, (list, tuple)):
            if not item:
                raise ValueError(f"OPD teacher log-prob entry {index} is malformed")
            value = item[0]
        else:
            value = item
        if value is None or not math.isfinite(float(value)):
            raise ValueError(f"OPD teacher log-prob entry {index} is non-finite")
        parsed.append(float(value))
    return parsed


def _task_reward(sample: Sample) -> float:
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    value = metadata.get("tau2_opd_task_reward")
    if value is None:
        raise ValueError("OPD sample is missing tau2_opd_task_reward")
    return float(value)


def post_process_rewards(args, samples: list[Sample], **kwargs):
    """Store teacher log-probs and expose task reward only as a diagnostic."""

    del args, kwargs
    task_rewards: list[float] = []
    training_rewards: list[float] = []
    for sample in samples:
        task_reward = _task_reward(sample)
        if sample.remove_sample:
            teacher_log_probs = [0.0] * sample.response_length
        elif sample.training_segments:
            metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
            segment_responses = metadata.get("tau2_opd_segment_responses")
            if segment_responses is None and isinstance(sample.reward, Mapping):
                segment_responses = sample.reward.get("tau2_opd_segment_responses")
            if not isinstance(segment_responses, list) or len(segment_responses) != len(sample.training_segments):
                raise ValueError(
                    "OPD sample is missing one teacher response per training segment"
                )
            teacher_log_probs = []
            for segment, teacher_response in zip(sample.training_segments, segment_responses, strict=True):
                if not isinstance(teacher_response, Mapping):
                    raise ValueError("OPD segment teacher response must be an object")
                segment_log_probs = extract_teacher_log_probs(
                    teacher_response,
                    tokens_length=len(segment["tokens"]),
                    response_length=int(segment["response_length"]),
                )
                segment["teacher_log_probs"] = segment_log_probs
                teacher_log_probs.extend(segment_log_probs)
        else:
            teacher_response = sample.reward
            if not isinstance(teacher_response, Mapping):
                teacher_response = (sample.metadata or {}).get("tau2_opd_teacher_response")
            if not isinstance(teacher_response, Mapping):
                raise ValueError("OPD sample reward must be the teacher response object")
            teacher_log_probs = extract_teacher_log_probs(
                teacher_response,
                tokens_length=len(sample.tokens),
                response_length=sample.response_length,
            )
        sample.teacher_log_probs = teacher_log_probs
        sample.reward = 0.0
        sample.metadata["raw_reward"] = task_reward
        sample.metadata["tau2_opd_teacher_tokens"] = len(teacher_log_probs)
        sample.train_metadata = {"domain": _sample_domain(sample)}
        task_rewards.append(task_reward)
        training_rewards.append(0.0)
    return task_rewards, training_rewards


def log_training_metrics(args, rollout_id, rollout_data):
    """Log domain means with the same whole-trajectory weights as the OPD loss.

    These are sampled log-ratios on collected contexts, not fresh-policy KL.
    Emit every configured domain on every rank so DP/CP reduction also works
    when one rank holds no trajectories from a domain.
    """
    import torch
    from megatron.core import mpu
    from slime.backends.megatron_utils.cp_utils import get_sum_of_sample_mean
    from slime.backends.megatron_utils.data import gather_log_data

    metadata = rollout_data["metadata"]
    if not mpu.is_pipeline_last_stage() or mpu.get_tensor_model_parallel_rank() != 0:
        return

    current = torch.cat(rollout_data["log_probs"]).detach()
    behavior = torch.cat(rollout_data["rollout_log_probs"]).detach()
    teacher = torch.cat(rollout_data["teacher_log_probs"]).detach()
    ratio = (current - behavior).exp()
    weights = ratio.clamp(min=args.tis_clip_low, max=args.tis_clip)
    values = {
        "sampled_current_teacher_log_ratio": current - teacher,
        "sampled_behavior_teacher_log_ratio": behavior - teacher,
        "tis_weighted_current_teacher_log_ratio": weights * (current - teacher),
        "tis_clipfrac": (weights != ratio).float(),
        "train_rollout_logprob_abs_diff": (current - behavior).abs(),
    }
    metrics = {}
    for domain in _DOMAIN_ENV_KEYS:
        masks = [
            mask if item["domain"] == domain else torch.zeros_like(mask)
            for item, mask in zip(metadata, rollout_data["loss_masks"], strict=True)
        ]
        reduce = get_sum_of_sample_mean(
            rollout_data["total_lengths"], rollout_data["response_lengths"],
            masks, rollout_data["rollout_mask_sums"],
        )
        # Sum partial trajectory shares across segments and DP/CP ranks.
        count = reduce(torch.ones_like(current)).item()
        for key, value in values.items():
            metrics[f"opd/{domain}/{key}"] = (reduce(value).item(), count)
    gather_log_data("rollout", args, rollout_id, metrics)
