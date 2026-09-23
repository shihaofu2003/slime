"""Loss segments built from actual inference tokens, without reserialization."""

from __future__ import annotations

import copy
import math


class BehaviorLogprobError(ValueError):
    """Inference did not provide usable probabilities for its generated tokens."""


def recorded_turn(prompt_token_ids, meta_info, policy_version):
    pairs = meta_info.get("output_token_logprobs")
    if not pairs or len(pairs) != meta_info.get("completion_tokens"):
        raise BehaviorLogprobError("Missing or incomplete SGLang output token logprobs")
    if any(p[0] is None or not math.isfinite(p[0]) for p in pairs):
        raise BehaviorLogprobError("Non-finite SGLang behavior logprob")
    return {
        "prompt_token_ids": list(prompt_token_ids),
        "output_token_ids": [p[1] for p in pairs],
        "log_probs": [p[0] for p in pairs],
        "policy_version": policy_version,
        "finish_reason": meta_info.get("finish_reason"),
    }


def training_segments(turns, *, record_spans=False):
    """Merge only exact prefixes; injected text always has a zero loss mask."""
    segments = []
    for turn_index, turn in enumerate(turns):
        prompt, output = turn["prompt_token_ids"], turn["output_token_ids"]
        probs = turn["log_probs"]
        if not prompt or not output or len(output) != len(probs):
            raise BehaviorLogprobError("Unaligned Agent turn tokens/logprobs")
        if segments and prompt[: len(segments[-1]["tokens"])] == segments[-1]["tokens"]:
            segment = segments[-1]
            gap = prompt[len(segment["tokens"]) :]
            segment["tokens"].extend(gap + output)
            segment["loss_mask"].extend([0] * len(gap) + [1] * len(output))
            segment["rollout_log_probs"].extend([0.0] * len(gap) + probs)
            segment["response_length"] += len(gap) + len(output)
        else:
            segments.append(
                {
                    "tokens": prompt + output,
                    "response_length": len(output),
                    "loss_mask": [1] * len(output),
                    "rollout_log_probs": list(probs),
                }
            )
        if record_spans:
            segment = segments[-1]
            end = segment["response_length"]
            segment.setdefault("assistant_spans", []).append({
                "turn_index": turn_index, "response_span": [end - len(output), end],
            })
    return segments


def expand_training_segments(samples, raw_rewards, rewards):
    """Expand AFTER K-trajectory GRPO normalization; preserve rollout identity."""
    expanded, expanded_raw, expanded_rewards = [], [], []
    for sample, raw, advantage in zip(samples, raw_rewards, rewards, strict=True):
        for segment in sample.training_segments or [None]:
            part = copy.copy(sample)
            part.rollout_id = sample.rollout_id if sample.rollout_id is not None else sample.index
            if segment is not None:
                for key, value in segment.items():
                    if key != "assistant_spans":
                        setattr(part, key, value)
            expanded.append(part)
            expanded_raw.append(raw)
            expanded_rewards.append(advantage)
    return expanded, expanded_raw, expanded_rewards
