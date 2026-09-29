"""Normalize comparable goal-progress stages with one contribution per trajectory."""

import json
import math
import os
from collections import defaultdict
from pathlib import Path

import reward_postprocess as credit


def goal_stage_advantages(score_sequences, gamma=0.98):
    gamma = float(gamma)
    if not math.isfinite(gamma) or not 0 <= gamma <= 1:
        raise ValueError("gamma must be finite and in [0, 1]")
    rtgs, buckets = [], defaultdict(lambda: defaultdict(list))
    for trajectory, scores in enumerate(score_sequences):
        values, running = [0.0] * (len(scores) - 1), 0.0
        for turn in range(len(values) - 1, -1, -1):
            running = scores[turn + 1] - scores[turn] + gamma * running
            values[turn] = running
            buckets[scores[turn]][trajectory].append(turn)
        rtgs.append(values)
    advantages = [[0.0] * len(values) for values in rtgs]
    for members in buckets.values():
        means = {i: sum(rtgs[i][t] for t in turns) / len(turns) for i, turns in members.items()}
        n = len(means)
        if n < 2:
            continue
        mean = sum(means.values()) / n
        std = math.sqrt(sum((v - mean) ** 2 for v in means.values()) / (n - 1))
        if std <= 1e-6:
            continue
        for i, turns in members.items():
            value = (means[i] - mean) / (std + 1e-6)
            for turn in turns:
                advantages[i][turn] = value
    return rtgs, advantages, buckets


def attach_advantages(args, samples):
    k = int(args.n_samples_per_prompt)
    if len(samples) % k:
        raise credit.TurnCreditAlignmentError("Goal credit requires complete task groups")
    if args.advantage_estimator != "grpo" or not args.rewards_normalization or not args.grpo_std_normalization:
        raise ValueError("Goal credit requires normalized GRPO with sample std")
    weight = float(os.environ.get("TAU2_PROGRESS_WEIGHT", "1.0"))
    format_weight = float(os.environ.get("TAU2_FORMAT_WEIGHT", "1.0"))
    gamma = float(os.environ.get("TAU2_PROGRESS_GAMMA", "0.98"))
    for start in range(0, len(samples), k):
        group = samples[start:start + k]
        identities = {(s.group_index, s.metadata.get("tau2_domain"), s.metadata.get("tau2_task_id")) for s in group}
        if len(identities) != 1 or any(s.remove_sample for s in group):
            raise credit.TurnCreditAlignmentError("Goal credit received mixed or invalid samples")
        outcome = credit._group_normalize([float(s.get_reward_value(args)) for s in group], valid_mask=[1.0] * k,
                                         n_samples_per_prompt=k, apply_std=True)
        payloads = [sample.metadata["tau2_progress"] for sample in group]
        details = [sample.metadata["tau2_turn_credits"] for sample in group]
        for turns, payload in zip(details, payloads):
            if [turn["simulation_message_index"] for turn in turns] != payload["source_indices"] or len(payload["scores"]) != len(turns) + 1:
                raise credit.TurnCreditAlignmentError("Goal boundaries and Assistant spans do not align")
        reasons = sorted({payload["unavailable_reason"] for payload in payloads if payload["unavailable_reason"]})
        if reasons:
            rtgs = [[None] * len(turns) for turns in details]
            dense = [[0.0] * len(turns) for turns in details]
            buckets = {}
        else:
            rtgs, dense, buckets = goal_stage_advantages([payload["scores"] for payload in payloads], gamma)
        group_signal = False
        for i, sample in enumerate(group):
            values = []
            for t, detail in enumerate(details[i]):
                local = -0.1 * float(any(error["type"] in {"malformed_json", "nonexistent_tool", "wrong_namespace_tool"} for error in detail["errors"]))
                advantage = outcome[i] + weight * dense[i][t] + format_weight * local
                stage = payloads[i]["scores"][t]
                members = buckets.get(stage, {})
                detail.update(outcome_advantage=outcome[i], progress_rtg=rtgs[i][t], progress_advantage=dense[i][t],
                              format_penalty=local, progress_gamma=gamma, advantage=advantage,
                              progress_reward=payloads[i]["scores"][t + 1] - stage if not reasons else None,
                              db_diff_count=payloads[i]["db_diff_counts"][t],
                              db_diff_side_counts=payloads[i]["db_diff_side_counts"][t],
                              goal_stage=stage, goal_stage_trajectories=len(members),
                              goal_stage_turns=sum(len(turns) for turns in members.values()))
                values.append(advantage)
            metadata = {"turn_credit_version": credit.PROGRESS_DB_COUNT_V1, "credit_recipe": "goal-progress-v4"}
            if sample.training_segments:
                tokens = []
                for segment_index, segment in enumerate(sample.training_segments):
                    segment_tokens = [0.0] * segment["response_length"]
                    for detail, value in zip(details[i], values):
                        if detail["segment_index"] == segment_index:
                            begin, end = detail["response_span"]
                            segment_tokens[begin:end] = [value] * (end - begin)
                    credit.validate_token_advantages(segment_tokens, response_length=segment["response_length"], loss_mask=segment["loss_mask"])
                    segment["train_metadata"] = dict(metadata, token_advantages=segment_tokens)
                    tokens.extend(segment_tokens)
                sample.train_metadata = sample.training_segments[0]["train_metadata"]
            else:
                tokens = credit.attach_turn_credit_v2_token_advantages(sample, values)
                sample.train_metadata = dict(metadata, token_advantages=tokens)
            sample.metadata["tau2_progress_group_unavailable_reasons"] = reasons
            sample.metadata["tau2_binary_zero_variance_group"] = not any(outcome)
            group_signal |= any(value != 0 for value in tokens)
        for sample in group:
            sample.metadata["tau2_turn_credit_group_has_signal"] = group_signal

    dump_path = os.environ.get("TAU2_PROGRESS_DIAGNOSTICS_PATH")
    if dump_path:
        path = Path(dump_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as stream:
            for sample in samples:
                stream.write(json.dumps(dict(
                    group_index=sample.group_index, sample_index=sample.index,
                    task_id=sample.metadata.get("tau2_task_id"), domain=sample.metadata.get("tau2_domain"),
                    reward=float(sample.get_reward_value(args)), credit_recipe="goal-progress-v4",
                    group_has_signal=sample.metadata["tau2_turn_credit_group_has_signal"],
                    trainable_tokens=sample.metadata.get("tau2_response_tokens", sum(sample.loss_mask)),
                    binary_zero_variance=sample.metadata["tau2_binary_zero_variance_group"],
                    progress_unavailable_reasons=sample.metadata["tau2_progress_group_unavailable_reasons"],
                    scoring_seconds=sample.metadata.get("tau2_progress_scoring_seconds", 0),
                    rollout_attempts=sample.metadata.get("tau2_rollout_attempts", 1),
                    turns=sample.metadata["tau2_turn_credits"],
                ), ensure_ascii=False) + "\n")
