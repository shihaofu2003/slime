"""Reward shaping and turn-level credit for tau2-bench GRPO rollouts.

Slime entry point::

    --custom-reward-post-process-path reward_postprocess.tau2_reward_post_process

Why this exists
---------------
tau2's official reward is the binary product of the task's configured
evaluation components (DB/environment state, reference actions, communication,
or natural-language checks, depending on the task).
On airline ~70% of failed trajectories fail *only* on the DB check — the agent
calls the right tools but with slightly wrong arguments — yet a binary reward
scores every failure as 0, indistinguishable from a trajectory that got nothing
right. With small batches (48) this makes GRPO's group-relative advantage
near-zero for most groups and the reward curve stays flat (verified on the
airline run: 98.6% of failed trajectories had ``action_match > 0``).

``turn-credit-v1`` adds field-level **partial credit** mined from rollout
metadata and tau2's ``reward_info``:

    global_score = task_reward + alpha(domain) * partial_score

``partial_score`` separately credits golden tool names, every compared
argument, the final DB assertion, environment assertions, and communication
checks. Behavior errors are attributed to the Assistant turn that emitted them
and converted to a response-length token penalty vector. The custom advantage
function subtracts that absolute local penalty *after* the global score is
group-normalized, so a group where every trajectory makes the same mistake
still receives a negative correction on the responsible turns.

Field shaping makes many raw-binary-zero-variance groups trainable. Groups
whose *shaped* rewards are still zero-variance remain uninformative, so the
dynamic sampler in :mod:`filters` requests replacement groups and excludes
unreplaced groups, alongside permanently-too-long groups.

``turn-credit-v2`` keeps the official binary outcome as the sole task reward.
It adds an AReW-style, zero-sum turn modifier: exact reference actions are
positive directional critiques, rule violations are negative critiques, and a
fixed budget is redistributed between the two sets.  The modifier is unitless
and has one explicit weight, so raw penalty magnitudes are never subtracted
from an already standardized trajectory score.

Weights/alpha are env-var configurable (defaults follow the public tau2 slime
cookbook, ``examples/tau-bench/tau-bench-example/tau2``).  V2's reallocation
weight is separately configurable.
"""

from __future__ import annotations

import json
import math
import os
from collections import Counter
from typing import Any

import torch

from slime.utils.types import Sample


TURN_CREDIT_V1 = "turn-credit-v1"
TURN_CREDIT_V2 = "turn-credit-v2"
# Backward-compatible name used by the selected v1 scripts and tests.
TURN_CREDIT_VERSION = TURN_CREDIT_V1
SUPPORTED_TURN_CREDIT_VERSIONS = frozenset({TURN_CREDIT_V1, TURN_CREDIT_V2})
_TURN_PENALTY_SPECS = {
    "nonexistent_tool": (0.15, 0.30),
    "malformed_json": (0.15, 0.30),
    "wrong_argument_field": (0.10, 0.40),
    "tool_execution_error": (0.25, 0.50),
    "repetition": (0.05, 0.20),
    "max_steps": (0.20, 0.20),
}
_TURN_CREDIT_V2_RULE_ERRORS = frozenset(
    {
        "wrong_namespace_tool",
        "nonexistent_tool",
        "malformed_json",
        "tool_execution_error",
        "repetition",
    }
)


class TurnCreditAlignmentError(ValueError):
    """A local error signal cannot be mapped safely to trainable tokens."""


def configured_turn_credit_version() -> str:
    version = os.environ.get("TAU2_TURN_CREDIT_VERSION", "")
    if version and version not in SUPPORTED_TURN_CREDIT_VERSIONS:
        raise ValueError(
            f"Unsupported TAU2_TURN_CREDIT_VERSION={version!r}; expected one of "
            f"{sorted(SUPPORTED_TURN_CREDIT_VERSIONS)}"
        )
    return version


def _turn_credit_v2_reallocation_weight() -> float:
    weight = float(os.environ.get("TAU2_TURN_CREDIT_REALLOCATION_WEIGHT", "0.1"))
    if weight < 0.0:
        raise ValueError(
            "TAU2_TURN_CREDIT_REALLOCATION_WEIGHT must be non-negative, "
            f"got {weight}"
        )
    return weight


def _alpha(domain: str | None) -> float:
    base = float(os.environ.get("TAU2_REWARD_ALPHA", "0.25"))
    if os.environ.get("TAU2_DOMAIN_ADAPTIVE_ALPHA", "1") != "1" or not domain:
        return base
    # Telecom carries extra dual-control communication overhead.
    mult = {"retail": 0.8, "airline": 1.0, "telecom": 1.6}.get(domain, 1.0)
    return base * mult


def _partial_weights(domain: str | None) -> dict[str, float]:
    tool_name = float(os.environ.get("TAU2_PARTIAL_TOOL_NAME_WEIGHT", "0.25"))
    argument = float(os.environ.get("TAU2_PARTIAL_ARGUMENT_WEIGHT", "0.35"))
    communicate = float(os.environ.get("TAU2_PARTIAL_COMMUNICATE_WEIGHT", "0.05"))
    env_assertion = float(os.environ.get("TAU2_PARTIAL_ENV_ASSERTION_WEIGHT", "0.10"))
    db = float(os.environ.get("TAU2_PARTIAL_DB_WEIGHT", "0.25"))
    if domain == "telecom" and os.environ.get("TAU2_TELECOM_COMMUNICATION_BOOST", "1") == "1":
        return {
            "tool_name": 0.20,
            "argument": 0.25,
            "communicate": 0.20,
            "env_assertion": 0.15,
            "db": 0.20,
        }
    return {
        "action": 0.5,
        "tool_name": tool_name,
        "argument": argument,
        "communicate": communicate,
        "env_assertion": env_assertion,
        "db": db,
    }


def _canonical_call(call: dict[str, Any]) -> str:
    return json.dumps(
        {"name": call.get("name"), "arguments": call.get("arguments") or {}},
        sort_keys=True,
        ensure_ascii=False,
        default=str,
    )


def compute_field_reward_signals(
    *,
    golden_actions: list[dict[str, Any]],
    predicted_calls: list[dict[str, Any]],
    attempted_calls: list[dict[str, Any]] | None = None,
    allowed_tool_names: set[str],
    user_tool_names: set[str] | None = None,
    parse_error_count: int,
    termination_reason: str | None,
    reward_info: dict[str, Any] | None,
    parse_error_turns: list[int] | None = None,
    last_trainable_turn_index: int | None = None,
) -> dict[str, Any]:
    """Score tool name and every compared argument with one-to-one matching."""
    unmatched = set(range(len(predicted_calls)))
    tool_matches = 0
    argument_matches = 0
    argument_total = 0
    matched_argument_fields = 0
    wrong_argument_fields = 0
    action_details = []
    turn_errors: list[dict[str, Any]] = []

    for gold in golden_actions:
        compare_args = gold.get("compare_args")
        if compare_args is None:
            compare_args = list((gold.get("arguments") or {}).keys())

        def candidate_score(index: int) -> tuple[int, int, int]:
            predicted = predicted_calls[index]
            name_match = predicted.get("name") == gold.get("name")
            matched_args = sum(
                (predicted.get("arguments") or {}).get(key)
                == (gold.get("arguments") or {}).get(key)
                for key in compare_args
            )
            full_argument_matches = sum(
                (predicted.get("arguments") or {}).get(key) == value
                for key, value in (gold.get("arguments") or {}).items()
            )
            return (
                int(name_match),
                matched_args if name_match else 0,
                full_argument_matches if name_match else 0,
            )

        best_index = max(unmatched, key=candidate_score, default=None)
        name_match = best_index is not None and candidate_score(best_index)[0] == 1
        predicted_args = (
            predicted_calls[best_index].get("arguments") or {} if name_match else {}
        )
        if name_match:
            unmatched.remove(best_index)
            tool_matches += 1

        per_argument = {
            key: bool(name_match and predicted_args.get(key) == (gold.get("arguments") or {}).get(key))
            for key in compare_args
        }
        all_arguments = {
            key: bool(name_match and predicted_args.get(key) == value)
            for key, value in (gold.get("arguments") or {}).items()
        }
        argument_matches += sum(per_argument.values())
        argument_total += len(per_argument)
        # Missing/wrong arguments are an argument error only when the model
        # selected the correct tool. A missing tool or a different tool is
        # already represented by the tool-name component and must not be
        # double-counted as one error per golden argument.
        if name_match:
            matched_argument_fields += len(per_argument)
            wrong_argument_fields += sum(
                int(not matched) for matched in per_argument.values()
            )
            predicted = predicted_calls[best_index]
            for field, matched in per_argument.items():
                if not matched:
                    turn_errors.append(
                        {
                            "type": "wrong_argument_field",
                            "turn_index": predicted.get("turn_index"),
                            "tool": predicted.get("name"),
                            "field": field,
                        }
                    )
        action_details.append(
            {
                "tool": gold.get("name"),
                "tool_name_match": name_match,
                "arguments": per_argument,
                "all_arguments": all_arguments,
                "predicted_turn_index": (
                    predicted_calls[best_index].get("turn_index")
                    if name_match
                    else None
                ),
            }
        )

    seen: set[str] = set()
    repetition_count = 0
    for call in predicted_calls:
        key = _canonical_call(call)
        repeated = key in seen
        repetition_count += int(repeated)
        if repeated:
            turn_errors.append(
                {
                    "type": "repetition",
                    "turn_index": call.get("turn_index"),
                    "tool": call.get("name"),
                    "call": key,
                }
            )
        seen.add(key)

    if parse_error_turns is None:
        parse_error_turns_with_orphans: list[int | None] = [None] * int(parse_error_count)
    else:
        if len(parse_error_turns) != int(parse_error_count):
            raise ValueError(
                "parse_error_turns must identify every parse error: "
                f"count={parse_error_count}, turns={parse_error_turns}"
            )
        parse_error_turns_with_orphans = list(parse_error_turns)
    turn_errors.extend(
        {"type": "malformed_json", "turn_index": turn_index}
        for turn_index in parse_error_turns_with_orphans
    )

    reward_info = reward_info or {}
    db_check = reward_info.get("db_check") or {}
    env_assertions = reward_info.get("env_assertions") or []
    communicate_checks = reward_info.get("communicate_checks") or []
    components: dict[str, float] = {}
    if golden_actions:
        components["tool_name"] = tool_matches / len(golden_actions)
    if argument_total:
        components["argument"] = argument_matches / argument_total
    if isinstance(db_check, dict) and "db_match" in db_check:
        components["db"] = float(bool(db_check.get("db_match")))
    if env_assertions:
        components["env_assertion"] = sum(bool(item.get("met")) for item in env_assertions) / len(
            env_assertions
        )
    if communicate_checks:
        components["communicate"] = sum(
            bool(item.get("met")) for item in communicate_checks
        ) / len(communicate_checks)

    user_tool_names = user_tool_names or set()
    classification_calls = predicted_calls if attempted_calls is None else attempted_calls
    wrong_namespace_calls = [
        call
        for call in classification_calls
        if call.get("name") in user_tool_names
        and call.get("name") not in allowed_tool_names
    ]
    nonexistent_calls = [
        call
        for call in classification_calls
        if call.get("name") not in allowed_tool_names
        and call.get("name") not in user_tool_names
    ]
    tool_execution_errors = [
        call
        for call in predicted_calls
        if call.get("name") in allowed_tool_names
        and bool(call.get("execution_error"))
    ]
    turn_errors.extend(
        {
            "type": "wrong_namespace_tool",
            "turn_index": call.get("turn_index"),
            "tool": call.get("name"),
        }
        for call in wrong_namespace_calls
    )
    turn_errors.extend(
        {
            "type": "nonexistent_tool",
            "turn_index": call.get("turn_index"),
            "tool": call.get("name"),
        }
        for call in nonexistent_calls
    )
    turn_errors.extend(
        {
            "type": "tool_execution_error",
            "turn_index": call.get("turn_index"),
            "tool": call.get("name"),
            "call_id": call.get("id"),
        }
        for call in tool_execution_errors
    )
    if termination_reason == "max_steps":
        turn_errors.append(
            {
                "type": "max_steps",
                "turn_index": last_trainable_turn_index,
            }
        )
    namespace_name_counts = Counter(str(call.get("name")) for call in wrong_namespace_calls)
    namespace_turns = sorted(
        {
            int(call["turn_index"])
            for call in wrong_namespace_calls
            if isinstance(call.get("turn_index"), int)
            and not isinstance(call.get("turn_index"), bool)
        }
    )
    return {
        "components": components,
        "action_details": action_details,
        "turn_errors": turn_errors,
        "counts": {
            "golden_actions": len(golden_actions),
            "predicted_calls": len(predicted_calls),
            "argument_fields": argument_total,
            "matched_argument_fields": matched_argument_fields,
            "wrong_argument_fields": wrong_argument_fields,
            "tool_execution_error": len(tool_execution_errors),
            "malformed_json": int(parse_error_count),
            "wrong_namespace_tool": len(wrong_namespace_calls),
            "nonexistent_tool": len(nonexistent_calls),
            "repetition": repetition_count,
            "max_steps": int(termination_reason == "max_steps"),
        },
        "wrong_namespace": {
            "count": len(wrong_namespace_calls),
            "names": dict(sorted(namespace_name_counts.items())),
            "affected_turns": namespace_turns,
            "termination_reason": termination_reason,
            "namespace_attributed_termination": bool(
                wrong_namespace_calls and termination_reason == "too_many_errors"
            ),
        },
        "wrong_arguments": {
            "matched_field_count": matched_argument_fields,
            "field_count": wrong_argument_fields,
            "tool_execution_error_count": len(tool_execution_errors),
        },
    }


def _turn_penalty(errors: list[dict[str, Any]]) -> tuple[float, dict[str, float]]:
    counts = Counter(str(error.get("type")) for error in errors)
    known_types = {"wrong_namespace_tool", *_TURN_PENALTY_SPECS}
    unknown_types = set(counts) - known_types
    if unknown_types:
        raise TurnCreditAlignmentError(
            f"Unsupported turn error types: {sorted(unknown_types)}"
        )

    namespace_count = counts.get("wrong_namespace_tool", 0)
    components = {
        "wrong_namespace_tool": (
            min(2.0, 1.0 + 0.1 * (namespace_count - 1))
            if namespace_count
            else 0.0
        )
    }
    for error_type, (per_event, cap) in _TURN_PENALTY_SPECS.items():
        components[error_type] = min(per_event * counts.get(error_type, 0), cap)
    return float(sum(components.values())), components


def validate_token_penalties(
    token_penalties: Any,
    *,
    response_length: int,
    loss_mask: Any,
) -> list[float]:
    """Validate the response-aligned training contract without repairing it."""

    if torch.is_tensor(token_penalties):
        token_penalties = token_penalties.detach().cpu().reshape(-1).tolist()
    if torch.is_tensor(loss_mask):
        loss_mask = loss_mask.detach().cpu().reshape(-1).tolist()
    if not isinstance(token_penalties, (list, tuple)):
        raise TurnCreditAlignmentError("token_penalties must be a response-length vector")
    if not isinstance(loss_mask, (list, tuple)):
        raise TurnCreditAlignmentError("loss_mask must be a response-length vector")
    if len(token_penalties) != response_length:
        raise TurnCreditAlignmentError(
            "token_penalties length does not match response_length: "
            f"{len(token_penalties)} != {response_length}"
        )
    if len(loss_mask) != response_length:
        raise TurnCreditAlignmentError(
            f"loss_mask length does not match response_length: {len(loss_mask)} != {response_length}"
        )

    penalties = [float(value) for value in token_penalties]
    for index, (penalty, mask) in enumerate(zip(penalties, loss_mask, strict=True)):
        if not math.isfinite(penalty) or penalty < 0:
            raise TurnCreditAlignmentError(
                f"token_penalties[{index}] must be finite and non-negative: {penalty}"
            )
        if not mask and penalty != 0.0:
            raise TurnCreditAlignmentError(
                f"token_penalties[{index}]={penalty} falls outside the loss mask"
            )
    return penalties


def validate_token_advantages(
    token_advantages: Any,
    *,
    response_length: int,
    loss_mask: Any,
) -> list[float]:
    """Validate a signed response-aligned advantage vector."""

    if torch.is_tensor(token_advantages):
        token_advantages = token_advantages.detach().cpu().reshape(-1).tolist()
    if torch.is_tensor(loss_mask):
        loss_mask = loss_mask.detach().cpu().reshape(-1).tolist()
    if not isinstance(token_advantages, (list, tuple)):
        raise TurnCreditAlignmentError(
            "token_advantages must be a response-length vector"
        )
    if not isinstance(loss_mask, (list, tuple)):
        raise TurnCreditAlignmentError("loss_mask must be a response-length vector")
    if len(token_advantages) != response_length:
        raise TurnCreditAlignmentError(
            "token_advantages length does not match response_length: "
            f"{len(token_advantages)} != {response_length}"
        )
    if len(loss_mask) != response_length:
        raise TurnCreditAlignmentError(
            f"loss_mask length does not match response_length: {len(loss_mask)} != {response_length}"
        )

    advantages = [float(value) for value in token_advantages]
    for index, (advantage, mask) in enumerate(
        zip(advantages, loss_mask, strict=True)
    ):
        if not math.isfinite(advantage):
            raise TurnCreditAlignmentError(
                f"token_advantages[{index}] must be finite: {advantage}"
            )
        if not mask and advantage != 0.0:
            raise TurnCreditAlignmentError(
                f"token_advantages[{index}]={advantage} falls outside the loss mask"
            )
    return advantages


def build_token_penalties(
    *,
    assistant_spans: list[dict[str, int | bool]],
    assistant_source_indices: dict[int, int],
    response_start: int,
    response_length: int,
    loss_mask: list[int],
    turn_errors: list[dict[str, Any]],
) -> tuple[list[float], list[dict[str, Any]]]:
    """Map every local error to exactly one trainable Assistant response span."""

    if response_start < 0:
        raise TurnCreditAlignmentError(f"response_start must be non-negative: {response_start}")
    if len(loss_mask) != response_length:
        raise TurnCreditAlignmentError(
            f"loss_mask length does not match response_length: {len(loss_mask)} != {response_length}"
        )

    covered = [False] * response_length
    details_by_source: dict[int, dict[str, Any]] = {}
    for span in assistant_spans:
        if not bool(span.get("trainable")):
            continue
        message_index = span.get("message_index")
        token_start = span.get("token_start")
        token_end = span.get("token_end")
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in (message_index, token_start, token_end)
        ):
            raise TurnCreditAlignmentError(f"Invalid Assistant span: {span}")
        if message_index not in assistant_source_indices:
            raise TurnCreditAlignmentError(
                f"Assistant message {message_index} has no simulation source index"
            )
        source_index = assistant_source_indices[message_index]
        if source_index in details_by_source:
            raise TurnCreditAlignmentError(
                f"Simulation Assistant turn {source_index} maps to multiple token spans"
            )
        response_span_start = token_start - response_start
        response_span_end = token_end - response_start
        if not 0 <= response_span_start < response_span_end <= response_length:
            raise TurnCreditAlignmentError(
                "Assistant span falls outside the response suffix: "
                f"source={source_index}, full=({token_start}, {token_end}), "
                f"response_start={response_start}, response_length={response_length}"
            )
        for token_index in range(response_span_start, response_span_end):
            if loss_mask[token_index] != 1:
                raise TurnCreditAlignmentError(
                    f"Assistant span includes loss-mask-zero token {token_index}"
                )
            if covered[token_index]:
                raise TurnCreditAlignmentError(
                    f"Assistant spans overlap at response token {token_index}"
                )
            covered[token_index] = True
        details_by_source[source_index] = {
            "simulation_message_index": source_index,
            "training_message_index": message_index,
            "response_span": [response_span_start, response_span_end],
            "errors": [],
        }

    uncovered_trainable = [
        index for index, mask in enumerate(loss_mask) if mask and not covered[index]
    ]
    if uncovered_trainable:
        raise TurnCreditAlignmentError(
            "Trainable tokens are not owned by an Assistant span: "
            f"{uncovered_trainable[:20]}"
        )

    for error in turn_errors:
        turn_index = error.get("turn_index")
        if isinstance(turn_index, bool) or not isinstance(turn_index, int):
            raise TurnCreditAlignmentError(f"Error event has no unique turn index: {error}")
        detail = details_by_source.get(turn_index)
        if detail is None:
            raise TurnCreditAlignmentError(
                f"Error event does not map to a trainable Assistant span: {error}"
            )
        detail["errors"].append(dict(error))

    token_penalties = [0.0] * response_length
    turn_details = sorted(
        details_by_source.values(),
        key=lambda detail: detail["response_span"][0],
    )
    for detail in turn_details:
        penalty, components = _turn_penalty(detail["errors"])
        detail["penalty_components"] = components
        detail["penalty"] = penalty
        span_start, span_end = detail["response_span"]
        for token_index in range(span_start, span_end):
            token_penalties[token_index] = penalty

    return (
        validate_token_penalties(
            token_penalties,
            response_length=response_length,
            loss_mask=loss_mask,
        ),
        turn_details,
    )


def build_turn_credit_v2(
    *,
    assistant_spans: list[dict[str, int | bool]],
    assistant_source_indices: dict[int, int],
    response_start: int,
    response_length: int,
    loss_mask: list[int],
    field_reward_signals: dict[str, Any],
) -> list[dict[str, Any]]:
    """Build AReW-style directional critiques for trainable Assistant turns.

    Golden actions are a reference path rather than the official success
    criterion for most tau2 tasks.  V2 therefore uses only exact tool-and-full-
    argument matches as positive *directional* critiques; partial and name-only
    matches remain diagnostics.  Rule violations take precedence and make the
    turn a negative critique.  The task utility remains the official outcome.
    """

    _, turn_details = build_token_penalties(
        assistant_spans=assistant_spans,
        assistant_source_indices=assistant_source_indices,
        response_start=response_start,
        response_length=response_length,
        loss_mask=loss_mask,
        turn_errors=list(field_reward_signals.get("turn_errors") or []),
    )
    details_by_source = {
        int(detail["simulation_message_index"]): detail for detail in turn_details
    }

    for action in field_reward_signals.get("action_details") or []:
        turn_index = action.get("predicted_turn_index")
        if isinstance(turn_index, bool) or not isinstance(turn_index, int):
            continue
        detail = details_by_source.get(turn_index)
        if detail is None:
            raise TurnCreditAlignmentError(
                f"Matched action does not map to a trainable Assistant turn: {action}"
            )
        argument_matches = list(
            (action.get("all_arguments") or action.get("arguments") or {}).values()
        )
        if argument_matches:
            argument_fraction = sum(bool(value) for value in argument_matches) / len(
                argument_matches
            )
        else:
            argument_fraction = 1.0
        if argument_fraction == 1.0:
            tier = "exact_action"
        elif argument_fraction > 0.0:
            tier = "partial_arguments"
        else:
            tier = "tool_name_only"
        detail.setdefault("matched_actions", []).append(
            {
                "tool": action.get("tool"),
                "tier": tier,
                "argument_fraction": float(argument_fraction),
                "credit": float(argument_fraction),
            }
        )

    for detail in turn_details:
        matched_actions = detail.get("matched_actions") or []
        exact_action = any(action["tier"] == "exact_action" for action in matched_actions)
        rule_errors = [
            error
            for error in detail.get("errors") or []
            if error.get("type") in _TURN_CREDIT_V2_RULE_ERRORS
        ]
        critique = -1 if rule_errors else 1 if exact_action else 0
        detail.update(
            {
                "matched_actions": matched_actions,
                "exact_reference_action": exact_action,
                "rule_errors": rule_errors,
                "critique": critique,
            }
        )
        # V1's raw magnitude is diagnostic only and must not be mistaken for
        # the v2 modifier.
        detail.pop("penalty", None)

    for detail, modifier in zip(
        turn_details,
        turn_credit_v2_modifiers(turn_details),
        strict=True,
    ):
        detail["reallocation_modifier"] = modifier
        detail["effective_critique"] = 1 if modifier > 0 else -1 if modifier < 0 else 0
        detail["complementary_neutral"] = bool(
            detail.get("critique") == 0 and modifier > 0
        )

    return turn_details


def turn_credit_v2_modifiers(turn_details: list[dict[str, Any]]) -> list[float]:
    """Allocate a fixed zero-sum budget across positive and negative turns.

    A negative-only trajectory uses non-final, rule-clean neutral turns as the
    positive counterpart.  This mirrors AReW's complementary-neutral fallback
    and lets definite format/tool violations carry local signal without
    declaring an unknown neutral turn to be task-correct.  Positive-only
    trajectories remain unchanged because treating unknown turns as negative
    would penalize valid paths that differ from the reference trajectory.
    """

    positive = [index for index, detail in enumerate(turn_details) if detail.get("critique") == 1]
    negative = [index for index, detail in enumerate(turn_details) if detail.get("critique") == -1]
    modifiers = [0.0] * len(turn_details)
    if not positive and negative:
        positive = [
            index
            for index, detail in enumerate(turn_details[:-1])
            if detail.get("critique") == 0
        ]
    if not positive or not negative:
        return modifiers
    for index in positive:
        modifiers[index] = 0.5 / len(positive)
    for index in negative:
        modifiers[index] = -0.5 / len(negative)
    return modifiers


def compute_partial_score(
    reward_info: dict[str, Any] | None,
    *,
    weights: dict[str, float] | None = None,
    normalize_over_present: bool = True,
) -> tuple[float, dict[str, float]]:
    """Fraction of satisfied golden action/env/communicate/db checks.

    Returns ``(score, components)`` where ``components`` maps each present
    sub-metric to its raw fraction in ``[0, 1]``.
    """
    reward_info = reward_info or {}
    weights = weights or _partial_weights(None)
    components: dict[str, float] = {}

    action_checks = reward_info.get("action_checks") or []
    if action_checks:
        components["action"] = sum(1 for ac in action_checks if ac.get("action_match")) / len(action_checks)

    communicate_checks = reward_info.get("communicate_checks") or []
    if communicate_checks:
        components["communicate"] = sum(1 for cc in communicate_checks if cc.get("met")) / len(communicate_checks)

    env_assertions = reward_info.get("env_assertions") or []
    if env_assertions:
        components["env_assertion"] = sum(1 for ea in env_assertions if ea.get("met")) / len(env_assertions)

    db_check = reward_info.get("db_check") or {}
    if isinstance(db_check, dict) and "db_match" in db_check:
        components["db"] = 1.0 if db_check.get("db_match") else 0.0

    if not components:
        return 0.0, {}

    if normalize_over_present:
        present = {k: weights.get(k, 0.0) for k in components}
        weight_sum = sum(present.values())
        if weight_sum <= 0:
            return 0.0, components
        score = sum(components[k] * present[k] for k in components) / weight_sum
        return float(score), components

    score = sum(components[k] * weights.get(k, 0.0) for k in components)
    return float(score), components


def compute_global_score(args, sample: Sample) -> tuple[float, dict[str, Any]]:
    """Compute positive trajectory credit before GRPO group normalization."""
    task_reward = float(sample.get_reward_value(args))
    metadata = sample.metadata or {}
    domain = metadata.get("tau2_domain") or metadata.get("domain")
    signals = metadata.get("tau2_field_reward_signals") or {}
    components = signals.get("components") or {}

    # Backward-compatible fallback for old trajectory fixtures.
    if not components:
        partial_score, components = compute_partial_score(
            metadata.get("tau2_reward_info") or metadata.get("reward_info") or {},
            weights={"action": 0.5, "communicate": 0.15, "env_assertion": 0.35, "db": 0.0},
        )
    else:
        weights = _partial_weights(domain)
        present_weight = sum(weights.get(key, 0.0) for key in components)
        partial_score = (
            sum(value * weights.get(key, 0.0) for key, value in components.items())
            / present_weight
            if present_weight > 0
            else 0.0
        )

    global_score = task_reward + _alpha(domain) * partial_score
    details = {
        "raw_reward": task_reward,
        "partial_score": partial_score,
        "partial_components": components,
        "tau2_global_score": global_score,
        "shaped_reward": global_score,
    }
    return float(global_score), details


def _legacy_trajectory_penalty(signals: dict[str, Any]) -> tuple[float, dict[str, float]]:
    """Compatibility path for existing non-turn-aware tau2 run scripts."""

    counts = signals.get("counts") or {}
    penalties = {
        "wrong_argument_fields": min(
            float(os.environ.get("TAU2_PENALTY_WRONG_ARGUMENT_FIELD", "0.10"))
            * counts.get("wrong_argument_fields", 0),
            0.40,
        ),
        "tool_execution_error": min(
            float(os.environ.get("TAU2_PENALTY_TOOL_EXECUTION_ERROR", "0.25"))
            * counts.get("tool_execution_error", 0),
            0.50,
        ),
        "malformed_json": min(
            float(os.environ.get("TAU2_PENALTY_MALFORMED_JSON", "0.15"))
            * counts.get("malformed_json", 0),
            0.30,
        ),
        "nonexistent_tool": min(
            float(os.environ.get("TAU2_PENALTY_NONEXISTENT_TOOL", "0.15"))
            * counts.get("nonexistent_tool", 0),
            0.30,
        ),
        "repetition": min(
            float(os.environ.get("TAU2_PENALTY_REPETITION", "0.05"))
            * counts.get("repetition", 0),
            0.20,
        ),
        "max_steps": float(os.environ.get("TAU2_PENALTY_MAX_STEPS", "0.20"))
        * int(bool(counts.get("max_steps"))),
    }
    return sum(penalties.values()), penalties


def compute_rollout_score(args, sample: Sample) -> tuple[float, dict[str, Any]]:
    """Select turn-aware global credit or the legacy trajectory shaper."""

    version = configured_turn_credit_version()
    if version == TURN_CREDIT_V1:
        return compute_global_score(args, sample)
    if version == TURN_CREDIT_V2:
        task_reward = float(sample.get_reward_value(args))
        _, field_details = compute_global_score(args, sample)
        return task_reward, {
            "raw_reward": task_reward,
            "partial_score_diagnostic": field_details["partial_score"],
            "partial_components_diagnostic": field_details["partial_components"],
            "tau2_global_score": task_reward,
            "shaped_reward": task_reward,
            "tau2_reward_recipe": TURN_CREDIT_V2,
        }

    global_score, details = compute_global_score(args, sample)
    metadata = sample.metadata or {}
    signals = metadata.get("tau2_field_reward_signals") or {}
    domain = metadata.get("tau2_domain") or metadata.get("domain")
    penalty, penalty_components = _legacy_trajectory_penalty(signals)
    base_shaped = global_score - _alpha(domain) * penalty
    wrong_namespace_count = int(
        (signals.get("counts") or {}).get("wrong_namespace_tool", 0)
    )
    namespace_penalty = (
        min(2.0, 1.0 + 0.1 * (wrong_namespace_count - 1))
        if wrong_namespace_count
        else 0.0
    )
    shaped = (
        min(base_shaped, 0.0) - namespace_penalty
        if wrong_namespace_count
        else base_shaped
    )
    details.update(
        {
            "penalty": penalty,
            "penalty_components": penalty_components,
            "base_shaped_reward": base_shaped,
            "wrong_namespace_tool": wrong_namespace_count,
            "namespace_penalty": namespace_penalty,
            "shaped_reward": shaped,
        }
    )
    return float(shaped), details


def _flatten(samples: list[Sample] | list[list[Sample]]) -> list[Sample]:
    if not samples:
        return []
    if isinstance(samples[0], list):
        return [s for group in samples for s in group]
    return list(samples)


def _group_normalize(
    rewards: list[float],
    *,
    valid_mask: list[float],
    n_samples_per_prompt: int,
    apply_std: bool,
) -> list[float]:
    """Per-prompt mean-center (and optional std) matching slime's default GRPO path.

    Invalid samples (``valid_mask == 0``) are excluded from the mean/std and get
    a centered reward of 0, so they contribute no advantage (their loss mask is
    already 0 from rollout).
    """
    if not rewards:
        return []
    if n_samples_per_prompt <= 0:
        raise ValueError(f"n_samples_per_prompt must be >= 1 (got {n_samples_per_prompt})")
    if len(rewards) % n_samples_per_prompt != 0:
        raise ValueError(
            "Reward count must be a multiple of n_samples_per_prompt "
            f"(count={len(rewards)}, n_samples_per_prompt={n_samples_per_prompt})."
        )

    rewards_t = torch.tensor(rewards, dtype=torch.float).view(-1, n_samples_per_prompt)
    mask_t = torch.tensor(valid_mask, dtype=torch.float).view(-1, n_samples_per_prompt)

    denom = mask_t.sum(dim=-1, keepdim=True).clamp(min=1.0)
    mean = (rewards_t * mask_t).sum(dim=-1, keepdim=True) / denom
    centered = (rewards_t - mean) * mask_t

    if apply_std:
        # Match slime's existing GRPO ``torch.std`` normalization exactly for
        # an all-valid group (sample standard deviation, correction=1).
        # Invalid rows are excluded from both the numerator and correction.
        variance_denom = (denom - 1.0).clamp(min=1.0)
        masked_std = torch.sqrt(
            (centered**2).sum(dim=-1, keepdim=True) / variance_denom
        )
        existing_grpo_std = centered.std(dim=-1, keepdim=True)
        all_valid = mask_t.bool().all(dim=-1, keepdim=True)
        std = torch.where(all_valid, existing_grpo_std, masked_std)
        centered = centered / (std + 1e-6)

    return centered.flatten().tolist()


def _arg_or_default(args, name: str, default: Any) -> Any:
    value = getattr(args, name, default)
    return default if value is None else value


def compute_turn_credit_v2_group_advantages(
    args,
    samples: list[Sample],
) -> list[list[float]]:
    """Combine official outcome advantage with reducer-scaled turn modifiers."""

    if not samples:
        return []
    valid_mask = [0.0 if sample.remove_sample else 1.0 for sample in samples]
    outcomes = [
        float(sample.get_reward_value(args)) if valid else 0.0
        for sample, valid in zip(samples, valid_mask, strict=True)
    ]
    if bool(_arg_or_default(args, "rewards_normalization", True)):
        outcome_advantages = _group_normalize(
            outcomes,
            valid_mask=valid_mask,
            n_samples_per_prompt=len(samples),
            apply_std=bool(_arg_or_default(args, "grpo_std_normalization", True)),
        )
    else:
        outcome_advantages = outcomes

    reallocation_weight = _turn_credit_v2_reallocation_weight()
    all_advantages: list[list[float]] = []
    for sample, valid, outcome_advantage in zip(
        samples, valid_mask, outcome_advantages, strict=True
    ):
        if not valid:
            all_advantages.append([])
            continue
        turn_details = (sample.metadata or {}).get("tau2_turn_credits")
        if not isinstance(turn_details, list):
            raise TurnCreditAlignmentError(
                "turn-credit-v2 sample is missing tau2_turn_credits"
            )
        trainable_token_count = sum(bool(value) for value in sample.loss_mask)
        modifiers = turn_credit_v2_modifiers(turn_details)
        turn_advantages = []
        for detail, modifier in zip(turn_details, modifiers, strict=True):
            span_start, span_end = detail["response_span"]
            turn_token_count = span_end - span_start
            # Tau2 has one training sample per rollout, so slime's policy-loss
            # reducer divides this sample by trainable_token_count. Multiplying
            # here makes the reducer-level local L1 budget equal to the explicit
            # reallocation weight instead of silently shrinking it by 1 / N.
            local_advantage = float(
                reallocation_weight
                * trainable_token_count
                * modifier
                / turn_token_count
            )
            advantage = float(
                outcome_advantage
                + local_advantage
            )
            detail["reallocation_modifier"] = modifier
            detail["weighted_reallocation_budget"] = reallocation_weight * modifier
            detail["per_token_reallocation_advantage"] = local_advantage
            detail["effective_critique"] = 1 if modifier > 0 else -1 if modifier < 0 else 0
            detail["complementary_neutral"] = bool(
                detail.get("critique") == 0 and modifier > 0
            )
            detail["outcome_advantage"] = float(outcome_advantage)
            detail["advantage"] = advantage
            turn_advantages.append(advantage)
        sample.metadata["tau2_turn_credit_summary"] = {
            "outcome_advantage": float(outcome_advantage),
            "reallocation_weight": reallocation_weight,
            "trainable_tokens": trainable_token_count,
            "raw_positive_turns": sum(detail.get("critique") == 1 for detail in turn_details),
            "raw_negative_turns": sum(detail.get("critique") == -1 for detail in turn_details),
            "effective_positive_turns": sum(value > 0 for value in modifiers),
            "effective_negative_turns": sum(value < 0 for value in modifiers),
            "complementary_neutral_turns": sum(
                detail.get("critique") == 0 and modifier > 0
                for detail, modifier in zip(turn_details, modifiers, strict=True)
            ),
            "modifier_sum": float(sum(modifiers)),
            "modifier_abs_sum": float(sum(abs(value) for value in modifiers)),
            "weighted_modifier_sum": float(
                reallocation_weight * sum(modifiers)
            ),
            "weighted_modifier_abs_sum": float(
                reallocation_weight * sum(abs(value) for value in modifiers)
            ),
        }
        all_advantages.append(turn_advantages)
    return all_advantages


def turn_credit_v2_group_has_signal(args, samples: list[Sample]) -> bool:
    advantages = compute_turn_credit_v2_group_advantages(args, samples)
    return any(
        abs(advantage) > 1e-6
        for sample_advantages in advantages
        for advantage in sample_advantages
    )


def attach_turn_credit_v2_token_advantages(
    sample: Sample,
    turn_advantages: list[float],
) -> list[float]:
    turn_details = (sample.metadata or {}).get("tau2_turn_credits")
    if not isinstance(turn_details, list) or len(turn_details) != len(turn_advantages):
        raise TurnCreditAlignmentError(
            "turn-credit-v2 turn detail and advantage counts do not match"
        )
    token_advantages = [0.0] * int(sample.response_length)
    for detail, advantage in zip(turn_details, turn_advantages, strict=True):
        span = detail.get("response_span")
        if (
            not isinstance(span, list)
            or len(span) != 2
            or any(isinstance(value, bool) or not isinstance(value, int) for value in span)
        ):
            raise TurnCreditAlignmentError(f"Invalid v2 response span: {span}")
        span_start, span_end = span
        if not 0 <= span_start < span_end <= sample.response_length:
            raise TurnCreditAlignmentError(
                f"V2 response span falls outside the response: {span}"
            )
        for token_index in range(span_start, span_end):
            token_advantages[token_index] = advantage
    return validate_token_advantages(
        token_advantages,
        response_length=int(sample.response_length),
        loss_mask=sample.loss_mask,
    )


def tau2_reward_post_process(
    args, samples: list[Sample] | list[list[Sample]]
) -> tuple[list[float], list[float]]:
    """Return ``(raw_rewards, group_normalized_global_scores)`` for slime GRPO.

    ``raw_rewards`` is the binary task reward (for ``rollout/raw_reward``
    logging); the second element is what feeds the advantage.
    """
    flat = _flatten(samples)
    version = configured_turn_credit_version()

    raw_rewards: list[float] = []
    global_scores: list[float] = []
    valid_mask: list[float] = []

    for sample in flat:
        task_reward = float(sample.get_reward_value(args))
        raw_rewards.append(task_reward)

        is_valid = not getattr(sample, "remove_sample", False)
        valid_mask.append(1.0 if is_valid else 0.0)

        global_score, details = compute_rollout_score(args, sample)
        global_scores.append(float(global_score) if is_valid else 0.0)

        sample.metadata.update(details)

    advantage_estimator = _arg_or_default(args, "advantage_estimator", "grpo")
    rewards_normalization = bool(_arg_or_default(args, "rewards_normalization", True))
    grpo_std_normalization = bool(
        _arg_or_default(args, "grpo_std_normalization", True)
    )
    n_samples_per_prompt = int(_arg_or_default(args, "n_samples_per_prompt", 1))

    if version == TURN_CREDIT_V2:
        if advantage_estimator != "grpo":
            raise ValueError("turn-credit-v2 requires --advantage-estimator grpo")
        if len(flat) % n_samples_per_prompt != 0:
            raise ValueError(
                "turn-credit-v2 sample count must be a multiple of "
                f"n_samples_per_prompt: {len(flat)} vs {n_samples_per_prompt}"
            )
        for group_start in range(0, len(flat), n_samples_per_prompt):
            group = flat[group_start : group_start + n_samples_per_prompt]
            group_advantages = compute_turn_credit_v2_group_advantages(args, group)
            for sample, turn_advantages in zip(group, group_advantages, strict=True):
                if sample.remove_sample:
                    token_advantages = [0.0] * int(sample.response_length)
                else:
                    token_advantages = attach_turn_credit_v2_token_advantages(
                        sample, turn_advantages
                    )
                sample.train_metadata = {
                    "turn_credit_version": TURN_CREDIT_V2,
                    "token_advantages": token_advantages,
                }

    if (
        advantage_estimator in {"grpo", "gspo", "reinforce_plus_plus_baseline"}
        and rewards_normalization
        and global_scores
    ):
        normalized = _group_normalize(
            global_scores,
            valid_mask=valid_mask,
            n_samples_per_prompt=n_samples_per_prompt,
            apply_std=(advantage_estimator in {"grpo", "gspo"} and grpo_std_normalization),
        )
        return raw_rewards, normalized

    return raw_rewards, global_scores


def turn_aware_grpo_advantage(args, rollout_data: dict[str, Any]) -> None:
    """Materialize v1 penalties or consume v2 response-token advantages."""

    if getattr(args, "advantage_estimator", None) != "grpo":
        raise ValueError("turn-aware tau2 credit requires --advantage-estimator grpo")

    metadata = rollout_data.pop("metadata", None)
    if not isinstance(metadata, list):
        raise TurnCreditAlignmentError(
            "turn-aware tau2 credit requires per-sample train metadata"
        )

    rewards = rollout_data.get("rewards")
    kl = rollout_data.get("kl")
    response_lengths = rollout_data.get("response_lengths")
    total_lengths = rollout_data.get("total_lengths")
    loss_masks = rollout_data.get("loss_masks")
    fields = {
        "metadata": metadata,
        "rewards": rewards,
        "kl": kl,
        "response_lengths": response_lengths,
        "total_lengths": total_lengths,
        "loss_masks": loss_masks,
    }
    lengths = {
        key: len(value) if isinstance(value, (list, tuple)) else None
        for key, value in fields.items()
    }
    if None in lengths.values() or len(set(lengths.values())) != 1:
        raise TurnCreditAlignmentError(
            f"turn-aware rollout fields have inconsistent sample counts: {lengths}"
        )

    from slime.backends.megatron_utils.cp_utils import slice_log_prob_with_cp

    advantages: list[torch.Tensor] = []
    for sample_index, (
        sample_metadata,
        reward,
        sample_kl,
        response_length,
        total_length,
        loss_mask,
    ) in enumerate(
        zip(
            metadata,
            rewards,
            kl,
            response_lengths,
            total_lengths,
            loss_masks,
            strict=True,
        )
    ):
        if not isinstance(sample_metadata, dict):
            raise TurnCreditAlignmentError(
                f"train metadata for sample {sample_index} is not a dictionary"
            )
        version = sample_metadata.get("turn_credit_version", TURN_CREDIT_V1)
        if version == TURN_CREDIT_V2:
            token_advantages = validate_token_advantages(
                sample_metadata.get("token_advantages"),
                response_length=int(response_length),
                loss_mask=loss_mask,
            )
            full_advantages = torch.tensor(
                token_advantages,
                dtype=torch.float32,
                device=sample_kl.device,
            )
            base_advantage = slice_log_prob_with_cp(
                full_advantages,
                int(total_length),
                int(response_length),
            )
            if base_advantage.shape != sample_kl.shape:
                raise TurnCreditAlignmentError(
                    "CP-sliced token advantages do not match the local log-prob shape: "
                    f"sample={sample_index}, advantages={tuple(base_advantage.shape)}, "
                    f"log_probs={tuple(sample_kl.shape)}"
                )
            advantages.append(base_advantage)
            continue
        if version != TURN_CREDIT_V1:
            raise TurnCreditAlignmentError(
                f"Unsupported train turn-credit version for sample {sample_index}: {version!r}"
            )

        penalties = validate_token_penalties(
            sample_metadata.get("token_penalties"),
            response_length=int(response_length),
            loss_mask=loss_mask,
        )
        base_advantage = torch.ones_like(sample_kl, dtype=torch.float32) * float(reward)
        if any(penalties):
            full_penalties = torch.tensor(
                penalties,
                dtype=torch.float32,
                device=sample_kl.device,
            )
            local_penalties = slice_log_prob_with_cp(
                full_penalties,
                int(total_length),
                int(response_length),
            )
            if local_penalties.shape != base_advantage.shape:
                raise TurnCreditAlignmentError(
                    "CP-sliced token penalties do not match the local advantage shape: "
                    f"sample={sample_index}, penalties={tuple(local_penalties.shape)}, "
                    f"advantage={tuple(base_advantage.shape)}"
                )
            base_advantage = base_advantage - local_penalties
        advantages.append(base_advantage)

    rollout_data["advantages"] = advantages
    rollout_data["returns"] = advantages
