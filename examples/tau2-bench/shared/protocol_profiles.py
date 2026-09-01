"""Shared tau2 Agent tool-call protocol profiles."""

from __future__ import annotations

import hashlib
import json
import re


PROTOCOL_CURRENT_SINGLE = "current-single"
PROTOCOL_STRICT_SINGLE_V1 = "strict-single-v1"
PROTOCOL_DEPENDENCY_SAFE_MULTI = "dependency-safe-multi"
PROTOCOL_AGENT_OWNED_DEPENDENCY_SAFE_MULTI = "agent-owned-dependency-safe-multi"
PROTOCOL_PROFILES = (
    PROTOCOL_CURRENT_SINGLE,
    PROTOCOL_STRICT_SINGLE_V1,
    PROTOCOL_DEPENDENCY_SAFE_MULTI,
    PROTOCOL_AGENT_OWNED_DEPENDENCY_SAFE_MULTI,
)
STRICT_SINGLE_V1_RULE = (
    "Use at most one native tool call per assistant message, without user-facing "
    "text. After a tool result, you may call another tool immediately."
)
DEPENDENCY_SAFE_MULTI_RULE = (
    "An assistant turn may contain multiple tool calls only when every call is "
    "independently justified by information available before the turn, no call requires "
    "another call's result, and every state-changing call is explicitly authorized and "
    "order-independent. Otherwise, make one tool call and wait for its result. If you make "
    "tool calls, do not respond to the user in the same turn."
)
PROTOCOL_DESCRIPTIONS = {
    PROTOCOL_CURRENT_SINGLE: "At most one tool call per assistant turn.",
    PROTOCOL_STRICT_SINGLE_V1: STRICT_SINGLE_V1_RULE,
    PROTOCOL_DEPENDENCY_SAFE_MULTI: DEPENDENCY_SAFE_MULTI_RULE,
    PROTOCOL_AGENT_OWNED_DEPENDENCY_SAFE_MULTI: (
        "Dependency-safe multi-call with an explicit Agent/User tool ownership boundary."
    ),
}
AGENT_TOOL_OWNERSHIP_RULES = """Tool ownership is a hard safety boundary:
- You may call only functions present in the Agent tool schemas supplied with this request.
- A function mentioned by the policy or technical-support manual but absent from those schemas is owned by the User/device, not by you.
- For a User/device action, ask the user in plain text to perform it, end your turn, and wait for the user's natural-language report before continuing.
- Never emit, simulate, or silently map a User/device tool call. Never claim that a User/device action succeeded without the user's report."""
PROTOCOL_IMPLEMENTATION_VERSION = "tau2-agent-tool-call-protocol-v1"
_SINGLE_CALL_POLICY_RE = re.compile(
    r"You should (?:only|at most) make one tool call at a time, and if you "
    r"(?:make|take) a tool call, you should not respond to the user "
    r"(?:simultaneously|at the same time)\. If you respond to the user, you should "
    r"not make a tool call at the same time\.",
    re.IGNORECASE,
)


def validate_protocol_profile(profile: str) -> str:
    if profile not in PROTOCOL_PROFILES:
        raise ValueError(
            f"unknown protocol profile {profile!r}; expected one of {PROTOCOL_PROFILES}"
        )
    return profile


def domain_policy_for_profile(policy: str, profile: str) -> tuple[str, bool]:
    """Apply one protocol profile to a tau2 domain policy."""

    validate_protocol_profile(profile)
    if profile == PROTOCOL_CURRENT_SINGLE:
        return policy, False
    if profile == PROTOCOL_STRICT_SINGLE_V1:
        amended, replacements = _SINGLE_CALL_POLICY_RE.subn("", policy, count=1)
        amended = re.sub(r"\n{3,}", "\n\n", amended).strip()
        return amended, bool(replacements)
    amended, replacements = _SINGLE_CALL_POLICY_RE.subn(
        DEPENDENCY_SAFE_MULTI_RULE,
        policy,
        count=1,
    )
    if not replacements and DEPENDENCY_SAFE_MULTI_RULE not in amended:
        amended = DEPENDENCY_SAFE_MULTI_RULE + "\n\n" + amended.strip()
    if _SINGLE_CALL_POLICY_RE.search(amended):
        raise ValueError("single-call policy clause remains after protocol amendment")
    return amended, bool(replacements)


def protocol_block_for_profile(profile: str) -> str:
    """Return the exact Agent system-prompt block for a profile."""

    validate_protocol_profile(profile)
    if profile in {PROTOCOL_CURRENT_SINGLE, PROTOCOL_STRICT_SINGLE_V1}:
        return ""
    return (
        "\n\n<tool_call_protocol>\n"
        f"{DEPENDENCY_SAFE_MULTI_RULE}\n"
        "</tool_call_protocol>"
    )


def protocol_signature(profile: str) -> str | None:
    """Hash every prompt-affecting input used by a protocol profile."""

    validate_protocol_profile(profile)
    if profile == PROTOCOL_STRICT_SINGLE_V1:
        return None
    payload = {
        "implementation_version": PROTOCOL_IMPLEMENTATION_VERSION,
        "profile": profile,
        "rule": (
            DEPENDENCY_SAFE_MULTI_RULE
            if profile
            in {
                PROTOCOL_DEPENDENCY_SAFE_MULTI,
                PROTOCOL_AGENT_OWNED_DEPENDENCY_SAFE_MULTI,
            }
            else None
        ),
        "protocol_block": protocol_block_for_profile(profile),
        "single_call_policy_pattern": (
            _SINGLE_CALL_POLICY_RE.pattern
            if profile
            in {
                PROTOCOL_DEPENDENCY_SAFE_MULTI,
                PROTOCOL_AGENT_OWNED_DEPENDENCY_SAFE_MULTI,
            }
            else None
        ),
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def strict_single_system_prompt(policy: str) -> str:
    """Return the unsigned native-tool prompt used by single-call-v1."""

    return (
        "You are a customer service agent. Follow the policy exactly.\n\n"
        f"{STRICT_SINGLE_V1_RULE}\n\n"
        f"{AGENT_TOOL_OWNERSHIP_RULES}\n\n"
        "<policy>\n"
        f"{policy.strip()}\n"
        "</policy>"
    )
