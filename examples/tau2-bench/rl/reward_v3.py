"""Opt-in value-equivalent tool execution and ACTION scoring for four-domain RL.

Reuses v2's synchronized reference states, scoring snapshots and Retail fix.
Only v3 worker processes install these hooks; earlier implementations stay intact.
"""

from __future__ import annotations

import inspect
import json
import logging
from functools import wraps
from typing import get_args, get_type_hints

import reward_v2
from tau2.data_model.tasks import Action
from tau2.domains.airline.tools import AirlineTools
from tau2.domains.banking_knowledge.tools import KnowledgeTools, KnowledgeUserTools
from tau2.domains.retail.tools import RetailTools
from tau2.domains.telecom.tools import TelecomTools
from tau2.domains.telecom.user_tools import TelecomUserTools

TOOL_CLASSES = (AirlineTools, RetailTools, TelecomTools, TelecomUserTools, KnowledgeTools, KnowledgeUserTools)
# These interfaces explicitly document `arguments` as an encoded JSON object.
JSON_ARGUMENT_TOOLS = {
    "call_discoverable_agent_tool", "call_discoverable_user_tool", "give_discoverable_user_tool",
}


def numeric_tool_methods():
    """Find declared numeric arguments, including Optional numeric arguments."""
    for toolkit in TOOL_CLASSES:
        for name, method in vars(toolkit).items():
            if not getattr(method, "__tool__", False):
                continue
            numeric = {}
            for parameter, annotation in get_type_hints(method).items():
                if parameter == "return":
                    continue
                choices = tuple(t for t in get_args(annotation) if t is not type(None))
                kind = choices[0] if len(choices) == 1 else annotation
                if kind in (int, float):
                    numeric[parameter] = kind
            if numeric:
                yield toolkit, name, method, numeric


def normalize_numeric_arguments(method, numeric):
    """Normalize at the actual method boundary, also reached by nested tools."""
    signature = inspect.signature(method)

    @wraps(method)
    def call(*args, **kwargs):
        bound = signature.bind_partial(*args, **kwargs)
        for name, kind in numeric.items():
            value = bound.arguments.get(name)
            # Do not coerce numeric strings, booleans, None or fractional
            # quantities into integers. Only equivalent representations change.
            if type(value) in (int, float):
                if kind is float or type(value) is int or value.is_integer():
                    bound.arguments[name] = kind(value)
        return method(*bound.args, **bound.kwargs)

    return call


def equivalent_values(left, right):
    """Exact values, numeric int/float equivalence, ordered lists, typed booleans."""
    if type(left) is bool or type(right) is bool:
        return type(left) is type(right) and left == right
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(equivalent_values(value, right[key]) for key, value in left.items())
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(equivalent_values(a, b) for a, b in zip(left, right))
    return left == right


def compare_with_tool_call(self, tool_call):
    """Preserve declared argument selection; compare JSON payloads as values."""
    if self.name != tool_call.name:
        return False
    compare_args = tool_call.arguments.keys() if self.compare_args is None else self.compare_args
    expected = {k: v for k, v in self.arguments.items() if k in compare_args}
    actual = {k: v for k, v in tool_call.arguments.items() if k in compare_args}
    if self.name in JSON_ARGUMENT_TOOLS:
        try:
            for arguments in (expected, actual):
                if "arguments" in arguments and isinstance(arguments["arguments"], str):
                    arguments["arguments"] = json.loads(arguments["arguments"])
        except json.JSONDecodeError:
            return False
    return equivalent_values(expected, actual)


def setup_worker():
    """Install the complete recipe before creating environments or target caches."""
    import progress
    import reward

    reward.EnvironmentEvaluator = reward_v2.EnvironmentEvaluator
    progress.reference_database = reward_v2.reference_database
    progress.database_snapshot = reward_v2.database_snapshot
    RetailTools.modify_pending_order_items = reward_v2.modify_pending_order_items
    Action.compare_with_tool_call = compare_with_tool_call
    count = 0
    for toolkit, name, method, numeric in numeric_tool_methods():
        setattr(toolkit, name, normalize_numeric_arguments(method, numeric))
        count += 1
    logging.getLogger(__name__).info(
        "tau2_reward_v3: synchronized references, semantic ACTION comparison and %d numeric tool methods enabled", count,
    )
