"""Opt-in synchronized reference states and semantic DB equality for Tau2 RL.

Reward components stay defined by each task. Runtime hooks are installed only
in workers of the separate v2 training entry point; legacy files are unchanged.
"""

from __future__ import annotations

import logging
import math
import re
from decimal import Decimal
from functools import partial, wraps

from tau2.data_model.simulation import DBCheck, EnvAssertionCheck, RewardInfo
from tau2.data_model.tasks import RewardType
from tau2.domains.banking_knowledge.utils import generate_application_id as _original_application_id
from tau2.domains.retail.data_model import GiftCard, OrderPayment
from tau2.domains.retail.tools import RetailTools

_REFUEL_DESCRIPTION = re.compile(r"Data refueling: (\S+) GB at \$(\S+)/GB")


def generate_application_id(card_type, customer_name, annual_income, rho_bank_subscription=False):
    """Use the declared numeric argument type before the existing business-ID generator."""
    return _original_application_id(card_type, customer_name, float(annual_income), rho_bank_subscription)


@wraps(RetailTools.modify_pending_order_items)
def modify_pending_order_items(self, order_id, item_ids, new_item_ids, payment_method_id):
    """Apply each old/new pair's own variant, preserving duplicate-item counts."""
    order = self._get_order(order_id)
    if order.status != "pending":
        raise ValueError("Non-pending order cannot be modified")
    all_item_ids = [item.item_id for item in order.items]
    for item_id in item_ids:
        if item_ids.count(item_id) > all_item_ids.count(item_id):
            raise ValueError(f"{item_id} not found")
    if len(item_ids) != len(new_item_ids):
        raise ValueError("The number of items to be exchanged should match")
    remaining = list(order.items)
    changes = []
    for item_id, new_item_id in zip(item_ids, new_item_ids):
        if item_id == new_item_id:
            raise ValueError("The new item id should be different from the old item id")
        item = next(item for item in remaining if item.item_id == item_id)
        remaining.remove(item)
        variant = self._get_variant(item.product_id, new_item_id)
        if not variant.available:
            raise ValueError(f"New item {new_item_id} not found or available")
        changes.append((item, new_item_id, variant))
    diff_price = round(math.fsum(variant.price - item.price for item, _, variant in changes), 2)
    payment_method = self._get_payment_method(order.user_id, payment_method_id)
    if isinstance(payment_method, GiftCard) and payment_method.balance < diff_price:
        raise ValueError("Insufficient gift card balance to pay for the new item")
    order.payment_history.append(OrderPayment(
        transaction_type="payment" if diff_price > 0 else "refund",
        amount=abs(diff_price), payment_method_id=payment_method_id,
    ))
    if isinstance(payment_method, GiftCard):
        payment_method.balance = round(payment_method.balance - diff_price, 2)
    for item, new_item_id, variant in changes:
        item.item_id = new_item_id
        item.price = variant.price
        item.options = variant.options
    order.status = "pending (item modified)"
    return order


def database_snapshot(environment):
    """Business state for both reward and progress; leave live tool output intact."""
    result = {"assistant": environment.tools.db.model_dump(mode="json")}
    if environment.user_tools is not None:
        result["user"] = environment.user_tools.db.model_dump(mode="json")
    if environment.domain_name in {"telecom", "telecom-workflow"}:
        for bill in result["assistant"]["bills"]:
            for item in bill["line_items"]:
                match = _REFUEL_DESCRIPTION.fullmatch(item["description"])
                if match:
                    amount, price = (str(Decimal(value).normalize()) for value in match.groups())
                    item["description"] = f"Data refueling: {amount} GB at ${price}/GB"
    return result


def reference_database(task, environment_constructor):
    """Replay reference calls with the same per-call synchronization as rollout."""
    target = environment_constructor()
    initial = task.initial_state
    target.set_state(
        initialization_data=initial.initialization_data if initial else None,
        initialization_actions=initial.initialization_actions if initial else None,
        message_history=(initial.message_history or []) if initial else [],
    )
    for action in task.evaluation_criteria.actions or []:
        target.make_tool_call(action.name, requestor=action.requestor, **action.arguments)
        target.sync_tools()
    return database_snapshot(target)


class EnvironmentEvaluator:
    @classmethod
    def calculate_reward(cls, environment_constructor, task, full_trajectory,
                         solo_mode=False, env_kwargs=None):
        criteria = task.evaluation_criteria
        if criteria is None:
            return RewardInfo(reward=1.0, info={"note": "No evaluation criteria"})
        if criteria.actions is None and criteria.env_assertions is None:
            return RewardInfo(reward=1.0, db_check=DBCheck(db_match=True, db_reward=1.0),
                              info={"note": "No expected actions or env assertions"})
        constructor = partial(environment_constructor, **(env_kwargs or {}))
        predicted = constructor(solo_mode=solo_mode)
        initial = task.initial_state
        predicted.set_state(
            initialization_data=initial.initialization_data if initial else None,
            initialization_actions=initial.initialization_actions if initial else None,
            message_history=list(full_trajectory),
        )
        target = reference_database(task, constructor)
        # Direct structured equality treats 2 and 2.0 as the same numeric value;
        # it does not round amounts or discard fields, list order or extra writes.
        match = database_snapshot(predicted) == target
        db_check = DBCheck(db_match=match, db_reward=float(match))
        checks = []
        for assertion in criteria.env_assertions or []:
            met = predicted.run_env_assertion(assertion, raise_assertion_error=False)
            checks.append(EnvAssertionCheck(env_assertion=assertion, met=met, reward=float(met)))
        reward, breakdown = 1.0, {}
        if RewardType.DB in criteria.reward_basis:
            breakdown[RewardType.DB] = db_check.db_reward
            reward *= db_check.db_reward
        if RewardType.ENV_ASSERTION in criteria.reward_basis:
            value = float(all(check.met for check in checks))
            breakdown[RewardType.ENV_ASSERTION] = value
            reward *= value
        return RewardInfo(reward=reward, db_check=db_check, env_assertions=checks,
                          reward_basis=criteria.reward_basis, reward_breakdown=breakdown)


def setup_worker():
    """Ray worker startup hook, before any trajectory or reference cache exists."""
    import progress
    import reward
    from tau2.domains.banking_knowledge import tools as banking_tools

    reward.EnvironmentEvaluator = EnvironmentEvaluator
    progress.reference_database = reference_database
    progress.database_snapshot = database_snapshot
    banking_tools.generate_application_id = generate_application_id
    RetailTools.modify_pending_order_items = modify_pending_order_items
    logging.getLogger(__name__).info("tau2_reward_v2: synchronized references and semantic DB comparison enabled")
