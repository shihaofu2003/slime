"""Prepare coherent training requests from existing task targets and real DB facts.

The adapters change simulator requests and restore declared evidence conditions;
they do not copy reference tool traces or answer labels into the Agent prompt.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
BANKING_CONTRACTS = ROOT.parent / "datasets/tau2/rl/banking_independent_synthetic/contracts/train.jsonl"
RETAIL_WRITES = {
    "cancel_pending_order", "exchange_delivered_order_items", "modify_pending_order_items",
    "modify_pending_order_address", "modify_user_address", "modify_pending_order_payment",
    "return_delivered_order_items",
}
SELECTION_RULE = re.compile(r"\b(cheapest|least expensive|most expensive|lowest[ -]priced|highest[ -]priced|lowest price|highest price|largest|smallest|biggest|longest|shortest)\b", re.I)


@lru_cache(maxsize=1)
def banking_task_settings():
    path = Path(os.environ.get("TAU2_BANKING_TASK_CONTRACTS", str(BANKING_CONTRACTS)))
    return {row["task_id"]: row for row in map(json.loads, path.open())}


def item_description(item):
    return {"product": item["name"], "options": dict(item["options"])}


def payment_preference(user, method_id):
    method = user["payment_methods"][method_id]
    source = method["source"]
    if source == "credit_card":
        return f"my {method['brand']} credit card ending in {method['last_four']}"
    if source == "gift_card":
        return f"my gift card ending in {method_id[-4:]}"
    return f"my {source.replace('_', ' ')} account"


def retail_goal_spec(metadata, database):
    """Business requests keyed by order, preserving per-item bindings/quantities."""
    task = metadata["task"]
    actions = task["evaluation_criteria"].get("actions") or []
    if any(a["name"] == "transfer_to_human_agents" for a in actions):
        return None, "handoff_task_kept"
    changes = [a for a in actions if a["name"] in RETAIL_WRITES]
    if not changes:
        return None, "no_business_write_kept"
    instructions = task["user_scenario"].get("instructions") or {}
    text = instructions.get("task_instructions", "") if isinstance(instructions, dict) else instructions
    # A selected target is not the customer's prior knowledge when the task
    # asks the Agent to optimize or compare alternatives. Keep that task intact.
    if SELECTION_RULE.search(text):
        return None, "selection_rule_kept"
    owners = {
        a["arguments"]["user_id"] if a["name"] == "modify_user_address"
        else database["orders"][a["arguments"]["order_id"]]["user_id"]
        for a in changes
    }
    if len(owners) != 1:
        return None, "multiple_customers_kept"
    user = database["users"][next(iter(owners))]
    goals = []
    for action in changes:
        name, arguments = action["name"], action["arguments"]
        order = database["orders"][arguments["order_id"]] if "order_id" in arguments else None
        goal = {"order": arguments["order_id"]} if order else {}
        if name in {"modify_pending_order_items", "exchange_delivered_order_items"}:
            goal["request"] = "change pending items" if name.startswith("modify") else "exchange delivered items"
            goal["items"] = []
            for (old_id, new_id), quantity in Counter(zip(arguments["item_ids"], arguments["new_item_ids"])).items():
                old = next(item for item in order["items"] if item["item_id"] == old_id)
                product = database["products"][old["product_id"]]
                new = product["variants"][new_id]
                goal["items"].append(dict(quantity=quantity, current=item_description(old),
                                          requested=dict(product=product["name"], options=dict(new["options"]))))
            goal["payment_or_refund_preference"] = payment_preference(user, arguments["payment_method_id"])
        elif name == "return_delivered_order_items":
            goal["request"] = "return delivered items"
            goal["items"] = [dict(quantity=count, current=item_description(next(item for item in order["items"] if item["item_id"] == item_id)))
                             for item_id, count in Counter(arguments["item_ids"]).items()]
            goal["refund_preference"] = payment_preference(user, arguments["payment_method_id"])
        elif name == "cancel_pending_order":
            goal.update(request="cancel pending order", reason=arguments["reason"])
        elif name == "modify_pending_order_payment":
            goal.update(request="change order payment method",
                        payment_preference=payment_preference(user, arguments["payment_method_id"]))
        else:
            goal["request"] = "change shipping address" if order else "change my default address"
            goal["requested_address"] = {k: v for k, v in arguments.items() if k not in {"order_id", "user_id"}}
        goals.append(goal)
    order_ids = list(dict.fromkeys(g["order"] for g in goals if "order" in g))
    receipts = [dict(order=key, items=[item_description(item) for item in database["orders"][key]["items"]])
                for key in order_ids]
    known = dict(name=user["name"], email=user["email"], zip=user["address"]["zip"],
                 address=user["address"], receipts=receipts, requested_changes=goals)
    return known, "retail_grounded"


def prepare_row(row):
    result = copy.deepcopy(row)
    metadata = result["metadata"]
    task = metadata["task"]
    status = "unchanged"
    if metadata["domain"] == "retail":
        database = json.loads(Path(metadata["db_path"]).read_text())
        known, status = retail_goal_spec(metadata, database)
        if known is not None:
            previous = task["user_scenario"].get("instructions") or {}
            text = previous.get("task_instructions", "") if isinstance(previous, dict) else previous
            staged = "[Pattern B]" in text
            timing = (
                "Begin by requesting the first change. Reveal the remaining changes after that request is completed."
                if staged else "Mention all requested changes when explaining why you are calling."
            )
            task["user_scenario"]["instructions"] = dict(
                domain="retail", reason_for_call="Complete the order and account changes listed in requested_changes.",
                known_info=json.dumps(known, ensure_ascii=False), unknown_info=None,
                task_instructions=(
                    "Your receipts and requested_changes below describe your actual items and fixed preferences. "
                    "Describe each item using its product name and options. Keep requests attached to the correct order, "
                    "including duplicate quantities and every requested change. "
                    "Use the stated payment/refund preference for each operation; do not change it because the agent offers alternatives. "
                    "If an agent proposes different product options, restate your original request and ask them to check availability. "
                    "Supply your identity, order number and payment preference when asked. Do not invent SKU IDs, tools, prices or policies. "
                    "Do not mention internal field names or a reference solution. "
                    + timing + " Ask for clarification if a requested change is missing from the summary, and confirm the correct changes explicitly. "
                    "When all your requests are completed, respond exactly ###STOP###."
                ),
            )
    if metadata["domain"] in {"banking", "banking_knowledge"}:
        settings = banking_task_settings().get(task["id"])
        if settings:
            metadata["retrieval_variant"] = settings["retrieval_variant"]
            status = "banking_" + settings["retrieval_variant"]
    return result, dict(task=task["id"], domain=metadata["domain"], status=status)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    details = []
    with args.input.open() as source, args.output.open("w") as target:
        for line in source:
            row, detail = prepare_row(json.loads(line))
            target.write(json.dumps(row, ensure_ascii=False) + "\n")
            details.append(detail)
    report = dict(tasks=len(details), counts=dict(Counter(d["status"] for d in details)), details=details)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "details"}, indent=2))


if __name__ == "__main__":
    main()
