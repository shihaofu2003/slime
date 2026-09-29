"""Check each modified Retail item against its requested catalog variant."""

import json
from pathlib import Path

from audit_reference_states import ROOT, initialized, make_environment_constructor, task_from_metadata
import reward_v2
from tau2.data_model.message import ToolCall


def main():
    rows = []
    source = ROOT / "output/experiments/tau2-mix-rl-sft4505-credit-signal-v1-20260927/data/train.jsonl"
    for line in source.open():
        metadata = json.loads(line)["metadata"]
        actions = metadata["task"]["evaluation_criteria"].get("actions") or []
        if metadata["domain"] != "retail" or not any(a["name"] == "modify_pending_order_items" for a in actions):
            continue
        task = task_from_metadata(metadata)
        constructor = make_environment_constructor(domain="retail", db_path=Path(metadata["db_path"]), task=task)
        record = dict(task=task.id, legacy_wrong_items=[], fixed_wrong_items=[],
                      paired_order_equal=None)
        states = {}
        for mode in ("legacy", "fixed", "reversed"):
            env = initialized(constructor, task)
            for action in task.evaluation_criteria.actions:
                arguments = dict(action.arguments)
                if action.name != "modify_pending_order_items":
                    env.get_response(ToolCall(id="call", name=action.name, requestor=action.requestor, arguments=arguments))
                    continue
                if mode == "reversed":
                    arguments["item_ids"] = list(reversed(arguments["item_ids"]))
                    arguments["new_item_ids"] = list(reversed(arguments["new_item_ids"]))
                order = env.tools._get_order(arguments["order_id"])
                remaining = list(order.items)
                expected = []
                for old, new in zip(arguments["item_ids"], arguments["new_item_ids"]):
                    item = next(item for item in remaining if item.item_id == old)
                    remaining.remove(item)
                    variant = env.tools._get_variant(item.product_id, new)
                    expected.append((item, new, variant.price, dict(variant.options)))
                function = (reward_v2.modify_pending_order_items.__wrapped__ if mode == "legacy"
                            else reward_v2.modify_pending_order_items)
                function(env.tools, **arguments)
                mismatches = [dict(item=new, expected_price=price, actual_price=item.price,
                                   expected_options=options, actual_options=item.options)
                              for item, new, price, options in expected
                              if item.item_id != new or item.price != price or item.options != options]
                if mode != "reversed":
                    record[mode + "_wrong_items"].extend(mismatches)
            states[mode] = reward_v2.database_snapshot(env)
        record["paired_order_equal"] = states["fixed"] == states["reversed"]
        rows.append(record)
    path = Path(__file__).with_name("retail_items_audit.jsonl")
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    print(json.dumps(dict(checked=len(rows), legacy_wrong_tasks=sum(bool(r["legacy_wrong_items"]) for r in rows),
                          fixed_wrong_tasks=sum(bool(r["fixed_wrong_items"]) for r in rows),
                          permutation_mismatches=sum(not r["paired_order_equal"] for r in rows)), indent=2))
    return any(r["fixed_wrong_items"] or not r["paired_order_equal"] for r in rows)


if __name__ == "__main__":
    raise SystemExit(main())
