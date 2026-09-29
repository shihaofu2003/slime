"""Extract group, trajectory and turn-credit evidence from the completed v3 run."""

import ast
import csv
import json
import math
import re
import time
from collections import Counter
from pathlib import Path

OUT = Path(__file__).resolve().parent
RUN = OUT.parent / "arms/async/20260927_reward-v3-train"


def write_csv(path, rows):
    with path.open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(key for row in rows for key in row)))
        writer.writeheader()
        writer.writerows(rows)


def main():
    started = time.monotonic()
    groups, updates, batches = {}, {}, {}
    for line in (RUN / "run.log").open(errors="replace"):
        match = re.search(r"tau2_credit_group (\{.*\})", line)
        if match:
            row = json.loads(match[1])
            row["completion_order"] = len(groups)
            groups[row["group"]] = row
        match = re.search(r"tau2_pool_batch update=(\d+) groups=(\[[^\]]*\])", line)
        if match:
            batches[int(match[1])] = json.loads(match[2])
        match = re.search(r"tau2_trainer batch=(\d+) start=([\d.]+) end=([\d.]+)", line)
        if match:
            update = int(match[1])
            updates.setdefault(update, dict(update=update)).update(start=float(match[2]), end=float(match[3]), completed=True)
        match = re.search(r"(?:step|perf) (\d+): (\{.*\})", line)
        if match:
            values = ast.literal_eval(match[2])
            if "train/loss" in values or "rollout/policy_lag_mean" in values:
                update = int(match[1])
                updates.setdefault(update, dict(update=update)).update(values)
    for update, keys in batches.items():
        for key in keys:
            groups[key].update(update=update, trained=bool(updates[update].get("completed")))
    write_csv(OUT / "groups.csv", list(groups.values()))
    write_csv(OUT / "updates.csv", list(updates.values()))

    writer = None
    count = 0
    with (RUN / "trajectories/train.jsonl").open("rb") as source, (OUT / "trajectories.csv").open("w") as output:
        for seq, line in enumerate(source):
            if not line.endswith(b"\n"):
                continue
            # Preserve one file offset per attempt for targeted dialogue review.
            offset = source.tell() - len(line)
            prefix = json.loads(line.split(b', "simulation":', 1)[0] + b"}")
            m = prefix["metadata"]
            task = m["task"]
            criteria = task["evaluation_criteria"]
            info = m.get("tau2_reward_info") or {}
            field = m.get("tau2_field_reward_signals") or {}
            potential = m.get("tau2_progress") or {}
            timing = m.get("tau2_timing") or {}
            actions = info.get("action_checks") or []
            env = info.get("env_assertions") or []
            comm = info.get("communicate_checks") or []
            scores = potential.get("scores") or []
            description = task.get("description") or {}
            labels = [a["name"] for a in criteria.get("actions") or []]
            form = re.search(r"_(l\d_[a-z_]+)$", task["id"])
            row = dict(seq=seq, file_offset=offset, group=prefix["group_index"], sample=prefix["sample_index"],
                       task=prefix["task_id"], domain=m["domain"], reward=info.get("reward"),
                       removed=prefix["remove_sample"], tokens=m.get("tau2_response_tokens"),
                       context=m.get("tau2_train_total_tokens"), turns=len(m.get("tau2_turn_credits") or []),
                       first_policy=m.get("tau2_earliest_policy_version"), last_policy=m.get("tau2_latest_policy_version"),
                       weighted_policy=m.get("tau2_token_weighted_policy_version"),
                       termination=m.get("tau2_termination_reason"), attempts=m.get("tau2_rollout_attempts"),
                       permanently_too_long=m.get("tau2_permanently_too_long"),
                       basis="+".join(criteria.get("reward_basis", ["DB", "COMMUNICATE"])),
                       difficulty=description.get("difficulty"), task_type=description.get("task_type"),
                       banking_form=form[1] if form else None, reference_tools=json.dumps(labels),
                       db=(info.get("db_check") or {}).get("db_reward"),
                       action_n=len(actions), action_passed=sum(bool(a.get("action_match")) for a in actions),
                       all_actions=all(a.get("action_match") for a in actions) if actions else None,
                       env_n=len(env), env_passed=sum(bool(a.get("met")) for a in env),
                       all_env=all(a.get("met") for a in env) if env else None,
                       comm_n=len(comm), comm_passed=sum(bool(a.get("met")) for a in comm),
                       progress_reason=potential.get("unavailable_reason"),
                       progress_initial=scores[0] if scores else None, progress_final=scores[-1] if scores else None,
                       progress_changed=len(set(s for s in scores if s is not None)) > 1,
                       failed_tools=json.dumps([a["action"]["name"] for a in actions if not a.get("action_match")]))
            for name in ("argument", "tool_name", "db", "env", "communication"):
                row["field_" + name] = field.get("components", {}).get(name)
            for name in ("predicted_calls", "malformed_json", "nonexistent_tool", "repetition", "tool_execution_error", "wrong_argument_fields", "golden_actions"):
                row["count_" + name] = field.get("counts", {}).get(name)
            for name in ("wall_seconds", "user_calls", "user_seconds", "agent_active_seconds", "agent_wait_seconds"):
                row[name] = timing.get(name)
            if writer is None:
                writer = csv.DictWriter(output, fieldnames=list(row))
                writer.writeheader()
            writer.writerow(row)
            count += 1
            if count % 5000 == 0:
                print(count, "trajectory records", round(time.monotonic() - started, 1), "seconds", flush=True)

    credits, turn_events = [], []
    for line in (RUN / "credit.jsonl").open():
        record = json.loads(line)
        turns = record.pop("turns")
        row = {key: value for key, value in record.items() if not isinstance(value, (list, dict))}
        row["progress_available"] = not bool(record["progress_unavailable_reasons"])
        row["progress_reasons"] = " | ".join(record["progress_unavailable_reasons"])
        row["turns"] = len(turns)
        row["outcome_advantage"] = turns[0]["outcome_advantage"] if turns else 0
        row.update({key: 0.0 for key in ("span_tokens", "progress_token_count", "positive_token_count",
                                        "negative_token_count", "zero_token_count", "outcome_sum_sq",
                                        "progress_sum_sq", "total_sum_sq", "adv_sum", "progress_adv_sum",
                                        "zero_delta_tokens", "zero_delta_progress_tokens", "opposite_sign_tokens",
                                        "wrong_arg_tokens", "wrong_arg_positive_tokens", "rule_error_tokens")})
        row["positive_progress_turns"] = 0
        row["negative_progress_turns"] = 0
        for index, turn in enumerate(turns):
            begin, end = turn["response_span"]
            n = end - begin
            outcome, dense, total = (turn[key] for key in ("outcome_advantage", "progress_advantage", "advantage"))
            delta = turn.get("progress_reward")
            types = {error["type"] for error in turn["errors"]}
            row["span_tokens"] += n
            row["progress_token_count"] += n * (dense != 0)
            row["positive_token_count"] += n * (total > 0)
            row["negative_token_count"] += n * (total < 0)
            row["zero_token_count"] += n * (total == 0)
            row["outcome_sum_sq"] += n * outcome ** 2
            row["progress_sum_sq"] += n * dense ** 2
            row["total_sum_sq"] += n * total ** 2
            row["adv_sum"] += n * total
            row["progress_adv_sum"] += n * dense
            row["zero_delta_tokens"] += n * (delta == 0)
            row["zero_delta_progress_tokens"] += n * (delta == 0 and dense != 0)
            row["opposite_sign_tokens"] += n * (outcome * total < 0)
            row["wrong_arg_tokens"] += n * ("wrong_argument_field" in types)
            row["wrong_arg_positive_tokens"] += n * ("wrong_argument_field" in types and total > 0)
            row["rule_error_tokens"] += n * bool(types & {"malformed_json", "nonexistent_tool", "wrong_namespace_tool"})
            row["positive_progress_turns"] += delta is not None and delta > 0
            row["negative_progress_turns"] += delta is not None and delta < 0
            if row["domain"] in {"retail", "airline"} and (types or (delta is not None and delta < 0)):
                turn_events.append(dict(group=row["group_index"], sample=row["sample_index"], domain=row["domain"],
                                        task=row["task_id"], reward=row["reward"], turn=index,
                                        message=turn["simulation_message_index"], tokens=n,
                                        outcome=outcome, dense=dense, total=total, delta=delta,
                                        errors=json.dumps(turn["errors"])))
        credits.append(row)
    write_csv(OUT / "credit.csv", credits)
    write_csv(OUT / "turn_events.csv", turn_events)
    print("done", count, "trajectory records", len(groups), "groups", len(credits), "credit records",
          round(time.monotonic() - started, 1), "seconds", flush=True)


if __name__ == "__main__":
    main()
