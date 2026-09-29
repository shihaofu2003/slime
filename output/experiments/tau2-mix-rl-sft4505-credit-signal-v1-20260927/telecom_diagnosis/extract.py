"""Read a bounded snapshot of live rollout metadata for domain diagnostics."""

import ast
import csv
import json
import re
import time
from pathlib import Path

OUT = Path(__file__).resolve().parent
RUN = OUT.parent / "arms/async/20260927_credit-signal-v1-train"


def write_csv(path, rows):
    with path.open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(k for row in rows for k in row)))
        writer.writeheader()
        writer.writerows(rows)


def main():
    groups, batches, updates = {}, {}, {}
    for line in (RUN / "run.log").open(errors="replace"):
        match = re.search(r"tau2_credit_group ({.*})", line)
        if match:
            row = json.loads(match[1])
            groups[row["group"]] = row
        match = re.search(r"tau2_pool_batch update=(\d+) groups=(\[[^\]]*\])", line)
        if match:
            batches[int(match[1])] = json.loads(match[2])
        match = re.search(r"tau2_trainer batch=(\d+) start=([\d.]+) end=([\d.]+)", line)
        if match:
            updates[int(match[1])] = dict(update=int(match[1]), start=float(match[2]), end=float(match[3]))
    for update, keys in batches.items():
        for key in keys:
            if key in groups:
                groups[key]["update"] = update
                groups[key]["trained"] = update in updates
    write_csv(OUT / "groups.csv", list(groups.values()))
    write_csv(OUT / "updates.csv", list(updates.values()))

    trace = RUN / "trajectories/train.jsonl"
    limit = trace.stat().st_size
    started = time.monotonic()
    writer = None
    examples = []
    with trace.open("rb") as stream, (OUT / "trajectories.csv").open("w") as output:
        seq = 0
        while stream.tell() < limit:
            line = stream.readline()
            if not line.endswith(b"\n") or stream.tell() > limit:
                break
            split = line.index(b', "simulation":')
            row = json.loads(line[:split] + b"}")
            metadata = row["metadata"]
            task = metadata["task"]
            criteria = task["evaluation_criteria"]
            info = metadata["tau2_reward_info"]
            progress = metadata.get("tau2_progress") or {}
            field = metadata.get("tau2_field_reward_signals") or {}
            description = task.get("description") or {}
            assertions = info.get("env_assertions") or []
            checks = info.get("action_checks") or []
            scores = progress.get("scores") or []
            valid_scores = [s for s in scores if s is not None]
            record = dict(
                seq=seq, group=row["group_index"], sample=row["sample_index"],
                domain=metadata["domain"], task=row["task_id"], reward=info["reward"],
                difficulty=description.get("difficulty", "unspecified"),
                task_type=description.get("task_type", "unspecified"),
                basis="+".join(info.get("reward_basis") or []),
                explicit_basis="reward_basis" in criteria,
                removed=row["remove_sample"], too_long=metadata.get("tau2_permanently_too_long"),
                tokens=metadata.get("tau2_response_tokens"), context=metadata.get("tau2_train_total_tokens"),
                turns=len(metadata.get("tau2_turn_credits") or []),
                messages=metadata.get("tau2_num_messages"), termination=metadata.get("tau2_termination_reason"),
                first_policy=metadata.get("tau2_earliest_policy_version"),
                last_policy=metadata.get("tau2_latest_policy_version"),
                wall_seconds=metadata.get("tau2_timing", {}).get("wall_seconds"),
                user_calls=metadata.get("tau2_timing", {}).get("user_calls"),
                db=(info.get("db_check") or {}).get("db_reward"),
                assertion_n=len(assertions), assertion_passed=sum(bool(a.get("met")) for a in assertions),
                all_assertions_passed=all(a.get("met") for a in assertions) if assertions else None,
                assertion_results=json.dumps({a["env_assertion"]["func_name"]: bool(a.get("met")) for a in assertions}),
                action_n=len(checks), action_passed=sum(bool(a.get("action_match")) for a in checks),
                user_actions=sum(a["action"].get("requestor") == "user" for a in checks),
                user_actions_passed=sum(a["action"].get("requestor") == "user" and bool(a.get("action_match")) for a in checks),
                assistant_actions=sum(a["action"].get("requestor") == "assistant" for a in checks),
                assistant_actions_passed=sum(a["action"].get("requestor") == "assistant" and bool(a.get("action_match")) for a in checks),
                progress_reason=progress.get("unavailable_reason"),
                progress_initial=valid_scores[0] if valid_scores else None,
                progress_final=valid_scores[-1] if valid_scores else None,
                progress_changed=len(set(valid_scores)) > 1,
            )
            for key in ("argument", "tool_name", "db", "env"):
                record["field_" + key] = field.get("components", {}).get(key)
            for key in ("predicted_calls", "malformed_json", "nonexistent_tool", "repetition", "tool_execution_error"):
                record["count_" + key] = field.get("counts", {}).get(key)
            if writer is None:
                writer = csv.DictWriter(output, fieldnames=list(record))
                writer.writeheader()
            writer.writerow(record)
            if (record["domain"] == "telecom" and record["all_assertions_passed"]
                    and record["reward"] == 0 and len(examples) < 3):
                simulation = json.loads(line)["simulation"]
                examples.append(dict(group=record["group"], sample=record["sample"], task=record["task"],
                                     reward_info=info, task_description=description,
                                     criterion=criteria, progress=progress,
                                     messages=[{key: value for key, value in m.items()
                                                if key in {"role", "content", "tool_calls", "tool_call_id"}}
                                               for m in simulation["messages"]]))
            seq += 1
            if seq % 5000 == 0:
                print(seq, "trajectories", round(time.monotonic() - started, 1), "seconds", flush=True)
    (OUT / "assertions_pass_reward_zero_examples.json").write_text(json.dumps(examples, ensure_ascii=False, indent=2) + "\n")
    print("done", seq, "trajectories", len(groups), "complete groups", round(time.monotonic() - started, 1), "seconds", flush=True)


if __name__ == "__main__":
    main()
