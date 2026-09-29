"""Extract small diagnostic tables from the three existing mix-RL runs."""

import ast
import csv
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[3]
BASE = ROOT / "output/experiments/tau2-mix-rl-sft4505-b128-20260925"
RUNS = {
    "credit": BASE / "arms/async/20260925_mix-retry1-train",
    "grpo": BASE / "arms/async/20260925_vanilla-grpo-train",
    "credit_repeat": OUT.parent / "arms/async/20260926_credit-no-replay-train",
}


def write_csv(path, rows):
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def extract(name, run):
    started = time.monotonic()
    updates, groups = {}, {}
    for line in (run / "run.log").open(errors="replace"):
        match = re.search(r"(?:step|perf) (\d+): ({.*})", line)
        if match:
            try:
                values = ast.literal_eval(match[2])
            except (ValueError, SyntaxError):
                continue
            if "train/loss" in values or "rollout/policy_lag_mean" in values:
                update = int(match[1])
                updates.setdefault(update, {"update": update}).update(values)
        match = re.search(r"tau2_pool_batch update=(\d+) groups=(\[[^\]]*\])", line)
        if match:
            update = int(match[1])
            for key in json.loads(match[2]):
                groups.setdefault(key, {"group": key})["update"] = update
        match = re.search(r"tau2_producer group=(\d+) start=([\d.]+) end=([\d.]+) accepted=(True|False)", line)
        if match:
            key = int(match[1])
            groups.setdefault(key, {"group": key}).update(
                start=float(match[2]), end=float(match[3]), accepted=match[4] == "True")
        match = re.search(r"tau2_trainer batch=(\d+) start=([\d.]+) end=([\d.]+)", line)
        if match:
            update = int(match[1])
            updates.setdefault(update, {"update": update}).update(
                start=float(match[2]), end=float(match[3]))
    write_csv(OUT / f"{name}_updates.csv", list(updates.values()))
    write_csv(OUT / f"{name}_groups.csv", list(groups.values()))

    writer = None
    with (OUT / f"{name}_trajectories.csv").open("w") as output:
        with (run / "trajectories/train.jsonl").open("rb") as stream:
            for seq, line in enumerate(stream):
                if not line.endswith(b"\n"):
                    continue
                # Dialogue messages dominate these 24 GB files. Decode only
                # the metadata and simulation header needed for this analysis.
                split = line.index(b', "simulation":')
                row = json.loads(line[:split] + b"}")
                tail = line[split + len(b', "simulation":'):]
                if tail.lstrip().startswith(b"{"):
                    header_end = tail.index(b', "messages":')
                    sim = json.loads(tail[:header_end] + b"}")
                else:
                    sim = {}
                meta = row["metadata"]
                info = meta.get("tau2_reward_info") or {}
                field = meta.get("tau2_field_reward_signals") or {}
                timing = meta.get("tau2_timing") or {}
                progress = meta.get("tau2_progress") or {}
                scores = progress.get("scores") or []
                actions = info.get("action_checks") or []
                record = {
                    "seq": seq, "sample": row["sample_index"],
                    "group": row["group_index"], "task": row["task_id"],
                    "domain": meta["domain"], "reward": row["reward"],
                    "official": info.get("reward"), "status": row["status"],
                    "removed": row["remove_sample"],
                    "response_length": row["response_length"],
                    "tokens": meta.get("tau2_response_tokens"),
                    "context": meta.get("tau2_train_total_tokens"),
                    "original_context": meta.get("tau2_original_total_tokens"),
                    "first_policy": meta.get("tau2_earliest_policy_version"),
                    "last_policy": meta.get("tau2_latest_policy_version"),
                    "weighted_policy": meta.get("tau2_token_weighted_policy_version"),
                    "termination": meta.get("tau2_termination_reason"),
                    "attempts": meta.get("tau2_rollout_attempts"),
                    "too_long": meta.get("tau2_permanently_too_long"),
                    "messages": meta.get("tau2_num_messages"),
                    "db": (info.get("db_check") or {}).get("db_reward"),
                    "action_n": len(actions),
                    "action_correct": sum(bool(a.get("action_match")) for a in actions),
                    "progress_reason": progress.get("unavailable_reason"),
                    "progress_initial": scores[0] if scores else None,
                    "progress_final": scores[-1] if scores else None,
                    "sim_start": sim.get("start_time"), "sim_end": sim.get("end_time"),
                }
                for key in ("tool_name", "argument", "db", "env", "communication"):
                    record["field_" + key] = field.get("components", {}).get(key)
                for key in ("golden_actions", "predicted_calls", "argument_fields",
                            "matched_argument_fields", "wrong_argument_fields", "tool_execution_error",
                            "malformed_json", "wrong_namespace_tool", "nonexistent_tool", "repetition", "max_steps"):
                    record["count_" + key] = field.get("counts", {}).get(key)
                for key in ("wall_seconds", "user_seconds", "user_calls", "agent_active_seconds",
                            "agent_wait_seconds", "step_wait_seconds", "simulation_seconds"):
                    record[key] = timing.get(key)
                if writer is None:
                    writer = csv.DictWriter(output, fieldnames=list(record))
                    writer.writeheader()
                writer.writerow(record)
                if (seq + 1) % 10000 == 0:
                    print(name, seq + 1, "trajectories", round(time.monotonic() - started, 1), "seconds", flush=True)
    print(name, "done", seq + 1, round(time.monotonic() - started, 1), "seconds", flush=True)


if __name__ == "__main__":
    with ThreadPoolExecutor(max_workers=3) as workers:
        list(workers.map(lambda item: extract(*item), RUNS.items()))
