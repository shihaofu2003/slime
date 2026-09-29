"""Read existing Airline evaluations, accepted groups, and optional rollout dumps.

Run from the repository root. No training, evaluation, or file mutation is done.
--scan-training also streams the large JSONL dumps, decoding only row headers.
"""

import argparse
import ast
import json
import random
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
MIX = ROOT / "output/experiments/tau2-mix-rl-sft4505-b128-20260925"
EXPERT = ROOT / "output/experiments/tau2-domain-experts-sft4505-b128"
SFT = ROOT / "output/experiments/tau2-sft-areal3-banking-simplified-full-20260920"
RUNS = {
    "progress": MIX / "arms/async/20260925_mix-retry1-train",
    "vanilla": MIX / "arms/async/20260925_vanilla-grpo-train",
    "expert": EXPERT / "arms/async/20260920_sft4505-airline-train",
}
WRITE = {
    "book_reservation", "cancel_reservation", "send_certificate",
    "update_reservation_flights", "update_reservation_baggages",
    "update_reservation_passengers",
}


def no_write(task):
    return not any(a["name"] in WRITE for a in task["evaluation_criteria"].get("actions") or [])


def emit(label, data):
    print(label, json.dumps(data, ensure_ascii=False, sort_keys=True), flush=True)


def read_log(path):
    batches, perf, train = {}, {}, {}
    for line in path.open(errors="replace"):
        match = re.search(r"tau2_pool_batch update=(\d+) groups=(\[[^\]]*\])", line)
        if match:
            batches[int(match[1])] = json.loads(match[2])
        match = re.search(r"perf (\d+): (\{.*\})", line)
        if match and "rollout/trained_groups_airline" in match[2]:
            perf[int(match[1])] = ast.literal_eval(match[2])
        match = re.search(r"step (\d+): (\{.*\})", line)
        if match and "train/loss" in match[2]:
            train[int(match[1])] = ast.literal_eval(match[2])
    return batches, perf, train


def evaluations():
    paths = {"sft": SFT / "eval/four-domain-sft-iter4505"}
    paths.update({f"expert{i}": EXPERT / f"eval/airline-iter{i}-four-domain" for i in (9, 19, 29, 39, 49, 59)})
    paths.update({f"progress{i}": MIX / f"eval/mix-iter{i}-four-domain" for i in (129, 139)})
    paths.update({f"vanilla{i}": MIX / f"eval/vanilla-grpo-iter{i}-four-domain" for i in (129, 139)})
    by_task, loaded = {}, {}
    for name, folder in paths.items():
        data = json.loads(next(folder.glob("trajectories/*_airline_test_4trials/results.json")).read_text())
        loaded[name] = data
        tasks = {t["id"]: t for t in data["tasks"]}
        counts, task_successes = Counter(), Counter()
        for sim in data["simulations"]:
            reward = sim["reward_info"]["reward"]
            kind = "no_write" if no_write(tasks[sim["task_id"]]) else "write"
            counts[kind + "_trials"] += 1
            counts[kind + "_successes"] += reward
            counts[sim["termination_reason"]] += 1
            task_successes[sim["task_id"]] += reward
            calls = [c for m in sim["messages"] if m["role"] == "assistant" for c in m.get("tool_calls") or []]
            counts["repeated_call_trials"] += len({(c["name"], json.dumps(c["arguments"], sort_keys=True)) for c in calls}) < len(calls)
        by_task[name] = dict(task_successes)
        emit("eval " + name, dict(counts))
    reference = loaded["expert29"]
    emit("eval_same_tasks", {k: v["tasks"] == reference["tasks"] for k, v in loaded.items()})
    keys = lambda d: {(s["task_id"], s["trial"], s["seed"]) for s in d["simulations"]}
    emit("eval_same_trial_keys", {k: keys(v) == keys(reference) for k, v in loaded.items()})
    emit("by_task_successes", by_task)
    for name in ("progress129", "progress139", "vanilla129", "vanilla139"):
        deltas = [(by_task[name][k] - value) / 4 for k, value in by_task["expert29"].items()]
        rng = random.Random(20260928)
        boot = sorted(statistics.mean(rng.choices(deltas, k=len(deltas))) * 100 for _ in range(20000))
        emit("exploratory_task_bootstrap " + name, {"delta_pp": statistics.mean(deltas) * 100, "ci95_pp": [boot[500], boot[19499]], "resamples": 20000})
    for name, folder in {
        "retail_expert9": EXPERT / "eval/retail-iter9-four-domain",
        "retail_progress129": MIX / "eval/mix-iter129-four-domain",
        "retail_progress139": MIX / "eval/mix-iter139-four-domain",
    }.items():
        data = json.loads(next(folder.glob("trajectories/*_retail_test_4trials/results.json")).read_text())
        tool_types = {c["action"]["name"]: c["tool_type"] for s in data["simulations"]
                      for c in (s.get("reward_info") or {}).get("action_checks") or []}
        readonly = {t["id"] for t in data["tasks"] if not any(
            tool_types[c["name"]] == "write" for c in t["evaluation_criteria"]["actions"] or [])}
        counts = Counter()
        for sim in data["simulations"]:
            kind = "no_write" if sim["task_id"] in readonly else "write"
            counts[kind + "_trials"] += 1
            counts[kind + "_successes"] += sim["reward_info"]["reward"]
        emit("eval " + name, dict(counts))


def training(pool, scan):
    tasks = {m["task"]["id"]: m for m in pool}
    airline = [m["task"] for m in pool if m["domain"] == "airline"]
    ro = {t["id"] for t in airline if no_write(t)}
    emit("training_pool", {"domains": dict(Counter(m["domain"] for m in pool)), "airline_no_write": len(ro), "no_write_reservations": dict(Counter(str(sorted({a["arguments"]["reservation_id"] for a in t["evaluation_criteria"].get("actions") or [] if "reservation_id" in a["arguments"]})) for t in airline if t["id"] in ro))})
    for name, folder in RUNS.items():
        batches, perf, train = read_log(folder / "run.log")
        emit("log_range " + name, {"first": min(train), "last": max(train), "updates": len(train)})
        for lo, hi in ((0, 30), (30, 60), (60, 90), (90, 130), (130, 140), (0, 140)):
            ps = [v for i, v in perf.items() if lo <= i < hi]
            ts = [v for i, v in train.items() if lo <= i < hi]
            if not ps:
                continue
            emit(f"training {name} [{lo},{hi})", {
                "logged_updates": len(ps),
                "groups": {d: sum(v["rollout/trained_groups_" + d] for v in ps) for d in ("airline", "retail", "telecom", "banking")},
                "mean_reference_kl": statistics.mean(v["train/kl_loss"] for v in ts),
                "mean_policy_lag": statistics.mean(v["rollout/policy_lag_mean"] for v in ps),
                "max_policy_lag": max(v["rollout/policy_lag_max"] for v in ps),
            })
        group_update = {g: i for i, gs in batches.items() if i in train for g in gs}
        credit_path = folder / "credit.jsonl"
        if credit_path.exists():
            groups, progress = {}, Counter()
            for line in credit_path.open():
                row = json.loads(line)
                groups[row["group_index"]] = (row["domain"], row["task_id"])
                if row["domain"] == "airline":
                    progress["unavailable_trajectories"] += bool(row["progress_unavailable_reasons"])
                    for turn in row["turns"]:
                        progress["turns"] += 1
                        progress["nonzero_progress"] += abs(turn["progress_advantage"]) > 1e-8
            emit("airline_credit " + name, dict(progress))
            ordered = [(i, g) for i, gs in sorted(batches.items()) for g in gs if groups.get(g, (None,))[0] == "airline"]
            for n in sorted({480, 960, len(ordered)}):
                part = ordered[:n]
                emit(f"coverage {name} first{len(part)}", {"unique_tasks": len({groups[g][1] for i, g in part}), "last_update": part[-1][0], "no_write_groups": sum(groups[g][1] in ro for i, g in part)})
        if scan:
            # Headers precede the large loss masks and embedded simulation.
            # Last record for a sample wins, matching the existing plotting script.
            headers = {}
            source = folder / "trajectories/train.jsonl"
            for n, line in enumerate(source.open(), 1):
                cut = line.index(', "loss_mask"')
                row = json.loads(line[:cut] + "}")
                headers[(row["group_index"], row["sample_index"])] = row
                if n % 10000 == 0:
                    emit("scan_rows " + name, n)
            grouped = defaultdict(list)
            for row in headers.values():
                grouped[row["group_index"]].append(row)
            stats = Counter()
            for g, rows in grouped.items():
                task = tasks[rows[0]["task_id"]]
                if task["domain"] != "airline" or len(rows) != 8:
                    continue
                kind = "no_write" if rows[0]["task_id"] in ro else "write"
                outcome = "mixed"
                if any(r["remove_sample"] for r in rows):
                    outcome = "removed"
                elif all(r["reward"] == 0 for r in rows):
                    outcome = "all_zero"
                elif all(r["reward"] == 1 for r in rows):
                    outcome = "all_one"
                stats[kind + "/" + outcome] += 1
                if g in group_update:
                    stats[kind + "/consumed_in_logged_updates"] += 1
            emit("dump_group_outcomes " + name, dict(stats))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan-training", action="store_true")
    args = parser.parse_args()
    evaluations()
    pool = [json.loads(line)["metadata"] for line in (MIX / "data/train.jsonl").open()]
    training(pool, args.scan_training)
