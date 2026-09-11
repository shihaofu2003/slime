"""Prepare matched concurrency arms and summarize their real pipeline outputs."""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

ARMS = {"c1": 1, "c2": 2, "c4": 4, "c4_repeat": 4}


def select_records(inventory):
    groups = defaultdict(list)
    pilot = set(inventory["pilot_ids"])
    for row in inventory["records"]:
        if row["id"] in pilot:
            groups[row["variant"], row["has_error"]].append(row)
    selected = []
    for key in sorted(groups):
        rows = sorted(groups[key], key=lambda r: (r["judge_prompt_tokens"], r["id"]))
        selected.extend(rows[i * (len(rows) - 1) // 3] for i in range(4))
    return selected


def summarize_arm(directory):
    stats = json.loads((directory / "data/pilot/stats.json").read_text())
    decisions = {}
    for path in sorted((directory / "decisions").glob("*.jsonl")):
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if row["finished_at"] >= decisions.get(row["id"], {}).get("finished_at", 0):
                decisions[row["id"]] = row
    start = min(r["finished_at"] - r["seconds"] for r in decisions.values())
    end = max(r["finished_at"] for r in decisions.values())
    requests = []
    for path in sorted((directory / "requests").glob("*.jsonl")):
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if not row["id"].startswith("control/"):
                requests.append(row)
    return {
        "counts": dict(Counter(r["status"] for r in decisions.values())),
        "wall_seconds": end - start,
        "trajectories_per_hour": len(decisions) * 3600 / (end - start),
        "output_tokens_per_second": sum((r.get("usage") or {}).get("completion_tokens", 0)
                                        for r in requests) / (end - start),
        "truncated_requests": sum(r["finish_reason"] == "length" for r in requests),
        "sft_rows": stats["counts"].get("sft_rows", 0),
        "export_errors": stats["counts"].get("export_error", 0),
        "sources": {key: {"status": r["status"], "candidate_tokens": r.get("candidate_tokens"),
                          "reason": r["reason"]} for key, r in decisions.items()},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "report"])
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.stage == "prepare":
        inventory = json.loads((args.source / "inventory.json").read_text())
        records = select_records(inventory)
        inventory["pilot_ids"] = [r["id"] for r in records]
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "samples.json").write_text(json.dumps(records, indent=2) + "\n")
        for arm in ARMS:
            directory = args.output / arm
            directory.mkdir(parents=True, exist_ok=True)
            (directory / "inventory.json").write_text(json.dumps(inventory) + "\n")
        print(json.dumps({"sample_count": len(records), "arms": ARMS}), flush=True)
    else:
        report = {arm: summarize_arm(args.output / arm) for arm in ARMS}
        baseline = report["c1"]
        for result in report.values():
            result["speedup"] = result["trajectories_per_hour"] / baseline["trajectories_per_hour"]
            result["status_disagreements_with_c1"] = [key for key, row in result["sources"].items()
                                                      if row["status"] != baseline["sources"][key]["status"]]
        (args.output / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({arm: {k: v for k, v in r.items() if k != "sources"}
                          for arm, r in report.items()}, indent=2), flush=True)


if __name__ == "__main__":
    main()
