"""Read serial job progress every 30 minutes; never submit, stop or retry jobs."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job_id")
    parser.add_argument("run_name")
    args = parser.parse_args()
    experiment = Path(__file__).resolve().parents[1] / "output/experiments/tau2-areal-async-rl"
    while True:
        observation = {"time": datetime.now(timezone.utc).isoformat(), "job": args.job_id}
        response = subprocess.run(["/mnt/afs/users/fush/job", "show", args.job_id],
                                  capture_output=True, text=True)
        try:
            state, _ = json.JSONDecoder().raw_decode(response.stdout.lstrip())
            observation["state"] = state["state"]
            observation["reason"] = state.get("blocked_reason") or state.get("last_error", "")
        except (ValueError, KeyError):
            observation["state"] = "QUERY_FAILED"
        run = experiment / "serial" / args.run_name
        stages = run / "stages.log"
        if stages.exists():
            observation["stage"] = stages.read_text().splitlines()[-1]
        for arm in ("sync", "async"):
            updates = []
            for log in run.glob(f"{arm}-train_*.log"):
                with log.open(errors="replace") as source:
                    for line in source:
                        match = re.search(r"tau2_trainer batch=(\d+) start=", line)
                        if match:
                            updates.append(int(match[1]))
            observation[f"{arm}_updates"] = len(set(updates))
            for seed in (300, 301):
                directory = experiment / "eval" / f"{arm}-iter99-{args.run_name}"
                summaries = sorted(directory.glob(f"seed{seed}_*_summary.json"))
                if summaries:
                    summary = json.loads(summaries[-1].read_text())
                    observation[f"{arm}_seed{seed}"] = summary["overall"]
                else:
                    completed = 0
                    for path in (directory / f"seed{seed}").glob("*/results.json"):
                        try:
                            completed += len(json.loads(path.read_text()).get("simulations", []))
                        except ValueError:
                            pass  # The evaluator may be writing this file now.
                    observation[f"{arm}_seed{seed}_completed"] = completed
        print(json.dumps(observation, ensure_ascii=False), flush=True)
        if observation["state"] in {"SUCCEEDED", "FAILED", "STOPPED", "CANCELLED"}:
            return
        time.sleep(1800)


if __name__ == "__main__":
    main()
