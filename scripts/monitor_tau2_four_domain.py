"""Check the four-domain experiment hourly and evaluate every 50 updates."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess
import time

RECIPES = {"vanilla-grpo": "v", "progress-db-count-v1": "db"}
ITERATIONS = tuple(range(49, 550, 50))
TERMINAL = {"SUCCEEDED", "FAILED", "STOPPED", "CANCELLED"}
JOB = "/mnt/afs/users/fush/job"


def job_status(job_id):
    result = subprocess.run([JOB, "show", str(job_id)], capture_output=True, text=True, check=True)
    return json.JSONDecoder().raw_decode(result.stdout.lstrip())[0]


def completed_checkpoints(log):
    updates, checkpoints = set(), set()
    if log.exists():
        with log.open(errors="replace") as stream:
            for line in stream:
                match = re.search(r"tau2_(trainer|checkpoint) batch=(\d+) start=[\d.]+ end=[\d.]+", line)
                if match:
                    (updates if match[1] == "trainer" else checkpoints).add(int(match[2]))
    return updates, checkpoints


def prior_eval_jobs(experiment, name):
    return sorted(int(path.name.split("-", 1)[0])
                  for path in (experiment / "jobs").glob(f"*-{name}-*")
                  if path.name.split("-", 1)[0].isdigit())


def submit_evaluation(root, experiment, stamp, recipe, iteration, name):
    run_name = f"{stamp}-{recipe}-train"
    command = ["bash", str(root / "scripts/submit.sh"), "--experiment", experiment.name,
               "--name", name, "--gpus", "8", "--cpus", "64", "--memory", "1024",
               "-e", f"TAU2_RUN_NAME={run_name}", "-e", f"RUN_STAMP={stamp}",
               str(root / "scripts/eval_tau2_async_four_domain.sh"), "--", recipe, str(iteration)]
    result = subprocess.run(command, cwd=root, capture_output=True, text=True, check=True)
    match = re.search(r"\bid=(\d+)\b", result.stdout)
    if match is None:
        raise RuntimeError(f"No job ID in submission response: {result.stdout}")
    job_id = int(match[1])
    state = job_status(job_id)
    # The authoritative filename comes from job-manager, including its date prefix.
    run_log = Path(state["run_log"].replace("run_*_", "run_0_"))
    readme = experiment / "README.md"
    entry = (f"- Periodic {recipe} iter{iteration} ({iteration + 1} updates), job {job_id}: "
             f"[run log]({run_log.relative_to(experiment)}). "
             "Four domains, seed300 × 4 trials; 8 GPU / 64 CPU / 1024 GB, normal priority.\n\n")
    readme.write_text(readme.read_text().replace("## Reporting", entry + "## Reporting"))
    index = root / "output/doc/INDEX.md"
    lines = index.read_text().splitlines()
    for i, line in enumerate(lines):
        if f"../experiments/{experiment.name}/README.md" in line:
            lines[i] = re.sub(r" Periodic eval latest job:.*$", "", line) + f" Periodic eval latest job: {job_id} ({recipe} iter{iteration})."
    index.write_text("\n".join(lines) + "\n")
    return job_id


def inspect(root, main_job, stamp, *, submit=True, recipes=RECIPES):
    experiment = root / "output/experiments/tau2-async-db-count-four-domain"
    observation = {"time": datetime.now(timezone.utc).isoformat(), "training_job": main_job,
                   "training_state": job_status(main_job)["state"], "arms": {}, "evaluations": {}}
    for recipe in recipes:
        short = RECIPES[recipe]
        run = experiment / "arms/async" / f"{stamp}-{recipe}-train"
        updates, checkpoints = completed_checkpoints(run / "run.log")
        observation["arms"][recipe] = {"updates": len(updates), "last_update": max(updates, default=None),
                                        "saved_checkpoints": sorted(checkpoints)}
        for iteration in ITERATIONS:
            name = f"four-db-{stamp}-{short}-{iteration:04d}"
            jobs = prior_eval_jobs(experiment, name)
            if not jobs and iteration in checkpoints and submit:
                jobs = [submit_evaluation(root, experiment, stamp, recipe, iteration, name)]
            if jobs:
                state = job_status(jobs[-1])
                entry = {"job": jobs[-1], "state": state["state"]}
                summary = experiment / "eval" / f"{recipe}-iter{iteration}" / f"seed300_{stamp}_summary.json"
                if summary.exists():
                    entry["overall"] = json.loads(summary.read_text())["overall"]
                observation["evaluations"][f"{recipe}-iter{iteration}"] = entry
    return observation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job_id", type=int)
    parser.add_argument("run_stamp")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--read-only", action="store_true")
    parser.add_argument("--recipe", choices=RECIPES, help="Monitor only this training arm")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    recipes = [args.recipe] if args.recipe else RECIPES
    while True:
        try:
            observation = inspect(root, args.job_id, args.run_stamp, submit=not args.read_only, recipes=recipes)
            print(json.dumps(observation, ensure_ascii=False), flush=True)
            evaluations = list(observation["evaluations"].values())
            if observation["training_state"] in TERMINAL and all(e["state"] in TERMINAL for e in evaluations):
                complete = (observation["training_state"] == "SUCCEEDED"
                            and len(evaluations) == len(recipes) * len(ITERATIONS)
                            and all(e["state"] == "SUCCEEDED" and "overall" in e for e in evaluations))
                raise SystemExit(0 if complete else 1)
        except (subprocess.SubprocessError, OSError, ValueError, KeyError, RuntimeError) as exc:
            print(json.dumps({"time": datetime.now(timezone.utc).isoformat(), "error": str(exc)}), flush=True)
            if args.once:
                raise
        if args.once:
            return
        time.sleep(3600)


if __name__ == "__main__":
    main()
