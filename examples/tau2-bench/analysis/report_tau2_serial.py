"""Summarize the two training arms and four official evaluations of one serial run."""

import json
import os
import re
from pathlib import Path
import subprocess
import sys

from inspect_tau2_ready_smoke import inspect


def main():
    run_name = sys.argv[1]
    root = Path(os.environ["PROJECT_ROOT"])
    experiment = root / "output/experiments/tau2-areal-async-rl"
    run = experiment / "serial" / run_name
    output = run / "comparison.json"
    command = [sys.executable, str(Path(__file__).with_name("summarize_tau2_areal_async.py")),
               "--output", str(output), "--results-root", str(experiment)]
    inspections = {}
    for arm in ("sync", "async"):
        logs = sorted(run.glob(f"{arm}-train_*.log"))
        command += [f"--{arm}-logs", *map(str, logs)]
        summaries = []
        for seed in (300, 301):
            directory = experiment / "eval" / f"{arm}-iter99-{run_name}"
            path = sorted(directory.glob(f"seed{seed}_*_summary.json"))[-1]
            summary = json.loads(path.read_text())
            if summary["overall"]["diagnostics"]["infrastructure_errors"]:
                raise RuntimeError(f"Invalid official evaluation: {path}")
            summaries.append(str(path))
        command += [f"--{arm}-eval", *summaries]
        artifacts = experiment / "arms" / arm / run_name
        inspection = inspect(logs, artifacts / "trajectories/train100.jsonl")
        (run / f"{arm}-inspection.json").write_text(json.dumps(inspection, indent=2) + "\n")
        inspections[arm] = inspection
    historical = root.parent / "slime/output/experiments/tau2-rl-sft-raw-vs-processed-vanilla-grpo/eval/processed-iter99"
    command += ["--historical-eval", str(historical / "seed300_0903_200624_summary.json"),
                str(historical / "seed301_0903_200257_summary.json")]
    subprocess.run(command, check=True)
    report = json.loads(output.read_text())
    lines = [f"# Serial GRPO 100 updates: {run_name}", "",
             "Purpose: matched single-node 8-GPU synchronous/asynchronous training and official iter99 evaluation.", "",
             "| Arm | Training seconds | pass@1 | pass@4(any) | pass^4 | Action accuracy | DB accuracy |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for arm in ("sync", "async"):
        metrics = report["evaluation"][arm]["overall"]
        values = " | ".join(f"{100 * metrics[key]:.2f}%" for key in (
            "pass_at_1", "pass_at_4_any", "pass_power_4", "action_accuracy", "db_accuracy"))
        lines.append(f"| {arm} | {report['timing'][arm]['end_to_end_seconds']} | {values} |")
    lines += ["", "| Stage (including service cleanup) | Seconds | Exit code |",
              "|---|---:|---:|"]
    for stage, seconds, code in re.findall(
        r"tau2_serial stage=(\S+) end=\d+ seconds=(\d+) exit=(\d+)",
        (run / "stages.log").read_text(),
    ):
        lines.append(f"| {stage} | {seconds} | {code} |")
    valid = all(not item["violations"] and item["accepted_groups"] == 500
                and item["accepted_trajectories"] == 4000 for item in inspections.values())
    lines += ["", f"Recorded-token/group inspection passed: {valid}. Speedup: {report['speedup']}. "
              f"Original 2x / 5pp acceptance: {report['acceptance']}.", "",
              "Training time excludes installation, conversion and official evaluation. A resumed run must also "
              "account for failed-attempt and restart time before making a formal speed claim.", "",
              "[Full metrics, domain results and paired intervals](comparison.json); "
              "[sync inspection](sync-inspection.json); [async inspection](async-inspection.json).", "",
              "[20-update comparison](../../ready-train20-final-results.md).", ""]
    (run / "README.md").write_text("\n".join(lines))
    if not valid:
        raise RuntimeError("Final recorded-token/group inspection failed; see inspection files")


if __name__ == "__main__":
    main()
