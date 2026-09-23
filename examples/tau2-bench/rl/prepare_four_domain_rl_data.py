#!/usr/bin/env python3
"""Build the fixed four-domain tau2 RL pool used by async experiments."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from tau2.data_model.tasks import Task

ROOT = Path("/mnt/afs/users/fush/projects/ServiceAgent")
THREE_DOMAIN = ROOT / "datasets/tau2/rl/areal_tasks_slime.jsonl"
BANKING = ROOT / "datasets/tau2/rl/banking_independent_synthetic/tasks/train.json"
EMPTY_DB = ROOT / "slime-async-tau2/examples/tau2-bench/analysis/banking_synthetic/assets/empty_db.json"
DEFAULT_OUTPUT = ROOT / "slime-async-tau2/output/experiments/tau2-async-db-count-four-domain/data/train.jsonl"


def _task_row(task: dict[str, Any], *, domain: str, db_path: Path, source: str, index: int) -> dict[str, Any]:
    task = dict(task)
    task.pop("db_path", None)
    Task.model_validate(task)
    return {"prompt": task["id"], "metadata": {
        "source": source, "source_row": index, "domain": domain,
        "db_path": str(db_path), "task": task,
    }}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--three-domain", type=Path, default=THREE_DOMAIN)
    parser.add_argument("--banking", type=Path, default=BANKING)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    rows: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()

    with args.three_domain.open(encoding="utf-8") as stream:
        for index, line in enumerate(stream):
            if line.strip():
                raw = json.loads(line)
                domain = raw["metadata"]["domain"]
                rows.append(_task_row(raw["metadata"]["task"], domain=domain,
                                      db_path=Path(raw["metadata"]["db_path"]),
                                      source="three_domain", index=index))
                counts[domain] += 1

    banking_tasks = json.loads(args.banking.read_text(encoding="utf-8"))
    for index, task in enumerate(banking_tasks):
        rows.append(_task_row(task, domain="banking", db_path=EMPTY_DB,
                              source="banking_train", index=index))
        counts["banking"] += 1

    identities = {(row["metadata"]["domain"], row["metadata"]["task"]["id"]) for row in rows}
    if len(identities) != len(rows):
        raise ValueError("Training inputs contain duplicate domain/task identities")
    banking_basis = Counter(basis for task in banking_tasks
                            for basis in task["evaluation_criteria"]["reward_basis"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(json.dumps({"output": str(args.output), "rows": len(rows),
                      "domains": dict(sorted(counts.items())), "banking_reward_basis": dict(banking_basis)}, indent=2))


if __name__ == "__main__":
    main()
