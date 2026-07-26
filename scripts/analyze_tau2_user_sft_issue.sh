#!/usr/bin/env bash
set -euo pipefail

SERVICE_AGENT_ROOT="${SERVICE_AGENT_ROOT:-/mnt/afs/users/fush/projects/ServiceAgent}"

python3 - <<'PY'
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

root = Path("/mnt/afs/users/fush/projects/ServiceAgent")
user_sft_dir = root / "datasets/tau2-bench-user-sft"
simulation_dir = root / "tau2-bench/data/simulations"

STOP = "###STOP###"
TRANSFER = "###TRANSFER###"
OUT_OF_SCOPE = "###OUT-OF-SCOPE###"


def load_runs(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data
    for key in ("simulations", "results", "runs"):
        value = data.get(key)
        if isinstance(value, list):
            return value
    return []


def termination_reason(run: dict[str, Any]) -> str:
    value = (
        run.get("termination_reason")
        or run.get("terminationReason")
        or (run.get("info") or {}).get("termination_reason")
    )
    if isinstance(value, dict):
        value = value.get("value") or value.get("name") or str(value)
    return str(value)


def reward_value(run: dict[str, Any]) -> float | None:
    candidates = [
        run.get("reward"),
        (run.get("metrics") or {}).get("reward")
        if isinstance(run.get("metrics"), dict)
        else None,
        (run.get("reward_info") or {}).get("reward")
        if isinstance(run.get("reward_info"), dict)
        else None,
    ]
    for value in candidates:
        if isinstance(value, (int, float)):
            return float(value)
    return None


def dataset_stats() -> None:
    print("## user SFT target stats")
    paths = sorted(user_sft_dir.glob("*user_sft_no_thinking.jsonl"))
    if not paths:
        print(f"no user SFT files found under {user_sft_dir}")
        return

    for path in paths:
        counts = Counter()
        finalish_examples: list[str] = []
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                messages = row.get("messages") or []
                target = (messages[-1].get("content") if messages else "") or ""
                low = target.lower()
                counts["total"] += 1
                counts["target_stop"] += int(STOP in target)
                counts["target_transfer"] += int(TRANSFER in target)
                counts["target_out_of_scope"] += int(OUT_OF_SCOPE in target)
                counts["target_tool"] += int("<tool_call>" in target)
                counts["target_text_only"] += int(bool(target.strip()) and "<tool_call>" not in target)
                finalish = any(
                    marker in low
                    for marker in (
                        "thank",
                        "thanks",
                        "appreciate",
                        "goodbye",
                        "bye",
                        "no need",
                        "that is all",
                        "that's all",
                    )
                )
                counts["finalish_text"] += int(finalish)
                if finalish and STOP not in target and len(finalish_examples) < 3:
                    finalish_examples.append(" ".join(target.split())[:300])

        print(path.name)
        print(
            "  "
            + " ".join(
                f"{key}={counts[key]}"
                for key in (
                    "total",
                    "target_stop",
                    "target_transfer",
                    "target_out_of_scope",
                    "target_tool",
                    "target_text_only",
                    "finalish_text",
                )
            )
        )
        for example in finalish_examples:
            print(f"  finalish_no_stop_example: {example}")


def simulation_stats() -> None:
    print("\n## user_sft eval termination stats")
    paths = sorted(simulation_dir.glob("*user_sft*/*results.json"))
    if not paths:
        print(f"no user_sft simulation results found under {simulation_dir}")
        return

    for path in paths:
        runs = load_runs(path)
        reasons = Counter(termination_reason(run) for run in runs)
        rewards = [reward for run in runs if (reward := reward_value(run)) is not None]
        markers = Counter()
        for run in runs:
            payload = json.dumps(run, ensure_ascii=False)
            markers["contains_stop"] += int(STOP in payload)
            markers["contains_transfer"] += int(TRANSFER in payload)
            markers["contains_out_of_scope"] += int(OUT_OF_SCOPE in payload)

        print(path.parent.name)
        print(f"  total={len(runs)} reasons={dict(sorted(reasons.items()))}")
        if rewards:
            print(f"  avg_reward={sum(rewards) / len(rewards):.4f} reward_count={len(rewards)}")
        print(
            "  markers="
            f"STOP:{markers['contains_stop']} "
            f"TRANSFER:{markers['contains_transfer']} "
            f"OUT_OF_SCOPE:{markers['contains_out_of_scope']}"
        )


def main() -> None:
    dataset_stats()
    simulation_stats()


if __name__ == "__main__":
    main()
PY
