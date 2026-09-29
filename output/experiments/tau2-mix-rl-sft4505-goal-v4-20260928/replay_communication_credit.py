"""Counterfactual credit on existing Banking trajectories, with no policy updates."""

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RL = ROOT / "examples/tau2-bench/rl"
sys.path[:0] = [str(ROOT), str(RL), str(ROOT / "output/experiments/tau2-areal-async-rl/dependencies/tau2/src")]
os.environ.setdefault("TAU2_DATA_DIR", str(ROOT.parent / "tau2-bench/data"))
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

import pandas as pd
from loguru import logger
logger.remove()
import goal_progress_v4 as goal
from goal_credit_v4 import goal_stage_advantages
from tau2.data_model.message import AssistantMessage, UserMessage, ToolMessage
from tau2.evaluator.evaluator_communicate import CommunicateEvaluator
from progress import group_normalize


def main():
    goal.setup_worker()
    directory = Path(__file__).resolve().parent
    previous = ROOT / "output/experiments/tau2-mix-rl-sft4505-reward-v3-20260927"
    diagnosis = previous / "domain_diagnosis_20260928"
    groups = pd.read_csv(diagnosis / "groups_enriched.csv")
    selected = groups[groups.domain.eq("banking") & ~groups.basis.str.contains("DB|ENV_ASSERTION")]
    trajectories = pd.read_csv(diagnosis / "trajectories.csv").drop_duplicates(["group", "sample"], keep="last")
    trajectories = trajectories[trajectories.group.isin(selected.group)]
    roles = {"assistant": AssistantMessage, "user": UserMessage, "tool": ToolMessage}
    rows = []
    with (previous / "arms/async/20260927_reward-v3-train/trajectories/train.jsonl").open("rb") as stream:
        for _, group in selected.iterrows():
            score_sequences, rewards, old_rewards, penalties = [], [], [], []
            for _, trace in trajectories[trajectories.group.eq(group.group)].sort_values("sample").iterrows():
                stream.seek(int(trace.file_offset))
                data = json.loads(stream.readline())
                metadata = data["metadata"]
                task = goal.task_from_metadata(metadata)
                messages = [roles[m["role"]].model_validate({k: v for k, v in m.items() if k != "raw_data"}) for m in data["simulation"]["messages"]]
                potential = goal.StatePotential(task, lambda: None)
                scores = []
                for index in metadata["tau2_progress"]["source_indices"] + [len(messages)]:
                    potential.messages = messages[:index]
                    scores.append(potential.score(None))
                score_sequences.append(scores)
                value = 1.0
                if "ACTION" in group.basis:
                    value *= goal.ActionEvaluator.calculate_reward(task, messages).reward
                if "COMMUNICATE" in group.basis:
                    value *= CommunicateEvaluator.calculate_reward(task, messages).reward
                if metadata["tau2_termination_reason"] not in {"user_stop", "agent_stop"}:
                    value = 0.0
                rewards.append(value)
                old_rewards.append(metadata["tau2_reward_info"]["reward"])
                penalties.append([-.1 * any(e["type"] in {"malformed_json", "nonexistent_tool", "wrong_namespace_tool"} for e in turn["errors"])
                                  for turn in metadata["tau2_turn_credits"]])
            _, dense, _ = goal_stage_advantages(score_sequences)
            outcome = group_normalize(rewards)
            new_signal = any(outcome[i] + credit + penalty != 0 for i, (credits, formats) in enumerate(zip(dense, penalties))
                             for credit, penalty in zip(credits, formats))
            rows.append(dict(group=int(group.group), task=group.task, form=group.banking_form, basis=group.basis,
                             old_kept=bool(group.kept), new_has_signal=new_signal, old_outcome=group.outcome,
                             old_reward=sum(old_rewards)/8, new_reward=sum(rewards)/8,
                             new_progress_signal=any(value != 0 for values in dense for value in values),
                             score_sequences=score_sequences))
    l1 = [row for row in rows if row["form"] == "l1_evidence"]
    summary = dict(groups=len(rows), recovered_groups=sum(not r["old_kept"] and r["new_has_signal"] for r in rows),
                   removed_old_signal_groups=sum(r["old_kept"] and not r["new_has_signal"] for r in rows),
                   terminal_reward_changed_groups=sum(r["old_reward"] != r["new_reward"] for r in rows),
                   l1_groups=len(l1), l1_recovered_groups=sum(not r["old_kept"] and r["new_has_signal"] for r in l1),
                   l1_all_zero_recovered=sum(not r["old_kept"] and r["new_has_signal"] and r["old_outcome"] == "all_zero" for r in l1),
                   l1_still_zero_signal=sum(not r["new_has_signal"] for r in l1))
    (directory / "communication_credit_replay.json").write_text(json.dumps(dict(summary=summary, groups=rows), indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
