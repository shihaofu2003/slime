"""Summarize the frozen Telecom snapshot and compare identical task draws."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[3]
groups = pd.read_csv(OUT / "groups.csv")
trace = pd.read_csv(OUT / "trajectories.csv")
trace = trace[trace.group.isin(groups.group)].copy()
pool = {}
for line in (OUT.parent / "data/train.jsonl").open():
    metadata = json.loads(line)["metadata"]
    if metadata["domain"] == "telecom":
        task = metadata["task"]
        pool[task["id"]] = "+".join(task["evaluation_criteria"].get("reward_basis", ["DB", "COMMUNICATE"]))
reference = json.load((OUT / "reference_state_example.json").open())
affected = {row["task"] for row in reference["default_db_reference_tasks_changed_by_final_sync"]}
telecom = trace[trace.domain == "telecom"].copy()
telecom["configured_basis"] = telecom.task.map(pool)
telecom["stale_reference"] = telecom.task.isin(affected)
telecom["env_pass"] = telecom.all_assertions_passed == True
domain = trace.groupby("domain").agg(trajectories=("reward", "size"), reward=("reward", "mean"),
                                    turns=("turns", "mean"), user_calls=("user_calls", "mean"),
                                    wall_seconds=("wall_seconds", "mean"), removed=("removed", "mean"))
basis = telecom.groupby(["configured_basis", "stale_reference"]).agg(
    trajectories=("reward", "size"), groups=("group", "nunique"), tasks=("task", "nunique"),
    successes=("reward", "sum"), reward=("reward", "mean"),
    env_passes=("env_pass", "sum"), env_pass_rate=("env_pass", "mean"))
domain.to_csv(OUT / "domain_comparison.csv")
basis.to_csv(OUT / "telecom_reward_bases.csv")

task_groups = telecom.groupby("group").agg(task=("task", "first"), reward=("reward", "mean"),
                                          env_pass=("env_pass", "mean"), stale=("stale_reference", "first"))
task_groups["draw_block"] = task_groups.index // 500 * 500
comparison_rows, pair_summary = [], {}
oldroot = ROOT / "output/experiments/tau2-mix-rl-sft4505-credit-no-replay-20260926/reward_diagnosis"
for name in ("credit", "grpo", "credit_repeat"):
    old = pd.read_csv(oldroot / (name + "_trajectories.csv"))
    old = old[old.domain == "telecom"].drop_duplicates(["group", "sample"], keep="last")
    old_groups = old.groupby("group").agg(task=("task", "first"), reward=("official", "mean"))
    paired = task_groups.join(old_groups, rsuffix="_old", how="inner")
    assert (paired.task == paired.task_old).all()
    pair_summary[name] = dict(groups=len(paired), correlation=paired.reward.corr(paired.reward_old),
                             old_affected_trajectories=int(old.task.isin(affected).sum()),
                             old_affected_successes=float(old.loc[old.task.isin(affected), "official"].sum()))
    for block, rows in paired.groupby("draw_block"):
        comparison_rows.append(dict(control=name, draw_block=block, groups=len(rows),
                                    current=rows.reward.mean(), control_reward=rows.reward_old.mean()))
comparisons = pd.DataFrame(comparison_rows)
comparisons.to_csv(OUT / "matched_task_draws.csv", index=False)

windows = []
for end in range(400, len(groups) + 1, 20):
    window = groups.iloc[end - 400:end]
    selected = task_groups.loc[window.loc[window.domain == "telecom", "group"]]
    windows.append(dict(completed_groups=end, telecom_groups=len(selected),
                        reward=selected.reward.mean(), env_pass=selected.env_pass.mean(),
                        stale_share=selected.stale.mean(), group_se=selected.reward.std() / np.sqrt(len(selected))))
windows = pd.DataFrame(windows)
windows.to_csv(OUT / "telecom_windows.csv", index=False)
summary = dict(
    completed_groups=len(groups), completed_trajectories=len(trace), telecom_trajectories=len(telecom),
    telecom_groups=len(task_groups), pool_tasks=len(pool), default_db_tasks=sum(v == "DB+COMMUNICATE" for v in pool.values()),
    stale_reference_tasks=len(affected), uniform_prefilter_reward_upper_bound=1 - len(affected) / len(pool),
    stale_reference_trajectories=int(telecom.stale_reference.sum()),
    stale_reference_successes=float(telecom.loc[telecom.stale_reference, "reward"].sum()),
    all_environment_passes=int(telecom.env_pass.sum()), reward_successes=int(telecom.reward.sum()),
    env_pass_but_reward_zero=int((telecom.env_pass & (telecom.reward == 0)).sum()),
    db_env_pass_but_reward_zero=int((telecom.env_pass & (telecom.reward == 0) & (telecom.configured_basis == "DB+COMMUNICATE")).sum()),
    matched_controls=pair_summary,
)
(OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

fig, axes = plt.subplots(3, 1, figsize=(11, 12), constrained_layout=True)
x = windows.completed_groups.to_numpy()
axes[0].plot(x, windows.reward * 100, label="Current terminal reward", color="#c86529", linewidth=2)
axes[0].plot(x, windows.env_pass * 100, label="All task environment assertions pass", color="#277da8", linewidth=2)
axes[0].plot(x, windows.stale_share * 100, label="Tasks with an inconsistent reference state", color="#9b647a", linestyle="--")
axes[0].set(title="Telecom: only 32-50 task groups in each mixed 400-group window",
            xlabel="Completed groups across all domains", ylabel="Percent", ylim=(0, 100))
axes[0].legend(frameon=False, fontsize=9)

labels = ["DB + stale reference", "DB + consistent reference", "ENV assertions", "ENV + ACTION"]
keys = [("DB+COMMUNICATE", True), ("DB+COMMUNICATE", False),
        ("ENV_ASSERTION", False), ("ENV_ASSERTION+ACTION", False)]
x = np.arange(4)
rows = [basis.loc[key] for key in keys]
axes[1].bar(x - .17, [r.reward * 100 for r in rows], .34, label="Current terminal reward", color="#c86529")
axes[1].bar(x + .17, [r.env_pass_rate * 100 for r in rows], .34, label="All environment assertions pass", color="#277da8")
axes[1].set(title="Scoring conditions explain much of the low Telecom reward", ylabel="Percent", ylim=(0, 100),
            xticks=x, xticklabels=[f"{label}\nn={int(row.trajectories)} trajectories" for label, row in zip(labels, rows)])
axes[1].legend(frameon=False, fontsize=9)

blocks = [0, 500, 1000]
labels = {"credit": "Old credit", "grpo": "Old GRPO", "credit_repeat": "Old credit repeat"}
current = task_groups.groupby("draw_block").reward.mean()
axes[2].plot(blocks, [current.loc[b] * 100 for b in blocks], marker="o", linewidth=2.5, label="Current credit-signal-v1", color="#c86529")
for name, label in labels.items():
    rows = comparisons[(comparisons.control == name) & comparisons.draw_block.isin(blocks)].sort_values("draw_block")
    axes[2].plot(rows.draw_block, rows.control_reward * 100, marker="o", label=label)
axes[2].set(title="The same sampled tasks produce the same trough across four runs",
            xlabel="Task-draw block (500 draws across all domains)", ylabel="Telecom reward (%)",
            xticks=blocks, xticklabels=["0-499\n54 Telecom groups", "500-999\n52 Telecom groups", "1000-1499\n51 Telecom groups"], ylim=(0, 60))
axes[2].legend(frameon=False, fontsize=9)
for ax in axes:
    ax.grid(alpha=.2)
fig.suptitle("Telecom reward diagnosis: a frozen live-run snapshot", fontsize=15)
fig.savefig(OUT / "telecom_diagnosis.png", dpi=170)
fig.savefig(OUT / "telecom_diagnosis.pdf")
plt.close(fig)
print(json.dumps(summary, indent=2))
