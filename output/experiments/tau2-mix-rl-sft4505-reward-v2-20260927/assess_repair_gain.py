"""Quantify score correction on fixed trajectories; this is not an RL ablation."""

import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
run = ROOT / "output/experiments/tau2-mix-rl-sft4505-credit-signal-v1-20260927/arms/async/20260927_credit-signal-v1-train"
scores = pd.read_csv(OUT / "telecom_counterfactual.csv")
assert scores.corrected_reward.notna().all(), "Inspect replay errors before reporting an aggregate"
paired = scores.groupby("group").agg(old=("old_reward", "mean"), corrected=("corrected_reward", "mean"), n=("sample", "size"))
assert (paired.n == 8).all()
groups, batches, completed = {}, {}, set()
for line in (run / "run.log").open(errors="replace"):
    match = re.search(r"tau2_credit_group ({.*})", line)
    if match:
        row = json.loads(match[1])
        groups[row["group"]] = row
    match = re.search(r"tau2_pool_batch update=(\d+) groups=(\[[^\]]*\])", line)
    if match:
        batches[int(match[1])] = json.loads(match[2])
    match = re.search(r"tau2_trainer batch=(\d+)", line)
    if match:
        completed.add(int(match[1]))
trained_ids = [key for update in sorted(completed) for key in batches[update] if key in paired.index]
trained = paired.loc[trained_ids]
windows = []
ordered = list(groups.values())
for end in range(400, len(ordered) + 1, 20):
    ids = [g["group"] for g in ordered[end - 400:end] if g["domain"] == "telecom"]
    window = paired.loc[ids]
    windows.append(dict(completed_groups=end, telecom_groups=len(window), old=window.old.mean(), corrected=window.corrected.mean()))
windows = pd.DataFrame(windows)
summary = dict(
    scope="Fixed v1 Telecom trajectories; only scoring corrected; no new rollouts or optimizer updates",
    trajectories=len(scores), groups=len(paired), distinct_tasks=scores.task.nunique(),
    old_reward=scores.old_reward.mean(), corrected_reward=scores.corrected_reward.mean(),
    score_delta=scores.corrected_reward.mean() - scores.old_reward.mean(),
    flips_0_to_1=int(((scores.old_reward == 0) & (scores.corrected_reward == 1)).sum()),
    flips_1_to_0=int(((scores.old_reward == 1) & (scores.corrected_reward == 0)).sum()),
    still_failed=int(((scores.old_reward == 0) & (scores.corrected_reward == 0)).sum()),
    binary_informative_groups_old=int(((paired.old > 0) & (paired.old < 1)).sum()),
    binary_informative_groups_corrected=int(((paired.corrected > 0) & (paired.corrected < 1)).sum()),
    trained_groups=len(trained),
    trained_all_zero_groups_recovering_binary_variance=int(((trained.old == 0) & (trained.corrected > 0) & (trained.corrected < 1)).sum()),
    whole_mix_score_delta_from_telecom_only=float((paired.corrected - paired.old).sum() / len(groups)),
    window=dict(mixed_groups=400, min_telecom_groups=int(windows.telecom_groups.min()), max_telecom_groups=int(windows.telecom_groups.max()),
                old_std=windows.old.std(), corrected_std=windows.corrected.std(),
                old_peak_to_trough=windows.old.max() - windows.old.min(),
                corrected_peak_to_trough=windows.corrected.max() - windows.corrected.min()),
)
(OUT / "repair_gain_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
windows.to_csv(OUT / "repair_gain_windows.csv", index=False)

fig, axes = plt.subplots(2, 1, figsize=(11, 7), constrained_layout=True)
for key, label, color in [("old", "Original scoring", "#ca6b32"), ("corrected", "Corrected scoring (same actions)", "#216aa7")]:
    axes[0].plot(windows.completed_groups, windows[key] * 100, color=color, linewidth=2, label=label)
    axes[1].plot(windows.completed_groups, (windows[key] - windows[key].mean()) * 100, color=color, linewidth=1.7, label=label)
axes[0].set(title="Telecom: trailing 400 mixed-domain groups", ylabel="Reward (%)", ylim=(0, 100))
axes[1].set(title="Each curve centered on its own mean", ylabel="Percentage points", xlabel="Completed groups across all domains", ylim=(-20, 20))
for ax in axes:
    ax.grid(alpha=.2)
    ax.legend(frameon=False)
fig.suptitle("Offline score correction: no policy training or new trajectories", fontsize=14)
fig.savefig(OUT / "repair_gain.png", dpi=170)
plt.close(fig)
print(json.dumps(summary, indent=2))
