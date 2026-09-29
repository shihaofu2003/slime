"""Reproduce the reward-curve diagnosis from extract.py's compact tables."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from extract import BASE, OUT, ROOT, RUNS

WEIGHTS = pd.Series({"airline": 1148 / 2524, "retail": 563 / 2524,
                     "telecom": 271 / 2524, "banking": 542 / 2524})
LABELS = {"credit": "Credit", "grpo": "GRPO (updates 41-140)",
          "credit_repeat": "Credit no-replay (same sampler)"}
COLORS = {"credit": "#1764ab", "grpo": "#dd8528", "credit_repeat": "#27936e"}
rng = np.random.default_rng(20260927)
summary, all_groups, curves, periods, domain_rows = {}, {}, {}, [], []

for name in RUNS:
    trace = pd.read_csv(OUT / f"{name}_trajectories.csv")
    raw_count = len(trace)
    # Match the existing plotter: latest value at the key's first insertion
    # position. Restarting GRPO reused 761 (group, sample) identities.
    first_seq = trace.groupby(["group", "sample"]).seq.min().rename("first_seq")
    trace = trace.drop_duplicates(["group", "sample"], keep="last")
    trace = trace.join(first_seq, on=["group", "sample"]).sort_values("first_seq")
    updates = pd.read_csv(OUT / f"{name}_updates.csv").sort_values("update")
    group_log = pd.read_csv(OUT / f"{name}_groups.csv").set_index("group")
    groups = trace.groupby("group").agg(
        n=("sample", "size"), task=("task", "first"), domain=("domain", "first"),
        reward=("official", "mean"), removed=("removed", "max"))
    groups = groups.join(group_log)
    all_groups[name] = groups
    if name != "grpo":
        trace["net_progress"] = trace.progress_final - trace.progress_initial
        progress_groups = trace.groupby("group").agg(
            maximum=("net_progress", "max"), std=("net_progress", "std"),
            unavailable=("progress_reason", lambda x: x.notna().any()))
        candidates = groups.join(progress_groups)
        candidates = candidates[(candidates.n == 8) & (candidates.reward == 0)
                                & ~candidates.removed & ~candidates.unavailable]
    accepted = trace.join(group_log[["update"]], on="group").dropna(subset=["update"])
    group_rewards = groups.loc[groups["update"].notna(), "reward"].to_numpy()
    rewards = accepted.groupby("update").official.mean()
    assert len(accepted) == len(updates) * 128
    assert (groups.loc[groups["update"].notna(), "n"] == 8).all()
    assert np.isfinite(updates["train/grad_norm"]).all()
    curves[name] = rewards

    # A diagnostic null: reorder the same empirical accepted task groups into
    # batches of 16 without imposing any temporal change in policy quality.
    simulated = np.array([rng.permutation(group_rewards).reshape(-1, 16).mean(1)
                          for _ in range(4000)])
    cumulative = np.cumsum(np.pad(simulated, ((0, 0), (1, 0))), axis=1)
    smooth = (cumulative[:, 10:] - cumulative[:, :-10]) / 10
    null_ranges = np.ptp(smooth, axis=1)
    observed_range = np.ptp(rewards.rolling(10).mean().dropna())
    stats = {
        "raw_rows": raw_count, "unique_rows": len(trace),
        "logged_updates": len(updates), "accepted_trajectories": len(accepted),
        "batch_reward_std": rewards.std(), "random_16_group_se": np.std(group_rewards, ddof=1) / 4,
        "ma10_range": observed_range,
        "null_ma10_range_5_50_95": np.quantile(null_ranges, [.05, .5, .95]).tolist(),
        "null_range_ge_observed_fraction": float(np.mean(null_ranges >= observed_range)),
        "overall_prefilter_reward": trace.official.mean(),
        "accepted_reward_first20": rewards.head(20).mean(),
        "accepted_reward_last20": rewards.tail(20).mean(),
        "policy_lag_mean": updates["rollout/policy_lag_mean"].mean(),
        "policy_lag_max": updates["rollout/policy_lag_max"].max(),
        "mixed_policy_trajectory_fraction": (accepted.first_policy != accepted.last_policy).mean(),
        "success_wall_seconds": trace.loc[trace.official == 1, "wall_seconds"].mean(),
        "failure_wall_seconds": trace.loc[trace.official == 0, "wall_seconds"].mean(),
        "no_agent_turn_trajectories": int((trace.tokens == 0).sum()),
    }
    for label, rows in [("first5000", trace.head(5000)), ("last5000", trace.tail(5000))]:
        means = rows.groupby("domain").official.mean()
        stats["prefilter_" + label] = dict(raw=rows.official.mean(),
                                          fixed_domain=float((means * WEIGHTS).sum()),
                                          domains=means.to_dict())
    for metric in ["train/grad_norm", "train/kl_loss", "train/entropy_loss",
                   "train/train_rollout_logprob_abs_diff", "train/tis_clipfrac",
                   "rollout/truncated_ratio", "rollout/zero_variance_group_rate"]:
        values = updates[metric]
        stats[metric] = dict(mean=values.mean(), max=values.max(),
                             first20=values.head(20).mean(), last20=values.tail(20).mean())
    filter_counts = updates.filter(like="rollout/filtered_groups/").fillna(0)
    increments = filter_counts.diff().fillna(filter_counts.iloc[0])
    stats["filter_totals_at_last_update"] = filter_counts.iloc[-1].to_dict()
    for label, segment in [("first20", increments.head(20)), ("last20", increments.tail(20))]:
        counts = segment.sum()
        total = counts.sum() + len(segment) * 16
        stats["filter_" + label] = {
            "all_one": counts["rollout/filtered_groups/official_outcome_all_one"] / total,
            "all_zero": counts["rollout/filtered_groups/official_outcome_all_zero"] / total,
            "accepted": len(segment) * 16 / total,
        }
    summary[name] = stats
    if name != "grpo":
        stats["discarded_all_zero_progress"] = dict(
            usable_groups=len(candidates),
            with_positive_net_progress=int((candidates.maximum > 0).sum()),
            with_positive_and_different_net_progress=int(((candidates.maximum > 0) & (candidates["std"] > 1e-6)).sum()))

    for domain in WEIGHTS.index:
        rows = trace[trace.domain == domain]
        used = accepted[accepted.domain == domain]
        complete = groups[(groups.domain == domain) & (groups.n == 8)]
        domain_rows.append(dict(
            run=name, domain=domain, source_share=WEIGHTS[domain],
            completed_share=len(rows) / len(trace), trained_share=len(used) / len(accepted),
            completed_reward=rows.official.mean(), trained_reward=used.official.mean(),
            complete_groups=len(complete), uniform_group_fraction=complete.reward.isin([0, 1]).mean(),
            unusable_group_fraction=complete.removed.mean(),
            wall_seconds=rows.wall_seconds.mean(),
            zero_domain_batches=int((updates["rollout/trained_groups_" + domain] == 0).sum())))

    # Draw-order bins include unusable groups and avoid assigning a policy
    # version to dialogues with no Agent turn (their metadata defaults to 0).
    complete = groups[groups.n == 8].copy()
    complete["period"] = (complete.index // 500).astype(int) * 500
    for period, rows in complete.groupby("period"):
        domains = rows.groupby("domain").reward.mean()
        bootstrap = np.zeros(2000)
        for domain, weight in WEIGHTS.items():
            dg = rows.loc[rows.domain == domain, "reward"].to_numpy()
            indices = rng.integers(len(dg), size=(2000, len(dg)))
            bootstrap += weight * dg[indices].mean(1)
        lower, upper = np.quantile(bootstrap, [.025, .975])
        periods.append(dict(run=name, task_draw_bin=int(period), groups=len(rows),
                            raw_reward=rows.reward.mean(),
                            fixed_domain_reward=float((domains * WEIGHTS).sum()),
                            ci_low=lower, ci_high=upper, **domains.to_dict()))

for name in ("grpo", "credit_repeat"):
    paired = all_groups["credit"].join(all_groups[name], lsuffix="_credit", rsuffix="_other")
    paired = paired[(paired.n_credit == 8) & (paired.n_other == 8)]
    assert (paired.task_credit == paired.task_other).all()
    a, b = paired.reward_credit, paired.reward_other
    ad = a.rolling(50).mean() - a.rolling(1000, min_periods=200, center=True).mean()
    bd = b.rolling(50).mean() - b.rolling(1000, min_periods=200, center=True).mean()
    summary[name]["paired_with_credit"] = dict(
        groups=len(paired), task_identity_agreement=1.0, group_reward_correlation=a.corr(b),
        ma50_correlation=a.rolling(50).mean().corr(b.rolling(50).mean()),
        detrended_ma50_correlation=ad.corr(bd))

for name in ("credit", "credit_repeat"):
    entries = [json.loads(line) for line in (RUNS[name] / "credit.jsonl").open()]
    by_group = {}
    outcome, progress, advantage, weights = [], [], [], []
    combine_error, reasons = 0.0, {}
    for row in entries:
        by_group.setdefault(row["group_index"], []).append(row)
        for reason in row["progress_unavailable_reasons"]:
            reasons[reason] = reasons.get(reason, 0) + 1
        turns = row["turns"]
        n_tokens = sum(t["response_span"][1] - t["response_span"][0] for t in turns)
        for turn in turns:
            o, p, a = (turn[k] for k in ("outcome_advantage", "progress_advantage", "advantage"))
            outcome.append(o); progress.append(p); advantage.append(a)
            weights.append((turn["response_span"][1] - turn["response_span"][0]) / n_tokens)
            combine_error = max(combine_error, abs(a - o - p - turn["format_penalty"]))
    norm_error = 0.0
    for group in by_group.values():
        values = np.array([r["reward"] for r in group])
        expected = (values - values.mean()) / (values.std(ddof=1) + 1e-6)
        norm_error = max(norm_error, max(abs(r["turns"][0]["outcome_advantage"] - e)
                                        for r, e in zip(group, expected)))
    o, p, a, w = map(np.asarray, [outcome, progress, advantage, weights])
    summary[name]["credit_check"] = dict(
        groups=len(by_group), trajectories=len(entries), turns=len(a),
        outcome_normalization_max_error=norm_error, addition_max_error=combine_error,
        outcome_rms=float(np.sqrt(np.average(o * o, weights=w))),
        progress_rms=float(np.sqrt(np.average(p * p, weights=w))),
        total_rms=float(np.sqrt(np.average(a * a, weights=w))),
        sign_flip_fraction=float(np.average(a * o < 0, weights=w)), unavailable_reasons=reasons)

pd.DataFrame(periods).to_csv(OUT / "draw_periods.csv", index=False)
pd.DataFrame(domain_rows).to_csv(OUT / "domain_statistics.csv", index=False)

fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
for name, series in curves.items():
    axes[0, 0].plot(series.index + 1, 100 * series.rolling(10).mean(),
                    label=LABELS[name], color=COLORS[name], linewidth=1.8)
axes[0, 0].set(title="Accepted training groups: trailing 10 updates", xlabel="Optimizer updates",
               ylabel="Official reward (%)", ylim=(45, 73))
axes[0, 0].legend(fontsize=8, frameon=False)
period_table = pd.DataFrame(periods)
for name in RUNS:
    rows = period_table[period_table.run == name]
    y = rows.fixed_domain_reward.to_numpy() * 100
    axes[0, 1].errorbar(rows.task_draw_bin + 250, y,
                       yerr=np.array([y - rows.ci_low * 100, rows.ci_high * 100 - y]),
                       color=COLORS[name], marker="o", capsize=3, linewidth=1.8)
axes[0, 1].set(title="Before filtering: fixed source-domain weights", xlabel="Task draw index (500-group bins)",
               ylabel="Official reward (%)", ylim=(45, 73))
for name, groups in all_groups.items():
    complete = groups[groups.n == 8]
    # Display at 10-group spacing to keep the plotted lines legible.
    series = complete.reward.rolling(50).mean().iloc[::10]
    axes[1, 0].plot(series.index, 100 * series, color=COLORS[name], alpha=.85, linewidth=1.2)
axes[1, 0].set(title="Same sampled tasks: trailing 50 complete groups", xlabel="Task draw / group index",
               ylabel="Reward before filtering (%)")
for i, (name, stats) in enumerate(summary.items()):
    low, median, high = np.array(stats["null_ma10_range_5_50_95"]) * 100
    axes[1, 1].errorbar(i - .08, median, yerr=[[median - low], [high - median]],
                       fmt="o", color=COLORS[name], capsize=7, label="Shuffled groups: median, 5-95%" if i == 0 else None)
    axes[1, 1].scatter(i + .08, stats["ma10_range"] * 100, marker="D", color="#333333",
                       label="Observed" if i == 0 else None)
axes[1, 1].set(title="Peak-to-trough range of the 10-update curve", ylabel="Percentage points",
               xticks=range(3), xticklabels=["Credit", "GRPO", "Credit repeat"], ylim=(0, 15))
axes[1, 1].legend(fontsize=8, frameon=False)
for ax in axes.flat:
    ax.grid(alpha=.2)
fig.suptitle("Mix-RL reward oscillations: sampling noise, filtering, and underlying improvement", fontsize=14)
fig.savefig(OUT / "diagnosis.png", dpi=180)
fig.savefig(OUT / "diagnosis.pdf")
plt.close(fig)

# Existing held-out evaluations: no new model calls or jobs.
eval_paths = {"sft": ROOT / "output/experiments/tau2-sft-areal3-banking-simplified-full-20260920/eval/four-domain-sft-iter4505/seed300_20260920_sft_iter4505_four_domain_summary.json"}
eval_paths.update({p.parent.name: p for p in BASE.glob("eval/**/*summary.json")})
eval_rows, task_scores = [], {}
for label, path in eval_paths.items():
    result = json.load(path.open())
    eval_rows.append(dict(model=label, domain="overall", **{k: result["overall"][k] for k in ("pass_at_1", "pass_at_4_any", "pass_power_4")}))
    task_scores[label] = {}
    for domain, data in result["domains"].items():
        eval_rows.append(dict(model=label, domain=domain, **data["pass_metrics"]))
        results_path = Path(data["results_file"])
        # Some historical summaries retain the container path.
        if not results_path.exists():
            results_path = ROOT / str(results_path).split("slime-async-tau2/", 1)[1]
        simulations = json.load(results_path.open())["simulations"]
        grouped = {}
        for sim in simulations:
            grouped.setdefault(str(sim["task_id"]), []).append(float((sim.get("reward_info") or {}).get("reward", 0.0)))
        task_scores[label][domain] = {key: float(np.mean(values)) for key, values in grouped.items()}
pd.DataFrame(eval_rows).to_csv(OUT / "official_evaluations.csv", index=False)
comparisons = {}
for end, start in [("mix-iter139-four-domain", "mix-iter129-four-domain"),
                   ("vanilla-grpo-iter139-four-domain", "vanilla-grpo-iter129-four-domain"),
                   ("mix-iter139-four-domain", "sft"),
                   ("vanilla-grpo-iter139-four-domain", "sft")]:
    samples, total, mean = np.zeros(5000), 0, 0.0
    for domain, scores in task_scores[end].items():
        before = task_scores[start][domain]
        assert scores.keys() == before.keys()
        delta = np.array([scores[k] - before[k] for k in scores])
        indices = rng.integers(len(delta), size=(5000, len(delta)))
        samples += delta[indices].sum(1)
        mean += delta.sum()
        total += len(delta)
    comparisons[end + " minus " + start] = dict(delta=mean / total,
                                                 paired_task_ci95=(np.quantile(samples, [.025, .975]) / total).tolist())
summary["official_comparisons"] = comparisons
serialized = json.dumps(summary, indent=2, default=lambda value: value.item())
(OUT / "summary.json").write_text(serialized + "\n")
print(serialized)
