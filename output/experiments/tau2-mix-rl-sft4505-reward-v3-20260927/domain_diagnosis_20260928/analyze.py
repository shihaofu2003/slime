"""Compare full windows, task-adjusted trends, training exposure and credit."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
DOMAINS = ["airline", "retail", "telecom", "banking"]
RNG = np.random.default_rng(20260928)


def interval(before, after):
    # Resample tasks jointly across windows, keeping each task's observations.
    ids = sorted(set(before.task) | set(after.task))
    a = before.groupby("task").reward.agg(["sum", "count"]).reindex(ids, fill_value=0)
    b = after.groupby("task").reward.agg(["sum", "count"]).reindex(ids, fill_value=0)
    w = RNG.multinomial(len(ids), np.full(len(ids), 1 / len(ids)), size=5000)
    delta = (w @ b["sum"].to_numpy()) / (w @ b["count"].to_numpy()) - (w @ a["sum"].to_numpy()) / (w @ a["count"].to_numpy())
    return np.quantile(delta, [.025, .975]).tolist()


def task_adjusted(rows):
    rows = rows.dropna(subset=["weighted_policy"]).copy()
    rows = rows[rows.groupby("task").task.transform("size") >= 2].copy()
    x = (rows.weighted_policy - rows.groupby("task").weighted_policy.transform("mean")) / 100
    y = rows.reward - rows.groupby("task").reward.transform("mean")
    sums = pd.DataFrame(dict(task=rows.task, xx=x * x, xy=x * y)).groupby("task").sum()
    sums = sums[sums.xx > 0]
    slope = sums.xy.sum() / sums.xx.sum()
    w = RNG.multinomial(len(sums), np.full(len(sums), 1 / len(sums)), size=5000)
    slopes = (w @ sums.xy.to_numpy()) / (w @ sums.xx.to_numpy())
    return dict(tasks=len(sums), groups=int(rows.task.isin(sums.index).sum()),
                gain_per_100_updates=slope, ci=np.quantile(slopes, [.025, .975]).tolist())


def main():
    groups = pd.read_csv(OUT / "groups.csv")
    groups["trained"] = groups.trained.fillna(False)
    trajectories = pd.read_csv(OUT / "trajectories.csv").drop_duplicates(["group", "sample"], keep="last")
    trajectories = trajectories[trajectories.group.isin(groups.group)].copy()
    group_meta = trajectories.groupby("group").agg(task=("task", "first"), weighted_policy=("weighted_policy", "mean"),
                                                    trajectory_mean=("reward", "mean"), reference_tools=("reference_tools", "first"),
                                                    banking_form=("banking_form", "first"), basis=("basis", "first"))
    groups = groups.merge(group_meta, on="group", validate="one_to_one")
    assert np.allclose(groups.reward, groups.trajectory_mean)
    trajectories = trajectories.merge(groups[["group", "completion_order", "trained", "update", "outcome"]], on="group")
    groups.to_csv(OUT / "groups_enriched.csv", index=False)
    credits = pd.read_csv(OUT / "credit.csv").drop_duplicates(["group_index", "sample_index"], keep="last")
    credits = credits.merge(groups[["group", "trained", "completion_order", "update", "outcome", "banking_form"]],
                            left_on="group_index", right_on="group")
    trained = credits[credits.trained].copy()
    assert len(trained) == 140 * 128
    assert np.allclose(trained.span_tokens, trained.trainable_tokens)
    pool = pd.DataFrame([json.loads(line)["metadata"] for line in (OUT.parent / "data/train.jsonl").open()])
    pool_counts = pool.domain.value_counts()
    summaries = []
    trends = {}
    for domain in DOMAINS:
        rows = groups[groups.domain == domain]
        early = rows[rows.completion_order < 400]
        late = rows[rows.completion_order >= len(groups) - 400]
        used = rows[rows.trained]
        c = trained[trained.domain == domain]
        t = trajectories[(trajectories.domain == domain) & trajectories.trained]
        all_t = trajectories[trajectories.domain == domain]
        ci = interval(early, late)
        row = dict(domain=domain, first400=early.reward.mean(), last400=late.reward.mean(),
                   delta=late.reward.mean() - early.reward.mean(), ci_low=ci[0], ci_high=ci[1],
                   first_groups=len(early), last_groups=len(late), completed_groups=len(rows), trained_groups=len(used),
                   pool_tasks=int(pool_counts[domain]), seen_tasks=rows.task.nunique(), trained_tasks=used.task.nunique(),
                   kept_rate=rows.kept.mean(), trained_share=len(used) / groups.trained.sum(),
                   pool_share=pool_counts[domain] / len(pool), all_zero_rate=(rows.outcome == "all_zero").mean(),
                   all_one_rate=(rows.outcome == "all_one").mean(),
                   trained_uniform_fraction=(used.outcome != "mixed").mean(),
                   progress_available=c.progress_available.mean(), mean_tokens=c.trainable_tokens.mean(),
                   outcome_rms=np.sqrt((c.outcome_sum_sq / c.span_tokens).mean()),
                   progress_rms=np.sqrt((c.progress_sum_sq / c.span_tokens).mean()),
                   total_advantage_rms=np.sqrt((c.total_sum_sq / c.span_tokens).mean()),
                   zero_credit_fraction=(c.zero_token_count / c.span_tokens).mean(),
                   opposite_sign_fraction=(c.opposite_sign_tokens / c.span_tokens).mean(),
                   zero_delta_progress_fraction=(c.zero_delta_progress_tokens / c.span_tokens).mean(),
                   mean_policy_lag=(t["update"] - t.first_policy).mean(),
                   invalid_termination=(~all_t.termination.eq("user_stop")).mean(),
                   failure_count=int(all_t.reward.eq(0).sum()))
        summaries.append(row)
        trends[domain] = task_adjusted(rows)
    summary = pd.DataFrame(summaries)
    summary.to_csv(OUT / "domain_summary.csv", index=False)

    bank = groups[groups.domain == "banking"].copy()
    bank_summary = bank.groupby("banking_form").agg(groups=("group", "size"), reward=("reward", "mean"),
                                                   trained=("trained", "sum"), kept=("kept", "sum"))
    for label in ("all_zero", "all_one", "mixed"):
        bank_summary[label] = bank[bank.outcome == label].groupby("banking_form").size().reindex(bank_summary.index, fill_value=0)
    bank_summary.to_csv(OUT / "banking_forms.csv")
    retail = groups[groups.domain == "retail"].copy()
    writes = {"cancel_pending_order", "modify_pending_order_items", "modify_pending_order_address", "modify_pending_order_payment",
              "modify_user_address", "return_delivered_order_items", "exchange_delivered_order_items"}
    retail["kind"] = retail.reference_tools.map(lambda s: "+".join(sorted(set(json.loads(s)) & writes)) or "no_write")
    retail["item_change"] = retail.reference_tools.str.contains("modify_pending_order_items")
    retail_summary = retail.groupby("kind").agg(groups=("group", "size"), reward=("reward", "mean"), trained=("trained", "sum"))
    retail_summary.to_csv(OUT / "retail_task_types.csv")
    conditional = []
    for label, lower, upper in (("first400", 0, 400), ("last400", len(groups) - 400, len(groups))):
        window = retail[retail.completion_order.between(lower, upper - 1)]
        for has_items, part in window.groupby("item_change"):
            conditional.append(dict(period=label, item_change=bool(has_items), groups=len(part), reward=part.reward.mean()))
    pd.DataFrame(conditional).to_csv(OUT / "retail_item_change_windows.csv", index=False)

    history = []
    for start in range(0, len(groups), 400):
        block = groups[groups.completion_order.between(start, min(start + 399, len(groups) - 1))]
        if len(block) < 200:
            continue
        for domain, rows in block.groupby("domain"):
            history.append(dict(start=start, end=int(block.completion_order.max()) + 1, domain=domain,
                                groups=len(rows), reward=rows.reward.mean(), policy=rows.weighted_policy.mean()))
    pd.DataFrame(history).to_csv(OUT / "nonoverlapping_windows.csv", index=False)
    updates = pd.read_csv(OUT / "updates.csv")
    health = {key: dict(first20=float(updates.head(20)[key].mean()), last20=float(updates.tail(20)[key].mean()),
                       minimum=float(updates[key].min()), maximum=float(updates[key].max()))
              for key in ("train/grad_norm", "train/entropy_loss", "train/kl_loss", "train/tis_clipfrac",
                          "train/tis_weight", "rollout/policy_lag_mean", "rollout/policy_lag_max")}
    result = dict(completed_groups=len(groups), trajectories=len(trajectories), trained_groups=int(groups.trained.sum()),
                  trained_trajectories=len(trained), task_adjusted=trends, health=health,
                  domains=summary.to_dict("records"), retail_item_windows=conditional)
    (OUT / "summary.json").write_text(json.dumps(result, indent=2) + "\n")

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    x = np.arange(4)
    axes[0, 0].bar(x - .18, summary.first400 * 100, .36, label="First full 400-group window")
    axes[0, 0].bar(x + .18, summary.last400 * 100, .36, label="Last full 400-group window")
    axes[0, 0].set(xticks=x, xticklabels=DOMAINS, ylim=(0, 100), ylabel="Reward (%)", title="Comparable windows: Airline gains the most")
    axes[0, 0].legend(fontsize=8, frameon=False)
    for i, row in summary.iterrows():
        axes[0, 0].text(i, max(row.first400, row.last400) * 100 + 2, f"{row.delta * 100:+.1f} pp", ha="center", fontsize=9)
    changes = np.array([trends[d]["gain_per_100_updates"] for d in DOMAINS]) * 100
    bounds = np.array([trends[d]["ci"] for d in DOMAINS]) * 100
    axes[0, 1].errorbar(x, changes, yerr=np.stack([changes - bounds[:, 0], bounds[:, 1] - changes]), fmt="o", capsize=5)
    axes[0, 1].axhline(0, color="grey", linewidth=1)
    axes[0, 1].set(xticks=x, xticklabels=DOMAINS, ylabel="Gain per 100 policy updates (pp)", title="Repeated-task trends; task bootstrap 95% intervals")
    forms = bank_summary.index.tolist()
    axes[1, 0].bar(np.arange(len(forms)), bank_summary.reward * 100)
    axes[1, 0].set(xticks=np.arange(len(forms)), xticklabels=[s.split("_")[0].upper() + "\n" + ("single" if "single" in s else "decision" if "decision" in s else "") for s in forms],
                   ylabel="Reward (%)", ylim=(0, 108), title="Banking: high average hides weak evidence quoting")
    for i, value in enumerate(bank_summary.reward):
        axes[1, 0].text(i, value * 100 + 2, f"{value * 100:.1f}", ha="center", fontsize=8)
    for j, period in enumerate(("first400", "last400")):
        rows = pd.DataFrame(conditional).query("period == @period").sort_values("item_change")
        axes[1, 1].bar(np.arange(2) + (j - .5) * .36, rows.reward * 100, .36, label=period)
    axes[1, 1].set(xticks=[0, 1], xticklabels=["Other retail tasks", "Includes item modification"],
                   ylabel="Reward (%)", ylim=(0, 100), title="Retail: gains vary by task type")
    axes[1, 1].legend(frameon=False)
    for axis in axes.flat:
        axis.grid(axis="y", alpha=.2)
    fig.suptitle("Reward v3: domain learning and training-signal diagnosis (not held-out evaluation)")
    fig.savefig(OUT / "domain_diagnosis.png", dpi=180)
    fig.savefig(OUT / "domain_diagnosis.pdf")
    print(summary.to_string(index=False))
    print(json.dumps(dict(task_adjusted=trends, retail_item_windows=conditional), indent=2))


if __name__ == "__main__":
    main()
