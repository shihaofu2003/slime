"""Rebuild the site's measured-result figures from the checked-in numeric data.

Run: python3 scripts/build_figures.py --font /path/to/SourceHanSansSC-Regular.otf
Each figure is exported as SVG and PNG. Concept diagrams in HTML are illustrative.
"""
import argparse
import json
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("--font", type=Path)
args = parser.parse_args()
if args.font:
    font_manager.fontManager.addfont(str(args.font))
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=args.font).get_name()
else:
    plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "Source Han Sans SC", "DejaVu Sans"]
plt.rcParams.update({"font.size": 11, "axes.titlesize": 12, "axes.labelsize": 10,
                     "text.color": "#233b34", "axes.labelcolor": "#54655e",
                     "xtick.color": "#54655e", "ytick.color": "#54655e",
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.spines.left": False, "axes.spines.bottom": False,
                     "axes.unicode_minus": False, "svg.fonttype": "path",
                     "figure.facecolor": "white", "axes.facecolor": "white"})
TEAL, BLUE, GOLD, GRAY, RED = "#246650", "#4e78a4", "#b4893e", "#9ba5ab", "#b46148"
data = json.loads((ROOT / "data/results.json").read_text())
studies = json.loads((ROOT / "data/supporting-studies.json").read_text())
models = {m["id"]: m for m in data["models"]}
metric_keys = ["pass_at_1", "pass_at_4_any", "pass_power_4"]
metric_names = ["pass@1", "pass@4(any)", "pass^4"]
OUT = ROOT / "assets/figures"
OUT.mkdir(parents=True, exist_ok=True)


def save(fig, name):
    fig.savefig(OUT / (name + ".svg"), bbox_inches="tight", pad_inches=0.24)
    fig.savefig(OUT / (name + ".png"), bbox_inches="tight", pad_inches=0.24, dpi=170)
    plt.close(fig)


def grid(ax, limit, label="成功率（%）"):
    ax.set_ylim(0, limit)
    ax.set_ylabel(label)
    ax.yaxis.grid(True, color="#e8ece7", linewidth=.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0, pad=8)


def annotate(ax, bars, suffix="", digits=2, fontsize=10):
    for bar in bars:
        v = bar.get_height()
        label = Decimal(str(round(float(v), 10))).quantize(Decimal(10) ** -digits, rounding=ROUND_HALF_UP)
        ax.annotate(f"{label}{suffix}",
                    (bar.get_x() + bar.get_width()/2, v),
                    xytext=(0, 6), textcoords="offset points", ha="center", fontsize=fontsize)


def grouped(name, labels, values, limit=100):
    fig, ax = plt.subplots(figsize=(9.2, 3.5), layout="constrained")
    grid(ax, limit)
    x = np.arange(3)
    for i, (label, vals) in enumerate(zip(labels, values)):
        bars = ax.bar(x + (i - .5) * .31, vals, .27, label=label,
                      color=[GRAY, TEAL][i], zorder=3)
        annotate(ax, bars)
    ax.set_xticks(x, metric_names)
    ax.legend(loc="upper center", bbox_to_anchor=(.5, 1.16), ncol=2, frameon=False)
    save(fig, name)


stages = ["raw", "sft", "mix", "opd59"]
stage_labels = ["Raw Instruct", "SFT · iter4505", "Mix RL · iter139", "OPD · iter59"]
for key, metric in zip(metric_keys, metric_names):
    fig, ax = plt.subplots(figsize=(9.2, 3.5), layout="constrained")
    grid(ax, 60)
    vals = [models[s]["overall"][key] * 100 for s in stages]
    bars = ax.bar(np.arange(4), vals, .54, color=[GRAY, GOLD, BLUE, TEAL], zorder=3)
    annotate(ax, bars, "%", fontsize=12)
    ax.set_xticks(np.arange(4), stage_labels)
    ax.text(.01, 1.08, metric + " · 四领域同协议", transform=ax.transAxes, fontsize=12)
    save(fig, "stages-" + key)

    fig, ax = plt.subplots(figsize=(9.2, 4.0), layout="constrained")
    grid(ax, 100)
    domains = ["airline", "retail", "telecom", "banking_knowledge"]
    x = np.arange(4)
    for i, s in enumerate(stages):
        vals = [models[s]["domains"][d][key] * 100 for d in domains]
        bars = ax.bar(x + (i - 1.5)*.20, vals, .175, label=stage_labels[i].split(" ·")[0],
                      color=[GRAY, GOLD, BLUE, TEAL][i], zorder=3)
        annotate(ax, bars, digits=2, fontsize=8)
    ax.set_xticks(x, ["Airline\n20 题", "Retail\n40 题", "Telecom\n40 题", "Banking\n97 题"])
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(.5, 1.15), frameon=False)
    save(fig, "domains-" + key)

s = studies["evaluation_speed"]
fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.5), layout="constrained")
for ax, field, label, limit in zip(axes, ["minutes", "gpu_hours"], ["作业耗时（分钟）", "资源占用（GPU-hours）"], [210, 15]):
    grid(ax, limit, label)
    bars = ax.bar(np.arange(3), s[field], .55, color=[GRAY, BLUE, TEAL])
    annotate(ax, bars)
    ax.set_xticks(np.arange(3), ["串行\n4 GPU", "固定并发\n8 GPU", "弹性并发\n8 GPU"])
save(fig, "evaluation-speed")

grouped("sft-quality", studies["sft_quality"]["labels"], studies["sft_quality"]["metrics"])
grouped("banking-sft", studies["banking_sft"]["labels"], studies["banking_sft"]["metrics"], 12)
grouped("credit-control", ["Vanilla GRPO · iter139", "状态进度 RL · iter139"],
        [[models[s]["overall"][k]*100 for k in metric_keys] for s in ("vanilla", "mix")], 60)

s = studies["banking_tasks"]
fig, ax = plt.subplots(figsize=(9.2, 3.9), layout="constrained")
left = np.zeros(7)
for k, label, color in zip(["train", "dev", "challenge"], ["训练 542", "开发 75", "挑战 30"], [TEAL, BLUE, GOLD]):
    bars = ax.barh(s["labels"], s[k], left=left, label=label, color=color, height=.62)
    for b, val in zip(bars, s[k]):
        if val:
            ax.text(b.get_x()+b.get_width()/2, b.get_y()+b.get_height()/2,
                    str(val), ha="center", va="center", color="white", fontsize=9)
    left += s[k]
ax.invert_yaxis()
ax.set_xlim(0, 200)
ax.set_xlabel("场景数")
ax.xaxis.grid(True, color="#e8ece7")
ax.set_axisbelow(True)
ax.tick_params(length=0)
ax.legend(loc="upper center", bbox_to_anchor=(.5, 1.15), ncol=3, frameon=False)
save(fig, "banking-tasks")

s = studies["rl_speed"]
fig, axes = plt.subplots(1, 3, figsize=(9.2, 3.2), layout="constrained")
for ax, field, label, limit in zip(axes, ["minutes", "ready_wait_seconds", "trainer_seconds"],
                                ["20 次更新（分钟）", "等一批数据（秒 / 更新）", "Trainer 计算（秒 / 更新）"],
                                [450, 1050, 110]):
    grid(ax, limit, label)
    bars = ax.bar([0, 1], s[field], .55, color=[GRAY, TEAL])
    annotate(ax, bars)
    ax.set_xticks([0, 1], ["优化前", "优化后"])
save(fig, "rl-speed")

s = studies["opd_buffer"]
fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.25), layout="constrained")
for ax, field, label, limit in zip(axes, ["mean_lag", "max_lag"],
                                ["实测平均滞后（更新数）", "实测最大滞后（更新数）"], [11, 25]):
    grid(ax, limit, label)
    bars = ax.bar([0, 1], s[field], .5, color=[GRAY, TEAL])
    annotate(ax, bars)
    ax.set_xticks([0, 1], s["labels"])
save(fig, "opd-buffer")

fig, axes = plt.subplots(1, 3, figsize=(10, 3.35), layout="constrained")
iters = [9, 19, 29, 39, 49, 59]
for ax, key, label in zip(axes, metric_keys, metric_names):
    vals = [models[f"opd{i}"]["overall"][key]*100 for i in iters]
    ax.plot(iters, vals, "o-", color=TEAL, linewidth=1.8, markersize=4, label="OPD")
    ax.scatter([59], [vals[-1]], s=80, color=TEAL, edgecolor="white", zorder=4)
    ax.axhline(models["mix"]["overall"][key]*100, color=BLUE, linestyle="--", linewidth=1, label="Mix 初始化")
    ax.set_title(label)
    ax.set_xticks(iters)
    ax.set_xlabel("Checkpoint iter")
    grid(ax, 60)
    ax.annotate(f"{vals[-1]:.2f}%", (59, vals[-1]), xytext=(-8, 11), textcoords="offset points", ha="right", color=TEAL)
    ax.axvspan(56, 62, color=TEAL, alpha=.06)
axes[1].legend(ncol=2, frameon=False, loc="upper center", bbox_to_anchor=(.5, 1.35))
save(fig, "opd-checkpoints")
print(f"Generated {len(list(OUT.glob('*.svg')))} measured-result figures (SVG + PNG).")
