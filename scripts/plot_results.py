"""
Learning curves and summary table across seeds.

    python scripts/plot_results.py

Reads results/<method>_s<seed>.csv (training curves) and
results/eval_pedestrian_crossing.json (evaluation), and writes
results/learning_curves.png and results/summary.md.
Curves show the mean over seeds (line) and the min-max range (band).
"""
import argparse
import csv
import glob
import json
import os
import re

import sys

import matplotlib
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from whew.scenarios import SCENARIOS  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

METHODS = {"ppo_from_coil": ("PPO from CoIL (timid)", "#e8452c"),
           "ppo_scratch": ("PPO from scratch", "#3a6ea5"),
           "ppo_switcher": ("PPO mode switcher (H-ReIL)", "#2a9d63")}
REFS = {"expert:aggressive": ("Aggressive expert", "--"), "expert:timid": ("Timid expert", ":")}
METRICS = [("success", "Success rate", True), ("collision", "Collision rate", True),
           ("timeout", "Timeout rate", True), ("mean_time_to_goal", "Time to goal (s)", False),
           ("mean_min_dist", "Closest distance (m)", False),
           ("mean_max_speed_kmh", "Top speed (km/h)", False), ("mean_return", "Return", False)]


def read_csv(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}


def method_of(spec):
    m = re.search(r"(ppo_[a-z_]+?)_s\d+", spec)
    return m.group(1) if m else spec


def plot_curves(evals, out, res_dir, title):
    panels = [("success", "Success rate", 1.0), ("collision", "Collision rate", 1.0),
              ("mean_time_to_goal", "Time to goal (s)", None), ("mean_return", "Episode return", None)]
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.2))
    n_seeds = {}
    for method, (label, color) in METHODS.items():
        runs = [read_csv(p) for p in sorted(glob.glob(f"{res_dir}/{method}_s*.csv"))]
        if not runs:
            continue
        n_seeds[method] = len(runs)
        n = min(len(r["steps"]) for r in runs)
        steps = runs[0]["steps"][:n] / 1000
        for ax, (col, _, _) in zip(axes, panels):
            y = np.stack([r[col][:n] for r in runs])
            ax.plot(steps, np.nanmean(y, 0), color=color, lw=2, label=f"{label} ({len(runs)} seeds)")
            ax.fill_between(steps, np.nanmin(y, 0), np.nanmax(y, 0), color=color, alpha=0.15, lw=0)
    for ax, (col, panel_title, ymax) in zip(axes, panels):
        for key, (label, ls) in REFS.items():
            if key in evals and evals[key].get(col) is not None:
                ax.axhline(evals[key][col], color="#8c8c8c", ls=ls, lw=1.2, label=label)
        ax.set_title(panel_title)
        ax.set_xlabel("Environment steps (thousands)")
        if ymax:
            ax.set_ylim(-0.02, ymax + 0.02)
        ax.grid(alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle(f"WHEW! - {title}: training curves (rolling 50 episodes; line = mean, band = min-max "
                 "over seeds)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print(f"Saved {out}")


def summary_table(evals, out):
    groups = {}
    for spec, r in evals.items():
        groups.setdefault(method_of(spec), []).append(r)
    names = {"expert:aggressive": "Aggressive expert", "expert:timid": "Timid expert",
             "coil:aggressive": "CoIL, aggressive branch", "coil:timid": "CoIL, timid branch",
             **{k: v[0] for k, v in METHODS.items()}}
    header = "| Policy | Runs | " + " | ".join(t for _, t, _ in METRICS) + " |"
    lines = [header, "|" + "---|" * (len(METRICS) + 2)]
    for key, rs in groups.items():
        cells = []
        for col, _, pct in METRICS:
            vals = np.array([r[col] for r in rs if r.get(col) is not None], dtype=float)
            if not len(vals):
                cells.append("-")
                continue
            fmt = (lambda v: f"{100 * v:.0f}%") if pct else (lambda v: f"{v:.1f}")
            cells.append(fmt(vals.mean()) + (f" ± {fmt(vals.std())}" if len(vals) > 1 else ""))
        episodes = rs[0]["episodes"]
        lines.append(f"| {names.get(key, key)} | {len(rs)} x {episodes} eps | " + " | ".join(cells) + " |")
    text = "\n".join(lines)
    with open(out, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print(text)
    print(f"Saved {out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scenario", choices=list(SCENARIOS), default="pedestrian")
    p.add_argument("--eval", default=None, help="default: the scenario's evaluation file")
    args = p.parse_args()
    sc = SCENARIOS[args.scenario]
    args.eval = args.eval or sc["eval"]
    res_dir = sc["results"]
    evals = json.load(open(args.eval)) if os.path.exists(args.eval) else {}
    plot_curves(evals, f"{res_dir}/learning_curves.png", res_dir, sc["title"])
    if evals:
        summary_table(evals, f"{res_dir}/summary.md")


if __name__ == "__main__":
    main()
