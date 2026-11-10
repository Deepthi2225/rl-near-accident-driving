"""
Plot PPO learning curves (from results/<run>.csv) against the expert / CoIL levels
(from results/eval_pedestrian_crossing.json).

    python scripts/plot_results.py --runs ppo_from_coil ppo_scratch
"""
import argparse
import csv
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

LABELS = {"ppo_from_coil": "PPO from CoIL (timid)", "ppo_scratch": "PPO from scratch"}
COLORS = {"ppo_from_coil": "#e8452c", "ppo_scratch": "#3a6ea5"}
REFS = {"expert:aggressive": ("Aggressive expert", "#8c8c8c", "--"),
        "expert:timid": ("Timid expert", "#8c8c8c", ":")}


def read_csv(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return {k: [float(r[k]) for r in rows] for k in rows[0]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", default=["ppo_from_coil", "ppo_scratch"])
    p.add_argument("--eval", default="results/eval_pedestrian_crossing.json")
    p.add_argument("--out", default="results/learning_curves.png")
    args = p.parse_args()

    refs = json.load(open(args.eval)) if os.path.exists(args.eval) else {}
    panels = [("success", "Success rate", "success", 1.0), ("collision", "Collision rate", "collision", 1.0),
              ("mean_time_to_goal", "Time to goal (s)", "mean_time_to_goal", None),
              ("mean_return", "Episode return", "mean_return", None)]
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.2))
    for ax, (col, title, ref_key, ymax) in zip(axes, panels):
        for run in args.runs:
            path = f"results/{run}.csv"
            if not os.path.exists(path):
                continue
            d = read_csv(path)
            ax.plot([s / 1000 for s in d["steps"]], d[col], color=COLORS.get(run), lw=2,
                    label=LABELS.get(run, run))
        for key, (label, color, ls) in REFS.items():
            if key in refs and refs[key].get(ref_key) is not None:
                ax.axhline(refs[key][ref_key], color=color, ls=ls, lw=1.2, label=label)
        ax.set_title(title)
        ax.set_xlabel("Environment steps (thousands)")
        if ymax:
            ax.set_ylim(-0.02, ymax + 0.02)
        ax.grid(alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle("WHEW! - pedestrian dart-out: rolling averages over the last 50 training episodes", fontsize=12)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=130)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
