"""Make the learning-curve figure and results table from results/*.json.

Usage:
    python analyze.py LunarLander-v3 200
(second argument = score that counts as solved)
"""
import glob
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

env = sys.argv[1] if len(sys.argv) > 1 else "LunarLander-v3"
solved = float(sys.argv[2]) if len(sys.argv) > 2 else 200.0
RES_DIR = sys.argv[3] if len(sys.argv) > 3 else "results"
METHODS = [("ppo", "PPO"), ("grpo", "GRPO"), ("bappo", "BA-PPO")]
COLORS = {"ppo": "#534AB7", "grpo": "#888780", "bappo": "#0F6E56"}


def moving_avg(x, w=100):
    x = np.asarray(x)
    c = np.cumsum(np.insert(x, 0, 0.0))
    out = np.empty(len(x))
    for i in range(len(x)):
        lo = max(0, i - w + 1)
        out[i] = (c[i + 1] - c[lo]) / (i + 1 - lo)
    return out


def load(method):
    runs = []
    for f in sorted(glob.glob(os.path.join(RES_DIR, f"{env}_{method}_s*.json"))):
        with open(f) as fh:
            runs.append(json.load(fh))
    return runs


def main():
    grid_max = None
    data = {}
    for m, _ in METHODS:
        runs = load(m)
        data[m] = runs
        for r in runs:
            end = r["episodes"][-1][0]
            grid_max = end if grid_max is None else min(grid_max, end)
    if grid_max is None:
        print("No results found.")
        return
    grid = np.linspace(0, grid_max, 200)

    fig, ax = plt.subplots(figsize=(8, 5))
    rows = []
    for m, label in METHODS:
        runs = data[m]
        if not runs:
            continue
        curves, finals, to_solve, times = [], [], [], []
        for r in runs:
            steps = np.array([s for s, _ in r["episodes"]])
            sm = moving_avg([x for _, x in r["episodes"]])
            curves.append(np.interp(grid, steps, sm))
            finals.append(sm[-1])  # mean of each run's last 100 episodes (same as stats.py)
            hit = np.where(sm >= solved)[0]
            to_solve.append(steps[hit[0]] if len(hit) else None)
            times.append(r["wall_time"] / 60)
        curves = np.array(curves)
        mean, std = curves.mean(0), curves.std(0)
        ax.plot(grid, mean, color=COLORS[m], label=f"{label} (n={len(runs)})", linewidth=2)
        ax.fill_between(grid, mean - std, mean + std, color=COLORS[m], alpha=0.15)

        reached = [s for s in to_solve if s is not None]
        solve_txt = (f"{np.mean(reached):,.0f} ({len(reached)}/{len(runs)} seeds)"
                     if reached else f"not reached (0/{len(runs)})")
        rows.append(f"| {label} | {np.mean(finals):.1f} ± {np.std(finals, ddof=1):.1f} | "
                    f"{solve_txt} | {np.mean(times):.1f} |")

    ax.axhline(solved, color="#888780", linestyle="--", linewidth=1)
    ax.set_xlabel("Environment steps")
    ax.set_ylabel("Episode return (100-episode moving average)")
    ax.set_title(f"{env}: mean ± std over seeds")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"learning_curves_{env}_{RES_DIR}.png", dpi=150)

    table = ["| Method | Final return (mean ± std) | Steps to reach solved | Wall time (min) |",
             "|---|---|---|---|"] + rows
    with open(f"results_{env}_{RES_DIR}.md", "w") as f:
        f.write("\n".join(table) + "\n")
    print("\n".join(table))


if __name__ == "__main__":
    main()
