"""Significance tests on the final returns of each method.

Usage:
    python stats.py results_2M
Each seed gives one number: the mean return of its last 100 training episodes.
"""
import glob
import json
import os
import sys

import numpy as np
from scipy import stats

RES_DIR = sys.argv[1] if len(sys.argv) > 1 else "results_2M"
ENV = "LunarLander-v3"
METHODS = ["ppo", "grpo", "bappo"]


def final_returns(method):
    vals = []
    for f in sorted(glob.glob(os.path.join(RES_DIR, f"{ENV}_{method}_s*.json"))):
        with open(f) as fh:
            ep = json.load(fh)["episodes"]
        vals.append(np.mean([r for _, r in ep[-100:]]))
    return np.array(vals)


def main():
    data = {m: final_returns(m) for m in METHODS}
    for m in METHODS:
        print(f"{m:<7} per-seed final: {np.round(data[m], 1)}  "
              f"mean {data[m].mean():.1f}  std {data[m].std(ddof=1):.1f}")
    print()
    for a, b in [("bappo", "ppo"), ("bappo", "grpo"), ("grpo", "ppo")]:
        t = stats.ttest_ind(data[a], data[b], equal_var=False)
        u = stats.mannwhitneyu(data[a], data[b], alternative="two-sided")
        diff = data[a].mean() - data[b].mean()
        print(f"{a} vs {b}: difference {diff:+.1f}  "
              f"Welch t-test p = {t.pvalue:.4f}  Mann-Whitney p = {u.pvalue:.4f}")


if __name__ == "__main__":
    main()
