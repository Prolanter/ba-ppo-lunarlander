"""Build the report figures from the saved results.

Run from the project root:
    python report/make_figures.py

Every figure is saved twice: as a vector PDF (used in the LaTeX report) and as a
PNG preview. Vector PDFs stay sharp at any zoom level.
"""
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
OUT = os.path.join(HERE, "figures")
os.makedirs(OUT, exist_ok=True)
ENV = "LunarLander-v3"
COL = {"ppo": "#534AB7", "grpo": "#888780", "bappo": "#0F6E56"}
NAME = {"ppo": "PPO", "grpo": "GRPO", "bappo": "BA-PPO"}
plt.rcParams.update({
    "font.size": 11,
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "mathtext.fontset": "dejavusans",
    "pdf.fonttype": 42,   # embed TrueType fonts so the PDF text stays selectable
})


def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".pdf"))
    fig.savefig(os.path.join(OUT, name + ".png"), dpi=200)
    plt.close(fig)


def path(*parts):
    return os.path.join(ROOT, *parts)


def final_return(file):
    ep = json.load(open(file))["episodes"]
    return float(np.mean([r for _, r in ep[-100:]]))


def test_scores(folder, method):
    files = sorted(glob.glob(path(folder, f"{ENV}_{method}_s[0-4].json")))
    return np.array([final_return(f) for f in files])


def tuning_scores(method, setting):
    files = sorted(glob.glob(path("results_tune", method, setting, f"{ENV}_{method}_s10[01].json")))
    return np.array([final_return(f) for f in files])


# ------------------------------------------------------------------ 1. Architecture
def fig_architecture():
    fig, ax = plt.subplots(figsize=(8.2, 8.8))
    ax.set_xlim(0, 10.6)
    ax.set_ylim(0, 11)
    ax.axis("off")

    def box(x, y, w, h, title, lines, fill, edge, tcol):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.18",
                                    facecolor=fill, edgecolor=edge, linewidth=1.2))
        ax.text(x + w / 2, y + h - 0.34, title, ha="center", va="center",
                fontsize=12, fontweight="bold", color=tcol)
        ax.text(x + w / 2, y + h / 2 - 0.22, "\n".join(lines), ha="center", va="center",
                fontsize=10.2, color="#2C2C2A", linespacing=1.5)

    def arrow(x1, y1, x2, y2, color="#444441"):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=1.3))

    box(2.5, 9.4, 5.0, 1.25, "Environment",
        ["8 state values in, 4 discrete actions out"],
        "#F1EFE8", "#5F5E5A", "#2C2C2A")
    arrow(5.0, 9.4, 5.0, 8.45)

    box(2.5, 7.2, 5.0, 1.25, "Rollouts: 8 episodes per start state",
        ["each episode replays the same start seed", "and its total return is recorded"],
        "#F1EFE8", "#5F5E5A", "#2C2C2A")
    arrow(3.8, 7.2, 2.4, 6.45)
    arrow(6.2, 7.2, 7.6, 6.45)

    box(0.3, 4.95, 4.2, 1.45, "Critic branch (GAE)",
        [r"$V(s)$ trained by regression",
         r"$\hat{A}^{\mathrm{GAE}}_t$ per step, normalized",
         "over the batch"],
        "#EEEDFE", "#534AB7", "#3C3489")
    box(5.5, 4.95, 4.2, 1.45, "Group branch (GRPO)",
        ["episode return minus group mean,",
         r"$\hat{A}^{\mathrm{GRPO}}_i$ per episode, divided",
         "by group std (no critic)"],
        "#F1EFE8", "#5F5E5A", "#2C2C2A")
    arrow(2.4, 4.95, 3.9, 3.85)
    arrow(7.6, 4.95, 6.1, 3.85)

    box(2.0, 2.55, 6.0, 1.3, "Blended advantage (BA-PPO)",
        [r"$\hat{A}_t = \alpha\,\hat{A}^{\mathrm{GRPO}}_i + (1-\alpha)\,\hat{A}^{\mathrm{GAE}}_t$,"
         r"   $\alpha = 0.5$"],
        "#E1F5EE", "#0F6E56", "#04342C")
    arrow(5.0, 2.55, 5.0, 1.75)

    box(2.5, 0.35, 5.0, 1.3, "Clipped PPO update",
        ["actor: clipped ratio, $\\epsilon = 0.2$, 4 epochs",
         "critic: regression onto GAE return targets"],
        "#F1EFE8", "#5F5E5A", "#2C2C2A")

    # Loop back to the environment, routed around the right-hand boxes.
    ax.plot([7.5, 10.05, 10.05, 7.5], [1.0, 1.0, 10.0, 10.0],
            color="#888780", lw=1.1, linestyle="--")
    arrow(10.05, 10.0, 7.52, 10.0, color="#888780")
    ax.text(10.2, 5.9, "next update", rotation=90, va="center", ha="left",
            fontsize=9.5, color="#5F5E5A")

    ax.text(0.3, 0.05, r"$\alpha = 0$: critic (GAE) term only.   $\alpha = 1$: group (GRPO) term only.",
            fontsize=9.5, color="#5F5E5A")
    fig.tight_layout()
    save(fig, "fig_architecture")


# ------------------------------------------------------------------ 2. Comparison, one minibatch per figure
def fig_comparison(folder, minibatch):
    fig, ax = plt.subplots(figsize=(5.8, 4.6))
    methods = ["ppo", "grpo", "bappo"]
    means, stds = [], []
    for i, m in enumerate(methods):
        vals = test_scores(folder, m)
        means.append(vals.mean())
        stds.append(vals.std(ddof=1))
        jitter = np.linspace(-0.12, 0.12, len(vals))
        ax.scatter(np.full(len(vals), i) + jitter, vals, s=26, color=COL[m], zorder=3,
                   edgecolor="white", linewidth=0.6)
    x = np.arange(len(methods))
    ax.bar(x, means, yerr=stds, color=[COL[m] for m in methods], alpha=0.25, capsize=5,
           width=0.55, edgecolor=[COL[m] for m in methods], error_kw=dict(ecolor="#2C2C2A"))
    for xi, mu in zip(x, means):
        ax.text(xi, -28, f"{mu:.0f}", ha="center", va="top", fontsize=10, color="#2C2C2A")
    ax.axhline(200, color="#888780", linestyle="--", linewidth=1)
    ax.text(2.45, 204, "solved (200)", color="#5F5E5A", fontsize=9, ha="right")
    ax.set_xticks(x, [NAME[m] for m in methods])
    ax.set_ylim(0, 290)
    ax.set_ylabel("Final return (mean of last 100 episodes)")
    ax.set_title(f"Minibatch {minibatch}: five test seeds")
    ax.grid(axis="y", alpha=0.3)
    fig.text(0.5, 0.01, "Bars: mean ± sample std. Dots: individual seeds.",
             ha="center", fontsize=9, color="#5F5E5A")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save(fig, f"fig_comparison_mb{minibatch}")


# ------------------------------------------------------------------ 3. Landing outcomes, one minibatch per figure
def outcome_rates(files):
    crash = landed = total = 0
    for f in files:
        p = json.load(open(f))
        crash += p["crashes"]
        landed += p["landed"]
        total += p["episodes_done"]
    return 100 * landed / total, 100 * crash / total, 100 * (total - crash - landed) / total


def fig_outcomes(minibatch):
    if minibatch == 512:
        files = {"ppo": glob.glob(path("progress", f"{ENV}_ppo_s[0-4].json")),
                 "grpo": glob.glob(path("progress", f"{ENV}_grpo_s[0-4].json")),
                 "bappo": glob.glob(path("progress", f"{ENV}_bappo_s[0-4].json"))}
    else:
        files = {"ppo": glob.glob(path("progress_tune", "ppo", "mb64", f"{ENV}_ppo_s[0-4].json")),
                 "grpo": glob.glob(path("progress_mb64", f"{ENV}_grpo_s[0-4].json")),
                 "bappo": glob.glob(path("progress_mb64", f"{ENV}_bappo_s[0-4].json"))}
    methods = [m for m in ["ppo", "grpo", "bappo"] if files[m]]
    rates = [outcome_rates(files[m]) for m in methods]
    landed = np.array([r[0] for r in rates])
    other = np.array([r[2] for r in rates])
    crashed = np.array([r[1] for r in rates])
    labels = [NAME[m] for m in methods]
    y = np.arange(len(labels))[::-1]

    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    ax.barh(y, landed, color="#0F6E56", label="Landed")
    ax.barh(y, other, left=landed, color="#B4B2A9", label="Other (time limit or out of view)")
    ax.barh(y, crashed, left=landed + other, color="#993C1D", label="Crashed")
    for yi, l in zip(y, landed):
        ax.text(l / 2, yi, f"{l:.1f}%", va="center", ha="center", color="white", fontsize=10)
    for yi, c in zip(y, crashed):
        ax.text(100 - c / 2, yi, f"{c:.1f}%", va="center", ha="center", color="white", fontsize=10)
    ax.set_yticks(y, labels)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of training episodes (%), five test seeds pooled")
    ax.set_title(f"How episodes end, minibatch {minibatch}")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=3, frameon=False, fontsize=9.5)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    save(fig, f"fig_outcomes_mb{minibatch}")


# ------------------------------------------------------------------ 4. Blend-weight ablation (minibatch 64 only)
def fig_ablation():
    settings = [("alpha0", 0.0), ("alpha25", 0.25), ("mb64", 0.5), ("alpha75", 0.75),
                ("alpha90", 0.9), ("alpha100", 1.0)]
    xs, means, seeds_x, seeds_y = [], [], [], []
    for name, a in settings:
        v = tuning_scores("bappo", name)
        if len(v) == 0:
            continue
        xs.append(a)
        means.append(v.mean())
        seeds_x += [a] * len(v)
        seeds_y += list(v)
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    ax.plot(xs, means, color=COL["bappo"], linewidth=2, marker="o",
            label="BA-PPO: mean of the two tuning seeds")
    ax.scatter(seeds_x, seeds_y, s=24, color=COL["bappo"], alpha=0.5, zorder=3, label="Individual tuning seeds")
    ppo = tuning_scores("ppo", "mb64")
    grpo = tuning_scores("grpo", "mb64")
    if len(ppo):
        ax.axhline(ppo.mean(), color=COL["ppo"], linestyle=":", linewidth=1.6,
                   label=f"PPO, minibatch 64 ({ppo.mean():.0f})")
    if len(grpo):
        ax.axhline(grpo.mean(), color=COL["grpo"], linestyle=":", linewidth=1.6,
                   label=f"GRPO, minibatch 64 ({grpo.mean():.0f})")
    ax.axhline(200, color="#888780", linestyle="--", linewidth=1)
    ax.set_xlabel(r"Blend weight $\alpha$ on the group term ($0$ = critic only, $1$ = group only)")
    ax.set_ylabel("Final return (tuning seeds)")
    ax.set_title("Blend-weight ablation, minibatch 64")
    ax.legend(frameon=False, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    save(fig, "fig_ablation_alpha")


def fig_environment():
    """Frames from the saved BA-PPO agent: start, mid-flight and touchdown."""
    import gymnasium as gym
    import torch
    import torch.nn as nn
    from torch.distributions import Categorical

    ck = torch.load(path("checkpoints", "bappo_s2.pt"), weights_only=False)
    pi = nn.Sequential(nn.Linear(ck["obs_dim"], 64), nn.Tanh(),
                       nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, ck["n_actions"]))
    pi.load_state_dict(ck["pi"])
    env = gym.make("LunarLander-v3", render_mode="rgb_array")
    torch.manual_seed(3)
    obs, _ = env.reset(seed=3)
    frames, states, done, t = [], [], False, 0
    while not done:
        frames.append(env.render())
        states.append(obs.copy())
        with torch.no_grad():
            a = int(Categorical(logits=pi(torch.as_tensor(obs, dtype=torch.float32))).sample())
        obs, r, term, trunc, _ = env.step(a)
        done = term or trunc
        t += 1
    frames.append(env.render())
    states.append(obs.copy())
    env.close()
    idx = [0, len(frames) // 2, len(frames) - 1]
    titles = ["Start: lander at the top", "Mid-flight", "Touchdown"]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 4.2))
    for ax, i, title in zip(axes, idx, titles):
        ax.imshow(frames[i])
        ax.set_title(f"{title}\nstep {i}", fontsize=10.5)
        ax.axis("off")
    fig.text(0.5, 0.01,
             "State vector (x, y, vx, vy, angle, angular velocity, left leg, right leg) at touchdown: "
             + ", ".join(f"{v:.2f}" for v in states[-1]),
             ha="center", fontsize=8.5, color="#5F5E5A")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save(fig, "fig_environment")


if __name__ == "__main__":
    fig_environment()
    fig_architecture()
    fig_comparison("results_2M", 512)
    fig_comparison("results_mb64", 64)
    fig_outcomes(512)
    fig_outcomes(64)
    fig_ablation()
    print("figures written to", OUT)
