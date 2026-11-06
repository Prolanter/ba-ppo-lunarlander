"""PPO vs GRPO vs a PPO+GRPO bappo for discrete-action Gymnasium tasks.

All three methods share the same policy/critic networks, rollout loop and clipped
PPO update. They differ only in how each action's advantage is estimated:

  ppo    : GAE advantage from a learned critic (per-step credit assignment)
  grpo   : group-relative episode return, no critic. Episodes are run G times
           from the same start state; each episode's return is compared with
           the group mean and that score is applied to every step of the episode.
  bappo : alpha * GRPO advantage + (1 - alpha) * GAE advantage

While training, a live progress file is written to progress/ so monitor.py can
show the current episode, its running return, and the crash/landing counts.
"""
import argparse
import json
import os
import time

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical


def mlp(inp, out):
    return nn.Sequential(
        nn.Linear(inp, 64), nn.Tanh(),
        nn.Linear(64, 64), nn.Tanh(),
        nn.Linear(64, out),
    )


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, n_actions, use_critic):
        super().__init__()
        self.pi = mlp(obs_dim, n_actions)
        self.v = mlp(obs_dim, 1) if use_critic else None

    def dist(self, obs):
        return Categorical(logits=self.pi(obs))

    def value(self, obs):
        return self.v(obs).squeeze(-1)


def run_episode(env, model, seed, live=None):
    """Play one full episode with the stochastic policy and record everything.

    live (optional) is a dict that is updated every step so progress can be shown.
    """
    obs, _ = env.reset(seed=seed)
    data = {"obs": [], "act": [], "logp": [], "val": [], "rew": []}
    done = False
    if live is not None:
        live["cur_return"], live["cur_steps"] = 0.0, 0
    while not done:
        o = torch.as_tensor(obs, dtype=torch.float32)
        with torch.no_grad():
            dist = model.dist(o)
            a = dist.sample()
            data["logp"].append(dist.log_prob(a).item())
            data["val"].append(model.value(o).item() if model.v is not None else 0.0)
        data["obs"].append(obs)
        data["act"].append(a.item())
        obs, r, term, trunc, _ = env.step(a.item())
        data["rew"].append(r)
        done = term or trunc  # truncation is treated as an episode end (no bootstrap)
        if live is not None:
            live["cur_return"] += float(r)
            live["cur_steps"] += 1
            if live.get("tick") and live["cur_steps"] % 200 == 0:
                live["tick"]()
    return {
        "obs": np.asarray(data["obs"], dtype=np.float32),
        "act": np.asarray(data["act"], dtype=np.int64),
        "logp": np.asarray(data["logp"], dtype=np.float32),
        "val": np.asarray(data["val"], dtype=np.float32),
        "rew": np.asarray(data["rew"], dtype=np.float32),
    }


def outcome(ep):
    """Classify a finished LunarLander episode from its last reward."""
    last = float(ep["rew"][-1])
    if last <= -100:
        return "crash"
    if last >= 100:
        return "landed"
    return "other"  # e.g. time limit or flew out of the screen


def gae(rew, val, gamma, lam):
    adv = np.zeros_like(rew)
    running, next_val = 0.0, 0.0  # episode ends, so nothing to bootstrap from
    for t in reversed(range(len(rew))):
        delta = rew[t] + gamma * next_val - val[t]
        running = delta + gamma * lam * running
        adv[t] = running
        next_val = val[t]
    return adv


def build_batch(groups, method, alpha, gamma, lam):
    """Turn a list of episode groups into flat training arrays with advantages."""
    obs, act, logp, adv_gae, adv_grp, ret = [], [], [], [], [], []
    for group in groups:
        R = np.array([ep["rew"].sum() for ep in group], dtype=np.float32)
        if len(group) > 1:
            grp = (R - R.mean()) / (R.std() + 1e-8)  # group-relative score
        else:
            grp = np.zeros(len(group), dtype=np.float32)
        for i, ep in enumerate(group):
            obs.append(ep["obs"])
            act.append(ep["act"])
            logp.append(ep["logp"])
            adv_grp.append(np.full(len(ep["rew"]), grp[i], dtype=np.float32))
            g = gae(ep["rew"], ep["val"], gamma, lam)
            adv_gae.append(g)
            ret.append(g + ep["val"])

    obs, act, logp = np.concatenate(obs), np.concatenate(act), np.concatenate(logp)
    adv_gae, adv_grp, ret = np.concatenate(adv_gae), np.concatenate(adv_grp), np.concatenate(ret)

    def norm(x):
        return (x - x.mean()) / (x.std() + 1e-8)

    if method == "ppo":
        adv = norm(adv_gae)
    elif method == "grpo":
        adv = adv_grp
    else:  # bappo
        adv = alpha * adv_grp + (1.0 - alpha) * norm(adv_gae)
    return obs, act, logp, adv.astype(np.float32), ret.astype(np.float32)


def ppo_update(model, opt, obs, act, logp_old, adv, ret, args):
    obs = torch.as_tensor(obs)
    act = torch.as_tensor(act)
    logp_old = torch.as_tensor(logp_old)
    adv = torch.as_tensor(adv)
    ret = torch.as_tensor(ret)
    n = len(obs)
    for _ in range(args.epochs):
        perm = torch.randperm(n)
        for s in range(0, n, args.minibatch):
            mb = perm[s:s + args.minibatch]
            dist = model.dist(obs[mb])
            ratio = torch.exp(dist.log_prob(act[mb]) - logp_old[mb])
            a = adv[mb]
            pg = -torch.min(ratio * a, ratio.clamp(1 - args.clip, 1 + args.clip) * a).mean()
            loss = pg - args.ent_coef * dist.entropy().mean()
            if model.v is not None:
                loss = loss + args.vf_coef * ((model.value(obs[mb]) - ret[mb]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            opt.step()


def write_json(path, data):
    """Write atomically so the monitor never reads a half-written file.

    On Windows the swap can fail while the monitor has the file open, so retry
    briefly and then skip. A missed progress update must never stop training.
    """
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f)
    for _ in range(20):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            time.sleep(0.05)


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--env", default="LunarLander-v3")
    p.add_argument("--method", choices=["ppo", "grpo", "bappo"], required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--total_steps", type=int, default=300_000)
    p.add_argument("--batch_steps", type=int, default=4096, help="env steps per update")
    p.add_argument("--group_size", type=int, default=8, help="GRPO episodes per start state")
    p.add_argument("--alpha", type=float, default=0.5, help="bappo weight on GRPO advantage")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--minibatch", type=int, default=512)
    p.add_argument("--ent_coef", type=float, default=0.01)
    p.add_argument("--vf_coef", type=float, default=0.5)
    p.add_argument("--out", default="results")
    p.add_argument("--progress_dir", default="progress")
    p.add_argument("--save_model", default="", help="path to save trained weights (optional)")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(2)
    rng = np.random.default_rng(args.seed)

    env = gym.make(args.env)
    obs_dim = env.observation_space.shape[0]
    n_act = env.action_space.n
    model = ActorCritic(obs_dim, n_act, use_critic=(args.method != "grpo"))
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-5)

    group_size = 1 if args.method == "ppo" else args.group_size
    steps, episodes, t0 = 0, [], time.time()
    crashes, landed, outcomes_seen = 0, 0, 0
    run_name = f"{args.env}_{args.method}_s{args.seed}"
    progress_path = os.path.join(args.progress_dir, run_name + ".json")
    os.makedirs(args.progress_dir, exist_ok=True)

    live = {"cur_return": 0.0, "cur_steps": 0, "episode_in_group": 0}

    # TensorBoard log: one folder per run, so each run shows as its own line on the dashboard.
    from torch.utils.tensorboard import SummaryWriter
    tb_name = f"{args.method}_mb{args.minibatch}_gamma{args.gamma}_seed{args.seed}"
    writer = SummaryWriter(log_dir=os.path.join("runs", args.env, tb_name), flush_secs=10)

    def write_progress(status="training"):
        recent = [r for _, r in episodes[-100:]]
        write_json(progress_path, {
            "run": run_name, "method": args.method, "seed": args.seed,
            "status": status,
            "steps": steps, "total_steps": args.total_steps,
            "episodes_done": len(episodes),
            "last100_return": float(np.mean(recent)) if recent else None,
            "crashes": crashes, "landed": landed, "other_endings": outcomes_seen - crashes - landed,
            "current_episode": len(episodes) + 1,
            "current_return": round(live["cur_return"], 1),
            "current_steps": live["cur_steps"],
            "elapsed_s": round(time.time() - t0),
        })

    live["tick"] = write_progress

    while steps < args.total_steps:
        groups, batch_steps = [], 0
        while batch_steps < args.batch_steps:
            seed = int(rng.integers(2**31 - 1))
            group = []
            for _ in range(group_size):
                ep = run_episode(env, model, seed, live)
                group.append(ep)
                kind = outcome(ep)
                outcomes_seen += 1
                crashes += kind == "crash"
                landed += kind == "landed"
                batch_steps += len(ep["rew"])
                steps += len(ep["rew"])
                episodes.append((steps, float(ep["rew"].sum())))
                writer.add_scalar("episode/return", episodes[-1][1], steps)
                writer.add_scalar("episode/last100_return",
                                  float(np.mean([r for _, r in episodes[-100:]])), steps)
                writer.add_scalar("episode/crashes", crashes, steps)
                writer.add_scalar("episode/landed", landed, steps)
                write_progress()
            groups.append(group)

        obs, act, logp, adv, ret = build_batch(groups, args.method, args.alpha, args.gamma, args.lam)
        ppo_update(model, opt, obs, act, logp, adv, ret, args)

        recent = np.mean([r for _, r in episodes[-100:]])
        print(f"steps {steps:>8} | episodes {len(episodes):>5} | "
              f"last100 {recent:8.1f} | crashes {crashes} | landed {landed} | "
              f"{time.time() - t0:6.0f}s", flush=True)

    write_progress(status="finished")
    writer.close()
    os.makedirs(args.out, exist_ok=True)
    fname = f"{run_name}.json"
    with open(os.path.join(args.out, fname), "w") as f:
        json.dump({"args": vars(args), "episodes": episodes,
                   "wall_time": time.time() - t0}, f)
    if args.save_model:
        os.makedirs(os.path.dirname(args.save_model) or ".", exist_ok=True)
        torch.save({"pi": model.pi.state_dict(), "obs_dim": obs_dim, "n_actions": n_act},
                   args.save_model)
    env.close()


if __name__ == "__main__":
    main()
