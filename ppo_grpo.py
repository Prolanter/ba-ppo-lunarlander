"""PPO vs GRPO vs a PPO+GRPO hybrid for discrete-action Gymnasium tasks.

All three methods share the same policy/critic networks, rollout loop and clipped
PPO update. They differ only in how each action's advantage is estimated:

  ppo    : GAE advantage from a learned critic (per-step credit assignment)
  grpo   : group-relative episode return, no critic. Episodes are run G times
           from the same start state; each episode's return is compared with
           the group mean and that score is applied to every step of the episode.
  hybrid : alpha * GRPO advantage + (1 - alpha) * GAE advantage (our idea)
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


def run_episode(env, model, seed):
    """Play one full episode with the stochastic policy and record everything."""
    obs, _ = env.reset(seed=seed)
    data = {"obs": [], "act": [], "logp": [], "val": [], "rew": []}
    done = False
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
    return {
        "obs": np.asarray(data["obs"], dtype=np.float32),
        "act": np.asarray(data["act"], dtype=np.int64),
        "logp": np.asarray(data["logp"], dtype=np.float32),
        "val": np.asarray(data["val"], dtype=np.float32),
        "rew": np.asarray(data["rew"], dtype=np.float32),
    }


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
    else:  # hybrid
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


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--env", default="LunarLander-v3")
    p.add_argument("--method", choices=["ppo", "grpo", "hybrid"], required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--total_steps", type=int, default=300_000)
    p.add_argument("--batch_steps", type=int, default=4096, help="env steps per update")
    p.add_argument("--group_size", type=int, default=8, help="GRPO episodes per start state")
    p.add_argument("--alpha", type=float, default=0.5, help="hybrid weight on GRPO advantage")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--minibatch", type=int, default=512)
    p.add_argument("--ent_coef", type=float, default=0.01)
    p.add_argument("--vf_coef", type=float, default=0.5)
    p.add_argument("--out", default="results")
    return p.parse_args()


def main():
    args = parse_args()
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

    while steps < args.total_steps:
        groups, batch_steps = [], 0
        while batch_steps < args.batch_steps:
            seed = int(rng.integers(2**31 - 1))
            group = [run_episode(env, model, seed) for _ in range(group_size)]
            groups.append(group)
            for ep in group:
                batch_steps += len(ep["rew"])
                steps += len(ep["rew"])
                episodes.append((steps, float(ep["rew"].sum())))

        obs, act, logp, adv, ret = build_batch(groups, args.method, args.alpha, args.gamma, args.lam)
        ppo_update(model, opt, obs, act, logp, adv, ret, args)

        recent = np.mean([r for _, r in episodes[-100:]])
        print(f"steps {steps:>8} | episodes {len(episodes):>5} | "
              f"last100 {recent:8.1f} | {time.time() - t0:6.0f}s", flush=True)

    os.makedirs(args.out, exist_ok=True)
    fname = f"{args.env}_{args.method}_s{args.seed}.json"
    with open(os.path.join(args.out, fname), "w") as f:
        json.dump({"args": vars(args), "episodes": episodes,
                   "wall_time": time.time() - t0}, f)
    env.close()


if __name__ == "__main__":
    main()
