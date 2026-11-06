"""Open a window and watch a trained agent fly the lander.

Usage:
    python watch.py checkpoints/bappo_s2.pt --episodes 5
"""
import argparse
import os

import gymnasium as gym
import pygame
import torch
import torch.nn as nn
from torch.distributions import Categorical


def mlp(inp, out):
    return nn.Sequential(
        nn.Linear(inp, 64), nn.Tanh(),
        nn.Linear(64, 64), nn.Tanh(),
        nn.Linear(64, out),
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint")
    p.add_argument("--env", default="LunarLander-v3")
    p.add_argument("--episodes", type=int, default=5)
    args = p.parse_args()

    ckpt = torch.load(args.checkpoint, weights_only=False)
    pi = mlp(ckpt["obs_dim"], ckpt["n_actions"])
    pi.load_state_dict(ckpt["pi"])
    pi.eval()

    env = gym.make(args.env, render_mode="human")
    label = f"{os.path.splitext(os.path.basename(args.checkpoint))[0]} | {args.env}"
    print(f"Watching saved model: {label}", flush=True)
    for ep in range(args.episodes):
        obs, _ = env.reset()
        pygame.display.set_caption(f"{label} | episode {ep + 1}")
        total, done = 0.0, False
        while not done:
            with torch.no_grad():
                logits = pi(torch.as_tensor(obs, dtype=torch.float32))
                # Sample like training does. Always taking the most likely action
                # scored far worse (about -38 vs 123 over 20 episodes), so the
                # trained policy relies on sampling.
                action = int(Categorical(logits=logits).sample().item())
            obs, r, term, trunc, _ = env.step(action)
            total += r
            done = term or trunc
        print(f"episode {ep + 1}: return {total:.1f}")
    env.close()


if __name__ == "__main__":
    main()
