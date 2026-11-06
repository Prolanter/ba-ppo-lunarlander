"""BA-PPO (GRPO + GAE blend, alpha = 0.5) with minibatch size 64 (phase 2 setting).

Usage:
    python experiments/bappo_mb64.py --seed 0
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from ppo_grpo import main  # noqa: E402

if __name__ == "__main__":
    main(["--method", "bappo", "--minibatch", "64"] + sys.argv[1:])
