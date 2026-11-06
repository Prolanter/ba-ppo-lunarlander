"""PPO with minibatch size 64 (phase 2 setting, chosen by the tuning check).

Usage:
    python experiments/ppo_mb64.py --seed 0
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from ppo_grpo import main  # noqa: E402

if __name__ == "__main__":
    main(["--method", "ppo", "--minibatch", "64"] + sys.argv[1:])
