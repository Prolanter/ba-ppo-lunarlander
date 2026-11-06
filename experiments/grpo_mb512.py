"""GRPO with minibatch size 512 (phase 1 setting).

Usage:
    python experiments/grpo_mb512.py --seed 0
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from ppo_grpo import main  # noqa: E402

if __name__ == "__main__":
    main(["--method", "grpo", "--minibatch", "512"] + sys.argv[1:])
