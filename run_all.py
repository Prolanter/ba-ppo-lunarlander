"""Run every (method, seed) combination in parallel. Finished runs are skipped.

Usage:
    python run_all.py --env LunarLander-v3 --total_steps 300000 --workers 8
"""
import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

METHODS = ["ppo", "grpo", "hybrid"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", default="LunarLander-v3")
    p.add_argument("--total_steps", type=int, default=300_000)
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--out", default="results")
    args = p.parse_args()

    os.makedirs("logs", exist_ok=True)
    jobs = []
    for seed in range(args.seeds):
        for method in METHODS:
            result = os.path.join(args.out, f"{args.env}_{method}_s{seed}.json")
            if os.path.exists(result):
                continue
            cmd = [sys.executable, "ppo_grpo.py", "--env", args.env, "--method", method,
                   "--seed", str(seed), "--total_steps", str(args.total_steps), "--out", args.out]
            jobs.append((cmd, f"logs/{args.env}_{method}_s{seed}.txt"))

    print(f"{len(jobs)} runs to do", flush=True)

    def run(job):
        cmd, log = job
        with open(log, "w") as f:
            code = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT)
        print(f"done {os.path.basename(log)} (exit {code})", flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(ex.map(run, jobs))


if __name__ == "__main__":
    main()
