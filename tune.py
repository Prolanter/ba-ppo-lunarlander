"""Per-method tuning on separate seeds, then optional final runs on the test seeds.

Every method gets the same three candidate settings, so the search is equal:
    mb512    : minibatch 512 (the original setting)
    mb64     : minibatch 64
    gamma999 : longer planning horizon (gamma 0.999, lambda 0.98)

Tuning uses seeds 100 and 101, which are never reported. The chosen setting is
then run on the test seeds 0 to 4.

Usage:
    python tune.py --methods ppo,grpo,bappo          # tuning runs
    python tune.py --final grpo mb64                  # rerun one setting on seeds 0-4
"""
import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

ENV = "LunarLander-v3"
SETTINGS = {
    "mb512": [],
    "mb64": ["--minibatch", "64"],
    "gamma999": ["--gamma", "0.999", "--lam", "0.98"],
    # Blend-weight test for BA-PPO, at the best minibatch setting (64).
    "alpha0": ["--minibatch", "64", "--alpha", "0.0"],   # group sampling, GAE advantage only
    "alpha25": ["--minibatch", "64", "--alpha", "0.25"],
    "alpha75": ["--minibatch", "64", "--alpha", "0.75"],
    "alpha90": ["--minibatch", "64", "--alpha", "0.9"],
    "alpha100": ["--minibatch", "64", "--alpha", "1.0"],
}
TUNE_SEEDS = [100, 101]
TEST_SEEDS = [0, 1, 2, 3, 4]


def job_cmd(method, setting, seed, out_root, progress_root, total_steps):
    return ([sys.executable, "ppo_grpo.py", "--env", ENV, "--method", method,
             "--seed", str(seed), "--total_steps", str(total_steps),
             "--out", os.path.join(out_root, method, setting),
             "--progress_dir", os.path.join(progress_root, method, setting)]
            + SETTINGS[setting])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--methods", default="ppo,grpo,bappo")
    p.add_argument("--settings", default=",".join(SETTINGS),
                   help="comma-separated subset of settings to run")
    p.add_argument("--final", nargs=2, metavar=("METHOD", "SETTING"),
                   help="run one chosen setting on seeds 0-4")
    p.add_argument("--total_steps", type=int, default=2_000_000)
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()
    os.makedirs("logs", exist_ok=True)

    jobs = []
    if args.final:
        method, setting = args.final
        for seed in TEST_SEEDS:
            cmd = job_cmd(method, setting, seed, "results_final", "progress_final",
                          args.total_steps)
            jobs.append((cmd, f"logs/final_{method}_{setting}_s{seed}.txt"))
    else:
        for method in args.methods.split(","):
            for setting in args.settings.split(","):
                for seed in TUNE_SEEDS:
                    cmd = job_cmd(method, setting, seed, "results_tune", "progress_tune",
                                  args.total_steps)
                    jobs.append((cmd, f"logs/tune_{method}_{setting}_s{seed}.txt"))

    def job(j):
        cmd, log = j
        with open(log, "w") as f:
            code = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT)
        print(f"done {os.path.basename(log)} (exit {code})", flush=True)

    print(f"{len(jobs)} runs", flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(ex.map(job, jobs))


if __name__ == "__main__":
    main()
