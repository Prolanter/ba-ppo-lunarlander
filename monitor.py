"""Live table of every training run: current episode, running return, crashes, landings.

Usage (leave it open in its own terminal while training runs):
    python monitor.py
    python monitor.py --progress_dir progress --every 2
Press Ctrl+C to stop. It only reads files, so it does not slow training.
"""
import argparse
import glob
import json
import os
import time

NAMES = {"ppo": "PPO", "grpo": "GRPO", "bappo": "BA-PPO"}


def read_runs(progress_dir):
    """Find progress files in the folder and in any subfolder (tuning runs use subfolders)."""
    runs = []
    for f in sorted(glob.glob(os.path.join(progress_dir, "**", "*.json"), recursive=True)):
        try:
            with open(f) as fh:
                run = json.load(fh)
        except (json.JSONDecodeError, OSError):
            continue  # file was being written; try again next refresh
        folder = os.path.relpath(os.path.dirname(f), progress_dir)
        run["config"] = "" if folder == "." else folder
        runs.append(run)
    return runs


def bar(done, total, width=20):
    frac = min(done / max(total, 1), 1.0)
    filled = int(frac * width)
    return "[" + "#" * filled + "-" * (width - filled) + f"] {frac * 100:5.1f}%"


def render(runs):
    lines = [f"{'Model':<16}{'Seed':>5}  {'Progress':<34}{'Episodes':>9}  {'Last 100':>9}  "
             f"{'Current ep':>10}  {'Running ret':>11}  {'Crashes':>8}  {'Landed':>7}  Status"]
    lines.append("-" * 130)
    for r in runs:
        last100 = "-" if r["last100_return"] is None else f"{r['last100_return']:.1f}"
        steps = f"{r['steps']:,}/{r['total_steps']:,}"
        name = NAMES.get(r["method"], r["method"]) + (f" [{r['config']}]" if r.get("config") else "")
        lines.append(
            f"{name:<24}{r['seed']:>5}  "
            f"{bar(r['steps'], r['total_steps']) + ' ' + steps:<34}"
            f"{r['episodes_done']:>9}  {last100:>9}  {r['current_episode']:>10}  "
            f"{r['current_return']:>11.1f}  {r['crashes']:>8}  {r['landed']:>7}  {r['status']}"
        )
    if not runs:
        lines.append("No training runs found yet. Start one with run_all.py or ppo_grpo.py.")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--progress_dir", default="progress")
    p.add_argument("--every", type=float, default=2.0, help="seconds between refreshes")
    args = p.parse_args()
    try:
        while True:
            os.system("cls" if os.name == "nt" else "clear")
            print(f"Live training monitor, updated {time.strftime('%H:%M:%S')}  (Ctrl+C to stop)\n")
            print(render(read_runs(args.progress_dir)))
            time.sleep(args.every)
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
