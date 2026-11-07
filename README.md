# BA-PPO: Blended Advantage PPO for LunarLander

This project compares three policy-gradient methods on the LunarLander-v3 task from Gymnasium:

- **PPO**: advantages from a learned critic (GAE).
- **GRPO**: advantages from comparing episodes that start from the same state, with no critic. From DeepSeekMath (Shao et al., 2024).
- **BA-PPO** (Blended Advantage PPO, our method): each action's advantage is a weighted blend of the GRPO group score and the PPO critic score, with weight α = 0.5.

All three share the same network (two layers of 64 units), rollout loop and clipped PPO update. They differ only in how the advantage is computed.

**Question:** does blending the group-relative signal from GRPO into PPO help on a control task?

## Results

Final return is the average of the last 100 training episodes, over 5 seeds (0 to 4), after 2 million environment steps. Mean ± sample standard deviation.

| Method | Minibatch 512 | Minibatch 64 |
|---|---|---|
| PPO | 89 ± 34 | 241 ± 17 |
| GRPO | 129 ± 44 | 167 ± 60 |
| BA-PPO | 214 ± 40 | **256 ± 14** |

- Minibatch 64 was chosen by tuning on separate seeds (100 and 101), and it was the best setting for all three methods.
- At minibatch 512, BA-PPO beats PPO (p ≈ 0.001) and GRPO (p ≈ 0.01).
- At minibatch 64, BA-PPO is ahead of PPO by about 15 points, which is not statistically significant (p ≈ 0.17). It beats GRPO (p ≈ 0.03).

Charts and tables:
- `learning_curves_LunarLander-v3_results_2M.png` and `results_LunarLander-v3_results_2M.md` (minibatch 512)
- `learning_curves_LunarLander-v3_results_mb64.png` and `results_LunarLander-v3_results_mb64.md` (minibatch 64)

## Limitations

- Only 5 test seeds, so the differences between the top two methods at minibatch 64 are not conclusive.
- Only the minibatch size was tuned per method. Learning rate, clipping, epochs and entropy weight were fixed for all three.
- BA-PPO's blend weight α was tested at 0.25, 0.5 and 0.75 on separate tuning seeds (100 and 101). α = 0.5 gave the best and most consistent results, so it was used for all reported runs. Only two tuning seeds were used, so this check is limited.
- GRPO has no critic, so the planning-horizon setting (gamma 0.999) has no effect on it. Its results match the minibatch-512 runs exactly. That setting was run for GRPO by mistake and is excluded from the comparison.
- Scores are training-episode returns, so they include the agent's random exploration.

## Setup

```bash
pip install -r requirements.txt
```

For CPU-only PyTorch, you can install it first with:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Tested on Windows 11 with Python 3.12, Gymnasium 1.4 and PyTorch 2.15 (CPU).

## Reproducing the results

Train all three methods on the five test seeds:

```bash
# Minibatch 512
python run_all.py --env LunarLander-v3 --total_steps 2000000 --seeds 5 --out results_2M --progress_dir progress

# Minibatch 64
python run_all.py --env LunarLander-v3 --total_steps 2000000 --seeds 5 --minibatch 64 --out results_mb64 --progress_dir progress_mb64
```

Make the charts and tables, and run the significance tests:

```bash
python analyze.py LunarLander-v3 200 results_2M
python analyze.py LunarLander-v3 200 results_mb64
python stats.py results_2M
python stats.py results_mb64
```

Repeat the tuning on seeds 100 and 101:

```bash
python tune.py --methods ppo,grpo,bappo
```

Repeat the blend-weight test (BA-PPO with α = 0.25 and 0.75, at minibatch 64):

```bash
python tune.py --methods bappo --settings alpha25,alpha75
```

Run one setting directly:

```bash
python experiments/bappo_mb64.py --seed 0 --total_steps 2000000 --out results_mb64 --progress_dir progress_mb64
```

Watch training live with the terminal monitor:

```bash
python monitor.py --progress_dir progress_mb64
```

Watch training curves in TensorBoard (open the printed link in a browser):

```bash
tensorboard --logdir runs --reload_interval 10
```

Watch a trained agent fly the lander. This needs a saved model, which is made when training runs with `--save_model`:

```bash
python watch.py checkpoints/bappo_s2.pt --episodes 5
```

## Files

| Path | Purpose |
|---|---|
| `ppo_grpo.py` | Shared training code for all three methods |
| `experiments/` | One small script per method and setting (for example `bappo_mb64.py`) |
| `run_all.py` | Runs every method on all seeds in parallel |
| `tune.py` | Tuning on seeds 100 and 101 |
| `analyze.py` | Learning curves and results tables |
| `stats.py` | Significance tests on final returns |
| `monitor.py` | Live terminal table of all runs |
| `watch.py` | Opens a window to watch a saved agent |
| `results_2M/`, `results_mb64/` | Per-episode results for the test seeds |
| `results_tune/` | Per-episode results for the tuning seeds |
| `results/` | Earlier 500k-step runs, kept for reference |
| `logs/` | Text output of each run |

## Method

Each update collects about 4,096 environment steps. Those steps are split into minibatches, and the networks are trained for 4 epochs with a clipped PPO objective (clip 0.2), a learning rate of 3e-4, and an entropy bonus of 0.01.

- **PPO:** the advantage is GAE with λ = 0.95 and γ = 0.99, normalized over the batch.
- **GRPO:** each start state is played 8 times, and each episode's total return is normalized within its group. Every step of an episode gets that episode's score.
- **BA-PPO:** advantage = 0.5 × (GRPO group score) + 0.5 × (GAE advantage, normalized over the batch). The GRPO score is already normalized within each group.
