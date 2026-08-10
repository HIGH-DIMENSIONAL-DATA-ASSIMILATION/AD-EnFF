# Lorenz-63 benchmark

The runner trains attention AD-EnFF and MNMEF in parallel, trains plain AD-EnFF afterward, then evaluates all three with analytical ENFF. Training uses arctan observations and balanced difficulty batches.

## Run

```bash
cd learned-enff/time-series-module/moe-v1-lorenz/algo/lorenz63
bash run_all.sh
```

Defaults:

```text
EPOCHS=100
BATCH_SIZE=64
TBTT=16
MAX_DIFFICULTY=7
```

`TBTT` cannot be lower than 16. Each batch is divided as evenly as possible across integer difficulties from zero through `MAX_DIFFICULTY`. Difficulty controls observation noise and the forecast-model rho and sigma mismatch.

Override only the values needed for a run:

```bash
EPOCHS=150 BATCH_SIZE=64 TBTT=16 MAX_DIFFICULTY=8 bash run_all.sh
```

Checkpoints and training logs are written under `results/<RUN_ID>/`. Evaluation reports RMSE, energy score, trajectory energy score, variogram, 95% coverage, climatology MMD, and latency. It also saves X/Y/Z trajectories and a 3D attractor plot.

## Evaluate another difficulty

Only the evaluation difficulty is required:

```bash
bash eval.sh 3
```

This reads the default training run and writes PNGs and `eval.log` to `results/l63_balanced_d0to7_ep100/eval_d3/`.

For a non-default training run, provide its training settings again:

```bash
EPOCHS=150 TRAIN_MAX_DIFFICULTY=8 bash eval.sh 4
```
