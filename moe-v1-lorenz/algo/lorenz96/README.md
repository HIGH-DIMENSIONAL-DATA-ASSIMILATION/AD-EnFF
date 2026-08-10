# Lorenz-96 benchmark

This folder contains AD-EnFF, MNMEF, the analytical ENFF baseline, and their shared Lorenz-96 protocol.

The default run trains these models sequentially:

1. AD-EnFF with particle attention
2. AD-EnFF with ensemble means and variances
3. MNMEF

It then evaluates all three trained models together with analytical ENFF.

## Run

```bash
cd learned-enff/time-series-module/moe-v1-lorenz/algo/lorenz96
bash run_all.sh
```

Default training values:

```text
DIM=40
EPOCHS=150
BATCH_SIZE=64
TBTT=16
MAX_DIFFICULTY=5
```

`TBTT` is the truncated backpropagation window in DA cycles and cannot be lower than 16. The batch is divided as evenly as possible across integer difficulties from `0` through `MAX_DIFFICULTY`. With the defaults, each difficulty from `0` to `5` receives at least 10 trajectories per epoch; the remaining four trajectories are assigned randomly and the batch is shuffled.

These values can be changed when launching the script:

```bash
DIM=80 EPOCHS=200 BATCH_SIZE=32 TBTT=16 MAX_DIFFICULTY=7 bash run_all.sh
```

Checkpoints, logs, metric tables, and trajectory plots are written under:

```text
results/l96_dim<DIM>_balanced_d0to<MAX_DIFFICULTY>_ep<EPOCHS>/
```

The benchmark uses arctan observations every eight model steps, 32 ensemble members, truth forcing `F=8`, difficulty-dependent forecast forcing, and difficulty-dependent observation noise.
