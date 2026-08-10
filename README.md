# AD-EnFF: Adaptive Ensemble Flow Filters

This repository contains the implementation of **Adaptive Ensemble Flow Filters (AD-EnFF)**, alongside baselines such as **MNMEF**, evaluated on both the Lorenz-63 and Lorenz-96 benchmark protocols.

## Overview

The codebase trains and evaluates AD-EnFF and MNMEF models for high-dimensional data assimilation tasks. Training uses arctan observations and balanced difficulty batches. Difficulty controls observation noise and the mismatch in forecast-model parameters (`rho` and `sigma` for Lorenz-63).

## Repository Structure

The actual model source code and evaluation scripts are located within `moe-v1-lorenz/algo/`:

- `moe-v1-lorenz/algo/lorenz63/`: Contains the complete implementation and execution scripts for Lorenz-63.
- `moe-v1-lorenz/algo/lorenz96/`: Contains the complete implementation and execution scripts for Lorenz-96.

Within each of these directories, you will find:
- `ad_enff/`: Implementation of the Adaptive Ensemble Flow Filter.
- `mnmef/`: Implementation of the MNMEF baseline model.
- `common/`: Shared utilities, configurations, physics simulators, and evaluation scripts.
- `run_all.sh`: The core bash script orchestrating the end-to-end training and evaluation loop.

## Quick Start (Smoke Tests)

If you want to verify that the environment is set up properly without waiting for a full training run, you can use the smoke test mode. This uses smaller batch sizes and significantly fewer epochs. 

By default, the scripts will attempt to use a CUDA-capable GPU. If you don't have one or prefer to run on CPU, set `CPU=1`.

```bash
# Quick local test for Lorenz-63 (on CPU)
SMOKE=1 CPU=1 ./run_l63.sh

# Quick local test for Lorenz-96 (on GPU)
SMOKE=1 ./run_l96.sh
```

## Full Training Protocol

To run the full benchmark protocol, use the provided wrapper bash scripts located at the root directory. You can override default parameters like `EPOCHS`, `BATCH_SIZE`, and `MAX_DIFFICULTY` using environment variables. The scripts will automatically run both AD-EnFF and MNMEF models, followed by a comprehensive evaluation.

For **Lorenz-63**:
```bash
# Defaults to: EPOCHS=150 BATCH_SIZE=64 TBTT=16 MAX_DIFFICULTY=3
./run_l63.sh
```

For **Lorenz-96**:
```bash
# Defaults to: EPOCHS=150 DIM=40 BATCH_SIZE=64 TBTT=16 MAX_DIFFICULTY=7
./run_l96.sh
```

### Outputs and Logs

Checkpoints, training logs, and evaluation results are written to the `results/` directory inside the respective algorithm folder:
- `moe-v1-lorenz/algo/lorenz63/results/`
- `moe-v1-lorenz/algo/lorenz96/results/`

## Evaluation Details

The automated evaluation step (`common/eval_all.py` triggered by the wrapper scripts) generates a comprehensive report containing:
- **Metrics**: RMSE, Energy Score, Trajectory Energy Score, Variogram, 95% Coverage, Climatology MMD, and Latency.
- **Visualizations**: Trajectory plots and a 3D attractor plot (for L-63), saved as `.png` files in the `results/.../eval/` directory.

*(To manually re-evaluate on specific difficulties, you can navigate to the specific algorithm directory and utilize the standalone evaluation scripts provided there).*
