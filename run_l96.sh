#!/usr/bin/env bash
set -euo pipefail

# This script runs the Lorenz-96 training and evaluation pipeline.
# You can pass environment variables to configure the run, e.g.:
# CPU=1 ./run_l96.sh

export EPOCHS="${EPOCHS:-150}"
export DIM="${DIM:-40}"
export BATCH_SIZE="${BATCH_SIZE:-64}"
export MAX_DIFFICULTY="${MAX_DIFFICULTY:-7}"
export TBTT="${TBTT:-16}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT/moe-v1-lorenz/algo/lorenz96"

exec bash run_all.sh "$@"
