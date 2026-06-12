#!/usr/bin/env bash
# Run FLAG on a HumanoidBench task.
#
# Usage:
#   ./run_flag.sh --env h1-walk-v0 [--seed 42] [--gpu 0] [extra hydra overrides...]
#
# Supported envs: h1-walk-v0  h1-run-v0  h1-hurdle-v0  h1-maze-v0  h1-stair-v0
#
# Examples:
#   ./run_flag.sh --env h1-walk-v0
#   ./run_flag.sh --env h1-run-v0 --seed 1 --gpu 1
#   ./run_flag.sh --env h1-maze-v0 total_steps=2000001
#   ./run_flag.sh --env all --seed 0          # sequential run over all 5 tasks

set -euo pipefail

# ── defaults ──────────────────────────────────────────────────────────────────
ENV_ID=""
SEED=42
GPU=0
CONDA_ENV=humanoidbench
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

VALID_ENVS=(h1-walk-v0 h1-run-v0 h1-hurdle-v0 h1-maze-v0 h1-stair-v0)

# ── argument parsing ──────────────────────────────────────────────────────────
HYDRA_OVERRIDES=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --env)   ENV_ID="$2";  shift 2 ;;
        --seed)  SEED="$2";    shift 2 ;;
        --gpu)   GPU="$2";     shift 2 ;;
        *)       HYDRA_OVERRIDES+=("$1"); shift ;;
    esac
done

if [[ -z "$ENV_ID" ]]; then
    echo "Error: --env is required." >&2
    echo "Usage: $0 --env <env_id|all> [--seed N] [--gpu N] [hydra overrides...]" >&2
    echo "Valid envs: ${VALID_ENVS[*]}" >&2
    exit 1
fi

# ── helpers ───────────────────────────────────────────────────────────────────
run_one() {
    local env="$1"
    local seed="$2"

    echo "========================================================"
    echo "  ENV : $env"
    echo "  SEED: $seed"
    echo "  GPU : $GPU"
    echo "========================================================"

    CUDA_VISIBLE_DEVICES="$GPU" \
    MUJOCO_GL=egl \
    conda run --no-capture-output -n "$CONDA_ENV" \
        python "$SCRIPT_DIR/train_flag.py" \
            env_id="$env" \
            seed="$seed" \
            "${HYDRA_OVERRIDES[@]+"${HYDRA_OVERRIDES[@]}"}"
}

# ── dispatch ──────────────────────────────────────────────────────────────────
if [[ "$ENV_ID" == "all" ]]; then
    for env in "${VALID_ENVS[@]}"; do
        run_one "$env" "$SEED"
    done
else
    run_one "$ENV_ID" "$SEED"
fi
