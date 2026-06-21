#!/usr/bin/env bash
# Run FLAG on a HumanoidBench task.
#
# Usage:
#   ./run_flag.sh --env h1-walk-v0 [--seed 42] [--gpu 0] [--video] [extra hydra overrides...]
#   ./run_flag.sh --env h1-walk-v0 --seed 0 21 42 63   # multiple seeds, run sequentially
#
# Supported envs: h1-walk-v0  h1-run-v0  h1-hurdle-v0  h1-maze-v0  h1-stair-v0
#
# Examples:
#   ./run_flag.sh --env h1-walk-v0
#   ./run_flag.sh --env h1-run-v0 --seed 1 --gpu 1
#   ./run_flag.sh --env h1-walk-v0 --seed 0 21 42 63 84 105 126 147 168 189 --gpu 0
#   ./run_flag.sh --env h1-walk-v0 --video --gpu 0
#   ./run_flag.sh --env h1-maze-v0 total_steps=2000001
#   ./run_flag.sh --env all --seed 0          # sequential run over all 5 tasks

set -euo pipefail

# ── defaults ──────────────────────────────────────────────────────────────────
ENV_ID=""
SEEDS=()
GPU=0
VIDEO=false
CONDA_ENV=humanoidbench
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

VALID_ENVS=(h1-walk-v0 h1-run-v0 h1-hurdle-v0 h1-maze-v0 h1-stair-v0)

# ── argument parsing ──────────────────────────────────────────────────────────
HYDRA_OVERRIDES=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --env)   ENV_ID="$2";  shift 2 ;;
        --seed)
            shift
            while [[ $# -gt 0 && "$1" != --* && "$1" != *=* ]]; do
                SEEDS+=("$1")
                shift
            done
            ;;
        --gpu)   GPU="$2";     shift 2 ;;
        --video) VIDEO=true;   shift ;;
        *)       HYDRA_OVERRIDES+=("$1"); shift ;;
    esac
done

if [[ ${#SEEDS[@]} -eq 0 ]]; then
    SEEDS=(42)
fi

if [[ -z "$ENV_ID" ]]; then
    echo "Error: --env is required." >&2
    echo "Usage: $0 --env <env_id|all> [--seed N [N ...]] [--gpu N] [--video] [hydra overrides...]" >&2
    echo "Valid envs: ${VALID_ENVS[*]}" >&2
    exit 1
fi

if [[ "$VIDEO" == true ]]; then
    HYDRA_OVERRIDES+=("save_video=true")
fi

# ── helpers ───────────────────────────────────────────────────────────────────
run_one() {
    local env="$1"
    local seed="$2"

    echo "========================================================"
    echo "  ENV : $env"
    echo "  SEED: $seed"
    echo "  GPU : $GPU (physical)"
    echo "  CUDA_VISIBLE_DEVICES=$GPU"
    echo "  VIDEO: $VIDEO"
    echo "========================================================"

    if [[ "$VIDEO" == true ]]; then
        CUDA_VISIBLE_DEVICES="$GPU" \
        MUJOCO_GL=egl \
        MUJOCO_EGL_DEVICE_ID="$GPU" \
        EGL_DEVICE_ID="$GPU" \
        XLA_PYTHON_CLIENT_PREALLOCATE=false \
        conda run --no-capture-output -n "$CONDA_ENV" \
                python "$SCRIPT_DIR/train_flag.py" \
                    env_id="$env" \
                    seed="$seed" \
                    "${HYDRA_OVERRIDES[@]+"${HYDRA_OVERRIDES[@]}"}"
    else
        CUDA_VISIBLE_DEVICES="$GPU" \
        XLA_PYTHON_CLIENT_PREALLOCATE=false \
        conda run --no-capture-output -n "$CONDA_ENV" \
                python "$SCRIPT_DIR/train_flag.py" \
                    env_id="$env" \
                    seed="$seed" \
                    "${HYDRA_OVERRIDES[@]+"${HYDRA_OVERRIDES[@]}"}"
    fi
}

# ── dispatch ──────────────────────────────────────────────────────────────────
if [[ "$ENV_ID" == "all" ]]; then
    for env in "${VALID_ENVS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            run_one "$env" "$seed"
        done
    done
else
    for seed in "${SEEDS[@]}"; do
        run_one "$ENV_ID" "$seed"
    done
fi
