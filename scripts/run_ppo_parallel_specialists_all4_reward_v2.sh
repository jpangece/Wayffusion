#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Reward-v2 experiment entrypoint. It delegates to the canonical parallel
# launcher so summary.csv, per-run logs, TensorBoard, media, and snapshots keep
# the same layout as run_ppo_parallel_specialists_all4.sh.
export CONFIG="${CONFIG:-configs/policy/ppo_cnn_deepsets_spatial_reward_v2.yaml}"
export ENV_CONFIG="${ENV_CONFIG:-configs/env/multitask.yaml}"
export RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)_reward_v2}"

exec bash "${SCRIPT_DIR}/run_ppo_parallel_specialists_all4.sh" "$@"
