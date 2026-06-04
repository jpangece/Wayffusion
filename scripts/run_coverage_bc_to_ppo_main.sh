#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

export MPLBACKEND="${MPLBACKEND:-Agg}"
PYTHON_BIN="${PYTHON_BIN:-/opt/conda/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="python3"
fi

RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)_coverage_bc_to_ppo_main}"
BC_CHECKPOINT="${BC_CHECKPOINT:-outputs/training/bc/20260601_034310/debug_bc_coverage_factorized_group_route_target_agents_perm_coverage_N4_multi_channel_field_plus_task_id/checkpoints/checkpoint_0032.pt}"
STAGE1_TOTAL_UPDATES="${STAGE1_TOTAL_UPDATES:-100}"
STAGE2_TOTAL_UPDATES="${STAGE2_TOTAL_UPDATES:-500}"
EVAL_EPISODES="${EVAL_EPISODES:-30}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
ENV_BACKEND="${ENV_BACKEND:-sync}"

SAFE_CONFIG="${SAFE_CONFIG:-configs/policy/debug_ppo_coverage_factorized_group_route_target_agents_safe.yaml}"
PPO_MAIN_CONFIG="${PPO_MAIN_CONFIG:-configs/policy/debug_ppo_coverage_factorized_group_route_target_agents_ppo_main.yaml}"
ENV_CONFIG="${ENV_CONFIG:-configs/env/debug_coverage_route_target_agents_canonical.yaml}"
OBS_VARIANT="${OBS_VARIANT:-multi_channel_field+task_id}"
AGENT_COUNTS="${AGENT_COUNTS:-4}"
SCALING_MODE="${SCALING_MODE:-fixed_map}"
CONSOLE_LOG_INTERVAL="${CONSOLE_LOG_INTERVAL:-10}"

STAGE1_RUN_NAME="${STAGE1_RUN_NAME:-coverage_stage1_safe_from_bc}"
STAGE2_RUN_NAME="${STAGE2_RUN_NAME:-coverage_stage2_ppo_main}"
OUTPUT_ROOT="outputs/training/bc_ppo/${RUN_TIMESTAMP}"
STAGE1_DIR="${OUTPUT_ROOT}/${STAGE1_RUN_NAME}"
STAGE2_DIR="${OUTPUT_ROOT}/${STAGE2_RUN_NAME}"
LOG_ROOT="outputs/debug_long/${RUN_TIMESTAMP}_coverage_bc_to_ppo_main"
mkdir -p "${LOG_ROOT}"

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "${LOG_ROOT}/run.log"
}

require_file() {
  local path="$1"
  if [[ ! -f "${path}" ]]; then
    log "missing required file: ${path}"
    exit 2
  fi
}

require_file "${BC_CHECKPOINT}"
require_file "${SAFE_CONFIG}"
require_file "${PPO_MAIN_CONFIG}"
require_file "${ENV_CONFIG}"

log "coverage BC warm-start + PPO-main experiment"
log "run_timestamp=${RUN_TIMESTAMP}"
log "bc_checkpoint=${BC_CHECKPOINT}"
log "stage1_total_updates=${STAGE1_TOTAL_UPDATES}"
log "stage2_total_updates=${STAGE2_TOTAL_UPDATES}"
log "eval_episodes=${EVAL_EPISODES}"
log "cuda_visible_devices=${CUDA_VISIBLE_DEVICES}"
log "env_backend=${ENV_BACKEND}"
log "stage1_dir=${STAGE1_DIR}"
log "stage2_dir=${STAGE2_DIR}"

run_ppo() {
  local config="$1"
  local init_checkpoint="$2"
  local total_updates="$3"
  local run_name="$4"
  local log_path="$5"

  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" "${PYTHON_BIN}" scripts/train_ppo.py \
    --config "${config}" \
    --env-config "${ENV_CONFIG}" \
    --tasks coverage \
    --agent_counts "${AGENT_COUNTS}" \
    --scaling_mode "${SCALING_MODE}" \
    --obs_variant "${OBS_VARIANT}" \
    --init_checkpoint "${init_checkpoint}" \
    --total_updates "${total_updates}" \
    --target_episodes 0 \
    --eval_episodes "${EVAL_EPISODES}" \
    --record_eval_episodes 0 \
    --console_log_interval "${CONSOLE_LOG_INTERVAL}" \
    --env_backend "${ENV_BACKEND}" \
    --final_eval_source best \
    --run_timestamp "${RUN_TIMESTAMP}" \
    --run_name "${run_name}" \
    --headless \
    --tensorboard \
    2>&1 | tee "${log_path}"
  return "${PIPESTATUS[0]}"
}

log "stage1 start: safe PPO from BC"
run_ppo "${SAFE_CONFIG}" "${BC_CHECKPOINT}" "${STAGE1_TOTAL_UPDATES}" "${STAGE1_RUN_NAME}" "${LOG_ROOT}/stage1_safe_from_bc.log"
log "stage1 finished"

STAGE1_BEST="${STAGE1_DIR}/checkpoints/checkpoint_best_eval.pt"
STAGE2_INIT_CHECKPOINT="${STAGE1_BEST}"
if [[ ! -f "${STAGE2_INIT_CHECKPOINT}" ]]; then
  log "warning: Stage1 best checkpoint not found at ${STAGE1_BEST}; falling back to BC checkpoint"
  STAGE2_INIT_CHECKPOINT="${BC_CHECKPOINT}"
fi

log "stage2 start: PPO-main fine-tuning"
log "stage2_init_checkpoint=${STAGE2_INIT_CHECKPOINT}"
run_ppo "${PPO_MAIN_CONFIG}" "${STAGE2_INIT_CHECKPOINT}" "${STAGE2_TOTAL_UPDATES}" "${STAGE2_RUN_NAME}" "${LOG_ROOT}/stage2_ppo_main.log"
log "stage2 finished"

log "stage1_training_csv=${STAGE1_DIR}/training_metrics.csv"
log "stage2_training_csv=${STAGE2_DIR}/training_metrics.csv"
log "stage1_eval_csv=${STAGE1_DIR}/eval_metrics.csv"
log "stage2_eval_csv=${STAGE2_DIR}/eval_metrics.csv"
log "stage1_best=${STAGE1_BEST}"
log "stage2_best=${STAGE2_DIR}/checkpoints/checkpoint_best_eval.pt"
log "done"
