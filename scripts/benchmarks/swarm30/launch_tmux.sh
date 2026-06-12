#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
RUNTIME="${RUNTIME:-$(date -u +%Y%m%d_%H%M)}"
SESSION="${SESSION:-wf_swarm30_backend_${RUNTIME}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/debug/benchmarks/swarm30_backend_training}"
LOG_DIR="${ROOT}/${OUTPUT_ROOT}/${RUNTIME}/launcher"
PYTHON="${PYTHON:-/opt/conda/bin/python}"
TRACKS="${TRACKS:-standard_specialist tuned_specialist generalist}"
SEEDS="${SEEDS:-0}"
DYNAMICS_BACKEND="${DYNAMICS_BACKEND:-waypoint_behavior_fast}"
ENV_BACKEND="${ENV_BACKEND:-process}"
NUM_ENVS="${NUM_ENVS:-8}"
ROLLOUT_STEPS="${ROLLOUT_STEPS:-128}"
SPECIALIST_UPDATES="${SPECIALIST_UPDATES:-1000}"
GENERALIST_UPDATES="${GENERALIST_UPDATES:-3000}"
EVAL_INTERVAL="${EVAL_INTERVAL:-100}"
EVAL_EPISODES="${EVAL_EPISODES:-10}"
RECORD_EVAL_EPISODES="${RECORD_EVAL_EPISODES:-1}"
MAX_PARALLEL="${MAX_PARALLEL:-3}"
GPUS="${GPUS:-0 1 2 3}"
CPU_THREADS_PER_JOB="${CPU_THREADS_PER_JOB:-3}"
SKIP_EVALUATION="${SKIP_EVALUATION:-1}"
SKIP_EMAIL="${SKIP_EMAIL:-0}"
mkdir -p "${LOG_DIR}"

read -r -a TRACK_ARGS <<< "${TRACKS}"
read -r -a SEED_ARGS <<< "${SEEDS}"
read -r -a GPU_ARGS <<< "${GPUS}"

command=(
  "${PYTHON}" scripts/benchmarks/swarm30/run_benchmark.py
  --output-root "${OUTPUT_ROOT}"
  --runtime "${RUNTIME}"
  --tracks "${TRACK_ARGS[@]}"
  --seeds "${SEED_ARGS[@]}"
  --dynamics-backend "${DYNAMICS_BACKEND}"
  --num-envs "${NUM_ENVS}"
  --rollout-steps "${ROLLOUT_STEPS}"
  --specialist-updates "${SPECIALIST_UPDATES}"
  --generalist-updates "${GENERALIST_UPDATES}"
  --eval-interval "${EVAL_INTERVAL}"
  --eval-episodes "${EVAL_EPISODES}"
  --record-eval-episodes "${RECORD_EVAL_EPISODES}"
  --env-backend "${ENV_BACKEND}"
  --gpus "${GPU_ARGS[@]}"
  --max-parallel "${MAX_PARALLEL}"
  --cpu-threads-per-job "${CPU_THREADS_PER_JOB}"
)
if [[ "${SKIP_EVALUATION}" == "1" || "${SKIP_EVALUATION}" == "true" ]]; then
  command+=(--skip-evaluation)
fi
if [[ "${SKIP_EMAIL}" == "1" || "${SKIP_EMAIL}" == "true" ]]; then
  command+=(--skip-email)
fi

printf '%q ' "${command[@]}" > "${LOG_DIR}/launch_command.sh"
printf '\n' >> "${LOG_DIR}/launch_command.sh"

tmux new-session -d -s "${SESSION}" \
  "cd '${ROOT}' && ${command[*]} 2>&1 | tee '${LOG_DIR}/queue_${RUNTIME}.log'"

printf 'session=%s\nruntime=%s\noutput=%s\nlog=%s\n' \
  "${SESSION}" "${RUNTIME}" "${ROOT}/${OUTPUT_ROOT}/${RUNTIME}" "${LOG_DIR}/queue_${RUNTIME}.log"
