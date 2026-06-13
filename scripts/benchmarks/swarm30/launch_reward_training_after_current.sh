#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${ROOT}"

PYTHON="${PYTHON:-/opt/conda/bin/python}"
RUNTIME="${RUNTIME:-rewardopt_$(date -u +%Y%m%d_%H%M)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/debug/benchmarks/swarm30_reward_optimized_training}"
LAUNCHER_DIR="${ROOT}/${OUTPUT_ROOT}/${RUNTIME}/launcher"
LOG_PATH="${LAUNCHER_DIR}/one_click_launcher.log"
PID_PATH="${LAUNCHER_DIR}/one_click_launcher.pid"

# Defaults are intentionally conservative and match the current N=30 reward-optimized training plan.
WAIT_SESSION="${WAIT_SESSION:-auto}"
WAIT_PATTERN="${WAIT_PATTERN:-scripts/benchmarks/swarm30/run_benchmark.py}"
POLL_SECONDS="${POLL_SECONDS:-120}"
MAX_WAIT_HOURS="${MAX_WAIT_HOURS:-72}"
MAX_LOAD_1M="${MAX_LOAD_1M:-18}"
SMTP_ENV_FILE="${SMTP_ENV_FILE:-.secrets/wayffusion_mail.env}"
TRACKS="${TRACKS:-tuned_specialist generalist}"
SEEDS="${SEEDS:-0}"
DYNAMICS_BACKEND="${DYNAMICS_BACKEND:-waypoint_behavior_fast}"
ENV_BACKEND="${ENV_BACKEND:-process}"
NUM_ENVS="${NUM_ENVS:-8}"
ROLLOUT_STEPS="${ROLLOUT_STEPS:-128}"
SPECIALIST_UPDATES="${SPECIALIST_UPDATES:-2000}"
GENERALIST_UPDATES="${GENERALIST_UPDATES:-2000}"
EVAL_INTERVAL="${EVAL_INTERVAL:-100}"
EVAL_EPISODES="${EVAL_EPISODES:-10}"
RECORD_EVAL_EPISODES="${RECORD_EVAL_EPISODES:-0}"
GPUS="${GPUS:-0 1 2 3}"
MAX_PARALLEL="${MAX_PARALLEL:-3}"
CPU_THREADS_PER_JOB="${CPU_THREADS_PER_JOB:-3}"
MONITOR_SECONDS="${MONITOR_SECONDS:-300}"
MONITOR_MAX_HOURS="${MONITOR_MAX_HOURS:-240}"
STATUS_EMAIL_INTERVAL_HOURS="${STATUS_EMAIL_INTERVAL_HOURS:-6}"
DISABLE_EMAIL="${DISABLE_EMAIL:-0}"
RUN_FINAL_EVALUATION="${RUN_FINAL_EVALUATION:-0}"
SKIP_EMAIL_FOR_BENCHMARK="${SKIP_EMAIL_FOR_BENCHMARK:-1}"

mkdir -p "${LAUNCHER_DIR}"
read -r -a TRACK_ARGS <<< "${TRACKS}"
read -r -a SEED_ARGS <<< "${SEEDS}"
read -r -a GPU_ARGS <<< "${GPUS}"

command=(
  "${PYTHON}" scripts/benchmarks/swarm30/wait_for_resources_and_launch_reward_training.py
  --runtime "${RUNTIME}"
  --output-root "${OUTPUT_ROOT}"
  --wait-session "${WAIT_SESSION}"
  --wait-pattern "${WAIT_PATTERN}"
  --poll-seconds "${POLL_SECONDS}"
  --max-wait-hours "${MAX_WAIT_HOURS}"
  --max-load-1m "${MAX_LOAD_1M}"
  --smtp-env-file "${SMTP_ENV_FILE}"
  --tracks "${TRACK_ARGS[@]}"
  --seeds "${SEED_ARGS[@]}"
  --dynamics-backend "${DYNAMICS_BACKEND}"
  --env-backend "${ENV_BACKEND}"
  --num-envs "${NUM_ENVS}"
  --rollout-steps "${ROLLOUT_STEPS}"
  --specialist-updates "${SPECIALIST_UPDATES}"
  --generalist-updates "${GENERALIST_UPDATES}"
  --eval-interval "${EVAL_INTERVAL}"
  --eval-episodes "${EVAL_EPISODES}"
  --record-eval-episodes "${RECORD_EVAL_EPISODES}"
  --gpus "${GPU_ARGS[@]}"
  --max-parallel "${MAX_PARALLEL}"
  --cpu-threads-per-job "${CPU_THREADS_PER_JOB}"
  --monitor-seconds "${MONITOR_SECONDS}"
  --monitor-max-hours "${MONITOR_MAX_HOURS}"
  --status-email-interval-hours "${STATUS_EMAIL_INTERVAL_HOURS}"
)

if [[ "${DISABLE_EMAIL}" == "1" || "${DISABLE_EMAIL}" == "true" ]]; then
  command+=(--disable-email)
fi
if [[ "${RUN_FINAL_EVALUATION}" == "1" || "${RUN_FINAL_EVALUATION}" == "true" ]]; then
  command+=(--run-final-evaluation)
fi
if [[ "${SKIP_EMAIL_FOR_BENCHMARK}" == "1" || "${SKIP_EMAIL_FOR_BENCHMARK}" == "true" ]]; then
  command+=(--skip-email-for-benchmark)
fi

printf '%q ' "${command[@]}" > "${LAUNCHER_DIR}/one_click_command.sh"
printf '\n' >> "${LAUNCHER_DIR}/one_click_command.sh"

nohup "${command[@]}" > "${LOG_PATH}" 2>&1 &
pid="$!"
echo "${pid}" > "${PID_PATH}"

cat <<EOF
Reward-optimized Swarm30 one-click launcher started.
Runtime: ${RUNTIME}
PID: ${pid}
Output: ${ROOT}/${OUTPUT_ROOT}/${RUNTIME}
Log: ${LOG_PATH}
State: ${ROOT}/${OUTPUT_ROOT}/${RUNTIME}/launcher/resource_wait_state.json
Monitor: ${ROOT}/${OUTPUT_ROOT}/${RUNTIME}/launcher/monitor_state.json
Command: ${ROOT}/${OUTPUT_ROOT}/${RUNTIME}/launcher/one_click_command.sh

Useful commands:
  tail -f ${LOG_PATH}
  cat ${ROOT}/${OUTPUT_ROOT}/${RUNTIME}/launcher/resource_wait_state.json
  cat ${ROOT}/${OUTPUT_ROOT}/${RUNTIME}/launcher/monitor_state.json
EOF
