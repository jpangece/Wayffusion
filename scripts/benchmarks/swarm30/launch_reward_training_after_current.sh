#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${ROOT}"

PYTHON="${PYTHON:-/opt/conda/bin/python}"
RUNTIME="${RUNTIME:-$(date -u +%Y%m%d_%H%M)_rewardopt}"
SESSION="${SESSION:-wf_swarm30_reward_${RUNTIME}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/training/benchmarks/swarm30_2000_direct}"
LAUNCHER_DIR="${ROOT}/${OUTPUT_ROOT}/${RUNTIME}/launcher"
LOG_PATH="${LAUNCHER_DIR}/one_click_launcher.log"
SESSION_PATH="${LAUNCHER_DIR}/one_click_launcher.session"
RUNNER_PATH="${LAUNCHER_DIR}/one_click_runner.sh"

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

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION}" >&2
  echo "Attach with: tmux attach -t ${SESSION}" >&2
  exit 1
fi

resolve_wait_session() {
  local requested="$1"
  if [[ "${requested}" == "none" || "${requested}" == "off" || "${requested}" == "false" || "${requested}" == "0" ]]; then
    printf ''
    return 0
  fi
  if [[ "${requested}" != "auto" ]]; then
    printf '%s' "${requested}"
    return 0
  fi
  if ! command -v tmux >/dev/null 2>&1; then
    printf ''
    return 0
  fi
  tmux list-sessions -F '#{session_name}' 2>/dev/null \
    | grep -E '^(wf_swarm30_backend_|wf_swarm30_reward_|wf_swarm30_)' \
    | grep -v -x "${SESSION}" \
    | sort \
    | tail -n 1 || true
}

WAIT_SESSION_RESOLVED="$(resolve_wait_session "${WAIT_SESSION}")"

command=(
  "${PYTHON}" scripts/benchmarks/swarm30/wait_for_resources_and_launch_reward_training.py
  --runtime "${RUNTIME}"
  --output-root "${OUTPUT_ROOT}"
  --wait-session "${WAIT_SESSION_RESOLVED:-none}"
  --exclude-wait-session "${SESSION}"
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
printf '%s\n' "${SESSION}" > "${SESSION_PATH}"
printf '%s\n' "${WAIT_SESSION_RESOLVED:-none}" > "${LAUNCHER_DIR}/wait_session.resolved"

cat > "${RUNNER_PATH}" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd '${ROOT}'
mkdir -p '${LAUNCHER_DIR}'
echo '[one-click] session=${SESSION}' | tee -a '${LOG_PATH}'
echo '[one-click] wait_session_resolved=${WAIT_SESSION_RESOLVED:-none}' | tee -a '${LOG_PATH}'
echo '[one-click] runtime=${RUNTIME}' | tee -a '${LOG_PATH}'
echo '[one-click] started_at='"\$(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a '${LOG_PATH}'
set +e
bash '${LAUNCHER_DIR}/one_click_command.sh' 2>&1 | tee -a '${LOG_PATH}'
status=\${PIPESTATUS[0]}
set -e
echo '[one-click] finished_at='"\$(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a '${LOG_PATH}'
echo "[one-click] exit_status=\${status}" | tee -a '${LOG_PATH}'
exit "\${status}"
EOF
chmod +x "${RUNNER_PATH}"

tmux new-session -d -s "${SESSION}" "bash '${RUNNER_PATH}'"

cat <<EOF
Reward-optimized Swarm30 one-click launcher started in tmux.
Runtime: ${RUNTIME}
Session: ${SESSION}
Waiting for previous session: ${WAIT_SESSION_RESOLVED:-none}
Output: ${ROOT}/${OUTPUT_ROOT}/${RUNTIME}
Log: ${LOG_PATH}
State: ${ROOT}/${OUTPUT_ROOT}/${RUNTIME}/launcher/resource_wait_state.json
Monitor: ${ROOT}/${OUTPUT_ROOT}/${RUNTIME}/launcher/monitor_state.json
Command: ${ROOT}/${OUTPUT_ROOT}/${RUNTIME}/launcher/one_click_command.sh
Runner: ${RUNNER_PATH}

Useful commands:
  tmux attach -t ${SESSION}
  tmux capture-pane -pt ${SESSION} -S -80
  tail -f ${LOG_PATH}
  cat ${ROOT}/${OUTPUT_ROOT}/${RUNTIME}/launcher/resource_wait_state.json
  cat ${ROOT}/${OUTPUT_ROOT}/${RUNTIME}/launcher/monitor_state.json
EOF
