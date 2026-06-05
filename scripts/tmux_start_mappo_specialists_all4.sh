#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# One-command tmux launcher for pure MAPPO single-task specialists.
# This intentionally does not pass --init_checkpoint because the goal is to
# evaluate MAPPO from random initialization, not BC warm-start or PPO continuation.

RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)_mappo_pure_all4_long}"
TMUX_SESSION="${TMUX_SESSION:-wayffusion_mappo_${RUN_TIMESTAMP}}"
QUEUE_LOG_ROOT="${QUEUE_LOG_ROOT:-outputs/training/parallel_mappo_specialists/${RUN_TIMESTAMP}}"

GOAL_TOTAL_UPDATES="${GOAL_TOTAL_UPDATES:-1000}"
COVERAGE_TOTAL_UPDATES="${COVERAGE_TOTAL_UPDATES:-1000}"
RISK_TOTAL_UPDATES="${RISK_TOTAL_UPDATES:-1000}"
FORMATION_TOTAL_UPDATES="${FORMATION_TOTAL_UPDATES:-1000}"

EVAL_EPISODES="${EVAL_EPISODES:-50}"
RECORD_EVAL_EPISODES="${RECORD_EVAL_EPISODES:-0}"

GOAL_CUDA_VISIBLE_DEVICES="${GOAL_CUDA_VISIBLE_DEVICES:-0}"
COVERAGE_CUDA_VISIBLE_DEVICES="${COVERAGE_CUDA_VISIBLE_DEVICES:-1}"
RISK_CUDA_VISIBLE_DEVICES="${RISK_CUDA_VISIBLE_DEVICES:-2}"
FORMATION_CUDA_VISIBLE_DEVICES="${FORMATION_CUDA_VISIBLE_DEVICES:-3}"

AGENT_COUNTS="${AGENT_COUNTS:-4}"
DEFAULT_ENV_BACKEND="${DEFAULT_ENV_BACKEND:-sync}"
TARGET_EPISODES="${TARGET_EPISODES:-0}"
CONSOLE_LOG_INTERVAL="${CONSOLE_LOG_INTERVAL:-10}"
DRY_RUN="${DRY_RUN:-0}"

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux is not installed or not on PATH" >&2
  exit 1
fi

if tmux has-session -t "${TMUX_SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${TMUX_SESSION}" >&2
  echo "Use TMUX_SESSION=<new_name> or attach with: tmux attach -t ${TMUX_SESSION}" >&2
  exit 2
fi

mkdir -p "${QUEUE_LOG_ROOT}"
LAUNCH_SCRIPT="${QUEUE_LOG_ROOT}/launch_inside_tmux.sh"

write_export() {
  local name="$1"
  local value="$2"
  printf 'export %s=%q\n' "${name}" "${value}"
}

{
  printf '#!/usr/bin/env bash\n'
  printf 'set -u -o pipefail\n'
  printf 'cd %q\n' "${REPO_ROOT}"
  write_export RUN_TIMESTAMP "${RUN_TIMESTAMP}"
  write_export QUEUE_LOG_ROOT "${QUEUE_LOG_ROOT}"
  write_export GOAL_TOTAL_UPDATES "${GOAL_TOTAL_UPDATES}"
  write_export COVERAGE_TOTAL_UPDATES "${COVERAGE_TOTAL_UPDATES}"
  write_export RISK_TOTAL_UPDATES "${RISK_TOTAL_UPDATES}"
  write_export FORMATION_TOTAL_UPDATES "${FORMATION_TOTAL_UPDATES}"
  write_export EVAL_EPISODES "${EVAL_EPISODES}"
  write_export RECORD_EVAL_EPISODES "${RECORD_EVAL_EPISODES}"
  write_export GOAL_CUDA_VISIBLE_DEVICES "${GOAL_CUDA_VISIBLE_DEVICES}"
  write_export COVERAGE_CUDA_VISIBLE_DEVICES "${COVERAGE_CUDA_VISIBLE_DEVICES}"
  write_export RISK_CUDA_VISIBLE_DEVICES "${RISK_CUDA_VISIBLE_DEVICES}"
  write_export FORMATION_CUDA_VISIBLE_DEVICES "${FORMATION_CUDA_VISIBLE_DEVICES}"
  write_export AGENT_COUNTS "${AGENT_COUNTS}"
  write_export DEFAULT_ENV_BACKEND "${DEFAULT_ENV_BACKEND}"
  write_export TARGET_EPISODES "${TARGET_EPISODES}"
  write_export CONSOLE_LOG_INTERVAL "${CONSOLE_LOG_INTERVAL}"
  printf 'bash scripts/run_mappo_parallel_specialists_all4.sh\n'
} > "${LAUNCH_SCRIPT}"
chmod +x "${LAUNCH_SCRIPT}"

cat > "${QUEUE_LOG_ROOT}/launch_summary.txt" <<EOF
run_timestamp=${RUN_TIMESTAMP}
tmux_session=${TMUX_SESSION}
queue_log_root=${QUEUE_LOG_ROOT}
launch_script=${LAUNCH_SCRIPT}
goal_total_updates=${GOAL_TOTAL_UPDATES}
coverage_total_updates=${COVERAGE_TOTAL_UPDATES}
risk_total_updates=${RISK_TOTAL_UPDATES}
formation_total_updates=${FORMATION_TOTAL_UPDATES}
eval_episodes=${EVAL_EPISODES}
record_eval_episodes=${RECORD_EVAL_EPISODES}
goal_cuda=${GOAL_CUDA_VISIBLE_DEVICES}
coverage_cuda=${COVERAGE_CUDA_VISIBLE_DEVICES}
risk_cuda=${RISK_CUDA_VISIBLE_DEVICES}
formation_cuda=${FORMATION_CUDA_VISIBLE_DEVICES}
bc_warm_start=0
EOF

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "DRY_RUN=1, not starting tmux."
  cat "${QUEUE_LOG_ROOT}/launch_summary.txt"
  exit 0
fi

tmux new-session -d -s "${TMUX_SESSION}" "bash '${LAUNCH_SCRIPT}' > '${QUEUE_LOG_ROOT}/launcher.out' 2>&1"

echo "started tmux session: ${TMUX_SESSION}"
echo "run_timestamp: ${RUN_TIMESTAMP}"
echo "queue_log_root: ${QUEUE_LOG_ROOT}"
echo "monitor: tail -f ${QUEUE_LOG_ROOT}/parallel.log"
echo "attach: tmux attach -t ${TMUX_SESSION}"
