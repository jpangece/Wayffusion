#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUNTIME="${RUN_TIMESTAMP:-$(date -u +%Y%m%d_%H%M)}"
SESSION="${TMUX_SESSION:-wayffusion_formal_${RUNTIME}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/outputs/training/mappo_direct_specialists}"
SUITE_ROOT="${OUTPUT_ROOT}/${RUNTIME}"
ENV_FILE="${ENV_FILE:-${ROOT}/.secrets/wayffusion_mail.env}"

mkdir -p "${SUITE_ROOT}"

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION}" >&2
  exit 1
fi

COMMAND=(
  /opt/conda/bin/python
  "${ROOT}/scripts/formal_mappo_specialists/run_formal_specialists.py"
  --runtime "${RUNTIME}"
  --output-root "${OUTPUT_ROOT}"
  --env-file "${ENV_FILE}"
  --num-envs "${NUM_ENVS:-16}"
  --rollout-steps "${ROLLOUT_STEPS:-256}"
  --eval-interval "${EVAL_INTERVAL:-100}"
  --eval-episodes "${EVAL_EPISODES:-20}"
  --record-eval-episodes "${RECORD_EVAL_EPISODES:-2}"
  --record-interval "${RECORD_INTERVAL:-5}"
  --cpu-threads-per-task "${CPU_THREADS_PER_TASK:-1}"
  --env-backend "${ENV_BACKEND:-process}"
)

printf '%q ' "${COMMAND[@]}" > "${SUITE_ROOT}/tmux_command.sh"
printf '\n' >> "${SUITE_ROOT}/tmux_command.sh"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  printf 'runtime=%s\nsession=%s\nsuite_root=%s\ncommand=' "${RUNTIME}" "${SESSION}" "${SUITE_ROOT}"
  printf '%q ' "${COMMAND[@]}"
  printf '\n'
  exit 0
fi

tmux new-session -d -s "${SESSION}" \
  "cd $(printf '%q' "${ROOT}") && exec $(printf '%q ' "${COMMAND[@]}") >> $(printf '%q' "${SUITE_ROOT}/suite_console.log") 2>&1"

cat > "${SUITE_ROOT}/tmux_info.json" <<EOF
{
  "runtime": "${RUNTIME}",
  "session": "${SESSION}",
  "suite_root": "${SUITE_ROOT}",
  "started_at_utc": "$(date -u --iso-8601=seconds)"
}
EOF

printf '%s\n' "${RUNTIME}" > "${OUTPUT_ROOT}/latest_runtime.txt"
echo "Started tmux session: ${SESSION}"
echo "Output: ${SUITE_ROOT}"
echo "Attach: tmux attach -t ${SESSION}"
echo "Status: ${ROOT}/scripts/formal_mappo_specialists/status.sh ${RUNTIME}"
