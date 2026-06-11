#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/opt/conda/bin/python}"
RUNTIME="${RUNTIME:-$(date -u +%Y%m%d_%H%M)}"
SESSION="${SESSION:-wf_phase31_${RUNTIME}}"
OUTPUT_ROOT="${ROOT}/outputs/debug/mappo_direct_specialists/phase31_connectivity_training_seed_reproducibility"
RUNTIME_ROOT="${OUTPUT_ROOT}/${RUNTIME}"

mkdir -p "${RUNTIME_ROOT}"
printf '%q ' "${PYTHON_BIN}" \
  "${ROOT}/scripts/debug_connectivity_soft_phases/run_phase31_training_seed_reproducibility.py" \
  --runtime "${RUNTIME}" \
  --output-root "${OUTPUT_ROOT}" \
  --seeds 1 2 3 \
  --gpus 0 1 2 > "${RUNTIME_ROOT}/launch_command.txt"
printf '\n' >> "${RUNTIME_ROOT}/launch_command.txt"

tmux new-session -d -s "${SESSION}" \
  "cd '${ROOT}' && '${PYTHON_BIN}' scripts/debug_connectivity_soft_phases/run_phase31_training_seed_reproducibility.py \
    --runtime '${RUNTIME}' \
    --output-root '${OUTPUT_ROOT}' \
    --seeds 1 2 3 \
    --gpus 0 1 2 \
    2>&1 | tee '${RUNTIME_ROOT}/controller.log'"

cat > "${RUNTIME_ROOT}/launch_metadata.json" <<EOF
{
  "runtime": "${RUNTIME}",
  "tmux_session": "${SESSION}",
  "output_root": "${RUNTIME_ROOT}",
  "training_seeds": [1, 2, 3],
  "gpu_assignment": {"1": 0, "2": 1, "3": 2}
}
EOF

echo "tmux_session=${SESSION}"
echo "runtime_root=${RUNTIME_ROOT}"
echo "attach: tmux attach -t ${SESSION}"
echo "logs: tail -f ${RUNTIME_ROOT}/seed01/orchestrator.log"
