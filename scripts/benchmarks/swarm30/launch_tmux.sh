#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
RUNTIME="${RUNTIME:-$(date -u +%Y%m%d_%H%M)}"
SESSION="${SESSION:-wf_swarm30_v1_${RUNTIME}}"
PHASE31_ROOT="${PHASE31_ROOT:-outputs/debug/mappo_direct_specialists/phase31_connectivity_training_seed_reproducibility/20260610_0954}"
LOG_DIR="${ROOT}/outputs/training/benchmarks/swarm30_v1"
PYTHON="${PYTHON:-/opt/conda/bin/python}"
mkdir -p "${LOG_DIR}"

tmux new-session -d -s "${SESSION}" \
  "cd '${ROOT}' && '${PYTHON}' scripts/benchmarks/swarm30/wait_for_phase31_and_launch.py \
    --phase31-root '${PHASE31_ROOT}' \
    --gpus 0 1 2 3 \
    --max-parallel 4 \
    2>&1 | tee '${LOG_DIR}/queue_${RUNTIME}.log'"

printf 'session=%s\nruntime=%s\nlog=%s\n' "${SESSION}" "${RUNTIME}" "${LOG_DIR}/queue_${RUNTIME}.log"
