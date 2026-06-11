#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
tmux ls 2>/dev/null | grep 'wf_swarm30_v1_' || true
if [[ -f "${ROOT}/outputs/training/benchmarks/swarm30_v1/queue_state.json" ]]; then
  cat "${ROOT}/outputs/training/benchmarks/swarm30_v1/queue_state.json"
fi
