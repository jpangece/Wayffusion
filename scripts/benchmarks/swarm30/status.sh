#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/debug/benchmarks/swarm30_backend_training}"
tmux ls 2>/dev/null | grep -E 'wf_swarm30_(v1|backend)_' || true
latest="$(find "${ROOT}/${OUTPUT_ROOT}" -maxdepth 2 -name suite_state.json -type f 2>/dev/null | sort | tail -n 1 || true)"
if [[ -n "${latest}" ]]; then
  printf 'latest_state=%s\n' "${latest}"
  cat "${latest}"
fi
