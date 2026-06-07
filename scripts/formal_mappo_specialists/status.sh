#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/outputs/training/mappo_direct_specialists}"
RUNTIME="${1:-}"
if [[ -z "${RUNTIME}" && -f "${OUTPUT_ROOT}/latest_runtime.txt" ]]; then
  RUNTIME="$(cat "${OUTPUT_ROOT}/latest_runtime.txt")"
fi
if [[ -z "${RUNTIME}" ]]; then
  echo "usage: $0 <runtime>" >&2
  exit 2
fi

SUITE_ROOT="${OUTPUT_ROOT}/${RUNTIME}"
echo "Suite: ${SUITE_ROOT}"
if [[ -f "${SUITE_ROOT}/suite_state.json" ]]; then
  /opt/conda/bin/python -m json.tool "${SUITE_ROOT}/suite_state.json"
else
  echo "suite_state.json not written yet"
fi
echo
for task in area_coverage belief_search priority_inspection connectivity_expansion; do
  metrics="${SUITE_ROOT}/${task}/s0/training_metrics.csv"
  if [[ -f "${metrics}" ]]; then
    updates=$(( $(wc -l < "${metrics}") - 1 ))
    echo "${task}: ${updates} metric rows"
  else
    echo "${task}: waiting for metrics"
  fi
done
