#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BASE="${ROOT}/outputs/debug/mappo_direct_specialists/phase31_connectivity_training_seed_reproducibility"
RUNTIME="${1:-$(find "${BASE}" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' 2>/dev/null | sort | tail -n 1)}"

if [[ -z "${RUNTIME}" ]]; then
  echo "No Phase31 runtime found." >&2
  exit 1
fi

RUN_ROOT="${BASE}/${RUNTIME}"
echo "runtime_root=${RUN_ROOT}"
for seed in 01 02 03; do
  echo
  echo "seed${seed}:"
  if [[ -f "${RUN_ROOT}/seed${seed}/state.json" ]]; then
    /opt/conda/bin/python - "${RUN_ROOT}/seed${seed}/state.json" <<'PY'
import json
import sys
from pathlib import Path

state = json.loads(Path(sys.argv[1]).read_text())
print(f"  status={state.get('status')}")
print(f"  current_stage={state.get('current_stage', 'not_started')}")
for key, stage in state.get("stages", {}).items():
    print(f"  {key}: final_update={stage.get('actual_final_update')} valid={stage.get('valid')}")
PY
  else
    echo "  state=not_created"
  fi
  latest_csv="$(find "${RUN_ROOT}/seed${seed}" -name training_metrics.csv -type f 2>/dev/null | sort | tail -n 1 || true)"
  if [[ -n "${latest_csv}" ]]; then
    /opt/conda/bin/python - "${latest_csv}" <<'PY'
import csv
import sys

with open(sys.argv[1], newline="") as handle:
    rows = list(csv.DictReader(handle))
if rows:
    print(f"  active_metrics={sys.argv[1]}")
    print(f"  latest_update={rows[-1].get('update')}")
PY
  fi
done
