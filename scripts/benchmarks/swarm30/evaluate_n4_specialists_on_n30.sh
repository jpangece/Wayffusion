#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${ROOT}"

PYTHON="${PYTHON:-/opt/conda/bin/python}"
RUNTIME="${RUNTIME:-$(date -u +%Y%m%d_%H%M)}"
SOURCE_RUN="${SOURCE_RUN:-outputs/training/mappo_direct_specialists/20260607_1237}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/eval/mappo_direct_specialists/${RUNTIME}}"
ENV_CONFIG="${ENV_CONFIG:-configs/env/waypoint_missions.yaml}"
TRACK="${TRACK:-n4_specialist_zero_shot}"
SEED="${SEED:-0}"
FIXED_EPISODES="${FIXED_EPISODES:-20}"
RANDOM_EPISODES="${RANDOM_EPISODES:-100}"
RECORD_EPISODES="${RECORD_EPISODES:-2}"
RECORD_FORMAT="${RECORD_FORMAT:-gif}"
RECORD_FPS="${RECORD_FPS:-8}"
SCALES="${SCALES:-30}"
MODES="${MODES:-fixed random}"
DEVICE="${DEVICE:-}"

TASKS=(
  area_coverage
  belief_search
  priority_inspection
  connectivity_expansion
)

declare -A CHECKPOINT_OVERRIDES=(
  [area_coverage]="${AREA_COVERAGE_CHECKPOINT:-}"
  [belief_search]="${BELIEF_SEARCH_CHECKPOINT:-}"
  [priority_inspection]="${PRIORITY_INSPECTION_CHECKPOINT:-}"
  [connectivity_expansion]="${CONNECTIVITY_EXPANSION_CHECKPOINT:-}"
)

read -r -a SCALE_ARGS <<< "${SCALES}"
read -r -a MODE_ARGS <<< "${MODES}"

mkdir -p "${OUTPUT_DIR}"

{
  printf 'runtime: %s\n' "${RUNTIME}"
  printf 'track: %s\n' "${TRACK}"
  printf 'source_run: %s\n' "${SOURCE_RUN}"
  printf 'env_config: %s\n' "${ENV_CONFIG}"
  printf 'scales: [%s]\n' "${SCALES// /, }"
  printf 'modes: [%s]\n' "${MODES// /, }"
  printf 'fixed_episodes: %s\n' "${FIXED_EPISODES}"
  printf 'random_episodes: %s\n' "${RANDOM_EPISODES}"
  printf 'record_episodes: %s\n' "${RECORD_EPISODES}"
  printf 'seed: %s\n' "${SEED}"
} > "${OUTPUT_DIR}/evaluation_manifest.yaml"

for task in "${TASKS[@]}"; do
  checkpoint="${CHECKPOINT_OVERRIDES[${task}]}"
  if [[ -z "${checkpoint}" ]]; then
    checkpoint="${SOURCE_RUN}/${task}/s0/checkpoints/checkpoint_best_eval.pt"
  fi
  if [[ ! -f "${checkpoint}" ]]; then
    printf 'Missing checkpoint for %s: %s\n' "${task}" "${checkpoint}" >&2
    exit 1
  fi

  task_output="${OUTPUT_DIR}/${task}/s${SEED}"
  mkdir -p "${task_output}"
  command=(
    "${PYTHON}" scripts/benchmarks/swarm30/evaluate.py
    --checkpoint "${checkpoint}"
    --track "${TRACK}"
    --tasks "${task}"
    --scales "${SCALE_ARGS[@]}"
    --modes "${MODE_ARGS[@]}"
    --fixed-episodes "${FIXED_EPISODES}"
    --random-episodes "${RANDOM_EPISODES}"
    --record-episodes "${RECORD_EPISODES}"
    --record-format "${RECORD_FORMAT}"
    --record-fps "${RECORD_FPS}"
    --seed "${SEED}"
    --env-config "${ENV_CONFIG}"
    --output-dir "${task_output}"
    --output-dir-is-task-root
  )
  if [[ -n "${DEVICE}" ]]; then
    command+=(--device "${DEVICE}")
  fi

  printf '[n4-to-n30] task=%s checkpoint=%s output=%s\n' \
    "${task}" "${checkpoint}" "${task_output}"
  "${command[@]}" 2>&1 | tee "${task_output}/evaluate.log"
done

"${PYTHON}" scripts/benchmarks/swarm30/build_report.py \
  --input-dir "${OUTPUT_DIR}" \
  --output-dir "${OUTPUT_DIR}/report"

printf '\nEvaluation complete.\n'
printf 'Output: %s\n' "${OUTPUT_DIR}"
printf 'Report: %s/report/report.md\n' "${OUTPUT_DIR}"
printf 'Note: this is a four-task N=4-to-N=30 zero-shot evaluation, not the full six-task benchmark.\n'
