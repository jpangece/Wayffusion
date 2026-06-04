#!/usr/bin/env bash
set -u -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

export MPLBACKEND="${MPLBACKEND:-Agg}"
export PYTHON_BIN="${PYTHON_BIN:-/opt/conda/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  export PYTHON_BIN="python3"
fi

BASE_RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
if [[ "${BASE_RUN_TIMESTAMP}" == *_parallel ]]; then
  export RUN_TIMESTAMP="${BASE_RUN_TIMESTAMP}"
else
  export RUN_TIMESTAMP="${BASE_RUN_TIMESTAMP}_parallel"
fi

# Keep the same experiment family as the original parallel script. These can be
# overridden from the shell without editing this file.
export CONFIG="${CONFIG:-configs/policy/ppo_cnn_deepsets_multitask_20k.yaml}"
export ENV_CONFIG="${ENV_CONFIG:-configs/env/multitask.yaml}"
export SCALING_MODE="${SCALING_MODE:-fixed_map}"
export OBS_VARIANT="${OBS_VARIANT:-multi_channel_field+task_id}"
export TARGET_EPISODES="${TARGET_EPISODES:-0}"
export AGENT_COUNTS="${AGENT_COUNTS:-4}"
export EVAL_EPISODES="${EVAL_EPISODES:-5}"
export RECORD_EVAL_EPISODES="${RECORD_EVAL_EPISODES:-1}"
export RECORD_FORMAT="${RECORD_FORMAT:-gif}"
export RECORD_FPS="${RECORD_FPS:-8}"
export RECORD_INTERVAL="${RECORD_INTERVAL:-4}"
export CONSOLE_LOG_INTERVAL="${CONSOLE_LOG_INTERVAL:-10}"
export ENV_BACKEND="${ENV_BACKEND:-thread}"
export ENVS_PER_TASK="${ENVS_PER_TASK:-4}"
export ENV_WORKERS="${ENV_WORKERS:-16}"

SMTP_CONFIG_FILE="${SMTP_CONFIG_FILE:-.secrets/wayffusion_mail.env}"
if [[ -f "${SMTP_CONFIG_FILE}" ]]; then
  # shellcheck disable=SC1090
  source "${SMTP_CONFIG_FILE}"
fi
SMTP_PRESET="${SMTP_PRESET:-qq}"
if [[ "${SMTP_PRESET}" == "qq" ]]; then
  SMTP_HOST="${SMTP_HOST:-smtp.qq.com}"
  SMTP_PORT="${SMTP_PORT:-465}"
  SMTP_SSL="${SMTP_SSL:-1}"
  SMTP_STARTTLS="${SMTP_STARTTLS:-0}"
fi
if [[ "${EMAIL_TO+x}" != "x" ]]; then
  EMAIL_TO="muadib@foxmail.com"
fi
NOTIFY_ONLY="${NOTIFY_ONLY:-0}"

PARALLEL_LOG_ROOT="${PARALLEL_LOG_ROOT:-outputs/training/ppo/${RUN_TIMESTAMP}}"
mkdir -p "${PARALLEL_LOG_ROOT}"
SUMMARY_CSV="${PARALLEL_LOG_ROOT}/summary.csv"
MAIN_LOG="${PARALLEL_LOG_ROOT}/parallel.log"

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "${MAIN_LOG}"
}

notify_email() {
  local subject="$1"
  local body="$2"
  local to="${EMAIL_TO}"
  if [[ -z "${to}" ]]; then
    log "email disabled: EMAIL_TO is empty"
    return 0
  fi
  if [[ -n "${SMTP_HOST:-}" ]]; then
    if [[ "${SMTP_ALLOW_NO_AUTH:-0}" != "1" && ( -z "${SMTP_USER:-}" || -z "${SMTP_PASSWORD:-}" ) ]]; then
      log "email not sent: SMTP_HOST is set but SMTP_USER or SMTP_PASSWORD is empty"
      return 1
    fi
    SMTP_HOST="${SMTP_HOST}" SMTP_PORT="${SMTP_PORT:-587}" SMTP_USER="${SMTP_USER:-}" SMTP_PASSWORD="${SMTP_PASSWORD:-}" SMTP_FROM="${SMTP_FROM:-}" SMTP_SSL="${SMTP_SSL:-0}" SMTP_STARTTLS="${SMTP_STARTTLS:-1}" EMAIL_TO="${to}" EMAIL_SUBJECT="${subject}" EMAIL_BODY="${body}" "${PYTHON_BIN}" - <<'PYMAIL'
from __future__ import annotations

import os
import smtplib
from email.message import EmailMessage

msg = EmailMessage()
msg["From"] = os.environ.get("SMTP_FROM", os.environ.get("SMTP_USER", "wayffusion@localhost"))
msg["To"] = os.environ["EMAIL_TO"]
msg["Subject"] = os.environ["EMAIL_SUBJECT"]
msg.set_content(os.environ["EMAIL_BODY"])

host = os.environ["SMTP_HOST"]
port = int(os.environ.get("SMTP_PORT", "587"))
use_ssl = os.environ.get("SMTP_SSL", "0") == "1"
use_starttls = os.environ.get("SMTP_STARTTLS", "1") == "1" and not use_ssl
client_cls = smtplib.SMTP_SSL if use_ssl else smtplib.SMTP
with client_cls(host, port, timeout=30) as client:
    if use_starttls:
        client.starttls()
    user = os.environ.get("SMTP_USER", "")
    if user:
        client.login(user, os.environ.get("SMTP_PASSWORD", ""))
    client.send_message(msg)
PYMAIL
    return $?
  fi
  log "email not sent: SMTP_HOST is not configured"
  return 1
}

# Same experiment types as the old run_ppo_parallel_specialists_all4.sh:
# one specialist per task plus one shared all-four-task policy.
JOBS=(
  "specialist_goal_nav|scripts/ppo_targets/train_specialist_goal_nav.sh"
  "specialist_coverage|scripts/ppo_targets/train_specialist_coverage.sh"
  "specialist_formation|scripts/ppo_targets/train_specialist_formation.sh"
  "specialist_risk_nav|scripts/ppo_targets/train_specialist_risk_nav.sh"
  "multi_all4|scripts/ppo_targets/train_all4.sh"
)

printf 'label,status,exit_code,start_time,end_time,duration_sec,log_path\n' > "${SUMMARY_CSV}"

if [[ "${NOTIFY_ONLY}" == "1" ]]; then
  if notify_email "[Wayffusion] PPO parallel notification test" "PPO parallel notification test.

repo_root=${REPO_ROOT}
parallel_log_root=${PARALLEL_LOG_ROOT}
email_to=${EMAIL_TO}
config=${CONFIG}
env_config=${ENV_CONFIG}
"; then
    log "notification test sent to ${EMAIL_TO}"
    exit 0
  fi
  log "notification test failed for ${EMAIL_TO}"
  exit 1
fi

queue_start_epoch="$(date +%s)"
log "parallel_log_root=${PARALLEL_LOG_ROOT}"
log "run_timestamp=${RUN_TIMESTAMP}"
log "python=${PYTHON_BIN}"
log "config=${CONFIG}"
log "env_config=${ENV_CONFIG}"
log "scaling_mode=${SCALING_MODE}"
log "obs_variant=${OBS_VARIANT}"
log "agent_counts=${AGENT_COUNTS}"
log "env_backend=${ENV_BACKEND}"
log "envs_per_task=${ENVS_PER_TASK}"
log "env_workers=${ENV_WORKERS}"
log "email_to=${EMAIL_TO}"

declare -A pids
declare -A start_epochs
declare -A start_times
declare -A log_paths

for job in "${JOBS[@]}"; do
  IFS='|' read -r label script_path <<< "${job}"
  run_log="${PARALLEL_LOG_ROOT}/${label}.log"
  start_epochs["${label}"]="$(date +%s)"
  start_times["${label}"]="$(date --iso-8601=seconds)"
  log_paths["${label}"]="${run_log}"
  log "start ${label}: ${script_path}"
  bash "${script_path}" > "${run_log}" 2>&1 &
  pid=$!
  pids["${label}"]="${pid}"
  log "pid ${pid} ${label}"
done

queue_status="success"
completed_runs=0
failed_runs=0

# Wait in JOBS order so summary.csv is stable and easy to compare across runs.
for job in "${JOBS[@]}"; do
  IFS='|' read -r label _script_path <<< "${job}"
  pid="${pids[${label}]}"
  if wait "${pid}"; then
    exit_code=0
    status="success"
    completed_runs=$((completed_runs + 1))
  else
    exit_code=$?
    status="failed"
    queue_status="failed"
    failed_runs=$((failed_runs + 1))
  fi
  end_epoch="$(date +%s)"
  end_time="$(date --iso-8601=seconds)"
  duration_sec=$((end_epoch - ${start_epochs[${label}]}))
  printf '"%s","%s","%s","%s","%s","%s","%s"\n' \
    "${label}" "${status}" "${exit_code}" "${start_times[${label}]}" "${end_time}" "${duration_sec}" "${log_paths[${label}]}" >> "${SUMMARY_CSV}"
  log "finish ${label}: ${status} exit_code=${exit_code} duration_sec=${duration_sec}"
done

queue_duration_sec=$(( $(date +%s) - queue_start_epoch ))
subject="[Wayffusion] PPO parallel ${queue_status}: ${completed_runs} completed, ${failed_runs} failed"
body="Wayffusion PPO parallel specialists + all4 finished.

status=${queue_status}
completed_runs=${completed_runs}
failed_runs=${failed_runs}
duration_sec=${queue_duration_sec}
repo_root=${REPO_ROOT}
parallel_log_root=${PARALLEL_LOG_ROOT}
summary_csv=${SUMMARY_CSV}
run_timestamp=${RUN_TIMESTAMP}
config=${CONFIG}
env_config=${ENV_CONFIG}
"

log "queue_status=${queue_status} completed=${completed_runs} failed=${failed_runs} duration_sec=${queue_duration_sec}"
if notify_email "${subject}" "${body}"; then
  log "email notification sent to ${EMAIL_TO}"
else
  log "email notification failed or unavailable for ${EMAIL_TO}"
fi

if [[ "${queue_status}" == "success" ]]; then
  exit 0
fi
exit 1
