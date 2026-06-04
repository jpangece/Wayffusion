#!/usr/bin/env bash
set -u -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

export MPLBACKEND="${MPLBACKEND:-Agg}"

# -----------------------------------------------------------------------------
# Sequential PPO task queue
#
# Edit this block for normal use. Each enabled row below launches one independent
# PPO training run. Fields are separated by "|":
#
# enabled | run_label | tasks | agent_counts | total_updates | eval_episodes |
# record_eval_episodes | record_interval | cuda_visible_devices | extra_args
#
# Notes:
# - tasks: space-separated task names, for example "goal_nav coverage".
# - record_interval counts evaluation passes, not updates. With eval_interval=25
#   in the default config, record_interval=4 saves GIFs every 100 updates.
# - extra_args can be empty, or contain train_ppo.py arguments such as:
#   --scaling_mode density_preserving --obs_variant single_channel_field
# -----------------------------------------------------------------------------

# Email notification configuration.
# Sensitive values should be provided by environment variables or by an ignored
# local file, not committed into this script. Example file:
#   .secrets/wayffusion_mail.env
#     EMAIL_TO="muadib@foxmail.com"
#     SMTP_USER="your_qq_or_foxmail_account"
#     SMTP_PASSWORD="your_qq_mail_smtp_authorization_code"
#     SMTP_FROM="your_qq_or_foxmail_account"
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
EMAIL_TO="${EMAIL_TO:-muadib@foxmail.com}"
NOTIFY_ONLY="${NOTIFY_ONLY:-0}"
CONFIG="${CONFIG:-configs/policy/ppo_cnn_deepsets_multitask_20k.yaml}"
ENV_CONFIG="${ENV_CONFIG:-configs/env/multitask.yaml}"
SCALING_MODE="${SCALING_MODE:-fixed_map}"
OBS_VARIANT="${OBS_VARIANT:-multi_channel_field+task_id}"
TARGET_EPISODES="${TARGET_EPISODES:-0}"
RECORD_FORMAT="${RECORD_FORMAT:-gif}"
RECORD_FPS="${RECORD_FPS:-8}"
CONSOLE_LOG_INTERVAL="${CONSOLE_LOG_INTERVAL:-10}"
CONTINUE_ON_FAILURE="${CONTINUE_ON_FAILURE:-0}"

# Long enough for meaningful debugging runs without being as expensive as full
# paper-scale sweeps. Raise per row when a combination looks promising.
DEFAULT_TOTAL_UPDATES="2000"
DEFAULT_EVAL_EPISODES="5"
DEFAULT_RECORD_EVAL_EPISODES="1"
DEFAULT_RECORD_INTERVAL="4"
DEFAULT_AGENT_COUNTS="4"
DEFAULT_CUDA_VISIBLE_DEVICES="0"

# Default queue: singles, representative pairs, triples, and all-task training.
# Add, remove, or duplicate rows to run multiple hyperparameter variants.
QUEUE=(
  "1|single_goal_nav|goal_nav|4|2000|5|1|4|0|"
  "1|single_coverage|coverage|4|2000|5|1|4|0|"
  "1|single_formation|formation|4|2000|5|1|4|0|"
  "1|single_risk_nav|risk_nav|4|2000|5|1|4|0|"
  "1|pair_goal_coverage|goal_nav coverage|4|2000|5|1|4|0|"
  "1|pair_goal_formation|goal_nav formation|4|2000|5|1|4|0|"
  "1|pair_coverage_risk|coverage risk_nav|4|2000|5|1|4|0|"
  "1|triple_goal_coverage_formation|goal_nav coverage formation|4|2000|5|1|4|0|"
  "1|all_tasks|goal_nav coverage formation risk_nav|4|2500|5|1|4|0|"
)

RUN_QUEUE_TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
QUEUE_LOG_ROOT="${QUEUE_LOG_ROOT:-outputs/training/task_queue/${RUN_QUEUE_TIMESTAMP}}"
mkdir -p "${QUEUE_LOG_ROOT}"
QUEUE_LOG="${QUEUE_LOG_ROOT}/queue.log"
SUMMARY_CSV="${QUEUE_LOG_ROOT}/summary.csv"

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "${QUEUE_LOG}"
}

shell_join() {
  local out=""
  local arg
  for arg in "$@"; do
    printf -v quoted '%q' "${arg}"
    out+=" ${quoted}"
  done
  echo "${out# }"
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
      log "email not sent: SMTP_HOST is set but SMTP_USER or SMTP_PASSWORD is empty; for QQ mail use an SMTP authorization code"
      return 1
    fi
    SMTP_HOST="${SMTP_HOST}" SMTP_PORT="${SMTP_PORT:-587}" SMTP_USER="${SMTP_USER:-}" SMTP_PASSWORD="${SMTP_PASSWORD:-}" SMTP_FROM="${SMTP_FROM:-}" SMTP_SSL="${SMTP_SSL:-0}" SMTP_STARTTLS="${SMTP_STARTTLS:-1}" EMAIL_TO="${to}" EMAIL_SUBJECT="${subject}" EMAIL_BODY="${body}" python - <<'PYMAIL'
from __future__ import annotations

import os
import smtplib
from email.message import EmailMessage

host = os.environ["SMTP_HOST"]
port = int(os.environ.get("SMTP_PORT", "587"))
user = os.environ.get("SMTP_USER", "")
password = os.environ.get("SMTP_PASSWORD", "")
mail_from = os.environ.get("SMTP_FROM", user or "wayffusion@localhost")
mail_to = os.environ["EMAIL_TO"]
use_ssl = os.environ.get("SMTP_SSL", "0") == "1"
use_starttls = os.environ.get("SMTP_STARTTLS", "1") == "1" and not use_ssl

msg = EmailMessage()
msg["From"] = mail_from
msg["To"] = mail_to
msg["Subject"] = os.environ["EMAIL_SUBJECT"]
msg.set_content(os.environ["EMAIL_BODY"])

client_cls = smtplib.SMTP_SSL if use_ssl else smtplib.SMTP
with client_cls(host, port, timeout=30) as client:
    if use_starttls:
        client.starttls()
    if user:
        client.login(user, password)
    client.send_message(msg)
PYMAIL
    return $?
  fi

  if command -v mail >/dev/null 2>&1; then
    printf '%s\n' "${body}" | mail -s "${subject}" "${to}"
    return $?
  fi

  if command -v mailx >/dev/null 2>&1; then
    printf '%s\n' "${body}" | mailx -s "${subject}" "${to}"
    return $?
  fi

  if command -v sendmail >/dev/null 2>&1; then
    {
      printf 'To: %s\n' "${to}"
      printf 'Subject: %s\n' "${subject}"
      printf '\n%s\n' "${body}"
    } | sendmail -t
    return $?
  fi

  log "email not sent: configure SMTP_HOST/SMTP_USER/SMTP_PASSWORD, set SMTP_CONFIG_FILE=${SMTP_CONFIG_FILE}, or install mail/mailx/sendmail"
  return 1
}

write_summary_header() {
  printf 'label,status,exit_code,start_time,end_time,duration_sec,tasks,agent_counts,total_updates,eval_episodes,record_eval_episodes,record_interval,cuda_visible_devices,log_path\n' > "${SUMMARY_CSV}"
}

append_summary() {
  local label="$1" status="$2" exit_code="$3" start_time="$4" end_time="$5" duration_sec="$6" tasks="$7" agent_counts="$8" total_updates="$9" eval_episodes="${10}" record_eval_episodes="${11}" record_interval="${12}" cuda_visible_devices="${13}" log_path="${14}"
  printf '"%s","%s","%s","%s","%s","%s","%s","%s","%s","%s","%s","%s","%s","%s"\n' \
    "${label}" "${status}" "${exit_code}" "${start_time}" "${end_time}" "${duration_sec}" "${tasks}" "${agent_counts}" "${total_updates}" "${eval_episodes}" "${record_eval_episodes}" "${record_interval}" "${cuda_visible_devices}" "${log_path}" >> "${SUMMARY_CSV}"
}

write_summary_header
log "queue_log_root=${QUEUE_LOG_ROOT}"
log "email_to=${EMAIL_TO}"
log "config=${CONFIG}"
log "env_config=${ENV_CONFIG}"
log "scaling_mode=${SCALING_MODE}"
log "obs_variant=${OBS_VARIANT}"

if [[ "${NOTIFY_ONLY}" == "1" ]]; then
  subject="[Wayffusion] PPO task queue notification test"
  body="This is a notification-only test from scripts/run_ppo_task_queue.sh.

repo_root=${REPO_ROOT}
queue_log_root=${QUEUE_LOG_ROOT}
email_to=${EMAIL_TO}
"
  if notify_email "${subject}" "${body}"; then
    log "notification test sent to ${EMAIL_TO}"
    exit 0
  fi
  log "notification test failed for ${EMAIL_TO}"
  exit 1
fi

queue_start_epoch="$(date +%s)"
queue_status="success"
failed_runs=0
completed_runs=0
skipped_runs=0

for row in "${QUEUE[@]}"; do
  IFS='|' read -r enabled label tasks agent_counts total_updates eval_episodes record_eval_episodes record_interval cuda_visible_devices extra_args <<< "${row}"

  enabled="${enabled:-0}"
  label="${label:-unnamed}"
  tasks="${tasks:-goal_nav coverage formation risk_nav}"
  agent_counts="${agent_counts:-${DEFAULT_AGENT_COUNTS}}"
  total_updates="${total_updates:-${DEFAULT_TOTAL_UPDATES}}"
  eval_episodes="${eval_episodes:-${DEFAULT_EVAL_EPISODES}}"
  record_eval_episodes="${record_eval_episodes:-${DEFAULT_RECORD_EVAL_EPISODES}}"
  record_interval="${record_interval:-${DEFAULT_RECORD_INTERVAL}}"
  cuda_visible_devices="${cuda_visible_devices:-${DEFAULT_CUDA_VISIBLE_DEVICES}}"
  extra_args="${extra_args:-}"

  if [[ "${enabled}" != "1" ]]; then
    skipped_runs=$((skipped_runs + 1))
    log "skip ${label}"
    continue
  fi

  run_start_epoch="$(date +%s)"
  run_start_time="$(date --iso-8601=seconds)"
  run_log="${QUEUE_LOG_ROOT}/${label}.log"
  log "start ${label}: tasks=${tasks}; agent_counts=${agent_counts}; total_updates=${total_updates}; cuda=${cuda_visible_devices}"

  cmd=(
    python scripts/train_ppo.py
    --config "${CONFIG}"
    --env-config "${ENV_CONFIG}"
    --tasks ${tasks}
    --agent_counts ${agent_counts}
    --scaling_mode "${SCALING_MODE}"
    --obs_variant "${OBS_VARIANT}"
    --total_updates "${total_updates}"
    --target_episodes "${TARGET_EPISODES}"
    --eval_episodes "${eval_episodes}"
    --record_eval_episodes "${record_eval_episodes}"
    --record_format "${RECORD_FORMAT}"
    --record_fps "${RECORD_FPS}"
    --record_interval "${record_interval}"
    --console_log_interval "${CONSOLE_LOG_INTERVAL}"
    --run_timestamp "${RUN_QUEUE_TIMESTAMP}"
    --run_name "${label}"
    --headless
    --tensorboard
  )

  if [[ -n "${extra_args}" ]]; then
    # Intentional shell splitting for the editable extra-args field.
    read -r -a extra_array <<< "${extra_args}"
    cmd+=("${extra_array[@]}")
  fi

  printf 'command=%s\n' "CUDA_VISIBLE_DEVICES=${cuda_visible_devices} $(shell_join "${cmd[@]}")" | tee "${run_log}"
  CUDA_VISIBLE_DEVICES="${cuda_visible_devices}" "${cmd[@]}" 2>&1 | tee -a "${run_log}"
  exit_code="${PIPESTATUS[0]}"

  run_end_epoch="$(date +%s)"
  run_end_time="$(date --iso-8601=seconds)"
  duration_sec=$((run_end_epoch - run_start_epoch))

  if [[ "${exit_code}" == "0" ]]; then
    completed_runs=$((completed_runs + 1))
    append_summary "${label}" "success" "${exit_code}" "${run_start_time}" "${run_end_time}" "${duration_sec}" "${tasks}" "${agent_counts}" "${total_updates}" "${eval_episodes}" "${record_eval_episodes}" "${record_interval}" "${cuda_visible_devices}" "${run_log}"
    log "finish ${label}: success duration_sec=${duration_sec}"
  else
    failed_runs=$((failed_runs + 1))
    queue_status="failed"
    append_summary "${label}" "failed" "${exit_code}" "${run_start_time}" "${run_end_time}" "${duration_sec}" "${tasks}" "${agent_counts}" "${total_updates}" "${eval_episodes}" "${record_eval_episodes}" "${record_interval}" "${cuda_visible_devices}" "${run_log}"
    log "finish ${label}: failed exit_code=${exit_code} duration_sec=${duration_sec}"
    if [[ "${CONTINUE_ON_FAILURE}" != "1" ]]; then
      log "stopping queue because CONTINUE_ON_FAILURE=${CONTINUE_ON_FAILURE}"
      break
    fi
  fi
 done

queue_end_epoch="$(date +%s)"
queue_duration_sec=$((queue_end_epoch - queue_start_epoch))
subject="[Wayffusion] PPO task queue ${queue_status}: ${completed_runs} completed, ${failed_runs} failed"
body="Wayffusion PPO task queue finished.

status=${queue_status}
completed_runs=${completed_runs}
failed_runs=${failed_runs}
skipped_runs=${skipped_runs}
duration_sec=${queue_duration_sec}
repo_root=${REPO_ROOT}
queue_log_root=${QUEUE_LOG_ROOT}
summary_csv=${SUMMARY_CSV}
config=${CONFIG}
env_config=${ENV_CONFIG}
"

log "queue_status=${queue_status} completed=${completed_runs} failed=${failed_runs} skipped=${skipped_runs} duration_sec=${queue_duration_sec}"
if notify_email "${subject}" "${body}"; then
  log "email notification sent to ${EMAIL_TO}"
else
  log "email notification failed or unavailable for ${EMAIL_TO}"
fi

if [[ "${queue_status}" == "success" ]]; then
  exit 0
fi
exit 1
