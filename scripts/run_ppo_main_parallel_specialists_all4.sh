#!/usr/bin/env bash
set -u -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

export MPLBACKEND="${MPLBACKEND:-Agg}"
PYTHON_BIN="${PYTHON_BIN:-/opt/conda/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="python3"
fi

RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)_ppo_main_all4}"
QUEUE_LOG_ROOT="${QUEUE_LOG_ROOT:-outputs/training/parallel_ppo_main_specialists/${RUN_TIMESTAMP}}"
mkdir -p "${QUEUE_LOG_ROOT}"
QUEUE_LOG="${QUEUE_LOG_ROOT}/parallel.log"
SUMMARY_CSV="${QUEUE_LOG_ROOT}/summary.csv"

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

AGENT_COUNTS="${AGENT_COUNTS:-4}"
SCALING_MODE="${SCALING_MODE:-fixed_map}"
OBS_VARIANT="${OBS_VARIANT:-multi_channel_field+task_id}"
TARGET_EPISODES="${TARGET_EPISODES:-0}"
EVAL_EPISODES="${EVAL_EPISODES:-100}"
RECORD_EVAL_EPISODES="${RECORD_EVAL_EPISODES:-0}"
CONSOLE_LOG_INTERVAL="${CONSOLE_LOG_INTERVAL:-10}"
DEFAULT_ENV_BACKEND="${DEFAULT_ENV_BACKEND:-sync}"
DEFAULT_CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

GOAL_TOTAL_UPDATES="${GOAL_TOTAL_UPDATES:-3000}"
COVERAGE_TOTAL_UPDATES="${COVERAGE_TOTAL_UPDATES:-5000}"
RISK_TOTAL_UPDATES="${RISK_TOTAL_UPDATES:-5000}"
FORMATION_TOTAL_UPDATES="${FORMATION_TOTAL_UPDATES:-3000}"

GOAL_BC_CHECKPOINT="${GOAL_BC_CHECKPOINT:-outputs/training/bc/20260601_125256/debug_bc_goal_nav_factorized_group_task_targets_goal_nav_N4_multi_channel_field_plus_task_id/checkpoints/checkpoint_0024.pt}"
COVERAGE_BC_CHECKPOINT="${COVERAGE_BC_CHECKPOINT:-outputs/training/bc/20260601_034310/debug_bc_coverage_factorized_group_route_target_agents_perm_coverage_N4_multi_channel_field_plus_task_id/checkpoints/checkpoint_0032.pt}"
RISK_BC_CHECKPOINT="${RISK_BC_CHECKPOINT:-outputs/training/bc/20260530_121227/debug_bc_risk_nav_factorized_group_risk_nav_N4_multi_channel_field_plus_task_id/checkpoints/checkpoint_0040.pt}"
FORMATION_BC_CHECKPOINT="${FORMATION_BC_CHECKPOINT:-outputs/training/bc/20260530_095515/debug_bc_formation_factorized_group_formation_N4_multi_channel_field_plus_task_id/checkpoints/checkpoint_0040.pt}"

GOAL_CONFIG="${GOAL_CONFIG:-configs/policy/debug_ppo_goal_nav_factorized_group_task_targets_ppo_main.yaml}"
GOAL_ENV_CONFIG="${GOAL_ENV_CONFIG:-configs/env/debug_goal_nav_task_targets.yaml}"
COVERAGE_CONFIG="${COVERAGE_CONFIG:-configs/policy/debug_ppo_coverage_factorized_group_route_target_agents_ppo_main.yaml}"
COVERAGE_ENV_CONFIG="${COVERAGE_ENV_CONFIG:-configs/env/debug_coverage_route_target_agents_canonical.yaml}"
RISK_CONFIG="${RISK_CONFIG:-configs/policy/debug_ppo_risk_nav_factorized_group_dagger_ppo_main.yaml}"
RISK_ENV_CONFIG="${RISK_ENV_CONFIG:-configs/env/debug_risk_nav_safety_completion.yaml}"
FORMATION_CONFIG="${FORMATION_CONFIG:-configs/policy/debug_ppo_formation_factorized_group_ppo_main.yaml}"
FORMATION_ENV_CONFIG="${FORMATION_ENV_CONFIG:-configs/env/formation.yaml}"

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
      log "email not sent: SMTP_HOST is set but SMTP_USER or SMTP_PASSWORD is empty"
      return 1
    fi
    SMTP_HOST="${SMTP_HOST}" SMTP_PORT="${SMTP_PORT:-587}" SMTP_USER="${SMTP_USER:-}" SMTP_PASSWORD="${SMTP_PASSWORD:-}" SMTP_FROM="${SMTP_FROM:-}" SMTP_SSL="${SMTP_SSL:-0}" SMTP_STARTTLS="${SMTP_STARTTLS:-1}" EMAIL_TO="${to}" EMAIL_SUBJECT="${subject}" EMAIL_BODY="${body}" "${PYTHON_BIN}" - <<'PYMAIL'
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
  log "email not sent: configure SMTP credentials or install mail/mailx/sendmail"
  return 1
}

write_summary_header() {
  printf 'label,status,exit_code,start_time,end_time,duration_sec,task,total_updates,eval_episodes,config,env_config,bc_checkpoint,run_dir,log_path\n' > "${SUMMARY_CSV}"
}

append_summary() {
  local label="$1" status="$2" exit_code="$3" start_time="$4" end_time="$5" duration_sec="$6" task="$7" total_updates="$8" eval_episodes="$9" config="${10}" env_config="${11}" bc_checkpoint="${12}" run_dir="${13}" log_path="${14}"
  printf '"%s","%s","%s","%s","%s","%s","%s","%s","%s","%s","%s","%s","%s","%s"\n' \
    "${label}" "${status}" "${exit_code}" "${start_time}" "${end_time}" "${duration_sec}" "${task}" "${total_updates}" "${eval_episodes}" "${config}" "${env_config}" "${bc_checkpoint}" "${run_dir}" "${log_path}" >> "${SUMMARY_CSV}"
}

check_required_files() {
  local missing=0
  local path
  for path in "$@"; do
    if [[ ! -f "${path}" ]]; then
      log "missing required file: ${path}"
      missing=1
    fi
  done
  return "${missing}"
}

# enabled|label|task|total_updates|config|env_config|bc_checkpoint|env_backend|cuda_visible_devices|extra_args
JOBS=(
  "1|goal_nav_ppo_main_from_bc|goal_nav|${GOAL_TOTAL_UPDATES}|${GOAL_CONFIG}|${GOAL_ENV_CONFIG}|${GOAL_BC_CHECKPOINT}|${GOAL_ENV_BACKEND:-${DEFAULT_ENV_BACKEND}}|${GOAL_CUDA_VISIBLE_DEVICES:-${DEFAULT_CUDA_VISIBLE_DEVICES}}|${GOAL_EXTRA_ARGS:-}"
  "1|coverage_ppo_main_from_bc|coverage|${COVERAGE_TOTAL_UPDATES}|${COVERAGE_CONFIG}|${COVERAGE_ENV_CONFIG}|${COVERAGE_BC_CHECKPOINT}|${COVERAGE_ENV_BACKEND:-${DEFAULT_ENV_BACKEND}}|${COVERAGE_CUDA_VISIBLE_DEVICES:-${DEFAULT_CUDA_VISIBLE_DEVICES}}|${COVERAGE_EXTRA_ARGS:-}"
  "1|risk_nav_ppo_main_from_bc|risk_nav|${RISK_TOTAL_UPDATES}|${RISK_CONFIG}|${RISK_ENV_CONFIG}|${RISK_BC_CHECKPOINT}|${RISK_ENV_BACKEND:-${DEFAULT_ENV_BACKEND}}|${RISK_CUDA_VISIBLE_DEVICES:-${DEFAULT_CUDA_VISIBLE_DEVICES}}|${RISK_EXTRA_ARGS:-}"
  "1|formation_ppo_main_from_bc|formation|${FORMATION_TOTAL_UPDATES}|${FORMATION_CONFIG}|${FORMATION_ENV_CONFIG}|${FORMATION_BC_CHECKPOINT}|${FORMATION_ENV_BACKEND:-${DEFAULT_ENV_BACKEND}}|${FORMATION_CUDA_VISIBLE_DEVICES:-${DEFAULT_CUDA_VISIBLE_DEVICES}}|${FORMATION_EXTRA_ARGS:-}"
)

write_summary_header

if ! check_required_files \
  "${GOAL_CONFIG}" "${GOAL_ENV_CONFIG}" "${GOAL_BC_CHECKPOINT}" \
  "${COVERAGE_CONFIG}" "${COVERAGE_ENV_CONFIG}" "${COVERAGE_BC_CHECKPOINT}" \
  "${RISK_CONFIG}" "${RISK_ENV_CONFIG}" "${RISK_BC_CHECKPOINT}" \
  "${FORMATION_CONFIG}" "${FORMATION_ENV_CONFIG}" "${FORMATION_BC_CHECKPOINT}"; then
  exit 2
fi

if [[ "${NOTIFY_ONLY}" == "1" ]]; then
  if notify_email "[Wayffusion] PPO-main all4 notification test" "Wayffusion PPO-main all4 notification test.

repo_root=${REPO_ROOT}
queue_log_root=${QUEUE_LOG_ROOT}
run_timestamp=${RUN_TIMESTAMP}
"; then
    log "notification test sent to ${EMAIL_TO}"
    exit 0
  fi
  log "notification test failed for ${EMAIL_TO}"
  exit 1
fi

run_one_job() {
  local row="$1"
  IFS='|' read -r enabled label task total_updates config env_config bc_checkpoint env_backend cuda_visible_devices extra_args <<< "${row}"
  local run_log="${QUEUE_LOG_ROOT}/${label}.log"
  local status_tsv="${QUEUE_LOG_ROOT}/${label}.status.tsv"
  local run_dir="outputs/training/bc_ppo/${RUN_TIMESTAMP}/${label}"
  local start_epoch start_time end_epoch end_time duration_sec exit_code status

  start_epoch="$(date +%s)"
  start_time="$(date --iso-8601=seconds)"

  if [[ "${enabled}" != "1" ]]; then
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' "skipped" "0" "${start_time}" "$(date --iso-8601=seconds)" "0" "${run_dir}" > "${status_tsv}"
    return 0
  fi

  cmd=(
    "${PYTHON_BIN}" scripts/train_ppo.py
    --config "${config}"
    --env-config "${env_config}"
    --tasks "${task}"
    --agent_counts "${AGENT_COUNTS}"
    --scaling_mode "${SCALING_MODE}"
    --obs_variant "${OBS_VARIANT}"
    --init_checkpoint "${bc_checkpoint}"
    --total_updates "${total_updates}"
    --target_episodes "${TARGET_EPISODES}"
    --eval_episodes "${EVAL_EPISODES}"
    --record_eval_episodes "${RECORD_EVAL_EPISODES}"
    --console_log_interval "${CONSOLE_LOG_INTERVAL}"
    --env_backend "${env_backend}"
    --final_eval_source best
    --run_timestamp "${RUN_TIMESTAMP}"
    --run_name "${label}"
    --headless
    --tensorboard
  )
  if [[ -n "${extra_args}" ]]; then
    read -r -a extra_array <<< "${extra_args}"
    cmd+=("${extra_array[@]}")
  fi

  {
    printf 'label=%s\n' "${label}"
    printf 'task=%s\n' "${task}"
    printf 'run_timestamp=%s\n' "${RUN_TIMESTAMP}"
    printf 'run_dir=%s\n' "${run_dir}"
    printf 'config=%s\n' "${config}"
    printf 'env_config=%s\n' "${env_config}"
    printf 'bc_checkpoint=%s\n' "${bc_checkpoint}"
    printf 'total_updates=%s\n' "${total_updates}"
    printf 'eval_episodes=%s\n' "${EVAL_EPISODES}"
    printf 'cuda_visible_devices=%s\n' "${cuda_visible_devices}"
    printf 'command=%s\n' "CUDA_VISIBLE_DEVICES=${cuda_visible_devices} $(shell_join "${cmd[@]}")"
  } > "${run_log}"

  set +e
  CUDA_VISIBLE_DEVICES="${cuda_visible_devices}" "${cmd[@]}" 2>&1 | tee -a "${run_log}"
  exit_code="${PIPESTATUS[0]}"
  set -u

  end_epoch="$(date +%s)"
  end_time="$(date --iso-8601=seconds)"
  duration_sec=$((end_epoch - start_epoch))
  if [[ "${exit_code}" == "0" ]]; then
    status="success"
  else
    status="failed"
  fi
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' "${status}" "${exit_code}" "${start_time}" "${end_time}" "${duration_sec}" "${run_dir}" > "${status_tsv}"
  return "${exit_code}"
}

queue_start_epoch="$(date +%s)"
log "parallel PPO-main all4 specialists started"
log "repo_root=${REPO_ROOT}"
log "run_timestamp=${RUN_TIMESTAMP}"
log "queue_log_root=${QUEUE_LOG_ROOT}"
log "email_to=${EMAIL_TO}"
log "eval_episodes=${EVAL_EPISODES}"
log "note=all jobs start from BC checkpoints, not PPO best checkpoints"

declare -A pids
for row in "${JOBS[@]}"; do
  IFS='|' read -r _enabled label task total_updates config env_config bc_checkpoint env_backend cuda_visible_devices extra_args <<< "${row}"
  log "start ${label}: task=${task}; updates=${total_updates}; config=${config}; env=${env_config}; bc=${bc_checkpoint}; cuda=${cuda_visible_devices}"
  run_one_job "${row}" &
  pids["${label}"]="$!"
  log "pid ${pids[${label}]} ${label}"
done

queue_status="success"
completed_runs=0
failed_runs=0
skipped_runs=0

for row in "${JOBS[@]}"; do
  IFS='|' read -r enabled label task total_updates config env_config bc_checkpoint env_backend cuda_visible_devices extra_args <<< "${row}"
  pid="${pids[${label}]}"
  if wait "${pid}"; then
    :
  else
    :
  fi
  status_tsv="${QUEUE_LOG_ROOT}/${label}.status.tsv"
  if [[ -f "${status_tsv}" ]]; then
    IFS=$'\t' read -r status exit_code start_time end_time duration_sec run_dir < "${status_tsv}"
  else
    status="failed"
    exit_code="missing_status"
    start_time=""
    end_time="$(date --iso-8601=seconds)"
    duration_sec="0"
    run_dir="outputs/training/bc_ppo/${RUN_TIMESTAMP}/${label}"
  fi
  case "${status}" in
    success) completed_runs=$((completed_runs + 1)) ;;
    skipped) skipped_runs=$((skipped_runs + 1)) ;;
    *)
      failed_runs=$((failed_runs + 1))
      queue_status="failed"
      ;;
  esac
  run_log="${QUEUE_LOG_ROOT}/${label}.log"
  append_summary "${label}" "${status}" "${exit_code}" "${start_time}" "${end_time}" "${duration_sec}" "${task}" "${total_updates}" "${EVAL_EPISODES}" "${config}" "${env_config}" "${bc_checkpoint}" "${run_dir}" "${run_log}"
  log "finish ${label}: ${status} exit_code=${exit_code} duration_sec=${duration_sec}"
done

queue_duration_sec=$(( $(date +%s) - queue_start_epoch ))
summary_text="$(cat "${SUMMARY_CSV}")"
subject="[Wayffusion] PPO-main all4 ${queue_status}: ${completed_runs} completed, ${failed_runs} failed"
body="Wayffusion PPO-main all4 specialists finished.

status=${queue_status}
completed_runs=${completed_runs}
failed_runs=${failed_runs}
skipped_runs=${skipped_runs}
duration_sec=${queue_duration_sec}
repo_root=${REPO_ROOT}
run_timestamp=${RUN_TIMESTAMP}
queue_log_root=${QUEUE_LOG_ROOT}
summary_csv=${SUMMARY_CSV}

Summary:
${summary_text}
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
