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

# Parallel long-run launcher for the four currently recommended single-task
# specialists. This intentionally does not run an all4 multitask policy.
RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)_best_specialists_all4}"
QUEUE_LOG_ROOT="${QUEUE_LOG_ROOT:-outputs/training/parallel_best_specialists/${RUN_TIMESTAMP}}"
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
EVAL_EPISODES="${EVAL_EPISODES:-50}"
RECORD_EVAL_EPISODES="${RECORD_EVAL_EPISODES:-0}"
RECORD_FORMAT="${RECORD_FORMAT:-gif}"
RECORD_FPS="${RECORD_FPS:-8}"
RECORD_INTERVAL="${RECORD_INTERVAL:-1}"
CONSOLE_LOG_INTERVAL="${CONSOLE_LOG_INTERVAL:-10}"
DEFAULT_ENV_BACKEND="${DEFAULT_ENV_BACKEND:-sync}"
DEFAULT_CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

GOAL_TOTAL_UPDATES="${GOAL_TOTAL_UPDATES:-160}"
COVERAGE_TOTAL_UPDATES="${COVERAGE_TOTAL_UPDATES:-200}"
RISK_TOTAL_UPDATES="${RISK_TOTAL_UPDATES:-240}"
FORMATION_TOTAL_UPDATES="${FORMATION_TOTAL_UPDATES:-160}"

GOAL_INIT_CHECKPOINT="${GOAL_INIT_CHECKPOINT:-outputs/training/bc_ppo/20260601_phase68_goalnav_task_targets_ppo/phase68_goalnav_task_targets_ppo/checkpoints/checkpoint_0040.pt}"
COVERAGE_INIT_CHECKPOINT="${COVERAGE_INIT_CHECKPOINT:-outputs/training/bc_ppo/20260601_phase60_coverage_route_target_agents_ppo/phase60_coverage_route_target_agents_ppo/checkpoints/checkpoint_best_eval.pt}"
RISK_INIT_CHECKPOINT="${RISK_INIT_CHECKPOINT:-outputs/training/bc_ppo/20260530_phase41_risknav_factorized_group_dagger_safe/phase41_risknav_factorized_group_dagger_safe/checkpoints/checkpoint_best_eval.pt}"
FORMATION_INIT_CHECKPOINT="${FORMATION_INIT_CHECKPOINT:-outputs/training/bc_ppo/20260530_phase39_formation_factorized_group_ppo/phase39_formation_factorized_group_ppo/checkpoints/checkpoint_best_eval.pt}"

GOAL_CONFIG="${GOAL_CONFIG:-configs/policy/debug_ppo_goal_nav_factorized_group_task_targets_safe.yaml}"
GOAL_ENV_CONFIG="${GOAL_ENV_CONFIG:-configs/env/debug_goal_nav_task_targets.yaml}"
COVERAGE_CONFIG="${COVERAGE_CONFIG:-configs/policy/debug_ppo_coverage_factorized_group_route_target_agents_safe.yaml}"
COVERAGE_ENV_CONFIG="${COVERAGE_ENV_CONFIG:-configs/env/debug_coverage_route_target_agents_canonical.yaml}"
RISK_CONFIG="${RISK_CONFIG:-configs/policy/debug_ppo_risk_nav_factorized_group_dagger_safe.yaml}"
RISK_ENV_CONFIG="${RISK_ENV_CONFIG:-configs/env/debug_risk_nav_safety_completion.yaml}"
FORMATION_CONFIG="${FORMATION_CONFIG:-configs/policy/debug_ppo_formation_factorized_group_ultra_strict.yaml}"
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
  printf 'label,status,exit_code,start_time,end_time,duration_sec,task,agent_counts,total_updates,eval_episodes,env_backend,cuda_visible_devices,config,env_config,init_checkpoint,run_dir,log_path\n' > "${SUMMARY_CSV}"
}

append_summary() {
  local label="$1" status="$2" exit_code="$3" start_time="$4" end_time="$5" duration_sec="$6" task="$7" agent_counts="$8" total_updates="$9" eval_episodes="${10}" env_backend="${11}" cuda_visible_devices="${12}" config="${13}" env_config="${14}" init_checkpoint="${15}" run_dir="${16}" log_path="${17}"
  printf '"%s","%s","%s","%s","%s","%s","%s","%s","%s","%s","%s","%s","%s","%s","%s","%s","%s"\n' \
    "${label}" "${status}" "${exit_code}" "${start_time}" "${end_time}" "${duration_sec}" "${task}" "${agent_counts}" "${total_updates}" "${eval_episodes}" "${env_backend}" "${cuda_visible_devices}" "${config}" "${env_config}" "${init_checkpoint}" "${run_dir}" "${log_path}" >> "${SUMMARY_CSV}"
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

# enabled|label|task|agent_counts|total_updates|eval_episodes|config|env_config|init_checkpoint|env_backend|envs_per_task|env_workers|cuda_visible_devices|extra_args
JOBS=(
  "1|goal_nav_task_targets_long|goal_nav|${AGENT_COUNTS}|${GOAL_TOTAL_UPDATES}|${GOAL_EVAL_EPISODES:-${EVAL_EPISODES}}|${GOAL_CONFIG}|${GOAL_ENV_CONFIG}|${GOAL_INIT_CHECKPOINT}|${GOAL_ENV_BACKEND:-${DEFAULT_ENV_BACKEND}}|${GOAL_ENVS_PER_TASK:-}|${GOAL_ENV_WORKERS:-}|${GOAL_CUDA_VISIBLE_DEVICES:-${DEFAULT_CUDA_VISIBLE_DEVICES}}|${GOAL_EXTRA_ARGS:-}"
  "1|coverage_route_targets_long|coverage|${AGENT_COUNTS}|${COVERAGE_TOTAL_UPDATES}|${COVERAGE_EVAL_EPISODES:-${EVAL_EPISODES}}|${COVERAGE_CONFIG}|${COVERAGE_ENV_CONFIG}|${COVERAGE_INIT_CHECKPOINT}|${COVERAGE_ENV_BACKEND:-${DEFAULT_ENV_BACKEND}}|${COVERAGE_ENVS_PER_TASK:-}|${COVERAGE_ENV_WORKERS:-}|${COVERAGE_CUDA_VISIBLE_DEVICES:-${DEFAULT_CUDA_VISIBLE_DEVICES}}|${COVERAGE_EXTRA_ARGS:-}"
  "1|risk_nav_dagger_safe_long|risk_nav|${AGENT_COUNTS}|${RISK_TOTAL_UPDATES}|${RISK_EVAL_EPISODES:-${EVAL_EPISODES}}|${RISK_CONFIG}|${RISK_ENV_CONFIG}|${RISK_INIT_CHECKPOINT}|${RISK_ENV_BACKEND:-${DEFAULT_ENV_BACKEND}}|${RISK_ENVS_PER_TASK:-}|${RISK_ENV_WORKERS:-}|${RISK_CUDA_VISIBLE_DEVICES:-${DEFAULT_CUDA_VISIBLE_DEVICES}}|${RISK_EXTRA_ARGS:-}"
  "1|formation_factorized_group_long|formation|${AGENT_COUNTS}|${FORMATION_TOTAL_UPDATES}|${FORMATION_EVAL_EPISODES:-${EVAL_EPISODES}}|${FORMATION_CONFIG}|${FORMATION_ENV_CONFIG}|${FORMATION_INIT_CHECKPOINT}|${FORMATION_ENV_BACKEND:-${DEFAULT_ENV_BACKEND}}|${FORMATION_ENVS_PER_TASK:-}|${FORMATION_ENV_WORKERS:-}|${FORMATION_CUDA_VISIBLE_DEVICES:-${DEFAULT_CUDA_VISIBLE_DEVICES}}|${FORMATION_EXTRA_ARGS:-}"
)

write_summary_header

if ! check_required_files \
  "${GOAL_CONFIG}" "${GOAL_ENV_CONFIG}" "${GOAL_INIT_CHECKPOINT}" \
  "${COVERAGE_CONFIG}" "${COVERAGE_ENV_CONFIG}" "${COVERAGE_INIT_CHECKPOINT}" \
  "${RISK_CONFIG}" "${RISK_ENV_CONFIG}" "${RISK_INIT_CHECKPOINT}" \
  "${FORMATION_CONFIG}" "${FORMATION_ENV_CONFIG}" "${FORMATION_INIT_CHECKPOINT}"; then
  exit 2
fi

if [[ "${NOTIFY_ONLY}" == "1" ]]; then
  if notify_email "[Wayffusion] best specialists notification test" "Wayffusion best-specialists notification test.

repo_root=${REPO_ROOT}
queue_log_root=${QUEUE_LOG_ROOT}
email_to=${EMAIL_TO}
"; then
    log "notification test sent to ${EMAIL_TO}"
    exit 0
  fi
  log "notification test failed for ${EMAIL_TO}"
  exit 1
fi

run_one_job() {
  local row="$1"
  IFS='|' read -r enabled label task agent_counts total_updates eval_episodes config env_config init_checkpoint env_backend envs_per_task env_workers cuda_visible_devices extra_args <<< "${row}"
  local run_log="${QUEUE_LOG_ROOT}/${label}.log"
  local status_tsv="${QUEUE_LOG_ROOT}/${label}.status.tsv"
  local run_dir="outputs/training/bc_ppo/${RUN_TIMESTAMP}/${label}"
  local start_epoch start_time end_epoch end_time duration_sec exit_code status

  start_epoch="$(date +%s)"
  start_time="$(date --iso-8601=seconds)"

  if [[ "${enabled}" != "1" ]]; then
    end_epoch="$(date +%s)"
    end_time="$(date --iso-8601=seconds)"
    duration_sec=$((end_epoch - start_epoch))
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' "skipped" "0" "${start_time}" "${end_time}" "${duration_sec}" "${run_dir}" > "${status_tsv}"
    return 0
  fi

  read -r -a agent_count_array <<< "${agent_counts}"
  cmd=(
    "${PYTHON_BIN}" scripts/train_ppo.py
    --config "${config}"
    --env-config "${env_config}"
    --tasks "${task}"
    --agent_counts "${agent_count_array[@]}"
    --scaling_mode "${SCALING_MODE}"
    --obs_variant "${OBS_VARIANT}"
    --init_checkpoint "${init_checkpoint}"
    --total_updates "${total_updates}"
    --target_episodes "${TARGET_EPISODES}"
    --eval_episodes "${eval_episodes}"
    --record_eval_episodes "${RECORD_EVAL_EPISODES}"
    --record_format "${RECORD_FORMAT}"
    --record_fps "${RECORD_FPS}"
    --record_interval "${RECORD_INTERVAL}"
    --console_log_interval "${CONSOLE_LOG_INTERVAL}"
    --env_backend "${env_backend}"
    --final_eval_source best
    --run_timestamp "${RUN_TIMESTAMP}"
    --run_name "${label}"
    --headless
    --tensorboard
  )

  if [[ -n "${envs_per_task}" ]]; then
    cmd+=(--envs_per_task "${envs_per_task}")
  fi
  if [[ -n "${env_workers}" ]]; then
    cmd+=(--env_workers "${env_workers}")
  fi
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
    printf 'init_checkpoint=%s\n' "${init_checkpoint}"
    printf 'total_updates=%s\n' "${total_updates}"
    printf 'eval_episodes=%s\n' "${eval_episodes}"
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
log "parallel best-specialist long training started"
log "repo_root=${REPO_ROOT}"
log "run_timestamp=${RUN_TIMESTAMP}"
log "queue_log_root=${QUEUE_LOG_ROOT}"
log "email_to=${EMAIL_TO}"
log "agent_counts=${AGENT_COUNTS}"
log "eval_episodes=${EVAL_EPISODES}"
log "record_eval_episodes=${RECORD_EVAL_EPISODES}"

declare -A pids
declare -A rows_by_label

for row in "${JOBS[@]}"; do
  IFS='|' read -r _enabled label task agent_counts total_updates eval_episodes config env_config init_checkpoint env_backend envs_per_task env_workers cuda_visible_devices extra_args <<< "${row}"
  rows_by_label["${label}"]="${row}"
  log "start ${label}: task=${task}; updates=${total_updates}; config=${config}; env=${env_config}; init=${init_checkpoint}; cuda=${cuda_visible_devices}"
  run_one_job "${row}" &
  pids["${label}"]="$!"
  log "pid ${pids[${label}]} ${label}"
done

queue_status="success"
completed_runs=0
failed_runs=0
skipped_runs=0

for row in "${JOBS[@]}"; do
  IFS='|' read -r enabled label task agent_counts total_updates eval_episodes config env_config init_checkpoint env_backend envs_per_task env_workers cuda_visible_devices extra_args <<< "${row}"
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
  append_summary "${label}" "${status}" "${exit_code}" "${start_time}" "${end_time}" "${duration_sec}" "${task}" "${agent_counts}" "${total_updates}" "${eval_episodes}" "${env_backend}" "${cuda_visible_devices}" "${config}" "${env_config}" "${init_checkpoint}" "${run_dir}" "${run_log}"
  log "finish ${label}: ${status} exit_code=${exit_code} duration_sec=${duration_sec}"
done

queue_duration_sec=$(( $(date +%s) - queue_start_epoch ))
summary_text="$(cat "${SUMMARY_CSV}")"
subject="[Wayffusion] best specialists ${queue_status}: ${completed_runs} completed, ${failed_runs} failed"
body="Wayffusion four-task best-specialist long training finished.

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
