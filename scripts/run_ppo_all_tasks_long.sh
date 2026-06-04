#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

export MPLBACKEND="${MPLBACKEND:-Agg}"

# Edit this block for normal use.
# Shell environment variables with the same names still override these defaults,
# for example:
#   CUDA_VISIBLE_DEVICES=1 TOTAL_UPDATES=3000 bash scripts/run_ppo_all_tasks_long.sh
DEFAULT_CUDA_VISIBLE_DEVICES="0"
DEFAULT_CONFIG="configs/policy/ppo_cnn_deepsets_multitask_20k.yaml"
DEFAULT_ENV_CONFIG="configs/env/multitask.yaml"
DEFAULT_TASKS="goal_nav coverage formation risk_nav"
DEFAULT_AGENT_COUNTS="4"
DEFAULT_SCALING_MODE="fixed_map"
DEFAULT_OBS_VARIANT="multi_channel_field+task_id"
DEFAULT_TOTAL_UPDATES="20000"
DEFAULT_TARGET_EPISODES="0"
DEFAULT_EVAL_EPISODES="5"
DEFAULT_RECORD_EVAL_EPISODES="1"
DEFAULT_RECORD_FORMAT="gif"
DEFAULT_RECORD_FPS="8"
DEFAULT_RECORD_INTERVAL="4"
DEFAULT_CONSOLE_LOG_INTERVAL="5"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${DEFAULT_CUDA_VISIBLE_DEVICES}}"

CONFIG="${CONFIG:-${DEFAULT_CONFIG}}"
ENV_CONFIG="${ENV_CONFIG:-${DEFAULT_ENV_CONFIG}}"
TASKS="${TASKS:-${DEFAULT_TASKS}}"
AGENT_COUNTS="${AGENT_COUNTS:-${DEFAULT_AGENT_COUNTS}}"
SCALING_MODE="${SCALING_MODE:-${DEFAULT_SCALING_MODE}}"
OBS_VARIANT="${OBS_VARIANT:-${DEFAULT_OBS_VARIANT}}"
TOTAL_UPDATES="${TOTAL_UPDATES:-${DEFAULT_TOTAL_UPDATES}}"
TARGET_EPISODES="${TARGET_EPISODES:-${DEFAULT_TARGET_EPISODES}}"
EVAL_EPISODES="${EVAL_EPISODES:-${DEFAULT_EVAL_EPISODES}}"
RECORD_EVAL_EPISODES="${RECORD_EVAL_EPISODES:-${DEFAULT_RECORD_EVAL_EPISODES}}"
RECORD_FORMAT="${RECORD_FORMAT:-${DEFAULT_RECORD_FORMAT}}"
RECORD_FPS="${RECORD_FPS:-${DEFAULT_RECORD_FPS}}"
RECORD_INTERVAL="${RECORD_INTERVAL:-${DEFAULT_RECORD_INTERVAL}}"
CONSOLE_LOG_INTERVAL="${CONSOLE_LOG_INTERVAL:-${DEFAULT_CONSOLE_LOG_INTERVAL}}"

echo "repo_root=${REPO_ROOT}"
echo "config=${CONFIG}"
echo "env_config=${ENV_CONFIG}"
echo "tasks=${TASKS}"
echo "agent_counts=${AGENT_COUNTS}"
echo "scaling_mode=${SCALING_MODE}"
echo "obs_variant=${OBS_VARIANT}"
echo "total_updates=${TOTAL_UPDATES}"
echo "target_episodes=${TARGET_EPISODES}"
echo "eval_episodes=${EVAL_EPISODES}"
echo "record_eval_episodes=${RECORD_EVAL_EPISODES}"
echo "record_interval=${RECORD_INTERVAL}"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES}"

python - <<'PYTORCH_CHECK'
import torch

print(f"torch={torch.__version__}")
print(f"cuda_available={torch.cuda.is_available()}")
print(f"torch_cuda={torch.version.cuda}")
if torch.cuda.is_available():
    print(f"gpu_0={torch.cuda.get_device_name(0)}")
PYTORCH_CHECK

python scripts/train_ppo.py \
  --config "${CONFIG}" \
  --env-config "${ENV_CONFIG}" \
  --tasks ${TASKS} \
  --agent_counts ${AGENT_COUNTS} \
  --scaling_mode "${SCALING_MODE}" \
  --obs_variant "${OBS_VARIANT}" \
  --total_updates "${TOTAL_UPDATES}" \
  --target_episodes "${TARGET_EPISODES}" \
  --eval_episodes "${EVAL_EPISODES}" \
  --record_eval_episodes "${RECORD_EVAL_EPISODES}" \
  --record_format "${RECORD_FORMAT}" \
  --record_fps "${RECORD_FPS}" \
  --record_interval "${RECORD_INTERVAL}" \
  --console_log_interval "${CONSOLE_LOG_INTERVAL}" \
  --headless \
  --tensorboard
