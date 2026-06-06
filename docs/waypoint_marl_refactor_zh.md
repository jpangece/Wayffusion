# Waypoint-Level Multi-UAV MARL 重构说明

本轮重构把 Wayffusion 主线从 centralized Gymnasium-style swarm-as-one-agent benchmark，切换为航点级多无人机 MARL mission suite。

## 核心变化

- 新主环境是 `envs/waypoint_marl_env.py::WaypointMultiUAVEnv`。
- API 遵循 PettingZoo ParallelEnv 语义：每架 UAV 是一个 agent。
- 新动作不是连续 `[N, 2]` joint delta，而是每架 UAV 选择一个候选航点 index。
- 候选航点本身仍是连续坐标，便于后续对接真机航点执行。
- 执行模型不是瞬移：`WaypointExecutionModel` 按 `max_speed * dt_decision` 限速移动，并检查 geofence、no-fly zone、最小机距。
- centralized critic 仍可通过 `env.get_global_state()` 获取全局状态，符合 CTDE/MAPPO。

## 新任务体系

新增任务都在 `tasks/waypoint/`：

- `area_coverage`：均匀区域覆盖。
- `belief_search`：概率图 / belief map 加权搜索。
- `priority_inspection`：带权 POI 巡检。
- `connectivity_expansion`：保持多跳连通的协同扩展。
- `dynamic_target_escort`：动态目标护航 / 持续监测。
- `target_interception`：动态目标 route-branch interception / containment。

旧 `goal_nav / coverage / risk_nav / formation` 不再是新主线任务族。旧 centralized 代码已经从当前工作树删除；如果需要复现实验，请从 git 历史回溯。

## MAPPO waypoint 训练

新 policy：`policies/mappo_waypoint_policy.py::MAPPOWaypointPolicy`

- CNN 编码 task field。
- 共享 agent encoder 编码 UAV state。
- 可选 self-attention 编码 UAV-UAV interaction。
- candidate encoder 编码每个候选航点与 candidate features。
- actor 输出 masked categorical logits `[B, N, K]`。
- `get_action_and_value` 返回：
  - action `[B, N]`
  - logprob `[B, N]`
  - entropy `[B, N]`
  - value `[B]`

新 trainer：`algorithms/mappo_waypoint.py::MAPPOWaypointTrainer`

- rollout buffer 存 agent-wise old logprob `[T, B, N]`。
- team reward 用于第一版 GAE，advantage broadcast 到 agent 维度。
- actor loss 使用 per-agent PPO ratio。
- critic loss 使用 centralized state value。
- 显式区分 `terminated` 和 `truncated`：
  - true termination 不 bootstrap；
  - time-limit truncation bootstrap terminal global state；
  - auto reset 后不会把下一 episode 的 advantage 串进上一 episode。
- 记录 CSV、TensorBoard、checkpoint、best eval summary 和 GIF/MP4 eval media。

## 关键文件

- `envs/waypoint/entities.py`
- `envs/waypoint/world.py`
- `envs/waypoint/execution.py`
- `envs/waypoint/candidates.py`
- `envs/waypoint/safety.py`
- `envs/waypoint/rendering.py`
- `envs/waypoint_marl_env.py`
- `tasks/waypoint/*.py`
- `utils/waypoint_vector_env.py`
- `utils/waypoint_evaluation.py`
- `policies/mappo_waypoint_policy.py`
- `algorithms/mappo_waypoint.py`
- `scripts/train_mappo_waypoint.py`
- `scripts/check/validate_waypoint_mappo.py`
- `configs/env/waypoint_missions.yaml`
- `configs/policy/mappo_waypoint.yaml`
- `configs/policy/mappo_waypoint_debug.yaml`

## 验证命令

```bash
python -m pytest tests/test_waypoint_env_api.py
python -m pytest tests/test_waypoint_scenarios.py
python -m pytest tests/test_waypoint_rewards.py
python -m pytest tests/test_waypoint_mappo_policy.py
python -m pytest tests/test_mappo_waypoint_trainer.py
python -m pytest tests/test_waypoint_rendering.py
```

```bash
python scripts/check/validate_waypoint_mappo.py \
  --tasks area_coverage belief_search connectivity_expansion \
  --num_agents 3 \
  --total_updates 5 \
  --eval_episodes 2 \
  --record_eval_episodes 1 \
  --record_format gif \
  --headless
```

```bash
python scripts/train_mappo_waypoint.py \
  --config configs/policy/mappo_waypoint_debug.yaml \
  --env-config configs/env/waypoint_missions.yaml \
  --tasks area_coverage belief_search priority_inspection connectivity_expansion \
  --num_agents 4 \
  --num_envs 4 \
  --total_updates 10 \
  --eval_episodes 2 \
  --record_eval_episodes 1 \
  --record_format gif \
  --headless
```

## 本轮 smoke 结果

执行日期：2026-06-05

- `pytest` 六个 waypoint 测试命令全部通过。
- `validate_waypoint_mappo.py` 通过：
  - output: `outputs/debug/waypoint_mappo_validation/20260605_refactor_smoke_evalcsv/`
  - `passed=true`
  - `action_validity_rate=1.0`
  - `candidate_mask_empty_count=0`
  - 已生成 random、greedy、MAPPO eval GIF。
  - `mappo_train/eval_metrics.csv` 已写入并包含 `recording_path`。
- `train_mappo_waypoint.py` debug run 通过：
  - output: `outputs/training/mappo_waypoint/20260605_refactor_smoke_evalcsv/waypoint_debug_train_4tasks/`
  - 10 updates 完成。
  - 最终日志：`eval_success_rate=0.250`, `eval_reward=-278.301`。
  - `eval_metrics.csv` 已写入并包含 `recording_path`。

这些结果证明新环境、候选航点 mask、MAPPO rollout/update、eval、GIF 和 checkpoint 链路可以闭环。它们不是长训练性能结论。

## 2026-06-06 清理

用户确认旧实现已经在 git 中可回溯，因此当前工作树删除了旧 centralized 主线资产，避免后续开发被旧 Gymnasium single-agent swarm benchmark 干扰。

删除范围包括：

- 旧 centralized env：`envs/centralized_env.py`、旧 dynamics/collision/reward/metrics helper。
- 旧四任务：`tasks/goal_nav.py`、`tasks/coverage.py`、`tasks/risk_nav.py`、`tasks/formation.py` 等。
- 旧连续动作 policy：MLP、CNNDeepSets、attention、factorized_group、旧 mappo_shared、SquashedNormal helper。
- 旧算法：PPO、SAC、TD3、BC、旧 minimal MAPPO、MTRL-CG。
- 旧 scripts：PPO/SAC/TD3/BC train/eval/debug/queue/tmux/helper scripts。
- 旧 configs：centralized env/policy/eval/example configs。
- 旧 tests：centralized env/policy/reward/training/config tests。
- 旧 `fields/` 和 `baselines/`。
- 旧 centralized docs，只保留当前 waypoint 重构说明。

保留后的主线结构：

```text
envs/waypoint*/
tasks/waypoint/
policies/mappo_waypoint_policy.py
algorithms/mappo_waypoint.py
utils/waypoint_*.py
scripts/train_mappo_waypoint.py
scripts/check/validate_waypoint_mappo.py
configs/env/waypoint_missions.yaml
configs/policy/mappo_waypoint*.yaml
configs/eval/waypoint_eval.yaml
tests/test_waypoint_*.py
tests/test_mappo_waypoint_trainer.py
```

清理后验证：

- `py_compile` waypoint 主线通过。
- `pytest` waypoint 测试全集通过：`12 passed`。
- `validate_waypoint_mappo.py` cleanup smoke 通过：
  - output: `outputs/debug/waypoint_mappo_validation/20260606_cleanup_smoke/`
  - `passed=true`
  - `action_validity_rate=1.0`
  - `candidate_mask_empty_count=0.0`
- `train_mappo_waypoint.py` cleanup smoke 通过：
  - output: `outputs/training/mappo_waypoint/20260606_cleanup_smoke/waypoint_cleanup_train_smoke/`
  - 写入 `training_metrics.csv`、`eval_metrics.csv`、`checkpoint_0002.pt`、`checkpoint_best_eval.pt` 和 eval GIF。
