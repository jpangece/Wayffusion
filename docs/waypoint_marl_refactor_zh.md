# Waypoint-Level Multi-UAV MARL 重构说明

本轮重构把 Wayffusion 主线从 centralized Gymnasium-style swarm-as-one-agent benchmark，切换为航点级多无人机 MARL mission suite。

## 核心变化

- 新主环境是 `envs/waypoint_marl_env.py::WaypointMultiUAVEnv`。
- API 遵循 PettingZoo ParallelEnv 语义：每架 UAV 是一个 agent。
- 新动作不是连续 `[N, 2]` joint delta，而是每架 UAV 选择一个候选航点 index。
- 候选航点本身仍是连续坐标，便于后续对接真机航点执行。
- 执行模型不是瞬移：`WaypointExecutionModel` 按 `max_speed * dt_decision` 限速移动，并检查 geofence、no-fly zone、最小机距。
- centralized critic 仍可通过 `env.get_global_state()` 获取全局状态，符合 CTDE/MAPPO。
- 更完整的环境说明见：`docs/waypoint_environment_guide_zh.md`。

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
python -m pytest tests/test_candidate_selection_policy.py
python -m pytest tests/test_direct_waypoint_policy.py
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

## 2026-06-06 final waypoint action refactor

用户指出上一版 waypoint MARL 仍然把候选航点生成和选择写进环境，导致环境和策略强耦合。本轮将主动作接口从 candidate index 改成 final waypoint。

新的主线契约：

- `WaypointMultiUAVEnv.step(actions)` 默认接收 `dict[str, np.ndarray]`。
- 每个 action 是该 UAV 的最终 waypoint，默认 shape `[2]`。
- `action_space(agent_id)` 是 `Box([0,0], [map_size,map_size], shape=(2,))`。
- 环境不再在默认 observation/global state 中暴露 `candidate_waypoints` 或 `candidate_mask`。
- `action_mode=candidate_index_legacy` 仅保留为 debug/legacy 分支，不是主线。

候选航点逻辑移动：

- 旧 candidate 工具迁到 `planners/waypoint_candidates.py`。
- `envs/waypoint/candidates.py` 已删除。
- scenario 中的 `generate_candidate_waypoints` 只作为可选 planner helper 保留，环境主流程不调用。

新增 policy 抽象：

- `policies/policy_output.py::PolicyOutput`
  - `env_action [B,N,D]`：最终传给环境的 waypoint。
  - `train_action`：policy 内部采样动作。
  - `logprob [B,N]`
  - `entropy [B,N]`
  - `value [B]`
  - `aux`

新增/重构 policy：

- `policies/candidate_selection_policy.py::CandidateSelectionWaypointPolicy`
  - policy 内部生成 candidates、features、mask；
  - categorical actor 采样 index；
  - index 映射为 final waypoint 后传给环境；
  - `train_action` 是 `[B,N]` index。
- `policies/direct_waypoint_policy.py::DirectWaypointPolicy`
  - actor 直接输出 Gaussian delta；
  - delta 按向量范数裁剪到 `max_delta`；
  - 转成 final waypoint；
  - `train_action` 是 `[B,N,2]` delta。
- `policies/mappo_waypoint_policy.py::MAPPOWaypointPolicy` 仅作为 candidate-selection alias 保留，避免旧 import 断裂。

MAPPO trainer 变化：

- rollout 存 `env_actions [T,B,N,D]` 和 `train_actions`。
- `env_batch.step(...)` 只接收 `env_action`。
- update 时调用 `policy.get_action_and_value(obs, train_action)` 重算 logprob。
- trainer 不再假设 action 是 int，也不关心 policy 是 categorical 还是 Gaussian。
- GAE 使用 `bootstrap_mask`，true termination 不 bootstrap，time-limit truncation bootstrap。

环境/安全修复：

- `SafetyLayer.filter_waypoints(...)` 新增为 final waypoint 主安全入口。
- `MissionWorld.reset_common(...)` 改为采样满足最小机距的初始 UAV 位置，避免 episode 初始状态天然非法。
- random/greedy validation baseline 改为 waypoint planner，并顺序避免多个 UAV 选择过近 waypoint。

配置变化：

- `configs/env/waypoint_missions.yaml`
  - 新增 `action_mode: waypoint`
  - 新增 `waypoint_dim: 2`
  - 新增 `execution_model: kinematic_point`
  - 删除 env 主配置中的 `candidate_count`
- `configs/policy/mappo_waypoint*.yaml`
  - `policy_class: candidate_selection_waypoint`
  - `candidate_count` 移到 policy config
- 新增 `configs/policy/direct_waypoint_debug.yaml`

验证结果：

- `py_compile` waypoint 主线通过。
- `pytest` 目标测试通过：`15 passed`。
- candidate-selection validate 通过：
  - command: `/opt/conda/bin/python scripts/check/validate_waypoint_mappo.py --tasks area_coverage belief_search connectivity_expansion --policy-class candidate_selection_waypoint --num_agents 3 --total_updates 5 --eval_episodes 2 --record_eval_episodes 1 --record_format gif --headless --run_timestamp 20260606_waypoint_action_candidate_validate_v2`
  - output: `outputs/debug/waypoint_mappo_validation/20260606_waypoint_action_candidate_validate_v2/`
  - `passed=true`
  - `action_validity_rate=0.9909722222222221`
  - `candidate_mask_empty_count=0.0`
  - `env_action_shape=[2,3,2]`
  - GIF 已生成。
- direct waypoint validate 通过：
  - command: `/opt/conda/bin/python scripts/check/validate_waypoint_mappo.py --tasks area_coverage belief_search --policy-class direct_waypoint --num_agents 3 --total_updates 2 --eval_episodes 1 --record_eval_episodes 1 --record_format gif --headless --run_timestamp 20260606_waypoint_action_direct_validate_v2`
  - output: `outputs/debug/waypoint_mappo_validation/20260606_waypoint_action_direct_validate_v2/`
  - `passed=true`
  - `action_validity_rate=0.9916666666666667`
  - `candidate_mask_empty_count=0.0`
  - `env_action_shape=[2,3,2]`
  - GIF 已生成。

这些结果证明 final waypoint action API、candidate-selection policy、direct waypoint policy、MAPPO rollout/update/eval/GIF 链路均可运行。它们不是长训练性能结论。

## 2026-06-06 MPE-like dynamics backend refactor

本轮目标是在不改变 final waypoint 外部动作接口的前提下，把旧的几何 waypoint execution 扩展为可配置 transition dynamics。

新增结构：

- `envs/waypoint/control/action_adapters.py`
  - `AdapterOutput`
  - `WaypointVelocityTracker`
  - `WaypointPDTracker`
  - `VelocityToForceAdapter`
  - `DirectAccelerationAdapter`
  - `build_action_adapter`
- `envs/waypoint/control/dynamics_backends.py`
  - `DynamicsInfo`
  - `MPEParticleBackend`
  - `KinematicPointBackend`
  - `build_dynamics_backend`
- `envs/waypoint/control/__init__.py`
- `tests/test_action_adapters.py`
- `tests/test_mpe_particle_backend.py`
- `docs/waypoint_mpe_dynamics_refactor.md`
- `configs/env/waypoint_missions_kinematic_debug.yaml`
- `configs/env/waypoint_missions_pd_adapter_debug.yaml`

默认 transition stack：

```text
final waypoint action
  -> SafetyLayer.validate_waypoints
  -> WaypointVelocityTracker
  -> MPEParticleBackend
  -> scenario.compute_rewards
```

默认配置：

```yaml
action_interface: waypoint
action_adapter:
  name: waypoint_velocity_tracker
dynamics_backend:
  name: mpe_particle
```

关键实现：

- `SafetyLayer` 新增 `SafetyResult` 和 `validate_waypoints(...)`。
- `WaypointMultiUAVEnv.step(...)` 现在合并 safety、adapter、dynamics 信息后传给 scenario reward。
- `UAVState` 增加 `mass`、`radius`、`last_control`、`last_desired_velocity`。
- `MissionWorld.from_config(...)` 读取 nested `dynamics_backend` 中的 `physics_dt/substeps/mass/damping/radius/max_speed`。
- 初始 UAV 布置改为按 particle radius 更保守地采样，避免 MPE contact radius 下 episode 起点天然过近。
- `CandidateSelectionWaypointPolicy.act_deterministic(...)` 增加 deterministic eval 去冲突，避免多个 UAV 在 eval 中选择过近 waypoint。
- `MAPPOWaypointTrainer` rollout/eval 记录 dynamics metrics：
  - `mean_speed`
  - `max_speed_observed`
  - `mean_control_norm`
  - `collision_count`
  - `no_fly_violation_count`
  - `geofence_violation_count`

验证结果：

- `py_compile` waypoint 主线通过。
- `git diff --check` 通过。
- pytest：
  - command: `/opt/conda/bin/python -m pytest -q tests/test_waypoint_env_api.py tests/test_action_adapters.py tests/test_mpe_particle_backend.py tests/test_candidate_selection_policy.py tests/test_direct_waypoint_policy.py tests/test_mappo_waypoint_trainer.py tests/test_waypoint_rewards.py tests/test_waypoint_rendering.py`
  - result: `18 passed`
- candidate-selection + MPE validate：
  - output: `outputs/debug/waypoint_mappo_validation/20260606_mpe_candidate_validate_final_v2/`
  - `passed=true`
  - `action_validity_rate=1.0`
  - `max_speed_observed=0.039999742060899734`
  - `adapter_finite=true`
  - `dynamics_finite=true`
  - GIFs generated.
- direct waypoint + MPE validate:
  - output: `outputs/debug/waypoint_mappo_validation/20260606_mpe_direct_validate_final/`
  - `passed=true`
  - `action_validity_rate=1.0`
  - `max_speed_observed=0.039999742060899734`
  - `adapter_finite=true`
  - `dynamics_finite=true`
  - GIFs generated.
- direct waypoint + kinematic debug validate:
  - output: `outputs/debug/waypoint_mappo_validation/20260606_kinematic_direct_validate_final/`
  - `passed=true`
  - `action_validity_rate=0.9916666666666667`
  - `max_speed_observed=0.07999999821186066`
  - GIFs generated.

这些结果证明 ActionAdapter + DynamicsBackend 集成、MPE-like particle 默认后端、kinematic debug 后端、candidate/direct policy、MAPPO rollout/update/eval/GIF 链路均可运行。它们不是长训练性能结论。
