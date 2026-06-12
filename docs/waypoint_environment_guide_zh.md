# WaypointMultiUAVEnv 环境说明

当前 Wayffusion 主线是航点级多无人机 MARL 环境，不再是旧 centralized Gymnasium single-agent swarm benchmark。

## 核心定位

- 每架 UAV 是一个独立 agent。
- 环境主动作接口是每架 UAV 的最终 waypoint 坐标，不是 candidate waypoint index。
- 环境职责是接收 waypoint、安全过滤、执行航点、更新任务状态、计算 reward/metric、渲染和保存 eval 媒体。
- 候选航点生成、候选航点选择、planner + RL correction 都属于 policy/planner/action-adapter 侧。
- 当前仍保留 `candidate_index_legacy` 作为 debug action mode，但默认和主线都是 `action_mode: waypoint`。

## 主入口

- 环境：`envs/waypoint_marl_env.py::WaypointMultiUAVEnv`
- 配置：`configs/env/waypoint_missions.yaml`
- 训练：`scripts/train_mappo_waypoint.py`
- 验证：`scripts/check/validate_waypoint_mappo.py`
- MAPPO trainer：`algorithms/mappo_waypoint.py::MAPPOWaypointTrainer`

## 动作接口

默认 2D 动作：

```python
actions = {
    "uav_0": np.asarray([0.32, 0.18], dtype=np.float32),
    "uav_1": np.asarray([0.40, 0.22], dtype=np.float32),
}
obs, rewards, terminations, truncations, infos = env.step(actions)
```

`action_space(agent_id)` 是：

```text
Box(low=[0, 0], high=[map_size, map_size], shape=(2,), dtype=float32)
```

如果后续启用 3D，可扩展成：

```text
Box(low=[0, 0, min_alt], high=[map_size, map_size, max_alt], shape=(3,), dtype=float32)
```

非法 waypoint 处理：

- 默认 `strict_action_validation=false`。
- waypoint 越界、进入 no-fly zone、超过 `max_waypoint_distance`、违反最小机距或可选连通性硬约束时，环境会替换为 safe fallback。
- 如果 `strict_action_validation=true`，非法动作会抛 `ValueError`。
- `infos[agent_id]` 会记录 `selected_waypoints`、`raw_waypoints`、`invalid_action_count` 和 `action_validity`。

## 默认世界配置

默认配置来自 `configs/env/waypoint_missions.yaml`：

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `action_mode` | `waypoint` | 环境主动作模式 |
| `map_size` | `1.0` | 二维连续地图大小 `[0,1] x [0,1]` |
| `grid_size` | `32` | 任务场栅格大小 |
| `num_agents` | `4` | UAV 数量 |
| `max_steps` | `80` | 每个 episode 最大步数 |
| `dt_decision` | `1.0` | 决策时间步长 |
| `max_speed` | `0.08` | 每步最大移动速度 |
| `max_waypoint_distance` | `0.22` | waypoint 相对当前位置最大距离 |
| `min_uav_distance` | `0.035` | UAV 最小安全间距 |
| `sensor_radius` | `0.12` | 传感/覆盖半径 |
| `comm_radius` | `0.36` | UAV-UAV 通信半径 |
| `base_comm_radius` | `0.45` | base-UAV 直连通信半径 |
| `base_position` | `[0.10, 0.10]` | base station 位置 |

默认 no-fly zone 是圆形区域：

```yaml
center: [0.55, 0.55]
radius: 0.08
```

## Observation 和 Global State

默认 agent observation 不再包含 candidates：

| key | shape | 含义 |
|---|---:|---|
| `self_state` | `[8]` | 当前 UAV 归一化状态 |
| `neighbor_relative_states` | `[N-1, 4]` | 其他 UAV 相对位置和相对速度 |
| `task_field` | `[5, H, W]` | 全局任务场 |
| `task_id` | `[6]` | 任务 one-hot |
| `global_summary` | `[8]` | 全局摘要 |
| `all_uav_states` | `[N, 8]` | 所有 UAV 状态 |
| `comm_adjacency` | `[N+1, N+1]` | base + UAV 通信图 |

centralized critic state 来自：

```python
env.get_global_state()
```

默认字段：

- `task_field`
- `global_task_field`
- `all_uav_states`
- `task_id`
- `global_info`
- `comm_adjacency`
- `agent_mask`

`self_state/all_uav_states` 8 维：

```text
x, y, vx, vy, battery, normalized_role_id, connected_to_base, last_action_valid
```

`task_field` 五个通道：

```text
channel 0: coverage_grid
channel 1: visit_count_grid
channel 2: belief_grid
channel 3: risk/no-fly grid
channel 4: target_or_poi grid
```

## 安全层和执行模型

当前 env.step 内部顺序：

```text
raw waypoint action
  -> SafetyLayer.validate_waypoints
  -> ActionAdapter.adapt
  -> DynamicsBackend.step
  -> scenario.compute_rewards
```

默认组合：

```yaml
action_interface: waypoint
action_adapter:
  name: waypoint_velocity_tracker
dynamics_backend:
  name: mpe_core
```

`SafetyLayer.validate_waypoints(...)` 是主线安全入口：

1. 检查 geofence。
2. 检查 no-fly zone。
3. 检查 waypoint 与当前位置距离是否超过 `max_command_distance`。
4. 可选检查选定 waypoint 后是否仍保持连通。
5. 检查 UAV 间最小距离。
6. 非法 waypoint 替换为 stay 或 base-safe fallback；过远 waypoint 会截断成 local waypoint。

`WaypointVelocityTracker` 是默认 ActionAdapter：

```text
e_i = w_i - p_i
d_i = ||e_i||
dir_i = e_i / (d_i + eps)
v_des_i = max_speed * min(1, d_i / slowdown_radius) * dir_i
a_cmd_i = velocity_gain * (v_des_i - v_i)
||a_cmd_i|| <= max_accel
```

如果 `d_i < acceptance_radius` 且 `hover_when_arrived=true`，`v_des_i=0`。`control_smoothing` 用于平滑控制变化。

`WaypointPDTracker` 是 debug/ablation adapter：

```text
a_cmd_i = kp * (w_i - p_i) - kd * v_i
||a_cmd_i|| <= max_accel
```

当前可手动选择三套动力学后端：

- `waypoint_behavior_fast`：大规模训练主线，不依赖 MPE，按上层航点行为快速推进。
- `waypoint_behavior_realistic`：中等真实度验证主线，不依赖 MPE，加入速度平滑、速度扰动、追踪噪声和命令延迟。
- `mpe_core`：高保真部署前校验主线，调用真实 MPE `World.step()`。

`MPECoreBackend` 不在 Wayffusion wrapper 内重写粒子积分，而是：

```text
Wayffusion UAVState
  -> MPE Agent.state
  -> agent.action.u
  -> real MPE World.step()
  -> sync back to Wayffusion UAVState
```

速度积分、damping、contact/collision force、max-speed clipping 来自真实 MPE core。Wayffusion 仍负责任务层 safety correction 记录、path length、trajectory、通信图和 coverage footprint 更新。

旧的自写 particle backend 和 kinematic waypoint backend 已从主线删除。`mpe_particle`、`kinematic_point` 等名字只会触发错误提示，不能作为可运行配置使用。新训练/验证后端详见 `docs/dynamics_backend_modes_zh.md`。

`WaypointExecutionModel` 仍存在，但主线 env 默认使用 `DynamicsBackend`。

## PolicyOutput 抽象

所有 MAPPO policy 都通过 `policies/policy_output.py::PolicyOutput` 对接 trainer：

```text
env_action:   [B, N, D]  最终传给环境的 waypoint
train_action: policy 内部采样动作
logprob:      [B, N]
entropy:      [B, N]
value:        [B]
aux:          调试信息
```

因此 trainer 不需要知道 policy 是 categorical candidate-selection 还是 Gaussian direct waypoint。

## CandidateSelectionWaypointPolicy

文件：`policies/candidate_selection_policy.py`

流程：

1. policy 根据 `task_field/all_uav_states/task_id/global_info` 内部生成 candidates。
2. 生成 `candidate_waypoints [B,N,K,2]`、`candidate_features [B,N,K,F]`、`candidate_mask [B,N,K]`。
3. neural actor 输出 masked logits `[B,N,K]`。
4. 每个 UAV 采样 candidate index。
5. 将 index 映射成最终 `env_action [B,N,2]`。
6. `train_action [B,N]` 保存 index，PPO update 时重算 categorical logprob。

这保持了候选选择策略的可解释性，但环境不再知道或依赖 candidates。

## DirectWaypointPolicy

文件：`policies/direct_waypoint_policy.py`

流程：

1. actor 直接输出每个 UAV 的 Gaussian delta mean。
2. 采样 delta 作为 `train_action [B,N,2]`。
3. 按向量范数裁剪到 `max_delta`，再映射为最终 waypoint。
4. `env_action [B,N,2]` 直接给环境。
5. logprob/entropy 来自 Normal distribution。

这个 policy 用于验证“不通过候选点、直接生成 waypoint”是否能训练。

## MAPPOWaypointTrainer

文件：`algorithms/mappo_waypoint.py`

关键点：

- rollout 存 `env_actions [T,B,N,D]` 和 `train_actions`。
- 环境 step 只使用 `env_action`。
- PPO update 使用 `train_action` 重算 per-agent logprob。
- actor ratio 是 per-agent：`exp(new_logprob_i - old_logprob_i)`。
- team reward 计算 GAE，team advantage broadcast 到 agents。
- critic 是 centralized state value。
- `terminated` 和 `truncated` 分离；true termination 不 bootstrap，time-limit truncation bootstrap。

## 任务体系

- `area_coverage`：最大化新覆盖，惩罚重复覆盖、距离和安全违规。
- `belief_search`：搜索概率/belief map 高质量区域，支持目标检测 bonus。
- `priority_inspection`：按权重访问 POI，惩罚重复访问和 deadline late。
- `connectivity_expansion`：保持到 base 多跳连通的同时向外扩展，避免“全部停在 base 附近”拿高分。
- `dynamic_target_escort`：动态目标持续监测，多视角分布有奖励。
- `target_interception`：覆盖目标未来高概率路径，实现检测/拦截/包围式成功。

每个 scenario 返回：

- `team_reward`
- `per_agent_rewards [N]`
- `components`
- `metrics`
- `success`

## 可视化和输出

环境支持：

```python
env.render("rgb_array")
env.render("human")
```

验证/训练可保存 GIF 或 MP4：

```text
outputs/debug/waypoint_mappo_validation/<run>/...
outputs/training/mappo_waypoint/<timestamp>/<run_name>/media/...
```

训练输出包括：

- `training_metrics.csv`
- `eval_metrics.csv`
- `tensorboard/`
- `checkpoints/`
- `best_eval_summary.json`
- `media/eval_*/`

## 当前验证命令

```bash
/opt/conda/bin/python -m pytest -q \
  tests/test_waypoint_env_api.py \
  tests/test_waypoint_scenarios.py \
  tests/test_waypoint_rewards.py \
  tests/test_candidate_selection_policy.py \
  tests/test_direct_waypoint_policy.py \
  tests/test_mappo_waypoint_trainer.py \
  tests/test_waypoint_rendering.py
```

```bash
/opt/conda/bin/python scripts/check/validate_waypoint_mappo.py \
  --tasks area_coverage belief_search connectivity_expansion \
  --policy-class candidate_selection_waypoint \
  --num_agents 3 \
  --total_updates 5 \
  --eval_episodes 2 \
  --record_eval_episodes 1 \
  --record_format gif \
  --headless
```

```bash
/opt/conda/bin/python scripts/check/validate_waypoint_mappo.py \
  --tasks area_coverage belief_search \
  --policy-class direct_waypoint \
  --num_agents 3 \
  --total_updates 2 \
  --eval_episodes 1 \
  --record_eval_episodes 1 \
  --record_format gif \
  --headless
```
