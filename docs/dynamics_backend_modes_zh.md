# Wayffusion Dynamics Backend 模式

Wayffusion 的主动作接口仍然是上层航点：policy 输出 `waypoint` / `waypoint_delta`，环境接收最终航点，任务和 reward 接口不变。`dynamics_backend.name` 只决定环境内部如何把航点执行为状态转移。

当前支持三套手动选择的 dynamics backend：

```yaml
dynamics_backend:
  name: waypoint_behavior_fast
```

```yaml
dynamics_backend:
  name: waypoint_behavior_realistic
```

```yaml
dynamics_backend:
  name: mpe_core
```

## 为什么需要三套 backend

`mpe_core` 调用真实 MPE `World.step()`，包含多 physics substeps、damping、max-speed clipping 和 pairwise collision/contact force。它适合高保真校验，但在 `N=30` 大规模训练中，MPE collision 和通信/coverage 计算会成为主要 wall-clock 瓶颈。

Wayffusion 的目标是多无人机上层航点调度算法验证，不是低层飞控控制。因此训练阶段可以使用 mission-level behavior simulation：只模拟无人机朝上层航点前进、受速度/安全约束限制，同时保持 env/task/policy/trainer 的接口不变。高保真部署前再切回 `mpe_core` 校验。

## `waypoint_behavior_fast`

用途：大规模训练主线。

特点：

- 不依赖 MPE。
- `substeps=1`。
- 基于 `adapter_output.clipped_waypoints` 直接按 `max_speed * dt` 推进 UAV。
- 保留航点接受半径、geofence/no-fly 处理、collision distance check。
- 更新 `UAVState`、通信图、coverage grid、visit count、trajectory、path length。
- 默认 deterministic，适合快速训练和 sweep。
- `uses_real_mpe_core=false`。

示例：

```yaml
dynamics_backend:
  name: waypoint_behavior_fast
  dt: 1.0
  max_speed: 0.04
  acceptance_radius: 0.025
  agent_size: 0.02
  enable_boundary_projection: true
  enable_no_fly_projection: true
```

## `waypoint_behavior_realistic`

用途：中等真实度验证主线。

特点：

- 不依赖 MPE。
- 不继承 `waypoint_behavior_fast`，实现独立。
- `substeps=1`。
- 基于航点行为推进，但加入 mission-level 执行不确定性：
  - `velocity_smoothing`
  - `speed_scale_range`
  - `tracking_noise_std`
  - `command_latency_steps`
  - 可选 `gps_noise_std` diagnostics
- 使用 `world.rng`，固定 seed 可复现。
- 保留速度限制、航点接受半径、geofence/no-fly、collision distance check、通信图和 coverage 更新。
- `uses_real_mpe_core=false`。

示例：

```yaml
dynamics_backend:
  name: waypoint_behavior_realistic
  dt: 1.0
  max_speed: 0.04
  acceptance_radius: 0.025
  agent_size: 0.02
  velocity_smoothing: 0.35
  speed_scale_range: [0.85, 1.15]
  tracking_noise_std: 0.005
  command_latency_steps: 1
  enable_boundary_projection: true
  enable_no_fly_projection: true
```

## `mpe_core`

用途：高保真部署前校验主线。

特点：

- 调用真实 MPE `World.step()`。
- 默认 source 是仓库内 vendored OpenAI MPE core：`third_party/openai_mpe/core.py`。
- 可手动设置 `source: mpe2` 或 `source: pettingzoo_mpe`，但没有自动 fallback。
- `uses_real_mpe_core=true`。

示例：

```yaml
dynamics_backend:
  name: mpe_core
  source: third_party_openai_mpe
  dt: 0.1
  substeps: 20
  contact_force: 100.0
  contact_margin: 0.01
```

## 接口不变

三套 backend 都实现同一接口：

```python
reset(world)
step(world, controls, adapter_output=None, safety_result=None) -> DynamicsInfo
```

`DynamicsInfo.to_transition_info()` 保持兼容，包含：

- `path_length_delta`
- `arrived_mask`
- `rejected_mask`
- `safety_violation_count`
- `min_pairwise_distance`
- `collision_count`
- `no_fly_violation_count`
- `geofence_violation_count`
- `mean_speed`
- `max_speed_observed`
- `mean_control_norm`
- `disturbance_norm`
- `substeps`
- `info`

上层 `WaypointMultiUAVEnv.step()`、task reward、policy、MAPPO trainer 不需要因为 backend 切换而修改。

## 已删除的旧 backend

以下旧名称不是可运行 backend：

- `mpe_particle`
- `mpe_like_particle`
- `kinematic_point`

如果配置这些名字，factory 会报错，并提示改用：

- `waypoint_behavior_fast`
- `waypoint_behavior_realistic`
- `mpe_core`
