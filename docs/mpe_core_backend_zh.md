# MPECoreBackend 真实 MPE 动力学后端

## 为什么删除自写后端

上一版 `MPEParticleBackend` 只是 Wayffusion 自己写的 particle dynamics。它虽然采用了类似 MPE 的变量名和公式，但没有调用 OpenAI MPE / MPE2 / PettingZoo MPE 的 `World.step()`。这会让动力学正确性、碰撞/contact force、damping、max-speed clipping 都变成 Wayffusion 自己维护的问题，不符合当前 MARL branch 的目标。

当前主线只支持：

```yaml
dynamics_backend:
  name: mpe_core
```

`mpe_particle`、`mpe_like_particle`、`kinematic_point` 已删除为可运行后端；传入这些名字会报错并提示使用 `mpe_core`。

## 当前 MPE source

`MPECoreBackend` 会按顺序尝试导入：

1. `mpe2.core`
2. `mpe2.mpe.core`
3. `pettingzoo.mpe._mpe_utils.core`
4. `third_party.openai_mpe.core`

本机当前没有安装 `mpe2` 或 `pettingzoo`，所以验证使用的是：

```text
third_party_openai_mpe
```

vendored 文件：

- `third_party/openai_mpe/core.py`
- `third_party/openai_mpe/UPSTREAM_README.md`
- `third_party/openai_mpe/README.md`
- `third_party/openai_mpe/LICENSE`

注意：upstream `openai/multiagent-particle-envs` 在 vendoring 时没有发现独立 `LICENSE` 文件，`third_party/openai_mpe/LICENSE` 记录了这个许可状态。外部分发前需要复核 upstream license。

## UAVState 到 MPE Agent 的映射

位置和速度同步发生在：

```text
envs/waypoint/control/dynamics_backends.py::MPECoreBackend._sync_wayffusion_to_mpe
```

每个 Wayffusion UAV 映射为一个 MPE `Agent`：

```text
uav.position -> agent.state.p_pos
uav.velocity -> agent.state.p_vel
uav.mass     -> agent.initial_mass
uav.radius   -> agent.size
```

MPE agent 参数由 env config 设置：

```yaml
dynamics_backend:
  source: mpe2
  dt: 0.1
  substeps: 10
  damping: 0.25
  contact_force: 100.0
  contact_margin: 0.01
  agent_size: 0.02
  agent_accel: 1.0
  max_speed: 0.08
  action_scale: 1.0
```

## Waypoint adapter 到 MPE action.u

Wayffusion 环境动作仍是 final waypoint `[x, y]`。ActionAdapter 是环境 transition dynamics 的一部分，负责把 waypoint command 转成 MPE physical action：

```text
safe_waypoint -> WaypointVelocityTracker / WaypointPDTracker -> controls [N,2]
controls * action_scale -> agent.action.u
```

写入位置：

```text
envs/waypoint/control/dynamics_backends.py::MPECoreBackend._set_mpe_actions
envs/waypoint/control/dynamics_backends.py::MPECoreBackend._assign_mpe_actions
```

ActionAdapter 可以计算 desired velocity、PD/velocity tracking control 和控制裁剪，但不更新位置和速度。

## MPE World.step 调用点

真实动力学调用发生在：

```text
envs/waypoint/control/dynamics_backends.py::MPECoreBackend.step
```

每个 decision step 内执行 `substeps` 次：

```text
agent.action.u = physical_action
mpe_world.step()
```

wrapper 不重写 MPE 的 velocity integration、damping、contact/collision force 或 max-speed clipping。这些来自 MPE core 的 `World.step()` / `integrate_state()` / `get_collision_force()`。

## Wayffusion 仍然负责什么

真实物理粒子更新来自 MPE core，但 Wayffusion 仍保留任务和工程逻辑：

- waypoint safety validation；
- geofence/no-fly hard correction 及其 violation 记录；
- task scenario；
- reward；
- task field；
- coverage/belief grids；
- connectivity graph；
- rendering；
- MAPPO rollout/eval/checkpoint/logging。

## 如何验证不是自写后端

测试文件：

```text
tests/test_mpe_core_backend.py
```

验证内容：

- `MPECoreBackend` 创建真实 `mpe_world` 和 MPE agents；
- monkeypatch/spy `backend.mpe_world.step`，确认每次 backend step 都调用 MPE `World.step()`；
- `build_dynamics_backend({"name": "mpe_particle"})` 和 `{"name": "kinematic_point"}` 必须失败；
- `dynamics_backends.py` 不包含自写积分/碰撞关键模式；
- step 后 Wayffusion `UAVState` 与 MPE `agent.state` 同步；
- `DynamicsInfo` finite。

validation summary 也会记录：

```text
uses_real_mpe_core: true
mpe_source: third_party_openai_mpe
mpe_world_step_calls: > 0
```
