# Waypoint Adapter 参数标定记录

日期：2026-06-06

本轮没有新增 adapter，也没有改变环境主接口。Wayffusion 仍使用 `waypoint_velocity_tracker`，但将它解释为“短期航点目标到 MPE physical action.u 的飞控近似器”，而不是低层速度/加速度控制环境。

## 尺度假设

- `map_size = 1.0` 对应真实 `100m x 100m` 任务区域。
- `0.01` map unit 约等于 `1m`。
- `max_waypoint_distance / max_command_distance` 表示每次重规划允许的短期航点半径，不表示单步物理位移。
- 单个 env step 的近似真实时间由 `dynamics_backend.dt * dynamics_backend.substeps` 决定。

当前默认值：

| 参数 | 默认值 | 物理含义 |
| --- | ---: | --- |
| `max_waypoint_distance` | `0.20` | 每次局部航点目标最多约 `20m` |
| `max_speed` | `0.04` | 约 `4m/s` 巡航速度上限 |
| `dt` | `0.1` | MPE core 子步时间 |
| `substeps` | `20` | 一个 env step 约 `2s` |
| 单步名义位移 | `0.08` | `4m/s * 2s = 8m` |
| `slowdown_radius` | `0.08` | 距目标约 `8m` 开始减速 |
| `acceptance_radius` | `0.020` | 约 `2m` 到点半径 |

## 为什么不默认使用 0.50

`max_waypoint_distance=0.50` 在 100m 地图中等价于每次重规划可以选择 50m 外目标。对覆盖、搜索、巡检这类局部任务，它会让 action space 过宽，policy 更容易输出穿越 no-fly / geofence / 已覆盖区域的大跨度目标。

小规模标定中，`wide_050_s04_a120_2s` 的 probe tracking error 明显高于 `0.20/0.25` 组，且 action validity 更低。因此 0.50 不适合作为默认值，只适合后续长距离转场或稀疏拦截任务单独验证。

## 标定结果

输出目录：

- `outputs/debug/waypoint_adapter_scale_calibration_20260606/`
- `outputs/debug/waypoint_adapter_scale_calibration_20260606_refined/`

第一轮 synthetic probes 包含越界/大转场压力测试，所有 setting 都触发了不同程度 safety clipping，不作为最终硬通过标准。第二轮改成局部短期航点 probes 后，额外补跑了 `0.20` 半径 + `2s` 决策步组合。

关键补跑结果：

| setting | score | arrival | progress | task_success | task no-fly/geofence |
| --- | ---: | ---: | ---: | ---: | --- |
| `def_020_s04_a080_2s` | `1.868` | `0.47` | `0.181` | `1.00` | `0 / 0` |
| `def_020_s04_a100_2s` | `1.922` | `0.50` | `0.191` | `1.00` | `0 / 0` |
| `def_020_s04_a120_2s` | `1.876` | `0.47` | `0.192` | `1.00` | `0 / 0` |
| `def_025_s04_a080_2s` | `1.643` | `0.44` | `0.188` | `0.875` | 有少量 no-fly 修正 |

因此默认选择 `0.20 / 0.04 / max_accel=0.10 / substeps=20`，原因是它在局部航点尺度下同时保持任务进度、安全性和较好的到点表现。

## 三档推荐配置

文件：

- `configs/env/adapters/waypoint_velocity_tracker_conservative.yaml`
- `configs/env/adapters/waypoint_velocity_tracker_default.yaml`
- `configs/env/adapters/waypoint_velocity_tracker_aggressive.yaml`

| profile | max command | speed | accel/control cap | smoothing | env step | 用途 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| conservative | `0.20` / `20m` | `0.03` / `3m/s` | `0.08` | `0.40` | `2s` | 稳定和安全优先 |
| default | `0.20` / `20m` | `0.04` / `4m/s` | `0.10` | `0.45` | `2s` | 当前主实验默认 |
| aggressive | `0.30` / `30m` | `0.05` / `5m/s` | `0.12` | `0.60` | `2s` | 上限/压力测试 |

注意：`max_accel` 在当前真实 MPE core 中实际限制的是传给 `World.step()` 的 `agent.action.u` 范围。由于 MPE core 每个子步有 damping，`0.015/0.025/0.040` 在本仓库动力学下过软，标定中不能有效达到期望航点跟踪速度。

## 后续真机日志最应微调的参数

- `max_command_distance`：由真实任务重规划频率和航点接受逻辑决定。
- `max_speed`：应根据搜索/覆盖任务期望巡航速度和电池策略设置。
- `slowdown_radius` 与 `acceptance_radius`：应根据道通飞控真实到点半径、减速行为、GPS/视觉定位误差校准。
- `control_smoothing`：用于模拟飞控对频繁航点更新的平滑程度。
- `dt/substeps`：决定 RL 决策频率，后续应和真实任务上传航点频率对齐。
