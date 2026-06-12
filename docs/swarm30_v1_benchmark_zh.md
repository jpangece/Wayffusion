# Swarm30 v1 多无人机基准

## 目标

`swarm30_v1` 用于评估 Direct MAPPO waypoint policy 从 4 架 UAV 扩展到 30 架 UAV 时的任务性能、泛化能力和安全性。协议冻结六个任务：

- `area_coverage`
- `belief_search`
- `priority_inspection`
- `connectivity_expansion`
- `dynamic_target_escort`
- `target_interception`

基准不会替代单任务专家调试。环境、奖励或专家策略仍可继续迭代，但必须用新的版本名重新跑 benchmark，不能覆盖 `swarm30_v1` 的结果。

## 三条训练轨道

1. `standardized_specialists`
   - 六个任务分别训练。
   - 使用同一网络结构、动作尺度和 2000 update 预算。
   - 用于公平比较不同任务的可学习性。

2. `tuned_specialists`
   - 每个任务使用当前任务专用 policy/reward 参数。
   - 每个任务仍训练 2000 update。
   - 作为任务专门化上限，不与 standardized/generalist 混淆为完全同条件比较。

3. `generalist`
   - 一个 policy 同时训练六个任务。
   - 总预算 12000 update，任务均匀采样。
   - 训练日志记录每个任务的实际 step 数与采样占比。

所有轨道均使用 Direct MAPPO、final waypoint action、真实 vendored OpenAI MPE core、随机化训练、无 BC、无 connectivity hard action filter。

## 密度归一化

| UAV | 地图边长 | 物理区域 | Grid | Max steps | POI | Belief peaks | Dynamic targets |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 1.00 | 100 m | 32 | 80 | 8 | 3 | 1 |
| 10 | 1.58 | 158 m | 56 | 128 | 20 | 8 | 2 |
| 20 | 2.24 | 224 m | 72 | 180 | 40 | 15 | 4 |
| 30 | 2.74 | 274 m | 88 | 220 | 60 | 23 | 5 |

地图面积和任务负载随 UAV 数量增长。以下物理能力保持绝对 map unit 不变：

- `sensor_radius`
- `comm_radius`
- `base_comm_radius`
- `max_speed`
- `max_delta`
- no-fly zone 物理尺寸
- spawn 半径
- task 中的探测、减速、拦截和目标速度参数

因此，扩大地图不会同时把传感器、通信或障碍物尺寸错误放大。

## 多目标动态任务

动态护航和目标拦截使用 `target_positions` 数组表示多个目标，同时保留第一个目标的 legacy alias。观测仍是统一的 5 通道 task field，所有目标写入同一个 target channel，不增加任务专用观测维度。

护航成功要求：

- 平均 target coverage 达到 `0.70`
- 最差目标 target coverage 达到 `0.50`

拦截成功要求：

- intercepted target ratio 达到 `0.80`

## 评估

每个训练 seed 的 checkpoint 都在 `N=4/10/20/30` 上评估：

- randomized: 100 episodes
- fixed: 20 episodes
- deterministic policy
- 每组保存 2 个 GIF

基线：

- hold
- random waypoint
- task-aware greedy planner

主排行榜使用六任务 macro success rate。不能把不同任务的 raw reward 相加。次级指标包括 normalized progress、action validity、collision 和 connectivity violation。

## 启动与状态

```bash
bash scripts/benchmarks/swarm30/launch_tmux.sh
```

该命令只启动等待器。等待器在 Phase31 三个 worker 正常结束后，才启动 benchmark 长训。

```bash
bash scripts/benchmarks/swarm30/status.sh
```

正式输出：

```text
outputs/training/benchmarks/swarm30_v1/<runtime>/
```

每个 run 保留 snapshot、TensorBoard、training metrics、checkpoints 和 eval media。

## Runtime Profiling

`swarm30_v1` 的主要 wall-clock 瓶颈是 CPU 侧环境步进和渲染，不是 PPO 网络更新。详细 profiling 见：

```text
docs/swarm30_runtime_profile_zh.md
```

关键结论：

- rollout collect / `env.step` 约占每 100 update 周期的 `86.2%`；
- eval stepping 约占 `9.1%`；
- render 约占 `3.7%`；
- PPO backward/update 约占 `0.8%`；
- N=30 connectivity 的热点包括 MPE collision force、通信图计算和重复 connectivity metrics。

因此后续提速应优先做 step-level metrics/cache、降低中途 media 频率和控制并发度；降低 MPE substeps 或关闭 collision force 必须作为单独 ablation，不能混入当前 frozen benchmark。
