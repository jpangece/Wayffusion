# Swarm30 Runtime Profiling 结果

日期：2026-06-11  
输出目录：

```text
outputs/debug/perf_profile/swarm30_runtime_breakdown/20260611_1526/
outputs/debug/perf_profile/env_step_cprofile/20260611_1526_connectivity/
```

本次 profiling 的目标是解释 `swarm30_v1` 训练和纯 evaluate 为什么慢。实验只新增诊断脚本，没有修改环境基础动力学、reward、policy 或任务实现。

相关脚本：

```text
scripts/benchmarks/swarm30/profile_runtime_breakdown.py
scripts/benchmarks/swarm30/profile_env_step_cprofile.py
```

## 核心结论

主要瓶颈不是神经网络，也不是 PPO backward，而是 CPU 侧环境仿真和渲染：

- `rollout collect / env.step` 约占每 100 update 周期的 `86.2%`；
- eval 环境步进约占 `9.1%`；
- render 约占 `3.7%`；
- GIF 写文件约占 `0.2%`；
- PPO backward/update 约占 `0.8%`。

因此，单纯换 GPU 或优化网络结构不能显著解决 wall-clock 速度问题。需要优先优化 N=30 环境步进、connectivity metrics 缓存、MPE collision/communication 计算和渲染频率。

## 主要测量值

| 项 | 测量值 |
|---|---:|
| N=4 单环境 step | `0.009-0.013 s/step` |
| N=30 area 单环境 step | `0.271 s/step` |
| N=30 belief 单环境 step | `0.221 s/step` |
| N=30 connectivity 单环境 step | `0.422 s/step` |
| N=30 8-env process batch step | `0.453 s/batch step` |
| 其中 policy forward | `0.011 s/batch step` |
| 其中 env batch step | `0.442 s/batch step` |
| render 单帧 | `1.14 s/frame` |
| GIF 写单帧 | `0.068 s/frame` |
| fake PPO minibatch backward | `0.066 s/minibatch` |

`N=30` 相比 `N=4` 的单环境 step 慢一个数量级以上，尤其是 `connectivity_expansion`。

## Benchmark 100 Update 周期估算

当前 `swarm30_v1` 训练设置：

```text
num_agents=30
num_envs=8
rollout_steps=256
eval_interval=100
eval_episodes=20
record_eval_episodes=2
env_backend=process
```

按实测时间估算每 100 update 周期：

| 环节 | 估算时间 | 占比 |
|---|---:|---:|
| collect | `116.0 s/update * 100` | `86.2%` |
| PPO update 近似 | `1.05 s/update * 100` | `0.8%` |
| eval stepping | `1221.6 s` | `9.1%` |
| eval render | `502.0 s` | `3.7%` |
| eval GIF write | `30.0 s` | `0.2%` |

注意：PPO update 是用 fake minibatch backward 近似估算，只用于判断量级。结论是 PPO backward 不是主要瓶颈。

## Evaluate 单任务耗时估算

N4 checkpoint 在 N30 上 evaluate 的默认设置是：

```text
fixed_episodes=20
random_episodes=100
record_episodes=2
N=30
max_steps≈220
```

按实测时间估算单任务 evaluate：

| 环节 | 时间 |
|---|---:|
| episode stepping | `7329.9 s` |
| render | `1003.9 s` |
| GIF write | `60.1 s` |
| 合计 | `8393.9 s ≈ 2.33 h` |

这与实际观察到 `area_coverage` N4-to-N30 evaluate 跑两个多小时吻合。

## env.step 内部热点

对 `connectivity_expansion, N=30` 做 `cProfile`，30 step 总计 `19.39 s`，均值 `0.647 s/step`。主要热点：

| 内部函数 | 累计耗时 | 含义 |
|---|---:|---|
| `dynamics_backends.py:MPECoreBackend.step` | `8.43 s` | MPE core 动力学 |
| `third_party/openai_mpe/core.py:World.step` | `5.83 s` | 真实 MPE world.step |
| `core.py:get_collision_force` | `4.80 s` | N=30 pairwise collision/contact |
| `world.py:communication_graph_for_positions` | `5.43 s` | 通信图 / pairwise 距离 |
| `connectivity_expansion.compute_rewards` | `4.85 s` | connectivity reward/metrics |
| `_build_agent_info` / `get_metrics` | `5.72 s` | 每个 agent info 重复构造 task metrics |

这说明 `connectivity_expansion` 的慢来自三类叠加：

- N=30 的 pairwise 计算接近 \(O(N^2)\)；
- MPE core 每个 env step 内有多 physics substeps 和 collision force；
- connectivity metrics 在 reward、info、agent info 中被重复计算。

## 优化优先级

不改变环境语义的优先优化：

1. 在 `WaypointMultiUAVEnv.step` 内缓存本 step 的 `task_metrics`，避免对 30 个 agent 重复计算同一批 team-level metrics。
2. 对 connectivity 的 communication graph、connected mask、margin、effective radius 做 step-level cache。
3. 训练中间评估减少 media 频率，例如 `record_eval_episodes=0/1`，最终评估再保存完整 GIF。
4. 降低 benchmark 并发度，例如 `max_parallel=4 -> 2`，避免 32 个 N=30 env worker 同时抢 CPU。
5. 纯 evaluate 不要和 benchmark training 同时跑。

会改变仿真语义、必须作为 ablation 的优化：

1. `dynamics_backend.substeps: 20 -> 10`。
2. 训练阶段禁用 MPE collision/contact force，最终 eval 再启用。
3. 降低 N=30 grid/horizon 或减少 evaluation episode 数。

## 已停止的进程

为避免 profiling 被正在运行的训练和 evaluate 干扰，本次 profiling 前停止了：

- `swarm30_v1` benchmark controller 和其训练子进程；
- N4-to-N30 evaluate 脚本；
- 等待 benchmark 结束的 Phase32 UVFA queue。

已有输出文件未删除。
