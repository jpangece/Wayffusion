# Swarm30 Reward / Rollout 严格等价优化

## 目标

本轮优化针对 `waypoint_behavior_fast` 下 N=30 rollout 的 CPU 热点。约束是：

- 不修改环境、task、policy、trainer 的对外接口。
- 不修改 reward 公式、终止条件、每步 metrics 定义。
- 不降低网格分辨率，不跳帧，不减少 reward 计算频率。
- 只允许缓存、共享中间量、向量化和等价的数据结构替换。
- 数值结果只允许浮点容差范围内的差异。

已有 profiling 表明，fast dynamics 下主要耗时不再是动力学，而是任务 reward/metric 几何：

- `area_coverage.compute_rewards`: 约占单步 `78%`。
- `belief_search.compute_rewards`: 约占单步 `72%`。
- `connectivity_expansion.compute_rewards`: 约占单步 `47%`。
- `target_interception.compute_rewards`: 约占单步 `82%`。

## 实现

### 1. 共享 coverage transition 统计

`MissionWorld.accumulate_sensor_footprints()` 本来已经计算每架 UAV 的传感器 footprint。现在它同时保存本步的：

- `sensor_masks`
- `newly_covered_mask`
- `per_agent_new_masks`
- `per_agent_new_coverage`
- `repeat_footprint_ratios`

area coverage 和 connectivity reward 直接复用这些结果，不再在同一步重复构造 `[N,H,W]` footprint。

缓存同时校验：

- 上一步 coverage / visit 状态
- 当前 UAV positions
- 当前 sensor radii

因此非标准调用中位置改变后不会错误复用旧统计。

### 2. uncovered approach 最近距离

旧实现构造完整距离张量：

```text
[N, uncovered_grid_cells, 2]
```

然后对 UAV 维取最近距离。新实现使用 `scipy.spatial.cKDTree` 查询每个未覆盖网格点到 UAV 集合的最近距离。

数学量保持不变：

```math
P(X) = \frac{1}{|U|}\sum_{x \in U}
\exp\left(-\frac{\min_i \lVert p_i-x\rVert_2}{s}\right)
```

reward 仍为：

```math
\max(P(X_{t+1}) - P(X_t), 0)
```

查询距离转回 `float32`，以匹配旧 NumPy dense 路径的计算精度。

### 3. belief search 批量精确 sensor mask

belief reward 仍使用精确欧氏网格 mask，不改用 disk stamp。变化仅为：

- 一次生成全部 UAV 的 exact masks。
- 保留每架 UAV 独立累加 belief mass 的旧语义。
- 重叠区域仍可被多架 UAV 分别计入本步 `newly_mass`。
- belief grid 清零使用 mask union，结果与逐 UAV 清零一致。

### 4. connectivity 共享 footprint 与 relay 等价化简

connected/disconnected coverage breakdown 复用本步 coverage transition masks。

旧 `_relay_credit()` 对每个已连接 UAV 做一次删点 DFS。当前公式实际会给每个已连接 UAV 记 1 分，因为删除该节点后，连通 UAV 数至少减少该节点自身。因此严格等价化简为：

```math
\text{relay\_credit} =
\begin{cases}
N_{\text{connected}}, & N_{\text{connected}} > 1 \\
0, & \text{otherwise}
\end{cases}
```

测试保留旧 DFS reference，并对比新结果。

### 5. target interception 向量化

- future route points 和 point weights 按旧 branch 顺序展开。
- future belief field 使用 `np.add.at` 写回网格。
- UAV 到未来路径点的覆盖判断改为批量距离矩阵。
- covered mass 和 total mass 仍按原顺序使用 Python float 累加。

### 6. 静态场和轻量 reward snapshot

- no-fly risk grid 在 episode world 内缓存。
- priority inspection 的 Gaussian POI kernels 按 POI positions 和 sigma 缓存。
- pairwise distance 按 positions 精确缓存。
- reward snapshot 不再复制完整 trajectory 历史，只保留当前点。
- coverage、belief、visit grid 和 UAV 动态状态仍做独立副本，reward 计算不会污染 live world。

## 等价性验证

新增 `tests/test_reward_geometry_equivalence.py`，旧实现公式作为 reference，覆盖：

- coverage transition 统计。
- N=4 与 N=30 uncovered approach。
- belief overlap/repeat/detection reward。
- connectivity connected/disconnected coverage 和 relay DFS。
- target interception future belief。
- priority Gaussian field。
- risk grid、pairwise cache 和 reward snapshot 隔离。

严格对照测试结果：

```text
7 passed
```

相关 reward、scenario、env、trainer、Swarm30 benchmark 回归结果：

```text
41 passed
```

动态后端、process vector env、随机化、policy、rendering 回归结果：

```text
25 passed
```

## 运行影响

Python 进程启动后已加载模块代码，因此本轮文件修改不会改变当前正在运行的正式 N=30 训练。只有后续新启动的训练或 evaluate 进程会使用优化实现。

当前没有在服务器空闲状态下重新跑完整 timing benchmark，以免和正式训练争抢 CPU。实际 wall-clock 加速比需要在当前训练结束后，用相同配置、相同 seed、相同并发度重新测量。

