# Waypoint 任务域随机化

## 修改前的实际情况

训练环境并非完全固定，但随机化不完整：

- UAV 出生点每个 episode 都会在固定基地附近随机采样。
- `belief_search` 的概率峰和隐藏目标会随机采样。
- `priority_inspection` 的 POI、权重和截止时间会随机采样。
- `dynamic_target_escort` 的目标初始位置和方向会随机采样。
- 地图边界、基地位置和禁飞区固定。
- 出生点只在较窄的基地邻域内变化。
- POI 或概率目标没有统一保证避开禁飞区。

因此，旧训练能够看到不同目标，但容易记住固定禁飞区布局，并且可能遇到不可达目标。

## 当前域随机化

`configs/env/waypoint_missions.yaml` 和
`configs/env/waypoint_missions_direct_tune_v2.yaml` 现在包含：

```yaml
domain_randomization:
  enabled: true
  spawn:
    min_radius: 0.03
    max_radius: 0.24
  no_fly_zones:
    enabled: true
    include_static: false
    min_count: 1
    max_count: 3
    circle_probability: 0.75
    radius_range: [0.04, 0.09]
    rectangle_size_range: [0.06, 0.14]
```

每个 episode reset 时会重新采样：

- 1 到 3 个圆形或矩形禁飞区；
- 基地附近不同半径和角度的 UAV 出生点；
- `belief_search` 的 2 到 5 个安全概率峰和隐藏目标；
- `priority_inspection` 的 6 到 10 个安全 POI。

随机禁飞区不会覆盖基地，并与其他禁飞区保持最小间隔。出生点、belief 峰、隐藏目标和 POI 都必须位于 geofence 内且不在禁飞区内。

当前没有单独的实体障碍物碰撞模型。任务中的静态空间障碍由禁飞区表达，底层 UAV-UAV 接触仍由真实 OpenAI MPE core 处理。

## 可复现性

域随机化使用环境自己的 `numpy.random.Generator`：

- 相同 seed 会生成相同禁飞区、出生点和目标；
- 不同 seed 会生成不同 episode 布局；
- vector env 中不同 env 使用不同 seed；
- 自动 reset 不重新固定 seed，因此后续 episode 会继续生成新布局；
- evaluation 使用固定的 held-out seed 序列，因此结果可以复现。

这意味着随机化不会破坏实验复现，但不能再只用一个 seed 判断鲁棒性。

## Coverage 分母

随机禁飞区面积不同。如果仍用整张栅格作为 coverage 分母，禁飞区更多的 episode 会天然更难，success threshold 也会失去一致含义。

当前 coverage 指标改为：

$$
\text{coverage ratio}
=
\frac{\text{已覆盖且可飞的栅格数}}
{\text{可飞栅格总数}}
$$

`area_coverage` 和 `connectivity_expansion` 的 coverage reward、metric 和 success 都使用该定义。禁飞区不计入必须覆盖的任务区域。

## 日志

环境 info 新增：

- `domain_randomization_enabled`
- `domain_randomization`
- `episode_seed`
- `no_fly_zone_count`

训练 CSV 新增：

- `mean_no_fly_zone_count`
- `domain_randomization_enabled`

评估 CSV 新增：

- `no_fly_zone_count`
- `domain_randomization_enabled`

## 使用建议

正式鲁棒性结论至少应报告：

1. 多个训练 seed；
2. 固定 held-out evaluation seeds；
3. success rate 和任务指标；
4. collision/no-fly/geofence violation；
5. 按禁飞区数量分组后的表现。

域随机化会提高训练方差，不保证立即提高训练集 reward。合理目标是降低训练布局记忆，并改善 held-out seed 的成功率。若随机化导致早期完全学不到，可使用 curriculum：先固定或减少禁飞区，再逐步切换到当前 1 到 3 个随机禁飞区，最终评估仍必须使用目标随机化配置。
