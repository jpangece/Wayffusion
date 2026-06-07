# Direct MAPPO Specialist v2 Debug Notes

本文记录 2026-06-06 开始的新一轮 Direct MAPPO specialist 调参。旧的 Direct full run 没有停止，新一轮在独立 phase 下并行运行。

## 为什么要开 v2

旧 formal run 中 `area_coverage` 单 seed 已经出现学习迹象，最高观察到 `eval_success_rate=1.0`、coverage 约 `0.56`。但这不能证明 Direct MAPPO 已经调通：

- 只有 `area_coverage` 的一个 seed 进入 formal，四任务和多 seed 还没完成。
- `delta_clip_fraction` 后期接近 `0.72`，说明 Gaussian raw delta 大量被 `max_delta` 裁剪，PPO 优化的动作分布和环境实际执行动作存在明显偏差。
- 其它任务还没有 target-config 多 seed 收敛证据。
- 任务场 `task_field` 一直给所有任务全部通道，可能让单任务 specialist 学到无关甚至有害的输入。
- `sensor_radius=0.12` 对 100m 地图等价于 12m，覆盖/巡检/search 可能过于容易靠随机误打误撞获得进展。

## v2 改了什么

v2 不改 MPE core dynamics，不加 BC，不回退 candidate policy，不新增 adapter。主要改三类内容：

1. Direct policy 观测

- `task_field_channel_mode: task_relevant`
- `use_spatial_field_context: true`
- 每个任务只保留相关 task-field channel。
- belief_search 默认遮掉 target channel，避免 hidden target 泄漏或干扰。
- actor 额外读取 UAV 周围多个半径上的局部 task-field 采样值。

2. 任务/奖励

- 新 env config: `configs/env/waypoint_missions_direct_tune_v2.yaml`
- `sensor_radius` 从 `0.12` 缩到 `0.09`。
- coverage/search/connectivity 增强任务进展奖励并降低距离惩罚。
- priority_inspection 增加可选 dense approach reward，避免只有访问瞬间才有信号。

3. 调参和输出

- 新 policy config dir: `configs/policy/direct_specialists_v2/`
- 新 full tuning 支持 `--specialist-config-dir`。
- TensorBoard 使用 `tensorboard_metric_mode: minimal`，CSV 保留完整指标。
- 输出布局按 phase 组织：
  - debug: `outputs/debug/<timestamp>_phase_change_on_direct_obs_reward_v2/<task>/<short_run>/`
  - formal: `outputs/training/<timestamp>_phase_change_on_direct_obs_reward_v2/<task>/<short_run>/`

## 当前 v2 运行

tmux:

```bash
tmux attach -t wayffusion_direct_v2_20260606_1611
```

TensorBoard:

```bash
/opt/conda/bin/tensorboard \
  --logdir_spec debug:outputs/debug/20260606_1611_phase_change_on_direct_obs_reward_v2,formal:outputs/training/20260606_1611_phase_change_on_direct_obs_reward_v2 \
  --bind_all \
  --port 6008
```

浏览器访问：

```text
http://<server-ip>:6008
```

## 早期观测

第一个 `area_coverage` smoke:

- run: `outputs/debug/20260606_1611_phase_change_on_direct_obs_reward_v2/area_coverage/smoke__md0p18__s0__md018_std21`
- 20 updates:
  - `eval_success_rate=0.0`
  - `eval_area_coverage_coverage_ratio_mean≈0.435`
  - `eval_reward≈-47.1`
  - `action_validity_rate≈0.953`
  - `eval_action_validity_rate≈0.998`
  - `delta_clip_fraction≈0.398`
  - `approx_kl≈4.9e-7`
  - `clip_frac=0.0`

解释：

- 动作裁剪比旧 formal 的 `≈0.72` 健康很多。
- coverage 有正向进展，但 success 还没有解决。
- `approx_kl` 仍很小，后续如果 task metric 也停滞，需要继续增强 PPO update 或检查 reward scale。

## 后续判断标准

不要把 smoke 当作收敛结果。只有 formal 多 seed target-config 达标后，任务才能标记为收敛。

重点看：

- `eval_success_rate`
- task metric:
  - `coverage_ratio`
  - `searched_probability_mass`
  - `weighted_poi_completion`
  - `effective_explored_radius`
- `delta_clip_fraction`
- `approx_kl`
- `clip_frac`
- `value_loss`
- `advantage_std`
- `action_validity_rate`

如果 `delta_clip_fraction > 0.50`，下一轮先降 `log_std` 或 `max_delta`。

如果 `delta_clip_fraction` 健康但 `approx_kl ~ 1e-7` 且任务指标不涨，下一轮增强 PPO update，例如 `learning_rate=3e-4` 或 `epochs=6`，但不要先改底层动力学。
