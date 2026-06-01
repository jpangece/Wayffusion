# 单任务专家基线表

更新时间：2026-06-01 phase68。

本表只记录当前新决策模式下应复用的单任务专家基线。新模式指 `factorized_group` / per-agent route-target style：critic 仍看全局 state，actor 共享参数但为每个 UAV/group 输出独立 waypoint，joint action shape 仍是 `[B, N, 2]`。

| 任务 | 状态 | checkpoint | policy config | env / eval config | 主要证据 | caveat |
| --- | --- | --- | --- | --- | --- | --- |
| `goal_nav` | 已修通/可复现 | `outputs/training/bc_ppo/20260601_phase68_goalnav_task_targets_ppo/phase68_goalnav_task_targets_ppo/checkpoints/checkpoint_0040.pt` | `configs/policy/debug_ppo_goal_nav_factorized_group_task_targets_safe.yaml` | `configs/env/debug_goal_nav_task_targets.yaml` | five-seed 40ep: `success_rate=0.965`, `goal_coverage_ratio=0.984333`, `pair_collision_rate=0.005324`; seed23 100ep: `success_rate=0.89` | 使用 `include_task_targets_in_agents`；比 phase36/65 更适合作为 robust goal_nav PPO expert。 |
| `coverage` | 已修通/可复现 | `outputs/training/bc_ppo/20260601_phase60_coverage_route_target_agents_ppo/phase60_coverage_route_target_agents_ppo/checkpoints/checkpoint_best_eval.pt` | `configs/policy/debug_ppo_coverage_factorized_group_route_target_agents_safe.yaml` | `configs/env/debug_coverage_route_target_agents_canonical.yaml`, `configs/env/debug_coverage_route_target_agents_seed23.yaml` | seed 7 100ep: `success_rate=0.72`, `coverage_ratio=0.801508`; seed 23 100ep: `success_rate=0.68`, `coverage_ratio=0.795507` | 这是 route-target-agent coverage 设计，不是原始 `configs/env/coverage.yaml` 无改动结果；重复覆盖仍高。 |
| `risk_nav` | 已修通/可复现 | `outputs/training/bc_ppo/20260530_phase41_risknav_factorized_group_dagger_safe/phase41_risknav_factorized_group_dagger_safe/checkpoints/checkpoint_best_eval.pt` | `configs/policy/debug_ppo_risk_nav_factorized_group_dagger_safe.yaml` | `configs/env/debug_risk_nav_safety_completion.yaml`, `configs/env/debug_risk_nav_seed23.yaml` | seed 7/23 100ep 都是 `success_rate=0.65`; seed 23 `goal_coverage_ratio=0.841667`, `collision_rate=0.018796` | 成功可复现，但 risk exposure 和 safety violation 仍偏高。 |
| `formation` | 已修通/可复现 | `outputs/training/bc_ppo/20260530_phase39_formation_factorized_group_ppo/phase39_formation_factorized_group_ppo/checkpoints/checkpoint_best_eval.pt` | `configs/policy/debug_ppo_formation_factorized_group_ultra_strict.yaml` | `configs/env/multitask.yaml`, `configs/env/debug_formation_seed23.yaml` | seed 7 100ep: `success_rate=0.77`; seed 23 100ep: `success_rate=0.78`; `formation_error≈0.073` | 依赖 template-aware success metric 修复；这不是降低阈值，而是修正 line/arc 的几何判据。 |

当前结论：

- 四个任务都可以作为 downstream multi-task 的专家基线使用。
- 若继续单任务 hardening，优先级是 `risk_nav` safety、`coverage` repeated coverage。
- 不建议回到旧的 shared-waypoint / non-factorized 架构。
