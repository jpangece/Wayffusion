# Coverage BC Warm-Start 到 PPO-Main 实验说明

这套实验用于回答一个具体问题：`coverage` 专家策略的性能是否主要来自 BC/teacher，还是 PPO 在 BC 专家基础上还能继续带来可观提升。

## 为什么旧 best-specialist 长跑不是严格 PPO-main

`scripts/tmux_start_best_specialists_all4.sh` 和 `scripts/run_ppo_parallel_best_specialists_all4.sh` 的定位是复现当前最稳专家，并做保守 continuation。

它们通常从已经 PPO 训练过的 `checkpoint_best_eval.pt` 或稳健 checkpoint 继续训练，并使用很保守的 safe config：

- `learning_rate` 很低；
- `clip_coef` 很小；
- `target_kl` 很小；
- `reference_policy_coef` 较大；
- `log_std` 被压得较窄。

这种设置适合“不要破坏已有专家”，但不适合验证“PPO 是否能作为主训练阶段继续优化 reward”。如果 `approx_kl` 长期很低、`reference_action_mse` 很小、`eval_reward` 只是震荡，那么更像是在维持原行为，而不是让 PPO 主导优化。

## 为什么要从 BC checkpoint 启动

要验证 PPO 的贡献，需要把起点定义清楚：

- BC checkpoint 表示 teacher / imitation 能达到的基础水平；
- Stage1 safe PPO 只验证 PPO 接上后不会立刻破坏 BC 行为；
- Stage2 PPO-main 从 BC 或 Stage1 best 出发，放宽 PPO 更新，让 PPO 有机会真正优化 reward。

如果直接从已有 `bc_ppo checkpoint_best_eval.pt` 继续训练，就无法区分：

- 性能来自 BC；
- 性能来自之前 PPO；
- 性能来自这次 PPO-main。

因此新实验默认从 coverage route-target-agent 的 BC checkpoint 启动：

```text
outputs/training/bc/20260601_034310/debug_bc_coverage_factorized_group_route_target_agents_perm_coverage_N4_multi_channel_field_plus_task_id/checkpoints/checkpoint_0032.pt
```

## 两阶段设计

### Stage 1: safe PPO from BC

配置：

```text
configs/policy/debug_ppo_coverage_factorized_group_route_target_agents_safe.yaml
```

目的：

- 从 BC checkpoint 出发；
- 短程 conservative PPO；
- 确认 PPO 接入后不会破坏 BC 行为；
- 生成 `checkpoint_best_eval.pt` 作为 Stage2 的默认起点。

默认：

```text
STAGE1_TOTAL_UPDATES=100
run_name=coverage_stage1_safe_from_bc
```

### Stage 2: PPO-main fine-tuning

配置：

```text
configs/policy/debug_ppo_coverage_factorized_group_route_target_agents_ppo_main.yaml
```

相对 safe config 的核心差异：

- `num_envs: 8`
- `epochs: 2`
- `learning_rate: 0.000008`
- `clip_coef: 0.03`
- `target_kl: 0.0025`
- `reference_policy_coef: 0.1`
- `ent_coef: 0.001`
- 更宽的 `log_std` 范围。

目的：

- 保留少量 reference regularization，避免完全漂移；
- 但显著放宽 PPO 更新，让 reward 优化能真正发生；
- 观察 `approx_kl`、`clip_frac`、`policy_std_mean`、`reference_action_mse` 是否显示策略在有效更新。

默认：

```text
STAGE2_TOTAL_UPDATES=500
run_name=coverage_stage2_ppo_main
```

## 如何运行

短程 smoke：

```bash
STAGE1_TOTAL_UPDATES=1 \
STAGE2_TOTAL_UPDATES=1 \
EVAL_EPISODES=2 \
RUN_TIMESTAMP=debug_coverage_bc_to_ppo_main_smoke \
bash scripts/run_coverage_bc_to_ppo_main.sh
```

正式 coverage 单任务实验：

```bash
RUN_TIMESTAMP="$(date +%Y%m%d_%H%M%S)_coverage_bc_to_ppo_main" \
STAGE1_TOTAL_UPDATES=100 \
STAGE2_TOTAL_UPDATES=500 \
EVAL_EPISODES=100 \
CUDA_VISIBLE_DEVICES=0 \
bash scripts/run_coverage_bc_to_ppo_main.sh
```

输出目录：

```text
outputs/training/bc_ppo/<RUN_TIMESTAMP>/coverage_stage1_safe_from_bc/
outputs/training/bc_ppo/<RUN_TIMESTAMP>/coverage_stage2_ppo_main/
```

辅助日志：

```text
outputs/debug_long/<RUN_TIMESTAMP>_coverage_bc_to_ppo_main/
```

## 如何判断 PPO 是否真正提升

不要只看 `eval_reward`。至少同时看：

- `eval_success_rate`
- `eval_reward`
- `eval_coverage_coverage_ratio` 或 `coverage_ratio_mean`
- `eval_coverage_repeated_coverage_ratio` 或 `repeated_coverage_ratio_mean`
- `eval_coverage_demand_revisit_excess` 或 `demand_revisit_excess_mean`
- `eval_collision_rate`
- `approx_kl`
- `kl_early_stop`
- `reference_action_mse`
- `policy_std_mean`
- `clip_frac`

解释标准：

- 如果 Stage2 的 `success_rate` 和 `coverage_ratio` 超过 BC/Stage1，同时 `collision_rate`、重复覆盖、`demand_revisit_excess` 没有恶化，才可以说 PPO-main 有净提升。
- 如果 Stage2 的 reward 提升但 success 或覆盖质量下降，说明 reward 优化方向可能不可靠。
- 如果 Stage2 没超过 BC/Stage1，说明当前 teacher/BC 是主要性能来源，PPO 暂时只起稳定微调作用。

## 对比脚本

使用：

```bash
/opt/conda/bin/python scripts/debug_long/compare_bc_safe_ppo_main.py \
  --bc-eval-csv outputs/training/bc/20260601_034310/debug_bc_coverage_factorized_group_route_target_agents_perm_coverage_N4_multi_channel_field_plus_task_id/eval_metrics.csv \
  --stage1-training-csv outputs/training/bc_ppo/<RUN_TIMESTAMP>/coverage_stage1_safe_from_bc/training_metrics.csv \
  --stage2-training-csv outputs/training/bc_ppo/<RUN_TIMESTAMP>/coverage_stage2_ppo_main/training_metrics.csv \
  --output-md outputs/debug_long/<RUN_TIMESTAMP>_coverage_bc_to_ppo_main/comparison.md
```

脚本会输出 Markdown 表格，并列出缺失字段。缺字段不会导致脚本失败。
