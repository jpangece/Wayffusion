# Specialist PPO 调试总结

本文总结本轮对 Wayffusion 单任务 specialist policy 收敛问题的排查、代码修改、实验路径和当前结论。

目标任务：

- `goal_nav`
- `coverage`
- `risk_nav`
- `formation`

目标不是改 benchmark 定义本身，而是在不降低 success threshold、不删除核心任务要素的前提下，把 specialist PPO 训练推进到可用收敛区间。

## 1. 问题判断

起始问题不是统一的“PPO 完全坏了”，而是不同任务卡在不同层面：

- `goal_nav`：
  - PPO 从零开始会学到一点局部追踪，但容易回退到 `success_rate≈0`
  - 说明 dense reward 有信号，但策略会被错误目标吸引或被分布偏移拖垮
- `coverage`：
  - PPO 能学到非随机覆盖行为，但长期卡在 `coverage_ratio≈0.45`
  - 说明探索和 credit assignment 有改善空间，但不是单纯 reward 为零
- `risk_nav`：
  - heuristic 本身够强，问题更像是“如何把 heuristic 行为稳定蒸馏进 policy”
- `formation`：
  - heuristic 很强，PPO/BC 需要更强的结构偏置才能稳定保持几何队形

## 2. 核心代码修改

### 2.1 `goal_nav` / `risk_nav` 的剩余目标可观测性修复

文件：

- [tasks/goal_nav.py](/workspace/Wayffusion/Wayffusion/tasks/goal_nav.py)
- [tasks/risk_nav.py](/workspace/Wayffusion/Wayffusion/tasks/risk_nav.py)

修改：

- `goal_reward` 不再固定为 reset 时的静态目标图
- 每步根据“尚未完成的目标”重建 `goal_reward`
- `goal_progress` 和 `distance` 相关 shaping 也改成只针对剩余目标

为什么要改：

- 原先已经完成的目标仍然会在 observation 里发亮
- policy 很容易继续追逐已经完成的目标，导致重复访问和 coverage 上限过低
- 这是观测错误，不是“降低任务难度”

### 2.2 PPO 诊断与稳定性增强

文件：

- [algorithms/ppo.py](/workspace/Wayffusion/Wayffusion/algorithms/ppo.py)

修改：

- trainer 初始化和 checkpoint 加载后都会强制 `log_std` clamp
- 新增 rollout 统计：
  - raw reward / normalized reward
  - action saturation
  - terminal success / coverage / collision / path
- 新增 update 统计：
  - `approx_kl`
  - `clip_frac`
  - `ratio_mean`
  - `grad_norm`
  - `explained_variance`
- 支持 `target_kl` early stop

为什么要改：

- 对这种多 UAV joint action 任务，仅看 `eval_reward` 不足以判断训练是否真的进入稳定区
- conservative fine-tune 非常依赖 `log_std` 在加载 checkpoint 后仍处于预期区间

### 2.3 Evaluation 记录增强

文件：

- [utils/evaluation.py](/workspace/Wayffusion/Wayffusion/utils/evaluation.py)

修改：

- eval episode 现在累计记录 episode 级 reward components

为什么要改：

- 需要明确知道 reward 改善来自于：
  - progress
  - completion
  - collision 下降
  - 还是别的副作用

### 2.4 BC warm-start 支持

文件：

- [scripts/train_bc.py](/workspace/Wayffusion/Wayffusion/scripts/train_bc.py)

修改：

- 增加 `--init_checkpoint`

为什么要改：

- DAgger 和 success-heavy repair 都需要在旧 policy 上继续做 imitation，而不是每次从随机初始化开始

### 2.5 Coverage 结构偏置

文件：

- [policies/cnn_deepsets_policy.py](/workspace/Wayffusion/Wayffusion/policies/cnn_deepsets_policy.py)
- [policies/__init__.py](/workspace/Wayffusion/Wayffusion/policies/__init__.py)

新增可选能力：

- `coordination_repulsion_strength`
- `use_spatial_action_head`
- `spatial_action_strength`
- `spatial_target_suppression_strength`
- `spatial_target_suppression_sigma`
- `use_angular_slot_embeddings`
- `slot_embedding_strength`

为什么要改：

- `coverage` 的核心瓶颈不是“找不到高价值区域”，而是多个 UAV 会反复拥挤到相近区域
- 纯 per-agent decoder 缺少足够强的几何结构偏置
- 这些改动都是默认关闭的可选分支，不破坏已有主干

### 2.6 Debug dataset 原子写盘

文件：

- [scripts/debug_long/generate_success_expert_dataset.py](/workspace/Wayffusion/Wayffusion/scripts/debug_long/generate_success_expert_dataset.py)
- [scripts/debug_long/collect_dagger_dataset.py](/workspace/Wayffusion/Wayffusion/scripts/debug_long/collect_dagger_dataset.py)
- [scripts/debug_long/collect_success_policy_dataset.py](/workspace/Wayffusion/Wayffusion/scripts/debug_long/collect_success_policy_dataset.py)
- [scripts/debug_long/generate_coverage_expert_v2_dataset.py](/workspace/Wayffusion/Wayffusion/scripts/debug_long/generate_coverage_expert_v2_dataset.py)

修改：

- 用临时文件写完后再 `os.replace(...)`

为什么要改：

- 长时运行时，连接中断或误触关线程会留下损坏 `.npz`
- 原子写盘可以保证：
  - 文件完整
  - 或根本不存在

## 3. 任务级实验结论

### 3.1 `goal_nav`

最有效路径：

1. 修复 remaining-goal observability
2. success-heavy BC
3. DAgger-style learner-state relabeling
4. ultra-strict PPO fine-tune

当前 canonical 结果：

- [outputs/training/bc_ppo/20260528_goalnav_dagger_finetine_ultra_strict/goalnav_dagger_finetine_ultra_strict/eval_metrics.csv](/workspace/Wayffusion/Wayffusion/outputs/training/bc_ppo/20260528_goalnav_dagger_finetine_ultra_strict/goalnav_dagger_finetine_ultra_strict/eval_metrics.csv)

关键指标：

- `success_rate_mean=0.8`
- `goal_coverage_ratio_mean=0.9425`
- `collision_rate_mean=0.0275`
- `return_mean=10.1225`

结论：

- `goal_nav` 已修通

### 3.2 `coverage`

从零 PPO 的问题：

- 可以学到部分覆盖结构
- 但长期卡在 `coverage_ratio≈0.45`
- success 几乎始终为 `0`

关键进展：

1. 设计更强的 `coverage expert v2`
2. 用 expert-v2 训练 `spatial-head BC`
3. 在 BC checkpoint 上做 ultra-strict PPO
4. 继续做 success-heavy / DAgger 风格强化
5. 再尝试 repulsion、suppression、reward-focus、sector bias 等轻量结构偏置

关键中间结果：

- `expert-v2` 探针优于旧 heuristic
- `spatial-head BC`：
  - `coverage_ratio_mean≈0.661`
  - `success_rate_mean≈0.05`
  - `collision_rate_mean≈0.00094`
- `spatial-head PPO`：
  - 连续多个 eval 点达到 `success≈0.10`
  - `coverage_ratio≈0.67~0.68`
  - `collision≈0.00025~0.00063`
- `coverage success-heavy BC -> PPO` 没有超过这条主线
- `repulsion + reward-focus` 没有把 success 稳定性再推进
- `sector target bias` 也没有优于 plain spatial-head branch

但最终点又会掉回 `success=0`

当前最值得复用的 coverage run：

- [outputs/training/bc_ppo/20260529_phase24_coverage_successonly_v2_bestfinal/phase24_coverage_successonly_v2_bestfinal](/workspace/Wayffusion/Wayffusion/outputs/training/bc_ppo/20260529_phase24_coverage_successonly_v2_bestfinal/phase24_coverage_successonly_v2_bestfinal)

当前 best-eval checkpoint：

- `checkpoints/checkpoint_best_eval.pt`
- best update: `120`
- `success_rate=0.15`
- `coverage_ratio≈0.673`
- `collision≈0.00019`

结论：

- `coverage` 已经从“不收敛”推进到“能稳定进入成功区间，但未最终锁住成功”
- 当前最优方向是：
  - `expert-v2 dataset -> spatial-head BC -> spatial-head PPO`
- 已验证“继续叠轻量偏置”的收益很有限
- 如果继续推进，下一步应升级到更强的 group-level coordination actor

### 3.3 `risk_nav`

最有效路径：

1. full-state heuristic dataset
2. BC
3. ultra-strict PPO

当前最好结果：

- [outputs/training/bc_ppo/20260528_phase13_risk_nav_ultra/phase13_risk_nav_ultra/eval_metrics.csv](/workspace/Wayffusion/Wayffusion/outputs/training/bc_ppo/20260528_phase13_risk_nav_ultra/phase13_risk_nav_ultra/eval_metrics.csv)

关键指标：

- `success_rate_mean=0.6`
- `goal_coverage_ratio_mean≈0.829`
- `collision_rate_mean≈0.049`

结论：

- `risk_nav` 已修通

### 3.4 `formation`

最有效路径目前是：

1. full-state heuristic dataset
2. slot-aware / spatial-head BC
3. ultra-strict PPO

当前最好结果：

- [outputs/training/bc_ppo/20260528_phase14_formation_ultra/phase14_formation_ultra/eval_metrics.csv](/workspace/Wayffusion/Wayffusion/outputs/training/bc_ppo/20260528_phase14_formation_ultra/phase14_formation_ultra/eval_metrics.csv)

关键现象：

- `update 40/60/80` 能稳定达到 `success≈0.4`
- 但 final eval 回到 `success=0.3`

结论：

- `formation` 已经达到可用且接近 heuristic 上限
- 但还不能像 `goal_nav / risk_nav` 那样判定为完全修通

## 4. 为什么这些调参/结构改造是合理的

### 4.1 Conservative PPO 是必要的

原因：

- 这些任务不是“完全没学到”，而是 warm-start 后容易被 PPO 更新冲坏
- 所以采用：
  - 更小 `learning_rate`
  - 更小 `clip_coef`
  - 更低 `target_kl`
  - 固定低 `log_std`

这不是玄学，而是为了让 PPO 先保住已有解，再慢慢扩展。

### 4.2 success-heavy BC / DAgger 是必要的

原因：

- heuristic 的 full-state 数据只能教“平均行为”
- 真正缺的是：
  - learner 自己会走到的状态
  - 成功轨迹尾段的 finishing behavior

所以：

- `goal_nav` 用 DAgger 真正修通
- `coverage / formation` 也显示出 success-heavy 数据比 generic heuristic 更有价值

### 4.3 Coverage 需要更强几何偏置

原因：

- coverage 的主问题不是目标找不到，而是 agent 会重复覆盖、互相挤压
- 所以单纯 reward scale 提升不够
- 更有效的是：
  - repulsion
  - spatial action head
  - slot / angular role bias

这些都在引导 “分工覆盖” 这个结构，而不是仅靠 loss 自己悟出来。

当前最新判断：

- `repulsion`
- `spatial action head`
- `slot / angular role bias`

这些轻量偏置确实把 coverage 从 `0.45` 平台推到了 `0.67` 左右，但还没把 success 稳稳锁住。  
因此下一步合理动作不是继续随意磨一个系数，而是升级到更明确的组级协调形式。

## 5. 当前任务总体状态

- `goal_nav`: 已修通
- `risk_nav`: 已修通
- `coverage`: 部分修通，已进入非零 success 区，但未最终稳定
- `formation`: 部分修通，已达 heuristic 水平附近，但 final 稳定性仍不足

## 6. 当前建议的后续顺序

1. 继续 `coverage` 的 success-heavy / DAgger repair
2. 继续 `formation` 的 success-policy repair
3. 覆盖这两条后，再刷新 canonical specialist runs 和 verification 文档

## 7. 2026-06-01 phase60-63 更新

本文件前面的状态是早期阶段总结。phase60-63 后，四个单任务专家状态更新为：

- `goal_nav`: 已在 phase67/68 的 task-target agent observation 模式下修通到 robust PPO expert；phase68 checkpoint_0040 five-seed 40ep `success_rate=0.965`，seed23 100ep `success_rate=0.89`。旧 phase36/65 只保留为历史对照。
- `coverage`: 已在 per-agent route-target observation/decision mode 下修通；两个 100-episode seed 分别 `success_rate=0.72` 和 `0.68`，coverage ratio 约 `0.80`，collision 低于 `0.005`。
- `risk_nav`: 已补齐 repeat-seed 验证；seed 7 和 seed 23 的 100-episode `success_rate` 都是 `0.65`，collision 约 `0.019-0.021`。
- `formation`: 已通过 template-aware success metric 修复；两个 100-episode seed 分别 `success_rate=0.77` 和 `0.78`。

这轮最重要的工程结论是：新决策模式不是“单个共享航点分配所有人”，而是 centralized 信息下的 per-agent/group waypoint actor。critic 仍看全局 state，actor 共享参数但每个 UAV/group 得到自己的 token/context，因此输出是可扩展的 `[B, N, 2]` joint waypoint action。

coverage 的关键变化不是继续调 PPO 小超参，而是把持久 route target 显式编码进每个 agent observation。这样 PPO/BC 学到的是“每个 agent 坚持自己的覆盖路线”，而不是每步从共享 field 里重新抢同一个高值区域。

formation 的关键变化不是降低阈值，而是修复 success 判据：`line` 和 `arc` 不是 full-circle template，不应使用和 `circle/diamond` 相同的 angular-uniformity 条件。

risk_nav 的关键链路仍是 learner-state DAgger：让 teacher 标注 learner 实际访问到的状态，再用 conservative PPO 微调。它已可复现，但如果继续优化，优先目标应是降低 risk exposure / safety violation，而不是只追 success。

如果进入下一轮单任务 hardening，优先级应是：

1. `risk_nav` 降低 risk exposure / safety violation，并优先尝试同样的 `include_task_targets_in_agents` 路线。
2. `coverage` 降低 repeated coverage 和 demand revisit excess。
3. 将 task-target convention 迁移成后续多任务训练的统一 observation 约定。
