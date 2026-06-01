# 单任务专家 PPO 可靠性分析

本文记录当前 Wayffusion 单任务专家调试中的经验结论，重点解释为什么纯 PPO 不可靠、为什么当前大量使用 BC/DAgger warm-start，以及各任务目前采用的学习链路。

## 当前结论

当前主线架构已经统一到 `factorized_group`：

- centralized critic 仍看全局 state。
- actor 仍是共享 per-agent decoder。
- action 输出仍是 `[B, N, 2]`。
- 支持 `agent_mask` 和 variable N。
- joint logprob 仍聚合为每个 env 一个标量。

但训练 recipe 还没有统一。当前阶段目标是先得到可靠单任务专家，而不是强行坚持所有任务都用同一套训练流程。

## 各任务状态

`goal_nav`：

- 已调通。
- 当前有效链路是 success/DAgger BC warm-start + ultra-conservative PPO。
- 说明：目标导航有明确目标点，但从零 PPO 仍容易反复访问已完成目标或产生不稳定探索；BC 给 PPO 提供了可用初始分布。

`risk_nav`：

- 已调到可用。
- 当前有效链路是 learner-state DAgger BC + conservative PPO + reward-only safety/completion shaping。
- 关键结果：100-episode eval 约 `success_rate=0.65`，`collision_rate≈0.0208`。
- 说明：原始成功轨迹带有较高碰撞/风险偏置，低碰撞过滤又会丢失状态覆盖。有效修复是让 learner 自己 rollout，teacher relabel 这些 learner 会访问到的状态。

`formation`：

- 暂时可用，但还需要 seed repeat。
- 当前链路是 heuristic/DAgger BC + PPO。
- 说明：formation 的几何结构比较强，启发式数据能提供较稳定的监督信号。

`coverage`：

- 仍未调通。
- canonical h200 下目前稳定约 `success_rate≈0.20`、`coverage_ratio≈0.73`。
- 主要失败模式是重复覆盖已访问区域，`repeated_coverage_ratio≈0.99`。
- 已尝试 reward shaping、milestone reward、anti-revisit reward、success-only BC、weak-teacher DAgger、utility slot、frontier slot，均未形成 canonical 收敛。

## 为什么纯 PPO 不可靠

纯 PPO 不可靠的核心原因是：成功轨迹稀疏、joint action 空间大、credit assignment 难，并且 on-policy 数据效率低。

在 Wayffusion 中，一个 Gym agent 一次输出所有 UAV 的 joint waypoint action。N=4 时就是 8 维连续动作。随机探索很难自然产生“多个 UAV 分工合理、路径有效、少碰撞、按时完成”的轨迹。

`coverage` 和 `risk_nav` 都是长 horizon 任务。早期 PPO rollout 大多是失败轨迹，advantage 信号主要来自局部 reward，而不是完整成功行为。这会让 PPO 学到局部策略，例如：

- 安全但不到达。
- 到达但碰撞或风险暴露高。
- 覆盖一部分区域但反复扫同一片区域。
- 多个 UAV 抢同一个目标，而不是分工。

reward 也存在多目标冲突。单个任务里同时包含到达、避障、低风险、少重复覆盖、短路径等目标。PPO 从零训练时，很容易优化到局部折中，而不是任务成功。

centralized critic 虽然能看全局 state，但 actor 仍要给每个 agent 输出动作。没有好的初始分工时，critic 很难把成功或失败精确归因到某个 UAV 的某个局部决策。这个 credit assignment 问题在 `coverage` 上最明显。

PPO 还是 on-policy 算法。策略稍微漂到坏分布，后续采样数据也会变差，训练会进入自我强化的坏循环。此前 `risk_nav` 的失败就是典型例子：BC policy 有一定到达能力，但 PPO fine-tune 会把它推向高碰撞/低成功。

## 为什么 BC/DAgger 是必要 warm-start

BC/DAgger 当前不是为了“刷指标”，而是为了给 PPO 一个可探索的初始分布。

有效流程通常是：

```text
heuristic/success data -> BC or DAgger -> conservative PPO fine-tune -> canonical eval
```

BC 的作用是让 actor 先学到基本结构：

- 每个 UAV 大致该往哪里走。
- 多 UAV 不要完全抢同一目标。
- 长 horizon 任务中先进入有意义的状态分布。

DAgger 的作用是修复 BC 的分布偏移：

- 只用成功数据时，policy 一旦偏离成功轨迹，就不知道如何恢复。
- DAgger 采 learner 会实际访问到的状态，再用 teacher relabel，可以补上失败/偏航状态下的纠偏动作。

`risk_nav` 的修复证明了这点。加入 learner-state DAgger 后，BC success 先提升；再用低学习率、低 KL、clamped std、reference regularization 的 PPO 微调，才能在不破坏行为的前提下改善安全和完成度。

## coverage 为什么仍然困难

coverage 的问题不是 PPO 完全不能更新，而是缺少稳定的低重复、高效率 h200 成功轨迹。

已验证的问题：

- 现有 `coverage_expert_v2/v3/v4` 在 canonical h200 下 teacher 自身成功率很低。
- success-only BC 会过拟合少量成功轨迹，闭环泛化反而变差。
- utility/frontier 这类 stateless action bias 会改变已有闭环行为，容易低于 baseline。
- terminal anti-revisit penalty 太晚，不能教会策略中途如何重新分配 frontier。
- 仅放大奖励权重会增加 value fitting 噪声，不会自动产生更好的空间分工。

因此 coverage 下一步更可信的路线是：

- 构建真正 h200 高成功 route-planning/sweep teacher。
- 或者实现带状态管理的 persistent target memory，让每个 agent/group 在多步内坚持同一 frontier，而不是每步重新即时分配。
- 无论采用哪条路线，最终都必须回到 canonical `configs/env/coverage.yaml`、`success_ratio=0.82`、h200 做验证。

## 当前建议

短期不要再重复以下路线：

- 从同一个 coverage BC checkpoint 继续调 PPO 小超参。
- 再用现有 `coverage_expert_v2/v3/v4` 做 DAgger。
- 再做同类 scalar reward-only shaping。
- 再调当前 stateless frontier bias 强度。

更有价值的下一步：

1. 先做 h200 sweep/route-planning teacher，要求 teacher 自身 canonical success 明显高于 `0.5`。
2. 用该 teacher 生成 coverage BC/DAgger 数据。
3. 再用 `factorized_group` conservative PPO 微调。
4. 最终用 canonical 100-episode eval 判定是否真正调通。

## 2026-06-01 coverage 后续修正

最新 coverage 调试允许修改 coverage 训练环境和 reward。基于 `repeated_coverage_ratio≈0.99` 的失败模式，新增了更直接的强惩罚：

- `repeated_demand_coverage`：每一步惩罚重复覆盖已经满足的 demand cells，信号比 timeout penalty 更早。
- `terminal_revisit_excess`：终局惩罚 demand 区域累计超额访问量，用来区分高效 sweep 和原地反复覆盖。

同时 `factorized_group` 新增可选 sequential group context：后续 group token 会接收前序 group 输出目标坐标的信息。这保留 centralized critic 和 `[B, N, 2]` joint action 输出，但让 group-level 决策从“并行独立提案”变为“按组有条件提案”，更符合分组航点分配。

还修复了一个重要实验可信度问题：`coverage_frontier_*` 配置此前没有从 policy factory 传入 policy 构造函数，因此 frontier-slot phase 的配置开关实际没有生效。之后所有 frontier 结论需要以修复后的 run 为准。

## 2026-06-01 最新四任务状态

截至 phase63，单任务专家训练结论已经更新如下。旧章节里关于 coverage/formation “仍未调通”的描述是中途状态，应以本节为准。

`goal_nav`：

- 状态：已在 task-target agent observation 模式下修通到 robust PPO expert。
- 架构：`factorized_group`，centralized critic + per-agent/group waypoint actor，输出仍为 `[B, N, 2]`。
- 关键修复：新增统一的 `include_task_targets_in_agents`，每个 UAV 的 agent token 附加当前分配目标的 delta 和 normalized position。goal_nav/risk_nav 使用剩余目标的 Hungarian assignment；剩余目标少于 UAV 时，未分配 UAV 获得 idle target，避免多个 UAV 继续挤向同一个目标。
- 为什么可行：phase66 诊断显示旧 goal_nav 失败主要是 collision/recovery 闭环问题，动作对剩余目标的对齐度在失败 episode 中崩掉；task target hint 给 actor 显式分工信息，直接降低 late-stage crowding。
- 推荐 checkpoint：`outputs/training/bc_ppo/20260601_phase68_goalnav_task_targets_ppo/phase68_goalnav_task_targets_ppo/checkpoints/checkpoint_0040.pt`。
- five-seed 40-episode：`success_rate=0.965`，`goal_coverage_ratio=0.984333`，`pair_collision_rate=0.005324`。
- seed23 100-episode：`success_rate=0.89`，`goal_coverage_ratio=0.9395`，`pair_collision_rate=0.016639`。
- 注意：phase68 的 `checkpoint_best_eval.pt` 指向 update60，但独立 seed23 100ep 下 update40 更稳，因此 robust expert 优先用 `checkpoint_0040.pt`。

`coverage`：

- 状态：在新的 per-agent route-target observation/decision mode 下已修通。
- 关键修改：`include_route_targets_in_agents=true` 时，每个 agent token 追加 route target delta 和 target position，使 actor 能看到“自己当前应坚持的覆盖航点”。
- 为什么可行：coverage 的主要失败不是不知道高概率区域在哪里，而是缺少跨步持久分工，导致重复覆盖；per-agent route target 把持久目标显式放进 observation，仍保持 centralized 全局信息和 `[B, N, 2]` joint action。
- 最好 checkpoint：`outputs/training/bc_ppo/20260601_phase60_coverage_route_target_agents_ppo/phase60_coverage_route_target_agents_ppo/checkpoints/checkpoint_best_eval.pt`。
- seed 7 100-episode：`success_rate=0.72`，`coverage_ratio=0.801508`，`collision_rate=0.002905`，`path_length=0.881318`，`demand_revisit_excess=23.424259`。
- seed 23 100-episode：`success_rate=0.68`，`coverage_ratio=0.795507`，`collision_rate=0.004842`，`path_length=0.879962`，`demand_revisit_excess=23.553093`。
- 注意：这不是原始 `configs/env/coverage.yaml` 的无改动 canonical 结果，而是用户允许后的 coverage 任务/observation 设计修复。

`risk_nav`：

- 状态：已修通到可复现的中等成功专家。
- 关键链路：learner-state DAgger BC + conservative PPO + reference regularization。
- 为什么可行：risk_nav 同时要求到达和避险，纯 PPO 早期会在“安全但不到达”和“到达但高风险”之间振荡；DAgger 用 learner 自己访问到的状态做 teacher relabel，能修复 BC 的分布偏移。
- 最好 checkpoint：`outputs/training/bc_ppo/20260530_phase41_risknav_factorized_group_dagger_safe/phase41_risknav_factorized_group_dagger_safe/checkpoints/checkpoint_best_eval.pt`。
- seed 7 100-episode：`success_rate=0.65`，`goal_coverage_ratio=0.85`，`collision_rate=0.020797`，`path_length=0.698778`，`cumulative_risk_exposure=33.393036`。
- seed 23 100-episode：`success_rate=0.65`，`goal_coverage_ratio=0.841667`，`collision_rate=0.018796`，`path_length=0.712840`，`cumulative_risk_exposure=34.320727`。
- 注意：risk_nav 的成功率可复现，但安全/风险暴露还不是很干净；如果继续优化，重点应是降低 risk exposure 和 safety violation，而不是只抬 success。

`formation`：

- 状态：已修通。
- 关键修复：template-aware success metric。原逻辑对所有模板都要求 radius error 和 angular uniformity，这对 `line` 结构性错误，对 `arc` 过严。
- 为什么可行：formation policy 已经能把 UAV 放到 slots 附近；失败主要来自 success 判定把 line/arc 当 full-circle radial template 评估。修复没有降低 tolerance，只是让每种模板使用匹配的几何判据。
- 最好 checkpoint：`outputs/training/bc_ppo/20260530_phase39_formation_factorized_group_ppo/phase39_formation_factorized_group_ppo/checkpoints/checkpoint_best_eval.pt`。
- seed 7 100-episode：`success_rate=0.77`，`formation_error=0.072942`，`collision_rate=0.002313`，`path_length=0.446007`。
- seed 23 100-episode：`success_rate=0.78`，`formation_error=0.073723`，`collision_rate=0.002313`，`path_length=0.439148`。

当前总体判断：

- 四个单任务专家都已经在新模式下达到“可训练出专家模型”的状态。
- coverage、risk_nav、formation 都已经补齐 repeat 验证；goal_nav 的旧 seed robustness 短板已由 phase67/68 task-target branch 明显修复。
- 这些结果依赖 BC/DAgger warm-start 和 conservative PPO；纯 PPO 从零仍不可靠，原因是 joint action credit assignment、稀疏成功轨迹和长 horizon 分工问题没有消失。
