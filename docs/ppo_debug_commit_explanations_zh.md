# PPO / Coverage / Target-Assignment 调试说明

本文按 commit 和主题解释近期 Wayffusion 单任务专家调试中的关键实现、数学含义和实验效果。结论基于本地 `git show`、当前代码和 `agent_memory/cards/03_recent_modifications.md`。

## 1. `0044742` 的 `SquashedNormal / log_std`

`SquashedNormal` 是把高斯策略压到环境动作范围 `[-1, 1]` 的分布。策略先输出未约束动作：

$$
u \sim \mathcal{N}(\mu, \sigma)
$$

$$
a = \tanh(u)
$$

所以真实 action 是 $a$。PPO 计算 log-prob 时不能直接用 $\mathcal{N}(a)$，必须做变量变换：

$$
\log \pi(a \mid s)
=
\log \mathcal{N}(\operatorname{atanh}(a); \mu, \sigma)
-
\sum_i \log(1 - a_i^2 + \epsilon)
$$

第二项是 `tanh` 的 Jacobian correction。否则 PPO 的 ratio 会算错：

$$
r =
\exp(\log \pi_{\text{new}}(a \mid s) - \log \pi_{\text{old}}(a \mid s))
$$

`log_std` 是高斯标准差的对数：

$$
\sigma = \exp(\log \sigma)
$$

它控制探索强度。

| 改动 | 含义 | 为什么重要 |
| --- | --- | --- |
| `log_std_min/max` | 把 `log_std` clamp 在范围内 | 防止探索方差爆炸或塌缩 |
| `base_entropy()` | 用未 squash 的高斯熵 $H(\mathcal{N})$ | 作为稳定 exploration 诊断和 entropy bonus |
| `get_action_and_value()` 内 clamp | 每次采样/评估都用受限 std | PPO fine-tune 不会被 checkpoint 里的异常 std 带偏 |

高斯熵为：

$$
H =
\sum_i \frac{1}{2}\log(2\pi e\sigma_i^2)
$$

后续 `64b9789` 又把 `log_std clamp` 扩展到 trainer init 和 checkpoint load 后，因为 BC/PPO checkpoint 继续训练时最容易把 std 带到不符合当前 fine-tune 配置的范围。

## 2. PPO 诊断指标

这些指标用于判断 PPO 是否真的有效更新，而不是只看 reward。

| 指标 | 数学/实现含义 | 怎么看 |
| --- | --- | --- |
| `approx_kl` | 当前策略和 rollout 老策略的近似 KL：$\mathbb{E}[(r - 1) - \log r]$ | 太低说明更新太弱；太高说明策略跳太远，容易破坏 BC 行为 |
| `clip_frac` | $\mathbb{E}[\mathbf{1}(|r - 1| > \epsilon_{\text{clip}})]$ | 接近 0 表示 PPO clip 基本没触发；太高表示大量样本被 clip，更新过猛 |
| `grad_norm` | `clip_grad_norm_` 返回的总梯度范数，通常是裁剪前 norm | 很高代表训练不稳；长期极低代表学习信号弱 |
| `explained_variance` | $1 - \frac{\mathrm{Var}(R - V)}{\mathrm{Var}(R)}$ | critic 拟合度；接近 1 好，接近 0 或负数说明 value 基本没解释 return |
| `rollout terminal stats` | on-policy rollout 中真实结束 episode 的终局指标 | 比 eval 更能看 PPO 当前采样分布是否已经崩了 |

`rollout_terminal_goal_coverage` 在 `goal_nav / risk_nav` 是目标完成比例，在 `coverage` 会用 `coverage_ratio` 兜底。

`rollout_terminal_collision_rate / path_length / success_rate` 是训练采样期的终局表现，不等同于 deterministic eval，但能提前暴露 PPO 正在把策略推坏。

## 3. `64b9789` 的 `goal_nav / risk_nav` remaining-goal 修复

旧逻辑的问题是：

- `goal_reward_map` 在 reset 时包含所有 goals；
- 目标达成后仍留在 field 里；
- reward shaping 里的 `goal_progress` 和 `distance_penalty` 也按所有 goals 算。

这会让策略被已经完成的目标继续吸引，尤其多 UAV 后期容易挤向旧目标。

修复后只看未完成目标：

$$
G_{\text{rem}} = \{g_k \mid \text{goal\_reached}_k = \text{false}\}
$$

field 改成：

$$
\text{goal\_reward} = \operatorname{gaussian\_map}(G_{\text{rem}})
$$

progress 改成：

$$
\text{progress}
=
C(G_{\text{rem}}^{\text{before}}, P_{\text{old}})
-
C(G_{\text{rem}}^{\text{before}}, P_{\text{new}})
$$

更新后：

$$
\text{goal\_progress}
=
C(G_{\text{rem}}^{\text{after}}, P_{\text{new}})
$$

distance penalty 也只按 remaining goals 算。

`goal_nav` 和 `risk_nav` 都改成 remaining-goal field 和 remaining-goal shaping。`risk_nav` 在此基础上仍保留风险惩罚：

$$
r =
r_{\text{progress}}
+
r_{\text{reached}}
+
r_{\text{coverage}}
+
r_{\text{distance}}
+
r_{\text{risk}}
+
r_{\text{repeated}}
+
r_{\text{success}}
$$

实验结论：

- remaining-goal 修复是必要条件；
- 但它不是充分条件；
- 真正让 `goal_nav` 进入 `success≈0.8` 的是 `success/DAgger BC -> conservative PPO`，不是纯 PPO 从零。

## 4. `9f1a6d0` 到 `62fcd66` 的 coverage 结构尝试

这一段是 coverage 最密集的结构探索。

| 方法 | 数学/机制 | 效果 |
| --- | --- | --- |
| `coverage expert v2` | 构造 utility：$U = 1.8D + 0.6P - 0.4V - 0.2R$，从 top-k cell 里贪心选高 utility 且相互分散的 targets，再 Hungarian 分配给 UAV | 比原 heuristic 好，probe `coverage≈0.579`；BC 到 `coverage≈0.512`，但 success 仍低 |
| coordination repulsion | 给 actor mean 加排斥项：$b_i = \alpha \cdot \operatorname{norm}\left(\sum_j \frac{p_i - p_j}{\lVert p_i - p_j\rVert^2}\right)$ | 降低拥挤，phase8 peak `coverage≈0.578`，后续 peak `≈0.664`，但 success 仍接近 0 |
| spatial action head | CNN field tokens $z_t$ 加坐标；agent query $q_i$ 对 tokens softmax：$w_{it} = \operatorname{softmax}\left(\frac{q_i^\top z_t}{\sqrt d}\right)$；target $c_i = \sum_t w_{it}\operatorname{coord}_t$；action bias $\alpha(c_i - p_i)$ | spatial-head BC 到 `coverage≈0.661, success≈0.05`；PPO 中期 `coverage≈0.682, success≈0.10`，但 final 不稳 |
| spatial target suppression | 后一个 agent 选 target 时对前面 agent 已选区域加核抑制：$\text{logits}_i(x) \leftarrow \text{logits}_i(x) - \beta \sum_j \exp\left(-\frac{\lVert x - x_j\rVert^2}{\sigma}\right)$ | 合理但没赢，coverage/reward 低于 plain spatial-head |
| sector bias / angular slots | 按 UAV 相对 swarm centroid 的角度排序，给每个 rank 一个 sector；logit 加角度相近 bonus | 没超过 spatial-head；说明简单扇区分配不足以形成闭环 coverage |
| global slot head | 从全局 state 直接预测 $N$ 个 slot 坐标：$S = \tanh(Wh)$，按 angular rank 分给 agents | 兼容四任务，但 coverage BC/PPO 没超过 spatial-head |
| global spatial slot head | slot query attend 到 spatial tokens：$S_k = \sum_t \operatorname{softmax}(q_k^\top z_t)\operatorname{coord}_t$ | 比纯 global slot 更空间化，但仍没超过 spatial-head |
| utility slot head | 从 field 算 $U = w_dD + w_pP + w_vV + w_rR + w_oO + w_aA$，softmax 成 target，再加 sector/suppression | untrained 很差；BC `success≈0.10, coverage≈0.666`；PPO 直接掉到 `success=0` |
| permutation-invariant BC loss | 不固定 UAV index 匹配 expert action，而是在 waypoint 空间取最小排列损失：$\min_{\pi}\frac{1}{N}\sum_i \lVert (p_i + a_i^{\text{pred}}s) - (p_{\pi(i)} + a_{\pi(i)}^{\text{exp}}s)\rVert^2$ | coverage BC 突破：`success≈0.25, coverage≈0.717`；说明固定编号监督严重伤害多 UAV 分配 |
| factorized_group | 全局 state 生成 group tokens；agent soft-assign 到 group；group 选 spatial coord；actor mean = decoder residual + group target bias | goal_nav 先验证成功；coverage h300 可到 `success≈0.40-0.50`，但 h200 仍弱 |
| expert v3 | local frontier utility：$U = (3D + 0.9P)(1 - V) - 0.9R - 2.2O - 0.35A$，加 local/spread/distance score 和 repulsion | teacher success 约 `0.07`，不是 solved teacher，但比 v2 更适合补成功数据 |
| expert v4 | stateful sector persistent targets，按 sector/utility/spread/distance 选目标并保持若干步 | direct eval success `0`，不如后来的 V5 |
| best-eval checkpoint | 保存 `checkpoint_best_eval.pt`，按 `eval_success_rate` 选，reward 破平 | 对 coverage/formation 很重要，因为常见“中期好、final 掉” |

核心结论：

- coverage 不是完全学不到；
- 它缺的是稳定的跨步目标分配；
- `permutation BC loss` 和 `spatial/slot/group` 几何 bias 有帮助；
- 真正修复还要靠后来的 route target。

## 5. `402e750` 的 frontier/utility slot、risk/formation factorized_group config、coverage anti-repeat

`402e750` 是把新架构推进到四任务，并继续压 coverage。

| 改动 | 数学/机制 | 效果 |
| --- | --- | --- |
| factorized_group risk/formation configs | 同一个 centralized critic + group-token actor，给 risk/formation 做 BC/PPO 配置 | formation 初步可用，BC/PPO `success≈0.44-0.50`；risk 初期失败，BC `0.45`、PPO 50ep 掉到 `0.24` |
| risk DAgger safe PPO | learner rollout states 用 heuristic relabel，再 conservative PPO + reference loss | 后续 phase41 修到 two-seed `success=0.65` |
| coverage completion reward | $\text{shortfall} = \max(C^\* - C, 0)$；失败 timeout penalty：$w_f \mathbf{1}[t=T \land \neg \text{success}]$ | 修正“部分覆盖也吃大量 level reward”的问题，但 h200 50ep 仍 `success≈0.16-0.20` |
| milestone reward | $\sum_l b_l \mathbf{1}[C \ge \tau_l \land \text{not paid}_l]$ | 中间 credit 更清楚，但 canonical 100ep 只有 `success≈0.17` |
| terminal repeated coverage | timeout 时加 $w \cdot \text{repeated\_coverage\_ratio}$ | phase46 canonical `success≈0.20`，重复覆盖仍 `≈0.991` |
| frontier slot | $U_f = w_d D(1 - V)^p + w_p P(1 - V)^p + w_o O + w_a A$，再 softmax/sector/distance/suppression 得 target | phase47/48 低强度也失败；会扰动已有 BC 闭环 |
| utility slot | $U = w_dD + w_pP + w_vV + w_rR + w_oO + w_aA$，偏向高价值低访问区域 | direct PPO 改 prior 太大，phase42 `success=0`；utility BC 也 `success=0` |

重要负结论：

- coverage 的 reward-only 或 stateless slot bias 不能自动教会“持久扫路线”；
- terminal anti-repeat 太晚；
- frontier/utility 每步重算，容易破坏已经学到的闭环。

## 6. `3c50ca8` 的 route hint / route target

`route hint` 和 `route target` 是两个不同层次。

### 6.1 Route Hint

`route hint` 是共享空间热图。

CoverageTask 在 env/task state 里维护每个 agent 的持久路线：

```text
demand cells -> x-stripes -> each UAV owns one stripe
within stripe -> y bins -> alternating sweep order
route_i = [c_i1, c_i2, ...]
target_i = route_i[index_i]
if near target_i or target demand fulfilled: index_i += 1
```

然后把所有当前 targets 画成一个 Gaussian map：

$$
H(x)
=
\sum_i \exp\left(
-\frac{\lVert x - \text{target}_i\rVert^2}{2\sigma^2}
\right)
$$

这个 map 被塞进已有 `formation_template` channel，避免改变 task field shape。

问题是：它是共享热图，没有告诉第 $i$ 个 UAV 应该追哪个 peak。

route-hint action head 从 $H$ 里 softmax 选点：

$$
\text{logits}_i(x)
=
\tau H(x)
-
\lambda \lVert x - p_i\rVert
$$

加 suppression 后仍然不够。

效果：

- 直接加 route hint 到旧 checkpoint 很差；
- route-hint-head BC 只有 `success≈0.26`；
- 不如 V5 BC。

### 6.2 Route Target

`route target` 是 per-agent 显式目标。

它不是共享 heatmap，而是每个 UAV token 直接附加自己的当前 route target：

$$
\text{agent}_i =
[x, y, v_x, v_y, \text{battery}, \text{role}, d_x, d_y, t_x, t_y]
$$

其中：

$$
[d_x, d_y]
=
\operatorname{clip}\left(
\frac{\text{target}_i - p_i}{\text{map\_size}},
-1,
1
\right)
$$

$$
[t_x, t_y]
=
\operatorname{clip}\left(
\frac{\text{target}_i}{\text{map\_size}},
0,
1
\right)
$$

它保留 centralized 全局信息，同时给每个 agent 明确“你现在该坚持的目标”。

效果是决定性的：

- BC `success=0.66`
- PPO 稳定保持 `0.667`
- independent 100ep seed7 `0.72`
- independent 100ep seed23 `0.68`

## 7. UAV 的 route target 信息是什么？

对 coverage 来说，UAV 的 route target 就是 `CoverageTask` 持久路线状态里该 UAV 当前要去的世界坐标：

```python
task_state["route_hint_targets"][i]
```

它不是 policy 输出，也不是 reward，也不是隐藏在 policy 里的 recurrent memory。

它是环境 observation 的一部分，使当前 observation 近似包含 route pointer。

没有 route target 时，policy 只能看到 visited map 和热图，难以知道自己该沿哪条持久路线继续扫。

后来这个思想泛化成 `include_task_targets_in_agents`：

| 任务 | task target 来源 |
| --- | --- |
| `goal_nav` | Hungarian assignment 到 remaining goals |
| `risk_nav` | Hungarian assignment 到 remaining goals |
| `coverage` | persistent route targets |
| `formation` | Hungarian assignment 到 formation slots |

## 8. `55356b2` 和 `51c2068`

### 8.1 `55356b2`

`55356b2` 是四任务状态收束和验证 commit。

它做了：

- formation success 修复；
- risk repeat-seed；
- coverage repeat-seed；
- goal seed audit；
- 四任务专家基线文档更新。

formation 旧 success 逻辑对所有模板都要求：

$$
\text{formation\_error} \le \text{tol}
$$

$$
\text{radius\_error} \le \text{tol}
$$

$$
\text{angular\_uniformity} \ge 1 - \text{angular\_tol}
$$

这对 `line` 是错的，因为 line 本来就不可能 full-circle angular uniform。

新逻辑：

```text
all templates:
  formation_error <= tol

circle / diamond:
  additionally radius_error <= tol
  additionally angular_uniformity >= 1 - angular_tol

arc:
  additionally radius_error <= tol

line:
  slot matching error only
```

效果：

| checkpoint | seed | episodes | success |
| --- | --- | --- | --- |
| phase39 old metric | seed7 | 100 | `0.50` |
| phase39 template-aware metric | seed7 | 100 | `0.77` |
| phase39 template-aware metric | seed23 | 100 | `0.78` |

这不是降低阈值，而是修正几何判据。

`55356b2` 还确认：

| 任务 | 验证结果 |
| --- | --- |
| `risk_nav` | phase41 seed7/seed23 都是 `success=0.65` |
| `coverage` | route-target seed7 `0.72`、seed23 `0.68` |
| 旧 `goal_nav` phase36 | seed23 只有 `0.66`，暴露 seed robustness 问题 |

### 8.2 `51c2068`

`51c2068` 是 goal_nav robustness 的真正修复。

先用诊断脚本确认旧 goal_nav 失败不是单一 seed 坏，而是失败 episode 中：

- pair collision / obstacle collision 高；
- action alignment 对 remaining goals 崩掉；
- action norm 下降；
- 最后 200 步 timeout。

然后引入 `include_task_targets_in_agents`：

```text
remaining goals -> Hungarian assign to UAVs
extra UAVs -> idle target = current position
append [target_delta, target_position] to each agent token
```

数学上，对 remaining goals $G_{\text{rem}}$ 和 UAV positions $P$ 做 Hungarian assignment：

$$
\pi^\*
=
\arg\min_{\pi}
\sum_i \lVert p_i - g_{\pi(i)} \rVert
$$

每个 UAV 得到 target：

$$
t_i =
\begin{cases}
g_{\pi^\*(i)}, & \text{if assigned} \\
p_i, & \text{if extra UAV should idle}
\end{cases}
$$

agent token 追加：

$$
\Delta_i =
\operatorname{clip}\left(
\frac{t_i - p_i}{\text{map\_size}},
-1,
1
\right)
$$

$$
\hat t_i =
\operatorname{clip}\left(
\frac{t_i}{\text{map\_size}},
0,
1
\right)
$$

效果非常明显：

| 分支 | 结果 |
| --- | --- |
| phase67 BC five-seed 40ep | `success=0.95` |
| phase68 PPO checkpoint_0040 five-seed 40ep | `success=0.965` |
| phase68 PPO checkpoint_0040 seed23 100ep | `success=0.89` |

结论：

- goal_nav 的 seed robustness 主要靠显式 per-agent target assignment 修复；
- 不是靠继续调 reward；
- 它和 coverage route-target solution 是同一个思想：给共享 per-agent actor 明确的个体任务指针，同时保持 centralized global state 和 joint action `[B, N, 2]`。
