# Wayffusion UVFA-Conditioned MAPPO

## 设计目标

Wayffusion 原有 generalist 使用六维 `task_id` 区分任务，但无法表达同一任务内不同成功阈值之间的关系。UVFA 版本在不改变 waypoint action、MPE 动力学和 MAPPO 更新公式的前提下，引入连续 mission goal。

第一版只覆盖四个共享静态动力学任务：

- `area_coverage`
- `belief_search`
- `priority_inspection`
- `connectivity_expansion`

## Mission goal

环境观测新增：

```text
mission_goal: [3]
goal_mask: [3]
```

三个槽位分别表示主完成度目标、空间扩展目标和最大违反预算。无关槽位通过 `goal_mask` 置零。

每个 episode reset 时采样一次 goal，episode 内不改变。`reset(options={"mission_goal": ..., "goal_split": ...})` 可以强制指定目标，供可复现评估使用。

旧训练默认使用 `fixed` split，即 YAML 中的正式成功阈值。只有显式传入 `--train-goal-split train` 才进行目标随机采样。

## 双流网络

UVFA policy class 为 `direct_waypoint_uvfa`：

\[
z_s=\phi(\text{task field},\text{UAV states},\text{global info})
\]

\[
z_g=\psi(\text{task id},\text{mission goal},\text{goal mask})
\]

\[
V(s,g)=h([z_s,z_g,z_s\odot z_g])
\]

actor 也接收 \(z_g\)，从而能够针对不同目标输出不同的 Direct Gaussian waypoint delta。PPO 的 per-agent log-prob、ratio、GAE 和 centralized team value 语义保持不变。

原 `direct_waypoint` 不创建 goal encoder，参数结构保持兼容，已有 checkpoint 可以继续严格加载。

## 训练与评估

UVFA 配置：

```text
configs/policy/uvfa/direct_waypoint_uvfa.yaml
configs/policy/uvfa/direct_waypoint_uvfa_debug.yaml
```

训练目标：

```bash
--train-goal-split train
```

评估目标：

```bash
--eval-goal-splits seen interpolation formal
```

输出会包含：

```text
eval_random_goal_seen_*
eval_random_goal_interpolation_*
eval_random_goal_formal_*
eval_goal_interpolation_gap
eval_goal_formal_gap
```

UVFA 配置使用 `reward_norm_scope: task`，避免不同任务的奖励尺度共享同一个 running mean/variance。

## Phase32 对照

```bash
python scripts/debug/phase32_uvfa_goal_conditioning.py
```

`trial01` 是看不到 mission goal 的原 Direct baseline；`trial02` 是 UVFA policy。二者使用相同的环境目标采样、任务归一化、训练预算和随机种子。

输出目录：

```text
outputs/debug/mappo_direct_specialists/
  phase32_uvfa_goal_conditioning/<runtime>/
```

## 当前边界

- 不宣称对未见任务类别零样本泛化。
- 第一版不包含 escort/interception。
- 不包含 Horde、矩阵分解或 hindsight goal relabeling。
- mission goal 只改变成功条件、终止和值函数，不修改原 reward shaping。
