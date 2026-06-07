# Wayffusion 运行输出目录规范

> Direct MAPPO specialist 使用更具体的新规范：
> `docs/direct_mappo_specialist_output_layout_zh.md`。对于 Direct specialist，
> 该规范优先于本文下面的通用约定。

本规范是新实验的最高优先级规则。仓库中已有的旧目录不迁移、不重命名；它们只用于历史追溯，不能作为新 run 的路径模板。

## Debug

所有调试、smoke、sweep、消融、参数搜索和短训练必须放在：

```text
outputs/debug/
  phaseNNN_debug_<需求简写>/
    run_registry.yaml
    YYYYMMDD_HHMM__<短run名>/
    YYYYMMDD_HHMM__<短run名>/
```

示例：

```text
outputs/debug/phase001_debug_randomized_spawn/
  run_registry.yaml
  20260607_1430__area_s0/
  20260607_1448__belief_s0/
```

规则：

- `phaseNNN` 使用三位递增编号，例如 `phase001`、`phase002`。
- 一个 phase 只对应一个明确的修改或调试需求。
- 同一需求的各次 run 直接放在 phase 目录下，不再增加 algorithm、formal、smoke、full_tuning 等重复层级。
- 时间戳精确到分钟。
- run 名保持简短，通常包含任务、seed 和一个区分 tag 即可。

## Formal Training

只有确认训练链路和配置健康后启动的正式长训练，才允许写入：

```text
outputs/training/
  run_registry.yaml
  YYYYMMDD_HHMM__<短run名>/
  YYYYMMDD_HHMM__<短run名>/
```

示例：

```text
outputs/training/
  run_registry.yaml
  20260607_1600__area_s0/
  20260607_1600__area_s1/
  20260607_1600__belief_s0/
```

新 run 不允许再创建以下层级：

```text
outputs/training/<algorithm>/<timestamp>/<phase>/<task>/<run>/
```

任务、算法、seed 和配置来源由 run 名、snapshot 以及 registry 记录，不依赖目录套层表达。

## Run Registry

每次 run 都必须写入统一元文件：

- debug：该 phase 根目录下的 `run_registry.yaml`；
- formal：`outputs/training/run_registry.yaml`。

建议结构：

```yaml
schema_version: 1
updated_at: 2026-06-07T14:30:00+00:00
runs:
  - run_id: 20260607_1430__area_s0
    purpose: 验证随机禁飞区下的 area coverage 学习趋势
    kind: debug
    tasks: [area_coverage]
    config: configs/policy/direct_specialists_v2/direct_area_coverage_v2.yaml
    env_config: configs/env/waypoint_missions_direct_tune_v2.yaml
    seed: 0
    output_dir: outputs/debug/phase001_debug_randomized_spawn/20260607_1430__area_s0
    status: RUNNING
    created_at: 2026-06-07T14:30:00+00:00
    updated_at: 2026-06-07T14:30:00+00:00
```

运行前写入 `PLANNED` 或 `RUNNING`；运行结束后更新为：

- `COMPLETED`
- `FAILED_RUNTIME`
- `STOPPED`
- `NEED_MORE_TUNING`

没有 registry 记录的 run 视为流程不完整。

## Run 内部结构

每个 run 继续保留：

```text
snapshot/
checkpoints/
training_metrics.csv
eval_metrics.csv
best_eval_summary.json
tensorboard/
media/
```

registry 只记录 run 的用途和索引信息，不替代 snapshot、CSV、checkpoint 或视频。

## 执行约束

- 旧实验不自动整理，避免破坏正在运行的进程和历史路径。
- 如果某个脚本仍默认生成旧式目录，启动时必须显式传 `--output-dir`。
- debug 结果通过后才能另开正式 run，不能把 debug 目录移动到 training 冒充正式结果。
- 正式训练出现问题时，其结果仍留在 training 并在 registry 中标记真实状态，不覆盖或删除。
