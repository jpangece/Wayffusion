# Direct MAPPO Specialist 输出目录规范

本规范适用于 `tune_direct_mappo_specialists.py` 和
`tune_direct_mappo_specialists_full.py`，并覆盖早期 Direct specialist
实验使用的长 run 名和时间戳 phase 目录。

## Debug

```text
outputs/debug/mappo_direct_specialists/
  phaseXX_<purpose>/
    YYYYMMDD_HHMM/
      <task_name>/
        trialXX/
          s<seed>/
```

例如：

```text
outputs/debug/mappo_direct_specialists/
  phase03_area_reward_probe/
    20260607_1900/
      area_coverage/
        trial02/
          s0/
```

`phase_name` 必须匹配 `phaseXX_<purpose>`。任务名使用完整 snake_case 名称。

## Formal Training

单一正式配置：

```text
outputs/training/mappo_direct_specialists/
  YYYYMMDD_HHMM/
    <task_name>/
      s<seed>/
```

若正式阶段比较多组 trial：

```text
outputs/training/mappo_direct_specialists/
  YYYYMMDD_HHMM/
    <task_name>/
      trialXX/
        s<seed>/
```

正式目录不包含 `phase_name`。

## Run Name

- 无 trial：`s0`、`s1`、`s2`
- 有 trial：`trial01_s0`、`trial02_s1`

`total_updates`、`num_envs`、`rollout_steps`、`max_delta`、`log_std_init`
和 `learning_rate` 不再写入 run 名。这些参数保存在 snapshot、summary 和
report 中。

## Summary / Report

每个 run 的记录至少包含：

- `phase_name`
- `runtime`
- `task_name`
- `trial_number`
- `seed`
- `output_dir`
- `total_updates`
- `num_envs`
- `rollout_steps`
- `max_delta`
- `log_std_init`
- `learning_rate`
- `best_eval_success_rate`
- `best_checkpoint`

Debug 汇总位于 phase/runtime 根目录。Formal 汇总位于 training runtime 根目录。

## 统一实现

路径和短名由 `utils/experiment_layout.py` 生成：

- `make_debug_output_dir(...)`
- `make_training_output_dir(...)`
- `make_run_name(...)`
- `debug_summary_dir(...)`
- `training_summary_dir(...)`

Direct specialist launcher 不应再自行拼接长路径或从长 run 名反推参数。
