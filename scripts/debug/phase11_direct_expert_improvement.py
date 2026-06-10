from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[2]
TASKS = [
    "area_coverage",
    "belief_search",
    "priority_inspection",
    "connectivity_expansion",
]
BASE_CONFIGS = {
    "area_coverage": "configs/policy/direct_specialists/direct_area_coverage.yaml",
    "belief_search": "configs/policy/direct_specialists/direct_belief_search.yaml",
    "priority_inspection": "configs/policy/direct_specialists/direct_priority_inspection.yaml",
    "connectivity_expansion": "configs/policy/direct_specialists/direct_connectivity_expansion.yaml",
}
TRIAL_OVERRIDES = {
    "area_coverage": {
        "max_delta": 0.18,
        "max_waypoint_distance": 0.18,
        "log_std_init": -2.6,
        "log_std_min": -3.5,
        "log_std_max": -1.3,
        "ent_coef": 0.0,
        "learning_rate": 0.0002,
    },
    "belief_search": {
        "max_delta": 0.18,
        "max_waypoint_distance": 0.18,
        "log_std_init": -2.3,
        "log_std_min": -3.2,
        "log_std_max": -1.0,
        "ent_coef": 0.001,
        "learning_rate": 0.0002,
    },
    "priority_inspection": {
        "max_delta": 0.16,
        "max_waypoint_distance": 0.16,
        "log_std_init": -2.6,
        "log_std_min": -3.5,
        "log_std_max": -1.3,
        "ent_coef": 0.0,
        "learning_rate": 0.00015,
    },
    "connectivity_expansion": {
        "max_delta": 0.10,
        "max_waypoint_distance": 0.10,
        "log_std_init": -2.8,
        "log_std_min": -3.5,
        "log_std_max": -1.5,
        "ent_coef": 0.0,
        "learning_rate": 0.0001,
    },
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def minute_runtime() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def f(row: dict[str, str], key: str, default: float | None = None) -> float | None:
    try:
        value = row.get(key, "")
        if value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def load_yaml(path: str | Path) -> dict[str, Any]:
    with (ROOT / path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def baseline_summary(formal_root: Path) -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}
    for task in TASKS:
        rows = read_csv(formal_root / task / "s0" / "training_metrics.csv")
        eval_rows = [row for row in rows if row.get("eval_success_rate") not in ("", None)]
        best = max(
            eval_rows,
            key=lambda row: (f(row, "eval_success_rate", 0.0) or 0.0, f(row, "eval_reward", -1.0e9) or -1.0e9),
            default={},
        )
        last = rows[-1] if rows else {}
        summary[task] = {
            "rows": len(rows),
            "last_update": f(last, "update"),
            "best_update": f(best, "update"),
            "best_success_rate": f(best, "eval_success_rate"),
            "best_eval_reward": f(best, "eval_reward"),
            "last_delta_clip_fraction": f(last, "delta_clip_fraction"),
            "last_action_validity_rate": f(last, "action_validity_rate"),
            "last_approx_kl": f(last, "approx_kl"),
            "last_policy_std_mean": f(last, "policy_std_mean"),
            "best_task_metric": task_metric(task, best),
        }
    return summary


def task_metric(task: str, row: dict[str, str]) -> float | None:
    keys = {
        "area_coverage": "eval_area_coverage_coverage_ratio_mean",
        "belief_search": "eval_belief_search_searched_probability_mass_mean",
        "priority_inspection": "eval_priority_inspection_weighted_poi_completion_mean",
        "connectivity_expansion": "eval_connectivity_expansion_effective_explored_radius_mean",
    }
    return f(row, keys[task])


def make_trial_config(task: str, args: argparse.Namespace, config_dir: Path) -> Path:
    config = load_yaml(BASE_CONFIGS[task])
    config.update(TRIAL_OVERRIDES[task])
    config.update(
        {
            "num_envs": int(args.num_envs),
            "rollout_steps": int(args.rollout_steps),
            "total_updates": int(args.total_updates),
            "eval_interval": int(args.eval_interval),
            "epochs": int(args.epochs),
            "minibatch_size": int(args.minibatch_size),
            "task_field_channel_mode": "task_mask",
            "use_spatial_field_context": True,
            "spatial_context_radii": [0.04, 0.08, 0.12],
            "tensorboard_metric_mode": "core",
        }
    )
    path = config_dir / f"direct_{task}_trial01.yaml"
    write_yaml(path, config)
    return path


def build_command(task: str, config_path: Path, output_dir: Path, args: argparse.Namespace) -> list[str]:
    return [
        sys.executable,
        str(ROOT / "scripts/train_mappo_waypoint.py"),
        "--config",
        str(config_path),
        "--env-config",
        "configs/env/waypoint_missions.yaml",
        "--eval-config",
        "configs/eval/waypoint_eval.yaml",
        "--tasks",
        task,
        "--num_agents",
        "4",
        "--num_envs",
        str(args.num_envs),
        "--rollout_steps",
        str(args.rollout_steps),
        "--total_updates",
        str(args.total_updates),
        "--eval_interval",
        str(args.eval_interval),
        "--eval_episodes",
        str(args.eval_episodes),
        "--record_eval_episodes",
        "1",
        "--record_format",
        "gif",
        "--seed",
        "0",
        "--run_name",
        "trial01_s0",
        "--output-dir",
        str(output_dir),
        "--train-randomization",
        "random",
        "--eval-randomization",
        "random",
        "--env_backend",
        "process",
        "--headless",
        "--tensorboard",
    ]


def summarize_run(task: str, output_dir: Path) -> dict[str, Any]:
    rows = read_csv(output_dir / "training_metrics.csv")
    eval_rows = [row for row in rows if row.get("eval_success_rate") not in ("", None)]
    best = max(
        eval_rows,
        key=lambda row: (f(row, "eval_success_rate", 0.0) or 0.0, f(row, "eval_reward", -1.0e9) or -1.0e9),
        default={},
    )
    last = rows[-1] if rows else {}
    return {
        "output_dir": str(output_dir),
        "rows": len(rows),
        "last_update": f(last, "update"),
        "best_update": f(best, "update"),
        "best_success_rate": f(best, "eval_success_rate"),
        "best_eval_reward": f(best, "eval_reward"),
        "best_task_metric": task_metric(task, best),
        "last_task_metric": task_metric(task, last),
        "last_delta_clip_fraction": f(last, "delta_clip_fraction"),
        "last_policy_std_mean": f(last, "policy_std_mean"),
        "last_approx_kl": f(last, "approx_kl"),
        "last_clip_frac": f(last, "clip_frac"),
        "last_action_validity_rate": f(last, "action_validity_rate"),
        "last_value_loss": f(last, "value_loss"),
        "checkpoint_count": len(list((output_dir / "checkpoints").glob("checkpoint_*.pt"))),
        "best_checkpoint": str(output_dir / "checkpoints" / "checkpoint_best_eval.pt"),
    }


def write_report(
    report_path: Path,
    baseline: dict[str, dict[str, Any]],
    results: dict[str, dict[str, Any]] | None,
    phase_root: Path,
    status: str,
) -> None:
    lines = [
        "# phase11 Direct MAPPO 专家策略改进报告",
        "",
        f"- 状态: `{status}`",
        f"- 生成时间: `{utc_now()}`",
        f"- phase 目录: `{phase_root}`",
        "",
        "## 现象",
        "",
        "正式 run `20260607_1237` 在随机化训练/随机化评估下没有达到专家收敛标准。",
        "四个任务的共同问题是 direct Gaussian 动作被大量裁剪，导致 PPO 优化的 raw delta 和环境实际执行的 clipped waypoint 不一致。",
        "",
        "| 任务 | best success | best task metric | best update | last clip | last KL |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for task in TASKS:
        item = baseline.get(task, {})
        lines.append(
            f"| {task} | {item.get('best_success_rate')} | {item.get('best_task_metric')} | "
            f"{item.get('best_update')} | {item.get('last_delta_clip_fraction')} | {item.get('last_approx_kl')} |"
        )
    lines.extend(
        [
            "",
            "## 原因分析",
            "",
            "1. `delta_clip_fraction` 长期处于约 `0.5-0.93`，说明大量 sampled delta 超过 `max_delta` 后被裁剪。",
            "2. direct policy 的 logprob 基于 raw delta，而环境执行 clipped waypoint；裁剪越多，策略梯度越偏离真实执行动作。",
            "3. `belief_search` 的 entropy 系数较高，log_std 在训练中增大，裁剪比例升到约 `0.93`，搜索质量和 reward 震荡。",
            "4. 单任务专家仍输入 5 通道 task field；当前 formal 配置没有启用任务通道 mask，也没有局部 spatial field context，actor 必须从全局 CNN embedding 间接学习每架 UAV 周围该去哪里。",
            "",
            "## 改进策略",
            "",
            "本 phase 不改 MPE 动力学、不加 BC、不回退 candidate policy。",
            "",
            "1. 降低 direct Gaussian 方差上限，目标是把动作裁剪降到更可控范围。",
            "2. 打开 `task_field_channel_mode: task_mask`，减少无关 field 通道对单任务专家的干扰。",
            "3. 打开 `use_spatial_field_context`，让每架 UAV 的 actor 看到自身附近多个半径上的 field 采样。",
            "4. 将 eval/checkpoint 间隔缩短到 20 updates，避免长时间没有可观察结果。",
            "",
            "## 改进操作",
            "",
            "- 每任务生成 phase-local policy config，不污染正式 config。",
            "- 输出只写入 `outputs/debug/mappo_direct_specialists/phase11_action_obs_scale/...`。",
            "- 训练和评估继续使用 domain randomization。",
            "",
        ]
    )
    if results is None:
        lines.extend(["## 实验结果", "", "实验仍在运行或尚未汇总。"])
    else:
        lines.extend(
            [
                "## 实验结果",
                "",
                "| 任务 | best success | best task metric | best update | last clip | last valid | checkpoints |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for task in TASKS:
            item = results.get(task, {})
            lines.append(
                f"| {task} | {item.get('best_success_rate')} | {item.get('best_task_metric')} | "
                f"{item.get('best_update')} | {item.get('last_delta_clip_fraction')} | "
                f"{item.get('last_action_validity_rate')} | {item.get('checkpoint_count')} |"
            )
        lines.extend(
            [
                "",
                "## 反馈和下一步",
                "",
                "如果本 phase 的裁剪比例明显下降但任务指标仍不涨，下一步应调整 reward/curriculum，而不是继续压低 log_std。",
                "如果裁剪仍然高于 `0.5`，下一步应考虑在 DirectWaypointPolicy 中改用有界分布或对 clipped delta 计算一致的 logprob。",
                "如果 area/connectivity 的 task metric 有进展但 success 仍为 0，说明 success threshold 与 dense reward 之间存在断层，需要单独做 reward shaping phase。",
            ]
        )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", default=minute_runtime())
    parser.add_argument("--phase-name", default="phase11_action_obs_scale")
    parser.add_argument("--total-updates", type=int, default=80)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--rollout-steps", type=int, default=128)
    parser.add_argument("--eval-interval", type=int, default=20)
    parser.add_argument("--eval-episodes", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=512)
    parser.add_argument("--formal-root", default="outputs/training/mappo_direct_specialists/20260607_1237")
    parser.add_argument("--output-root", default="outputs/debug/mappo_direct_specialists")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    phase_root = ROOT / args.output_root / args.phase_name / args.runtime
    config_dir = phase_root / "configs"
    report_path = phase_root / "report.md"
    baseline = baseline_summary(ROOT / args.formal_root)
    write_report(report_path, baseline, None, phase_root, "RUNNING")

    processes: dict[str, subprocess.Popen] = {}
    commands: dict[str, list[str]] = {}
    for task in TASKS:
        config_path = make_trial_config(task, args, config_dir)
        output_dir = phase_root / task / "trial01" / "s0"
        output_dir.mkdir(parents=True, exist_ok=True)
        command = build_command(task, config_path, output_dir, args)
        commands[task] = command
        (output_dir / "command.json").write_text(json.dumps(command, indent=2) + "\n", encoding="utf-8")
        log_handle = (output_dir / "stdout.log").open("a", encoding="utf-8")
        processes[task] = subprocess.Popen(command, cwd=ROOT, stdout=log_handle, stderr=subprocess.STDOUT, text=True)

    (phase_root / "commands.json").write_text(json.dumps(commands, indent=2) + "\n", encoding="utf-8")
    return_codes = {task: process.wait() for task, process in processes.items()}
    results = {
        task: {
            **summarize_run(task, phase_root / task / "trial01" / "s0"),
            "return_code": return_codes[task],
        }
        for task in TASKS
    }
    summary = {
        "phase_name": args.phase_name,
        "runtime": args.runtime,
        "created_at": utc_now(),
        "baseline": baseline,
        "results": results,
    }
    (phase_root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    status = "COMPLETED" if all(code == 0 for code in return_codes.values()) else "FAILED_RUNTIME"
    write_report(report_path, baseline, results, phase_root, status)
    print(json.dumps({"status": status, "phase_root": str(phase_root), "return_codes": return_codes}, indent=2), flush=True)
    return 0 if status == "COMPLETED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
