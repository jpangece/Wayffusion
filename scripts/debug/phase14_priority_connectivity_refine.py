from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.debug import phase12_direct_reward_curriculum as p12
from utils.experiment_layout import make_debug_output_dir, make_run_name, validate_phase_name, validate_runtime


TASKS = ["priority_inspection", "connectivity_expansion"]

POLICY_OVERRIDES: dict[str, dict[str, Any]] = {
    "priority_inspection": {
        "max_delta": 0.20,
        "max_waypoint_distance": 0.20,
        "log_std_init": -1.9,
        "log_std_min": -3.2,
        "log_std_max": -0.8,
        "ent_coef": 0.002,
        "learning_rate": 0.00025,
        "lr_schedule": "constant",
        "vf_coef": 1.0,
        "use_field_moment_context": True,
        "field_moment_include_inverse": False,
    },
    "connectivity_expansion": {
        "max_delta": 0.16,
        "max_waypoint_distance": 0.16,
        "log_std_init": -2.4,
        "log_std_min": -3.5,
        "log_std_max": -1.1,
        "ent_coef": 0.001,
        "learning_rate": 0.00018,
        "lr_schedule": "constant",
        "target_kl": 0.03,
        "vf_coef": 0.75,
        "use_field_moment_context": True,
        "field_moment_include_inverse": True,
    },
}

ENV_TASK_OVERRIDES: dict[str, dict[str, Any]] = {
    "priority_inspection": {
        "num_pois": 6,
        "num_pois_range": [5, 7],
        "poi_min_separation": 0.08,
        "poi_target_field": "gaussian",
        "poi_target_sigma": 0.10,
        "w_visit": 12.0,
        "w_approach": 40.0,
        "approach_scale": 0.26,
        "w_repeat": 0.5,
        "w_late": 0.2,
        "w_distance": 0.03,
        "w_safety": 2.0,
    },
    "connectivity_expansion": {
        "w_expand": 4.0,
        "w_coverage": 500.0,
        "w_connected_radius_hold": 0.05,
        "w_connected_spread": 2.0,
        "w_relay": 1.0,
        "w_disconnect": 10.0,
        "w_distance": 0.03,
        "w_safety": 2.0,
    },
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def minute_runtime() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")


def write_report(
    phase_root: Path,
    results: dict[str, Any] | None,
    return_codes: dict[str, int] | None,
    status: str,
    args: argparse.Namespace,
) -> None:
    lines = [
        "# phase14 priority/connectivity 专项调试报告",
        "",
        f"- 状态: `{status}`",
        f"- 生成时间: `{utc_now()}`",
        f"- phase 目录: `{phase_root}`",
        f"- total_updates: `{args.total_updates}`",
        f"- num_envs: `{args.num_envs}`",
        f"- rollout_steps: `{args.rollout_steps}`",
        "",
        "## 1. 现象",
        "",
        "phase13 已让 `area_coverage` 达到 debug success `1.0`，`belief_search` 达到 `0.8333`。剩余问题是 `priority_inspection` 只在 easy curriculum 下部分成功，`connectivity_expansion` 半径达标但 coverage 不足。",
        "",
        "## 2. 原因分析",
        "",
        "- `priority_inspection` 的稀疏 one-hot POI target field 不利于 CNN/field moment 提取连续方向；访问成功后又有大量 repeat/late penalty，eval reward 很负。",
        "- `connectivity_expansion` 学会了连通外扩半径，但没有覆盖足够多的区域，说明 reward 仍偏向拉远而不是铺开搜索。",
        "",
        "## 3. 改进策略",
        "",
        "- priority：把 POI target channel 改成 phase-local Gaussian weighted potential field，给 Direct actor 连续方向信号；降低 repeat/late penalty，避免已经访问 POI 的策略因负 reward 被压制。",
        "- connectivity：降低半径奖励，显著提高 coverage reward，加入 connected angular spread reward，鼓励 UAV 在保持连通的同时朝不同方向铺开。",
        "",
        "## 4. 改进操作",
        "",
        "| 任务 | 操作 |",
        "|---|---|",
        "| priority_inspection | `poi_target_field=gaussian`, `poi_target_sigma=0.10`, `num_pois_range=[5,7]`, `w_repeat=0.5`, `w_late=0.2` |",
        "| connectivity_expansion | `w_coverage=500`, `w_expand=4`, `w_connected_spread=2`, `w_disconnect=10`, `w_connected_radius_hold=0.05` |",
        "",
        "## 5. 开展实验",
        "",
        "- 单任务 Direct MAPPO specialist。",
        "- 训练随机化: `random`。",
        "- 评估随机化: `random`。",
        "- 只跑 phase13 未达标任务。",
        "- 输出只在 `outputs/debug`。",
        "",
    ]
    if results is None:
        lines.extend(["## 6. 观察结果", "", "实验仍在运行或尚未汇总。"])
    else:
        lines.extend(
            [
                "## 6. 观察结果",
                "",
                "| 任务 | return | best success | best task metric | best update | last clip | last valid | last KL |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for task in TASKS:
            item = results.get(task, {})
            lines.append(
                f"| {task} | {return_codes.get(task) if return_codes else None} | "
                f"{item.get('best_success_rate')} | {item.get('best_task_metric')} | "
                f"{item.get('best_update')} | {item.get('last_delta_clip_fraction')} | "
                f"{item.get('last_action_validity_rate')} | {item.get('last_approx_kl')} |"
            )
        lines.extend(["", "### 任务细节", ""])
        for task in TASKS:
            item = results.get(task, {})
            lines.extend(
                [
                    f"#### {task}",
                    "",
                    f"- output_dir: `{item.get('output_dir')}`",
                    f"- best_checkpoint: `{item.get('best_checkpoint')}`",
                    f"- best_eval_reward: `{item.get('best_eval_reward')}`",
                ]
            )
            for key in p12.secondary_metric_keys(task):
                lines.append(f"- best {key}: `{item.get(f'best_{key}')}`")
            lines.append("")
        lines.extend(
            [
                "## 7. 反馈和下一步",
                "",
                "- 如果 priority 达标，应做 target POI 数量恢复验证；如果仍不达标，下一步需要把最近未访问 POI 相对坐标显式加入 agent/global observation。",
                "- 如果 connectivity coverage 仍低，下一步应考虑 connectivity 专项课程：先低 success_coverage/长 max_steps 学会铺开，再恢复 target threshold 评估。",
            ]
        )
    (phase_root / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase-name", default="phase14_priority_connectivity_refine")
    parser.add_argument("--runtime", default=minute_runtime())
    parser.add_argument("--output-root", default="outputs/debug/mappo_direct_specialists")
    parser.add_argument("--total-updates", type=int, default=140)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--rollout-steps", type=int, default=128)
    parser.add_argument("--eval-interval", type=int, default=20)
    parser.add_argument("--eval-episodes", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=512)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    p12.TASKS = TASKS
    p12.POLICY_OVERRIDES = POLICY_OVERRIDES
    p12.ENV_TASK_OVERRIDES = ENV_TASK_OVERRIDES
    phase_name = validate_phase_name(args.phase_name)
    runtime = validate_runtime(args.runtime)
    phase_root = ROOT / args.output_root / phase_name / runtime
    config_dir = phase_root / "configs"
    phase_root.mkdir(parents=True, exist_ok=True)
    write_report(phase_root, None, None, "RUNNING", args)

    processes: dict[str, subprocess.Popen] = {}
    commands: dict[str, list[str]] = {}
    output_dirs: dict[str, Path] = {}
    config_paths: dict[str, dict[str, str]] = {}
    log_handles = []
    try:
        for task in TASKS:
            policy_config = p12.make_policy_config(task, args, config_dir)
            env_config = p12.make_env_config(task, config_dir)
            output_dir = make_debug_output_dir(ROOT / args.output_root, phase_name, runtime, task, 1, 0)
            output_dir.mkdir(parents=True, exist_ok=True)
            command = p12.build_command(task, policy_config, env_config, output_dir, args)
            commands[task] = command
            output_dirs[task] = output_dir
            config_paths[task] = {"policy_config": str(policy_config), "env_config": str(env_config)}
            (output_dir / "command.json").write_text(json.dumps(command, indent=2) + "\n", encoding="utf-8")
            log_handle = (output_dir / "stdout.log").open("a", encoding="utf-8")
            log_handles.append(log_handle)
            processes[task] = subprocess.Popen(command, cwd=ROOT, stdout=log_handle, stderr=subprocess.STDOUT, text=True)
        (phase_root / "commands.json").write_text(json.dumps(commands, indent=2) + "\n", encoding="utf-8")
        (phase_root / "phase_config_paths.json").write_text(json.dumps(config_paths, indent=2) + "\n", encoding="utf-8")
        return_codes = {task: process.wait() for task, process in processes.items()}
    finally:
        for handle in log_handles:
            handle.close()

    results: dict[str, Any] = {}
    for task, output_dir in output_dirs.items():
        result = p12.summarize_csv(output_dir / "training_metrics.csv", task)
        result.update(
            {
                "output_dir": str(output_dir),
                "return_code": return_codes.get(task),
                "best_checkpoint": str(output_dir / "checkpoints/checkpoint_best_eval.pt"),
                "checkpoint_count": len(list((output_dir / "checkpoints").glob("checkpoint_*.pt"))),
            }
        )
        results[task] = result
    summary = {
        "phase_name": phase_name,
        "runtime": runtime,
        "created_at": utc_now(),
        "status": "COMPLETED" if all(code == 0 for code in return_codes.values()) else "FAILED_RUNTIME",
        "return_codes": return_codes,
        "config_paths": config_paths,
        "prior_phase13": {
            task: p12.summarize_csv(
                ROOT
                / "outputs/debug/mappo_direct_specialists/phase13_moment_constant_lr/20260607_1618"
                / task
                / "trial01/s0/training_metrics.csv",
                task,
            )
            for task in TASKS
        },
        "results": results,
    }
    (phase_root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_report(phase_root, results, return_codes, str(summary["status"]), args)
    print(json.dumps({"status": summary["status"], "phase_root": str(phase_root), "return_codes": return_codes}, indent=2), flush=True)
    return 0 if summary["status"] == "COMPLETED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
