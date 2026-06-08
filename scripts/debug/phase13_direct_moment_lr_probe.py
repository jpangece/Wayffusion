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


POLICY_OVERRIDES: dict[str, dict[str, Any]] = {
    "area_coverage": {
        "max_delta": 0.18,
        "max_waypoint_distance": 0.18,
        "log_std_init": -2.5,
        "log_std_min": -3.5,
        "log_std_max": -1.2,
        "ent_coef": 0.0,
        "learning_rate": 0.00025,
        "lr_schedule": "constant",
        "vf_coef": 0.75,
        "use_field_moment_context": True,
        "field_moment_include_inverse": True,
    },
    "belief_search": {
        "max_delta": 0.18,
        "max_waypoint_distance": 0.18,
        "log_std_init": -2.2,
        "log_std_min": -3.2,
        "log_std_max": -1.0,
        "ent_coef": 0.001,
        "learning_rate": 0.00025,
        "lr_schedule": "constant",
        "vf_coef": 0.5,
        "use_field_moment_context": True,
        "field_moment_include_inverse": False,
    },
    "priority_inspection": {
        "max_delta": 0.20,
        "max_waypoint_distance": 0.20,
        "log_std_init": -2.0,
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
    "area_coverage": {
        "w_new": 220.0,
        "w_uncovered_approach": 4000.0,
        "uncovered_approach_scale": 0.24,
        "w_overlap": 1.0,
        "w_repeat_footprint": 0.3,
        "w_distance": 0.03,
        "w_safety": 2.0,
    },
    "belief_search": {
        "w_prob": 150.0,
        "w_repeat": 2.0,
        "w_distance": 0.05,
        "w_safety": 2.0,
        "detection_bonus": 10.0,
    },
    "priority_inspection": {
        "num_pois": 5,
        "num_pois_range": [4, 6],
        "poi_min_separation": 0.08,
        "w_visit": 12.0,
        "w_approach": 80.0,
        "approach_scale": 0.26,
        "w_repeat": 2.0,
        "w_late": 1.0,
        "w_distance": 0.03,
        "w_safety": 2.0,
    },
    "connectivity_expansion": {
        "w_expand": 8.0,
        "w_coverage": 300.0,
        "w_connected_radius_hold": 0.2,
        "w_relay": 0.7,
        "w_disconnect": 8.0,
        "w_distance": 0.04,
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
        "# phase13 Direct MAPPO moment-context / constant-LR 调试报告",
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
        "phase12 证明 reward/curriculum 对 area 和 connectivity 有阶段性提升，但末端 `approx_kl` 再次接近 `1e-7`；priority 即使 easy POI curriculum 和 approach reward 也几乎没有访问 POI。",
        "",
        "## 2. 原因分析",
        "",
        "- 短 debug 也使用 `lr_schedule: linear`，120 updates 后学习率基本衰减到 0，后期 PPO 更新过弱。",
        "- Direct policy 的全局 CNN 使用池化表示，远处 POI、高 belief 区域、未覆盖区域的坐标关系被压缩，actor 不容易直接输出目标方向。",
        "- priority 的 target field 是稀疏点，局部 spatial context 在 UAV 离 POI 很远时看不到目标，所以仅靠局部采样和全局池化不够。",
        "",
        "## 3. 改进策略",
        "",
        "- 保留 Direct waypoint 输出，不回退 candidate-selection。",
        "- 保留 MPE dynamics 和 adapter。",
        "- 使用 `lr_schedule: constant` 保持短 debug 后半段仍有 PPO 更新。",
        "- 启用可选 `use_field_moment_context`：把任务 field 的质量和重心相对每个 UAV 的方向作为 actor 输入。",
        "- 对 area/connectivity 使用 inverse field moment，给未覆盖区域的紧凑方向提示；对 belief/priority 使用高值 field moment，给 belief/POI 方向提示。",
        "",
        "## 4. 改进操作",
        "",
        "| 任务 | policy 操作 | env/reward 操作 |",
        "|---|---|---|",
        "| area_coverage | constant LR, field moment + inverse moment | 提高 `w_uncovered_approach` 到 `4000`，降低 repeat footprint 到 `0.3`，距离惩罚 `0.03` |",
        "| belief_search | constant LR, high-value moment | 保留 phase12 belief shaping |",
        "| priority_inspection | constant LR, high-value POI target moment, max_delta `0.20` | easy POI curriculum，`w_approach=80`, `w_visit=12` |",
        "| connectivity_expansion | constant LR, field moment + inverse moment, max_delta `0.16` | coverage 权重 `300`，断连惩罚 `8`，弱化半径 hold |",
        "",
        "## 5. 开展实验",
        "",
        "- 单任务 Direct MAPPO specialist。",
        "- 训练随机化: `random`。",
        "- 评估随机化: `random`。",
        "- 后端: process vector env + real vendored MPE core。",
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
                "| 任务 | return | best success | best task metric | best update | last clip | last valid | last KL | last value loss |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for task in p12.TASKS:
            item = results.get(task, {})
            lines.append(
                f"| {task} | {return_codes.get(task) if return_codes else None} | "
                f"{item.get('best_success_rate')} | {item.get('best_task_metric')} | "
                f"{item.get('best_update')} | {item.get('last_delta_clip_fraction')} | "
                f"{item.get('last_action_validity_rate')} | {item.get('last_approx_kl')} | "
                f"{item.get('last_value_loss')} |"
            )
        lines.extend(["", "### 任务细节", ""])
        for task in p12.TASKS:
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
                "- 若 priority 仍失败，说明只给 field moment 还不够，需要把 POI 相对坐标/最近未访问 POI summary 显式写入 observation 或任务 field，而不是继续盲目调 PPO。",
                "- 若 area coverage 提高但 success 仍为 0，需要继续处理 repeat footprint 恒为 1 的问题，可能要把 repeat penalty 改成 marginal repeated area 而不是每步常数惩罚。",
                "- 若 connectivity coverage 仍低，下一步应单独做 coverage-vs-radius curriculum，而不是继续增大半径奖励。",
                "- belief 若保持 success >= 0.8，可以进入多 seed 长 debug 验证。",
            ]
        )
    (phase_root / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase-name", default="phase13_moment_constant_lr")
    parser.add_argument("--runtime", default=minute_runtime())
    parser.add_argument("--output-root", default="outputs/debug/mappo_direct_specialists")
    parser.add_argument("--total-updates", type=int, default=120)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--rollout-steps", type=int, default=128)
    parser.add_argument("--eval-interval", type=int, default=20)
    parser.add_argument("--eval-episodes", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=512)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
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
        for task in p12.TASKS:
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
        "prior_phase12": {
            task: p12.summarize_csv(
                ROOT
                / "outputs/debug/mappo_direct_specialists/phase12_reward_curriculum/20260607_1600"
                / task
                / "trial01/s0/training_metrics.csv",
                task,
            )
            for task in p12.TASKS
        },
        "results": results,
    }
    (phase_root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_report(phase_root, results, return_codes, str(summary["status"]), args)
    print(json.dumps({"status": summary["status"], "phase_root": str(phase_root), "return_codes": return_codes}, indent=2), flush=True)
    return 0 if summary["status"] == "COMPLETED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
