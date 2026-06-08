from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.debug import phase12_direct_reward_curriculum as p12
from utils.experiment_layout import make_debug_output_dir, make_run_name, validate_phase_name, validate_runtime


TASKS = ["connectivity_expansion"]

POLICY_OVERRIDES: dict[str, dict[str, Any]] = {
    "connectivity_expansion": {
        "max_delta": 0.14,
        "max_waypoint_distance": 0.14,
        "log_std_init": -2.5,
        "log_std_min": -3.5,
        "log_std_max": -1.1,
        "ent_coef": 0.001,
        "learning_rate": 0.00018,
        "lr_schedule": "constant",
        "target_kl": 0.03,
        "vf_coef": 0.75,
        "use_field_moment_context": True,
        "field_moment_include_inverse": True,
    }
}

ENV_TASK_OVERRIDES: dict[str, dict[str, Any]] = {
    "connectivity_expansion": {
        "w_expand": 10.0,
        "w_coverage": 360.0,
        "w_uncovered_approach": 1500.0,
        "uncovered_approach_scale": 0.26,
        "w_repeat_footprint": 0.2,
        "w_connected_radius_hold": 0.15,
        "w_connected_spread": 0.0,
        "w_relay": 1.5,
        "w_disconnect": 4.0,
        "w_distance": 0.03,
        "w_safety": 2.0,
    }
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def minute_runtime() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")


def patch_env_config(path: Path) -> None:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    data["connectivity_action_filter"] = True
    data["strict_action_validation"] = False
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def write_report(phase_root: Path, results: dict[str, Any] | None, return_codes: dict[str, int] | None, status: str, args: argparse.Namespace) -> None:
    lines = [
        "# phase16 connectivity hard-action-filter 调试报告",
        "",
        f"- 状态: `{status}`",
        f"- 生成时间: `{utc_now()}`",
        f"- phase 目录: `{phase_root}`",
        f"- total_updates: `{args.total_updates}`",
        "",
        "## 1. 现象",
        "",
        "phase15 将 coverage 推到约 `0.54`，但 connectivity violation 高达约 `0.77`，说明策略学到的是断连覆盖，不是连通覆盖。",
        "",
        "## 2. 原因分析",
        "",
        "Direct waypoint 主线默认没有启用 `connectivity_action_filter`；已有的 `connectivity_candidate_filter` 主要作用于旧 candidate path。只靠 reward 惩罚，策略可以先断连覆盖再承受惩罚。",
        "",
        "## 3. 改进策略",
        "",
        "启用已有 waypoint-level `connectivity_action_filter: true`，让会导致断连的 final waypoint 被 safety layer 替换为 fallback。该 phase 验证 hard filter 是否能把 violation 降下来，同时保持 coverage 增长。",
        "",
        "## 4. 改进操作",
        "",
        "- `connectivity_action_filter=true`",
        "- `max_delta=0.14`",
        "- `w_coverage=360`",
        "- `w_uncovered_approach=1500`",
        "- `w_expand=10`",
        "- `w_connected_radius_hold=0.15`",
        "",
        "## 5. 开展实验",
        "",
        "- 单任务 Direct MAPPO connectivity specialist。",
        "- 训练随机化: `random`。",
        "- 评估随机化: `random`。",
        "- 输出只在 `outputs/debug`。",
        "",
    ]
    if results is None:
        lines.extend(["## 6. 观察结果", "", "实验仍在运行或尚未汇总。"])
    else:
        item = results["connectivity_expansion"]
        lines.extend(
            [
                "## 6. 观察结果",
                "",
                "| best success | best radius | best coverage | best violation | best update | last clip | last valid | last KL |",
                "|---:|---:|---:|---:|---:|---:|---:|---:|",
                (
                    f"| {item.get('best_success_rate')} | {item.get('best_task_metric')} | "
                    f"{item.get('best_eval_connectivity_expansion_coverage_ratio_mean')} | "
                    f"{item.get('best_eval_connectivity_expansion_connectivity_violation_rate_mean')} | "
                    f"{item.get('best_update')} | {item.get('last_delta_clip_fraction')} | "
                    f"{item.get('last_action_validity_rate')} | {item.get('last_approx_kl')} |"
                ),
                "",
                f"- output_dir: `{item.get('output_dir')}`",
                f"- best_checkpoint: `{item.get('best_checkpoint')}`",
                f"- best_eval_reward: `{item.get('best_eval_reward')}`",
                f"- best uncovered approach: `{item.get('best_eval_connectivity_expansion_uncovered_approach_gain_mean')}`",
                f"- best repeat footprint: `{item.get('best_eval_connectivity_expansion_repeat_footprint_ratio_mean')}`",
                "",
                "## 7. 反馈和下一步",
                "",
                "如果 hard filter 降低 violation 但 coverage/radius 不够，下一步应做 curriculum：先用更宽 `comm_radius/base_comm_radius` 或更多 agents 学会连通覆盖，再恢复 target 参数。若 hard filter 导致 action_validity 过低，需要降低 max_delta 或训练 action-validity shaping。",
            ]
        )
    (phase_root / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase-name", default="phase16_connectivity_hard_filter")
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
    p12.TASKS = TASKS
    p12.POLICY_OVERRIDES = POLICY_OVERRIDES
    p12.ENV_TASK_OVERRIDES = ENV_TASK_OVERRIDES
    phase_name = validate_phase_name(args.phase_name)
    runtime = validate_runtime(args.runtime)
    phase_root = ROOT / args.output_root / phase_name / runtime
    config_dir = phase_root / "configs"
    phase_root.mkdir(parents=True, exist_ok=True)
    write_report(phase_root, None, None, "RUNNING", args)

    commands: dict[str, list[str]] = {}
    output_dirs: dict[str, Path] = {}
    config_paths: dict[str, dict[str, str]] = {}
    log_handles = []
    processes: dict[str, subprocess.Popen] = {}
    try:
        for task in TASKS:
            policy_config = p12.make_policy_config(task, args, config_dir)
            env_config = p12.make_env_config(task, config_dir)
            patch_env_config(env_config)
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
        "results": results,
    }
    (phase_root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_report(phase_root, results, return_codes, str(summary["status"]), args)
    print(json.dumps({"status": summary["status"], "phase_root": str(phase_root), "return_codes": return_codes}, indent=2), flush=True)
    return 0 if summary["status"] == "COMPLETED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
