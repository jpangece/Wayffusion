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


TASKS = ["connectivity_expansion"]

POLICY_OVERRIDES: dict[str, dict[str, Any]] = {
    "connectivity_expansion": {
        "max_delta": 0.16,
        "max_waypoint_distance": 0.16,
        "log_std_init": -2.35,
        "log_std_min": -3.5,
        "log_std_max": -1.0,
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
        "w_expand": 8.0,
        "w_coverage": 320.0,
        "w_uncovered_approach": 3000.0,
        "uncovered_approach_scale": 0.26,
        "w_repeat_footprint": 0.2,
        "w_connected_radius_hold": 0.20,
        "w_connected_spread": 0.0,
        "w_relay": 1.5,
        "w_disconnect": 8.0,
        "w_distance": 0.03,
        "w_safety": 2.0,
    }
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
        "# phase15 connectivity uncovered-approach 专项调试报告",
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
        "phase13 的 connectivity 能达到半径 `0.56` 但 coverage 只有约 `0.27`；phase14 的 angular spread 方案退化为低覆盖保守策略，coverage 约 `0.22`。",
        "",
        "## 2. 原因分析",
        "",
        "- 只奖励半径会让 UAV 形成连通外扩但不覆盖足够区域。",
        "- 只奖励 angular spread 会让 UAV 在 base 附近安全铺开，仍然不产生 coverage。",
        "- connectivity 需要同时有“保持连通外扩”和“朝未覆盖区域推进”的 dense signal。",
        "",
        "## 3. 改进策略",
        "",
        "- 移除 phase14 的 `w_connected_spread`。",
        "- 启用 connectivity 自己的 `w_uncovered_approach` 和 `w_repeat_footprint`。",
        "- 保留 field moment inverse，让 actor 得到未覆盖区域的紧凑方向提示。",
        "- 保持 target success threshold，不降低 `success_coverage/success_radius/max_violation_rate`。",
        "",
        "## 4. 改进操作",
        "",
        "- `w_expand=8`",
        "- `w_coverage=320`",
        "- `w_uncovered_approach=3000`",
        "- `w_repeat_footprint=0.2`",
        "- `w_connected_radius_hold=0.20`",
        "- `w_connected_spread=0`",
        "- `w_relay=1.5`",
        "- `w_disconnect=8`",
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
                "如果 coverage 仍低于 `0.55`，下一步不应继续堆 reward 权重，而应做 connectivity curriculum：先用较宽 comm 或更长 max_steps 学会 connected coverage，再恢复 target config 评估。",
            ]
        )
    (phase_root / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase-name", default="phase15_connectivity_uncovered")
    parser.add_argument("--runtime", default=minute_runtime())
    parser.add_argument("--output-root", default="outputs/debug/mappo_direct_specialists")
    parser.add_argument("--total-updates", type=int, default=160)
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
