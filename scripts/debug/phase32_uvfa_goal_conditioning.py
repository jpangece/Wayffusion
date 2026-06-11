from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.experiment_layout import make_debug_output_dir, make_run_name, validate_phase_name, validate_runtime


TASKS = [
    "area_coverage",
    "belief_search",
    "priority_inspection",
    "connectivity_expansion",
]
TASK_METRICS = {
    "area_coverage": "coverage_ratio_mean",
    "belief_search": "searched_probability_mass_mean",
    "priority_inspection": "weighted_poi_completion_mean",
    "connectivity_expansion": "coverage_ratio_mean",
}


def minute_runtime() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def write_trial_config(base: dict, path: Path, policy_class: str, trial: int, args: argparse.Namespace) -> Path:
    config = deepcopy(base)
    config.update(
        {
            "name": "direct_waypoint_goal_blind" if trial == 1 else "direct_waypoint_uvfa",
            "policy_class": policy_class,
            "num_envs": args.num_envs,
            "rollout_steps": args.rollout_steps,
            "total_updates": args.total_updates,
            "eval_interval": args.eval_interval,
            "reward_norm": True,
            "reward_norm_scope": "task",
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def build_command(
    config_path: Path,
    output_dir: Path,
    trial: int,
    seed: int,
    args: argparse.Namespace,
) -> list[str]:
    return [
        sys.executable,
        "scripts/train_mappo_waypoint.py",
        "--config",
        str(config_path.relative_to(ROOT)),
        "--env-config",
        "configs/env/waypoint_missions.yaml",
        "--eval-config",
        "configs/eval/waypoint_eval.yaml",
        "--tasks",
        *TASKS,
        "--num_agents",
        "4",
        "--envs_per_task",
        str(args.num_envs // len(TASKS)),
        "--total_updates",
        str(args.total_updates),
        "--rollout_steps",
        str(args.rollout_steps),
        "--eval_interval",
        str(args.eval_interval),
        "--eval_episodes",
        str(args.eval_episodes),
        "--record_eval_episodes",
        "1",
        "--record_interval",
        str(max(args.total_updates // args.eval_interval, 1)),
        "--record_format",
        "gif",
        "--train-randomization",
        "random",
        "--eval-randomization",
        "random",
        "--train-goal-split",
        "train",
        "--eval-goal-splits",
        "seen",
        "interpolation",
        "formal",
        "--env_backend",
        "process",
        "--seed",
        str(seed),
        "--run_name",
        make_run_name(seed, trial),
        "--phase-name",
        args.phase_name,
        "--output-dir",
        str(output_dir),
        "--headless",
        "--tensorboard",
    ]


def last_numeric(rows: list[dict[str, str]], key: str) -> float | None:
    for row in reversed(rows):
        value = row.get(key, "")
        if value not in {"", None}:
            try:
                number = float(value)
            except ValueError:
                continue
            if math.isfinite(number):
                return number
    return None


def summarize_run(output_dir: Path, trial: int, seed: int, return_code: int) -> dict[str, Any]:
    metrics_path = output_dir / "training_metrics.csv"
    rows: list[dict[str, str]] = []
    if metrics_path.exists():
        with metrics_path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    result: dict[str, Any] = {
        "trial": trial,
        "seed": seed,
        "return_code": return_code,
        "output_dir": str(output_dir),
        "metrics_exists": metrics_path.exists(),
        "best_checkpoint": str(output_dir / "checkpoints" / "checkpoint_best_eval.pt"),
        "formal_success": last_numeric(rows, "eval_random_goal_formal_success_rate"),
        "formal_progress": last_numeric(rows, "eval_random_goal_formal_goal_progress"),
        "interpolation_success": last_numeric(rows, "eval_random_goal_interpolation_success_rate"),
        "interpolation_progress": last_numeric(rows, "eval_random_goal_interpolation_goal_progress"),
        "seen_success": last_numeric(rows, "eval_random_goal_seen_success_rate"),
        "seen_progress": last_numeric(rows, "eval_random_goal_seen_goal_progress"),
        "approx_kl": last_numeric(rows, "approx_kl"),
        "clip_frac": last_numeric(rows, "clip_frac"),
        "goal_embedding_norm": last_numeric(rows, "goal_embedding_norm"),
        "goal_embedding_variance": last_numeric(rows, "goal_embedding_variance"),
    }
    for task_name, metric in TASK_METRICS.items():
        result[f"{task_name}_formal_progress"] = last_numeric(
            rows,
            f"eval_random_goal_formal_{task_name}_goal_progress_mean",
        )
        result[f"{task_name}_formal_metric"] = last_numeric(
            rows,
            f"eval_random_goal_formal_{task_name}_{metric}",
        )
    return result


def mean_available(items: list[dict[str, Any]], key: str) -> float | None:
    values = [float(item[key]) for item in items if item.get(key) is not None]
    return float(sum(values) / len(values)) if values else None


def build_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    baseline = [item for item in results if item["trial"] == 1]
    uvfa = [item for item in results if item["trial"] == 2]
    comparisons = {}
    improved_tasks = 0
    for task in TASKS:
        key = f"{task}_formal_progress"
        baseline_mean = mean_available(baseline, key)
        uvfa_mean = mean_available(uvfa, key)
        delta = None if baseline_mean is None or uvfa_mean is None else uvfa_mean - baseline_mean
        comparisons[task] = {
            "baseline_formal_progress": baseline_mean,
            "uvfa_formal_progress": uvfa_mean,
            "delta": delta,
        }
        if delta is not None and delta >= 0.02:
            improved_tasks += 1

    baseline_formal = mean_available(baseline, "formal_progress")
    uvfa_formal = mean_available(uvfa, "formal_progress")
    baseline_interp = mean_available(baseline, "interpolation_progress")
    uvfa_interp = mean_available(uvfa, "interpolation_progress")
    all_healthy = all(
        item["return_code"] == 0
        and item["metrics_exists"]
        and item["formal_progress"] is not None
        and item["approx_kl"] is not None
        for item in results
    )
    formal_delta = None if baseline_formal is None or uvfa_formal is None else uvfa_formal - baseline_formal
    interp_delta = None if baseline_interp is None or uvfa_interp is None else uvfa_interp - baseline_interp
    if not all_healthy:
        status = "NEED_FIX"
    elif (
        formal_delta is not None
        and interp_delta is not None
        and formal_delta >= -0.02
        and interp_delta >= -0.02
        and improved_tasks >= 2
    ):
        status = "READY_FOR_LONG_TRAINING"
    else:
        status = "NEED_MORE_TUNING"
    return {
        "status": status,
        "baseline_formal_progress": baseline_formal,
        "uvfa_formal_progress": uvfa_formal,
        "formal_progress_delta": formal_delta,
        "baseline_interpolation_progress": baseline_interp,
        "uvfa_interpolation_progress": uvfa_interp,
        "interpolation_progress_delta": interp_delta,
        "tasks_improved_by_0p02": improved_tasks,
        "task_comparisons": comparisons,
        "runs": results,
    }


def write_report(phase_root: Path, summary: dict[str, Any], args: argparse.Namespace) -> None:
    lines = [
        "# phase32 UVFA goal conditioning 对照报告",
        "",
        f"- 状态: `{summary['status']}`",
        f"- 运行目录: `{phase_root}`",
        f"- 训练预算: `{args.total_updates}` updates × 2 trials × {len(args.seeds)} seeds",
        "- trial01: 原 Direct policy，环境随机目标，但 policy 不编码 mission goal。",
        "- trial02: 双流 UVFA Direct policy，actor/critic 编码 task_id、mission_goal、goal_mask。",
        "- 两个 trial 均使用 task-scoped reward normalization。",
        "",
        "## 总体对比",
        "",
        "| 指标 | baseline | UVFA | delta |",
        "|---|---:|---:|---:|",
        f"| formal goal progress | {summary['baseline_formal_progress']} | {summary['uvfa_formal_progress']} | {summary['formal_progress_delta']} |",
        f"| interpolation goal progress | {summary['baseline_interpolation_progress']} | {summary['uvfa_interpolation_progress']} | {summary['interpolation_progress_delta']} |",
        "",
        "## 分任务 formal progress",
        "",
        "| 任务 | baseline | UVFA | delta |",
        "|---|---:|---:|---:|",
    ]
    for task, item in summary["task_comparisons"].items():
        lines.append(
            f"| {task} | {item['baseline_formal_progress']} | "
            f"{item['uvfa_formal_progress']} | {item['delta']} |"
        )
    lines.extend(["", "## Run 明细", ""])
    for item in summary["runs"]:
        lines.extend(
            [
                f"### trial{item['trial']:02d} seed{item['seed']}",
                "",
                f"- output_dir: `{item['output_dir']}`",
                f"- return_code: `{item['return_code']}`",
                f"- best_checkpoint: `{item['best_checkpoint']}`",
                f"- seen/interpolation/formal progress: `{item['seen_progress']}` / `{item['interpolation_progress']}` / `{item['formal_progress']}`",
                f"- approx_kl / clip_frac: `{item['approx_kl']}` / `{item['clip_frac']}`",
                f"- goal embedding norm/variance: `{item['goal_embedding_norm']}` / `{item['goal_embedding_variance']}`",
                "",
            ]
        )
    lines.extend(
        [
            "## 结论规则",
            "",
            "- `READY_FOR_LONG_TRAINING`: 链路健康，UVFA 未显著退化，且至少两个任务 formal progress 提升 0.02。",
            "- `NEED_MORE_TUNING`: 链路健康但目标条件化优势尚不明确。",
            "- `NEED_FIX`: 运行失败、指标缺失或出现非有限训练诊断。",
        ]
    )
    (phase_root / "report_zh.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase-name", default="phase32_uvfa_goal_conditioning")
    parser.add_argument("--runtime", default=minute_runtime())
    parser.add_argument("--output-root", default="outputs/debug/mappo_direct_specialists")
    parser.add_argument("--total-updates", type=int, default=400)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--rollout-steps", type=int, default=128)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--eval-episodes", type=int, default=3)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--max-parallel", type=int, default=2)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.phase_name = validate_phase_name(args.phase_name)
    args.runtime = validate_runtime(args.runtime)
    if args.num_envs % len(TASKS) != 0:
        raise ValueError("num_envs must be divisible by the four static tasks")
    phase_root = ROOT / args.output_root / args.phase_name / args.runtime
    config_dir = phase_root / "configs"
    phase_root.mkdir(parents=True, exist_ok=True)
    base_config = load_yaml(ROOT / "configs/policy/uvfa/direct_waypoint_uvfa_debug.yaml")
    configs = {
        1: write_trial_config(base_config, config_dir / "trial01_baseline.yaml", "direct_waypoint", 1, args),
        2: write_trial_config(base_config, config_dir / "trial02_uvfa.yaml", "direct_waypoint_uvfa", 2, args),
    }

    pending = []
    for trial in [1, 2]:
        for seed in args.seeds:
            output_dir = make_debug_output_dir(
                ROOT / args.output_root,
                args.phase_name,
                args.runtime,
                "four_task_generalist",
                trial,
                seed,
            )
            output_dir.mkdir(parents=True, exist_ok=True)
            command = build_command(configs[trial], output_dir, trial, seed, args)
            (output_dir / "command.json").write_text(json.dumps(command, indent=2) + "\n", encoding="utf-8")
            pending.append((trial, seed, output_dir, command))

    active: list[tuple[int, int, Path, subprocess.Popen, Any]] = []
    completed: list[tuple[int, int, Path, int]] = []
    while pending or active:
        while pending and len(active) < max(int(args.max_parallel), 1):
            trial, seed, output_dir, command = pending.pop(0)
            log_handle = (output_dir / "stdout.log").open("a", encoding="utf-8")
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
            )
            active.append((trial, seed, output_dir, process, log_handle))
        for item in list(active):
            trial, seed, output_dir, process, log_handle = item
            return_code = process.poll()
            if return_code is None:
                continue
            log_handle.close()
            completed.append((trial, seed, output_dir, int(return_code)))
            active.remove(item)
        if active:
            import time

            time.sleep(2.0)

    results = [summarize_run(output_dir, trial, seed, code) for trial, seed, output_dir, code in completed]
    results.sort(key=lambda item: (item["trial"], item["seed"]))
    summary = build_summary(results)
    (phase_root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_report(phase_root, summary, args)
    print(f"[phase32] output_dir={phase_root}")
    print(f"[phase32] status={summary['status']}")
    return 0 if summary["status"] != "NEED_FIX" else 1


if __name__ == "__main__":
    raise SystemExit(main())
