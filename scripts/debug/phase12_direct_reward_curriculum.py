from __future__ import annotations

import argparse
import csv
import json
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

BASE_POLICY_CONFIGS = {
    "area_coverage": "configs/policy/direct_specialists/direct_area_coverage.yaml",
    "belief_search": "configs/policy/direct_specialists/direct_belief_search.yaml",
    "priority_inspection": "configs/policy/direct_specialists/direct_priority_inspection.yaml",
    "connectivity_expansion": "configs/policy/direct_specialists/direct_connectivity_expansion.yaml",
}

PHASE11_ROOT = ROOT / "outputs/debug/mappo_direct_specialists/phase11_action_obs_scale/20260607_1510"
FORMAL_ROOT = ROOT / "outputs/training/mappo_direct_specialists/20260607_1237"

POLICY_OVERRIDES: dict[str, dict[str, Any]] = {
    "area_coverage": {
        "max_delta": 0.18,
        "max_waypoint_distance": 0.18,
        "log_std_init": -2.6,
        "log_std_min": -3.5,
        "log_std_max": -1.3,
        "ent_coef": 0.0,
        "learning_rate": 0.0002,
        "vf_coef": 0.75,
    },
    "belief_search": {
        "max_delta": 0.18,
        "max_waypoint_distance": 0.18,
        "log_std_init": -2.3,
        "log_std_min": -3.2,
        "log_std_max": -1.0,
        "ent_coef": 0.001,
        "learning_rate": 0.0002,
        "vf_coef": 0.5,
    },
    "priority_inspection": {
        "max_delta": 0.16,
        "max_waypoint_distance": 0.16,
        "log_std_init": -2.6,
        "log_std_min": -3.5,
        "log_std_max": -1.3,
        "ent_coef": 0.0,
        "learning_rate": 0.00015,
        "vf_coef": 1.0,
    },
    "connectivity_expansion": {
        "max_delta": 0.14,
        "max_waypoint_distance": 0.14,
        "log_std_init": -2.5,
        "log_std_min": -3.5,
        "log_std_max": -1.2,
        "ent_coef": 0.001,
        "learning_rate": 0.00012,
        "target_kl": 0.025,
        "vf_coef": 0.75,
    },
}

ENV_TASK_OVERRIDES: dict[str, dict[str, Any]] = {
    "area_coverage": {
        "w_new": 240.0,
        "w_uncovered_approach": 80.0,
        "uncovered_approach_scale": 0.22,
        "w_overlap": 2.0,
        "w_repeat_footprint": 3.0,
        "w_distance": 0.05,
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
        "w_visit": 10.0,
        "w_approach": 12.0,
        "approach_scale": 0.22,
        "w_repeat": 1.5,
        "w_late": 1.0,
        "w_distance": 0.05,
        "w_safety": 2.0,
    },
    "connectivity_expansion": {
        "w_expand": 12.0,
        "w_coverage": 160.0,
        "w_connected_radius_hold": 0.35,
        "w_relay": 0.7,
        "w_disconnect": 6.0,
        "w_distance": 0.05,
        "w_safety": 2.0,
    },
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def minute_runtime() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")


def load_yaml(path: str | Path) -> dict[str, Any]:
    full_path = ROOT / path if not isinstance(path, Path) or not path.is_absolute() else path
    with full_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def finite_float(row: dict[str, str], key: str, default: float | None = None) -> float | None:
    try:
        value = row.get(key, "")
        if value == "":
            return default
        output = float(value)
        return output if output == output else default
    except (TypeError, ValueError):
        return default


def task_metric_key(task: str) -> str:
    return {
        "area_coverage": "eval_area_coverage_coverage_ratio_mean",
        "belief_search": "eval_belief_search_searched_probability_mass_mean",
        "priority_inspection": "eval_priority_inspection_weighted_poi_completion_mean",
        "connectivity_expansion": "eval_connectivity_expansion_effective_explored_radius_mean",
    }[task]


def secondary_metric_keys(task: str) -> list[str]:
    return {
        "area_coverage": [
            "eval_area_coverage_overlap_ratio_mean",
            "eval_area_coverage_repeat_footprint_ratio_mean",
            "eval_area_coverage_uncovered_approach_gain_mean",
        ],
        "belief_search": [
            "eval_belief_search_detection_rate_mean",
            "eval_belief_search_remaining_belief_mass_mean",
        ],
        "priority_inspection": [
            "eval_priority_inspection_visited_poi_ratio_mean",
            "eval_priority_inspection_repeat_visit_count_mean",
        ],
        "connectivity_expansion": [
            "eval_connectivity_expansion_coverage_ratio_mean",
            "eval_connectivity_expansion_connectivity_violation_rate_mean",
            "eval_connectivity_expansion_connected_radius_hold_mean",
            "eval_connectivity_expansion_connected_angular_spread_mean",
            "eval_connectivity_expansion_uncovered_approach_gain_mean",
            "eval_connectivity_expansion_repeat_footprint_ratio_mean",
        ],
    }[task]


def best_eval_row(rows: list[dict[str, str]], task: str) -> dict[str, str]:
    eval_rows = [row for row in rows if row.get("eval_success_rate") not in ("", None)]
    key = task_metric_key(task)
    return max(
        eval_rows,
        key=lambda row: (
            finite_float(row, "eval_success_rate", 0.0) or 0.0,
            finite_float(row, key, 0.0) or 0.0,
            finite_float(row, "eval_reward", -1.0e9) or -1.0e9,
        ),
        default={},
    )


def summarize_csv(path: Path, task: str) -> dict[str, Any]:
    rows = read_csv(path)
    best = best_eval_row(rows, task)
    last = rows[-1] if rows else {}
    task_key = task_metric_key(task)
    result: dict[str, Any] = {
        "rows": len(rows),
        "last_update": finite_float(last, "update"),
        "best_update": finite_float(best, "update"),
        "best_success_rate": finite_float(best, "eval_success_rate"),
        "best_eval_reward": finite_float(best, "eval_reward"),
        "best_task_metric_name": task_key,
        "best_task_metric": finite_float(best, task_key),
        "last_task_metric": finite_float(last, task_key),
        "last_delta_clip_fraction": finite_float(last, "delta_clip_fraction"),
        "last_policy_std_mean": finite_float(last, "policy_std_mean"),
        "last_approx_kl": finite_float(last, "approx_kl"),
        "last_clip_frac": finite_float(last, "clip_frac"),
        "last_action_validity_rate": finite_float(last, "action_validity_rate"),
        "last_value_loss": finite_float(last, "value_loss"),
        "last_eval_reward": finite_float(last, "eval_reward"),
    }
    for key in secondary_metric_keys(task):
        result[f"best_{key}"] = finite_float(best, key)
        result[f"last_{key}"] = finite_float(last, key)
    return result


def make_policy_config(task: str, args: argparse.Namespace, config_dir: Path) -> Path:
    config = load_yaml(BASE_POLICY_CONFIGS[task])
    config.update(POLICY_OVERRIDES[task])
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
    path = config_dir / f"policy_{task}_trial01.yaml"
    write_yaml(path, config)
    return path


def make_env_config(task: str, config_dir: Path) -> Path:
    config = load_yaml("configs/env/waypoint_missions.yaml")
    config = deepcopy(config)
    task_cfg = config.setdefault("tasks", {}).setdefault(task, {})
    task_cfg.update(ENV_TASK_OVERRIDES[task])
    if task == "priority_inspection":
        config["max_steps"] = max(int(config.get("max_steps", 80)), 100)
    config["domain_randomization"] = deepcopy(config.get("domain_randomization", {}))
    config["domain_randomization"]["enabled"] = True
    path = config_dir / f"env_{task}_trial01.yaml"
    write_yaml(path, config)
    return path


def build_command(task: str, policy_config: Path, env_config: Path, output_dir: Path, args: argparse.Namespace) -> list[str]:
    return [
        sys.executable,
        str(ROOT / "scripts/train_mappo_waypoint.py"),
        "--config",
        str(policy_config),
        "--env-config",
        str(env_config),
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
        "0",
        "--seed",
        "0",
        "--run_name",
        make_run_name(0, 1),
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


def prior_summary(task: str) -> dict[str, Any]:
    formal = summarize_csv(FORMAL_ROOT / task / "s0" / "training_metrics.csv", task)
    phase11 = summarize_csv(PHASE11_ROOT / task / "trial01" / "s0" / "training_metrics.csv", task)
    return {"formal_20260607_1237": formal, "phase11_action_obs_scale": phase11}


def write_report(
    phase_root: Path,
    results: dict[str, Any] | None,
    return_codes: dict[str, int] | None,
    status: str,
    args: argparse.Namespace,
) -> None:
    lines = [
        "# phase12 Direct MAPPO reward/curriculum 调试报告",
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
        "正式 run `20260607_1237` 和 phase11 的共同结论是：动作裁剪问题可被缓解，但 area、priority、connectivity 仍没有学到任务关键结构。",
        "",
        "| 任务 | formal best success | formal best metric | phase11 best success | phase11 best metric | phase11 last clip |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for task in TASKS:
        prior = prior_summary(task)
        formal = prior["formal_20260607_1237"]
        phase11 = prior["phase11_action_obs_scale"]
        lines.append(
            f"| {task} | {formal.get('best_success_rate')} | {formal.get('best_task_metric')} | "
            f"{phase11.get('best_success_rate')} | {phase11.get('best_task_metric')} | {phase11.get('last_delta_clip_fraction')} |"
        )
    lines.extend(
        [
            "",
            "## 2. 原因分析",
            "",
            "- `area_coverage`：phase11 coverage 能到约 `0.55`，但 success 仍为 0，且 overlap 较高，说明策略会移动但大量重复覆盖；原 reward 主要依赖新覆盖增量，缺少对“靠近未覆盖区域”和“当前 footprint 重复覆盖”的直接信号。",
            "- `belief_search`：phase11 已经达到 debug success `0.8333`，主要需要验证更稳的动作尺度，而不是重写任务。",
            "- `priority_inspection`：weighted completion 有短暂上升但不稳定；原 `w_approach` 默认为 0，只有真正访问 POI 才给强奖励，Direct Gaussian 很难从随机探索稳定发现 POI 访问序列。",
            "- `connectivity_expansion`：phase11 把动作压得过保守，半径不如 formal；原 reward 的 radius/coverage 都是增量式，缺少对“保持连通且已经外扩”的稳定强化。",
            "",
            "## 3. 改进策略",
            "",
            "- 不改 MPE core、adapter 或环境动作 API。",
            "- 不加 BC，不回退 candidate-selection。",
            "- 保留 phase11 的低裁剪动作尺度和 spatial context。",
            "- 对任务学习信号做 phase-local reward/curriculum probe，所有改动只写入本 phase 的 generated env config。",
            "",
            "## 4. 改进操作",
            "",
            "| 任务 | 操作 | 预期作用 |",
            "|---|---|---|",
            "| area_coverage | `w_new=240`, `w_uncovered_approach=80`, `w_repeat_footprint=3`, `w_overlap=2`, `w_distance=0.05` | 奖励向未覆盖区域移动，惩罚当前重复 footprint，降低探索距离惩罚 |",
            "| belief_search | 保持低裁剪动作尺度，`w_prob=150`, `w_repeat=2`, `w_distance=0.05` | 稳住 phase11 的成功趋势，减少重复搜索 |",
            "| priority_inspection | `num_pois_range=[4,6]`, `w_visit=10`, `w_approach=12`, `w_repeat=1.5`, `w_distance=0.05` | 先验证 easy curriculum 下能否学到 POI 接近/访问行为 |",
            "| connectivity_expansion | `max_delta=0.14`, `w_expand=12`, `w_coverage=160`, `w_connected_radius_hold=0.35`, `w_distance=0.05` | 放松过保守动作，强化连通外扩和覆盖进展 |",
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
    )
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
                    f"- last_value_loss: `{item.get('last_value_loss')}`",
                ]
            )
            for key in secondary_metric_keys(task):
                lines.append(f"- best {key}: `{item.get(f'best_{key}')}`")
            lines.append("")
        lines.extend(
            [
                "## 7. 反馈和下一步",
                "",
                "本 phase 不是正式收敛声明。若某任务达到阶段性改善，下一步应使用相同假设做更长 debug 和多 seed；若仍失败，应继续缩小假设而不是进入正式 training。",
                "",
                "- `area_coverage` 若 coverage 上升但 success 仍为 0：继续调 anti-repeat 与 frontier shaping，并检查 sensor radius 是否让 repeat footprint 占比过高。",
                "- `belief_search` 若保持高 success：进入多 seed 长 debug 验证。",
                "- `priority_inspection` 若 easy curriculum 能学会：下一 phase 做 target POI 数量 curriculum；若 easy 也学不会，需要检查 target field/POI approach shaping。",
                "- `connectivity_expansion` 若半径恢复但 coverage 低：继续调 coverage vs disconnect reward；若半径仍低，动作尺度还太保守。",
            ]
        )
    (phase_root / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase-name", default="phase12_reward_curriculum")
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
            policy_config = make_policy_config(task, args, config_dir)
            env_config = make_env_config(task, config_dir)
            output_dir = make_debug_output_dir(ROOT / args.output_root, phase_name, runtime, task, 1, 0)
            output_dir.mkdir(parents=True, exist_ok=True)
            command = build_command(task, policy_config, env_config, output_dir, args)
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
        result = summarize_csv(output_dir / "training_metrics.csv", task)
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
        "prior": {task: prior_summary(task) for task in TASKS},
        "results": results,
    }
    (phase_root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_report(phase_root, results, return_codes, str(summary["status"]), args)
    print(json.dumps({"status": summary["status"], "phase_root": str(phase_root), "return_codes": return_codes}, indent=2), flush=True)
    return 0 if summary["status"] == "COMPLETED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
