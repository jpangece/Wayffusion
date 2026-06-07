from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import traceback
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algorithms.mappo_waypoint import MAPPOWaypointTrainer, write_metrics_csv
from policies import build_policy
from utils.waypoint_vector_env import make_waypoint_env_batch
from utils.output_layout import minute_timestamp, resolve_output_root, safe_slug
from utils.tensorboard_metrics import should_log_tensorboard_metric


TASKS = [
    "area_coverage",
    "belief_search",
    "priority_inspection",
    "connectivity_expansion",
]
SPECIALIST_CONFIGS = {
    "area_coverage": "configs/policy/specialists/mappo_area_coverage.yaml",
    "belief_search": "configs/policy/specialists/mappo_belief_search.yaml",
    "priority_inspection": "configs/policy/specialists/mappo_priority_inspection.yaml",
    "connectivity_expansion": "configs/policy/specialists/mappo_connectivity_expansion.yaml",
}
SUCCESS_THRESHOLDS = {
    "area_coverage": 0.80,
    "belief_search": 0.75,
    "priority_inspection": 0.70,
    "connectivity_expansion": 0.65,
}


@dataclass(frozen=True)
class TrialSpec:
    tag: str
    learning_rate: float
    rollout_steps: int
    epochs: int
    minibatch_size: int
    ent_coef: float
    vf_coef: float
    target_kl: float
    reward_norm: bool
    env_overrides: dict[str, Any] | None = None


def load_yaml(path: str | Path) -> dict:
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = ROOT / resolved
    with open(resolved, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def write_yaml(path: str | Path, data: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def deep_update(base: dict, override: dict) -> dict:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value
    return result


def safe_name(value: str) -> str:
    return safe_slug(value)


def format_lr(value: float) -> str:
    text = f"{float(value):.0e}".replace("e-0", "e-").replace("e+0", "e")
    return text.replace("e", "e")


def sync_policy_env_config(train_config: dict, env_config: dict) -> None:
    for key in [
        "map_size",
        "grid_size",
        "max_waypoint_distance",
        "min_uav_distance",
        "comm_radius",
        "base_comm_radius",
        "base_position",
        "no_fly_zones",
        "connectivity_candidate_filter",
    ]:
        if key in env_config:
            train_config[key] = env_config[key]
    if "max_delta" not in train_config and "max_waypoint_distance" in env_config:
        train_config["max_delta"] = env_config["max_waypoint_distance"]


def trial_specs(task: str, stage: str, debug_rollout_steps: int) -> list[TrialSpec]:
    base_rollout = int(debug_rollout_steps) if stage == "debug" else 256
    common = [
        TrialSpec("dbg01", 2e-4, base_rollout, 4, 256, 0.01, 0.50, 0.03, True),
        TrialSpec("lr1e-4_safe", 1e-4, 128 if stage == "debug" else 256, 4, 256, 0.01, 0.75, 0.02, True),
        TrialSpec("lr3e-4_ent", 3e-4, 128 if stage == "debug" else 256, 4, 256, 0.02, 0.50, 0.05, True),
        TrialSpec("low_ent", 2e-4, 128 if stage == "debug" else 256, 4, 512, 0.003, 0.50, 0.03, True),
        TrialSpec("epochs6_vf", 2e-4, 256, 6, 512, 0.01, 0.75, 0.03, True),
        TrialSpec("no_rnorm", 1e-4, 256, 4, 512, 0.01, 0.75, 0.02, False),
    ]
    if task == "belief_search":
        return [
            TrialSpec("belief_ent", 2e-4, base_rollout, 4, 256, 0.02, 0.50, 0.03, True),
            common[2],
            common[5],
            common[1],
            common[4],
            common[3],
        ]
    if task == "priority_inspection":
        return [
            TrialSpec("poi_vf", 2e-4, base_rollout, 4, 256, 0.01, 0.75, 0.03, True),
            common[2],
            common[5],
            common[1],
            common[3],
            common[4],
        ]
    if task == "connectivity_expansion":
        safe_env = {"connectivity_action_filter": True}
        aggressive_env = {
            "connectivity_action_filter": True,
            "max_waypoint_distance": 0.30,
            "max_speed": 0.05,
            "action_adapter": {
                "max_command_distance": 0.30,
                "max_speed": 0.05,
                "slowdown_radius": 0.10,
                "max_accel": 0.12,
                "control_smoothing": 0.60,
                "acceptance_radius": 0.020,
            },
            "dynamics_backend": {
                "max_speed": 0.05,
                "dt": 0.1,
                "substeps": 20,
            },
        }
        return [
            TrialSpec("safe01", 1e-4, base_rollout, 4, 256, 0.02, 0.75, 0.02, True, safe_env),
            TrialSpec("safe_ent", 2e-4, 128 if stage == "debug" else 256, 4, 256, 0.02, 0.75, 0.03, True, safe_env),
            TrialSpec("safe_lr3e-4", 3e-4, 128 if stage == "debug" else 256, 4, 256, 0.02, 0.50, 0.05, True, safe_env),
            TrialSpec("safe_aggressive", 2e-4, 128 if stage == "debug" else 256, 4, 256, 0.02, 0.75, 0.03, True, aggressive_env),
            TrialSpec("safe_lowent_move", 5e-4, 128 if stage == "debug" else 256, 6, 256, 0.003, 0.50, 0.08, True, aggressive_env),
            TrialSpec("safe_lowkl", 1e-4, 256, 4, 512, 0.01, 0.75, 0.02, True, safe_env),
            TrialSpec("safe_nonorm", 1e-4, 256, 4, 512, 0.02, 0.75, 0.02, False, safe_env),
            TrialSpec("safe_epochs6", 1e-4, 256, 6, 512, 0.02, 0.75, 0.02, True, safe_env),
        ]
    return common


def apply_trial(train_config: dict, env_config: dict, task: str, spec: TrialSpec, args, stage: str) -> tuple[dict, dict]:
    train = deepcopy(train_config)
    env = deepcopy(env_config)
    env["task_name"] = task
    env["task_names"] = [task]
    env["task_sampling_probs"] = {task: 1.0}
    env["num_agents"] = int(args.num_agents)
    env["seed"] = int(args.seed)
    if spec.env_overrides:
        env = deep_update(env, spec.env_overrides)
    train["learning_rate"] = float(spec.learning_rate)
    train["rollout_steps"] = int(spec.rollout_steps)
    train["epochs"] = int(spec.epochs)
    train["minibatch_size"] = int(spec.minibatch_size)
    train["ent_coef"] = float(spec.ent_coef)
    train["vf_coef"] = float(spec.vf_coef)
    train["target_kl"] = float(spec.target_kl)
    train["reward_norm"] = bool(spec.reward_norm)
    train["num_envs"] = int(args.debug_num_envs if stage == "debug" else args.training_num_envs)
    train["total_updates"] = int(args.debug_total_updates if stage == "debug" else args.training_total_updates)
    train["eval_interval"] = int(args.debug_eval_interval if stage == "debug" else args.training_eval_interval)
    sync_policy_env_config(train, env)
    return train, env


def policy_short(policy_class: str) -> str:
    return "cand" if policy_class == "candidate_selection_waypoint" else str(policy_class).replace("_waypoint", "").replace("_", "-")


def run_name(task: str, train_config: dict, env_config: dict, spec: TrialSpec, stage: str) -> str:
    return safe_name(
        f"{task}__{policy_short(train_config.get('policy_class', 'policy'))}"
        f"__N{env_config['num_agents']}__E{train_config['num_envs']}"
        f"__rs{train_config['rollout_steps']}__lr{format_lr(train_config['learning_rate'])}"
        f"__seed{env_config.get('seed', 0)}__{spec.tag if stage == 'debug' else 'train_' + spec.tag}"
    )


def read_csv_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8", newline="") as handle:
        rows = []
        for row in csv.DictReader(handle):
            parsed = {}
            for key, value in row.items():
                if value is None:
                    continue
                try:
                    parsed[key] = float(value)
                except ValueError:
                    parsed[key] = value
            rows.append(parsed)
        return rows


def finite_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def metric(row: dict, task: str, key: str, default: float = 0.0) -> float:
    safe = task.replace("-", "_")
    candidates = [
        f"eval_{safe}_{key}_mean",
        f"eval_{safe}_{key}",
        f"eval_{key}",
        f"{key}_mean",
        key,
    ]
    for item in candidates:
        if item in row:
            return finite_float(row[item], default)
    return default


def eval_records(records: list[dict]) -> list[dict]:
    return [row for row in records if "eval_success_rate" in row or "eval_reward" in row]


def best_record(records: list[dict], task: str | None = None) -> dict:
    candidates = eval_records(records)
    if not candidates:
        return records[-1] if records else {}
    return max(
        candidates,
        key=lambda row: (
            finite_float(row.get("eval_success_rate", 0.0)),
            task_progress(row, task) if task is not None else 0.0,
            finite_float(row.get("eval_reward", -1e9)),
        ),
    )


def recent_reward_ok(records: list[dict]) -> bool:
    candidates = eval_records(records)
    if len(candidates) < 3:
        return True
    rewards = [finite_float(row.get("eval_reward", 0.0)) for row in candidates[-3:]]
    return rewards[-1] >= max(rewards[:-1]) - max(abs(max(rewards[:-1])) * 0.25, 1.0)


def task_progress(row: dict, task: str) -> float:
    if task == "area_coverage":
        return metric(row, task, "coverage_ratio", 0.0)
    if task == "belief_search":
        return max(metric(row, task, "searched_probability_mass", 0.0), metric(row, task, "detection_rate", 0.0))
    if task == "priority_inspection":
        return metric(row, task, "weighted_poi_completion", 0.0)
    if task == "connectivity_expansion":
        return max(metric(row, task, "coverage_ratio", 0.0), metric(row, task, "effective_explored_radius", 0.0))
    return 0.0


def positive_trend(records: list[dict], task: str) -> bool:
    candidates = eval_records(records)
    if len(candidates) < 2:
        return False
    first, last = candidates[0], candidates[-1]
    reward_gain = finite_float(last.get("eval_reward", 0.0)) > finite_float(first.get("eval_reward", 0.0))
    progress_gain = task_progress(last, task) > task_progress(first, task)
    return bool(reward_gain or progress_gain)


def evaluate_status(task: str, records: list[dict]) -> tuple[str, dict[str, Any], str]:
    if not records:
        return "FAILED_RUNTIME", {}, "No training metrics were produced."
    best = best_record(records, task)
    success = finite_float(best.get("eval_success_rate", 0.0))
    action_validity = finite_float(best.get("eval_action_validity_rate", best.get("action_validity_rate", 1.0)), 1.0)
    reward_ok = recent_reward_ok(records)
    reason = []
    if success < SUCCESS_THRESHOLDS[task]:
        reason.append(f"eval_success_rate {success:.3f} < {SUCCESS_THRESHOLDS[task]:.2f}")
    if action_validity < 0.95:
        reason.append(f"action_validity_rate {action_validity:.3f} < 0.95")
    if not reward_ok:
        reason.append("recent eval_reward declined")
    if task == "area_coverage":
        coverage = metric(best, task, "coverage_ratio", 0.0)
        if coverage < 0.55:
            reason.append(f"coverage_ratio {coverage:.3f} < 0.55")
    elif task == "belief_search":
        searched = metric(best, task, "searched_probability_mass", 0.0)
        detection = metric(best, task, "detection_rate", 0.0)
        if max(searched, detection) < 0.55:
            reason.append(f"searched/detection max {max(searched, detection):.3f} < 0.55")
    elif task == "priority_inspection":
        weighted = metric(best, task, "weighted_poi_completion", 0.0)
        if weighted < 0.70:
            reason.append(f"weighted_poi_completion {weighted:.3f} < 0.70")
    elif task == "connectivity_expansion":
        violation = metric(best, task, "connectivity_violation_rate", 1.0)
        progress = task_progress(best, task)
        if violation > 0.10:
            reason.append(f"connectivity_violation_rate {violation:.3f} > 0.10")
        if progress <= 0.0:
            reason.append("coverage/expansion metric did not increase")
    status = "CONVERGED" if not reason else "NEED_MORE_TUNING"
    return status, best, "; ".join(reason) if reason else "Success criteria met."


def next_suggestion(task: str, records: list[dict], status_reason: str) -> str:
    if not records:
        return "Check runtime errors and snapshot configs before launching another trial."
    last = records[-1]
    kl = finite_float(last.get("approx_kl", 0.0))
    clip = finite_float(last.get("clip_frac", 0.0))
    entropy = finite_float(last.get("entropy", 0.0))
    action_validity = finite_float(last.get("action_validity_rate", last.get("eval_action_validity_rate", 1.0)), 1.0)
    value_loss = finite_float(last.get("value_loss", 0.0))
    if action_validity < 0.95:
        return "Do not keep tuning PPO first; inspect candidate scale, safety clipping, no-fly/geofence, and connectivity filter."
    if kl > 0.05 or clip > 0.35:
        return "PPO update is too aggressive; reduce learning_rate/epochs or increase minibatch_size."
    if entropy < 0.2:
        return "Policy may be over-deterministic; raise ent_coef or reduce training pressure."
    if value_loss > 50.0:
        return "Value fit is poor; try reward_norm toggle, lower learning_rate, or vf_coef=0.75."
    if "success_rate" in status_reason and task_progress(best_record(records, task), task) > 0.0:
        return "Task metric is moving but success is low; continue with longer run before changing reward."
    return "Try the next scheduled specialist trial; if task metric remains flat, inspect task_field/candidate scores/reward components."


def write_trial_snapshot(output_dir: Path, train_config: dict, env_config: dict, spec: TrialSpec, args: argparse.Namespace) -> None:
    snapshot = output_dir / "snapshot"
    snapshot.mkdir(parents=True, exist_ok=True)
    write_yaml(snapshot / "train_config.yaml", train_config)
    write_yaml(snapshot / "env_config.yaml", env_config)
    write_yaml(snapshot / "tuning_trial.yaml", {"trial": spec.__dict__, "args": vars(args)})
    (snapshot / "metadata.json").write_text(
        json.dumps({"created_at": datetime.now().astimezone().isoformat(), "main_env": "WaypointMultiUAVEnv", "launcher": "tune_mappo_specialists"}, indent=2) + "\n",
        encoding="utf-8",
    )


def build_tensorboard_writer(output_dir: Path, enabled: bool = True):
    if not enabled:
        return None
    try:
        from torch.utils.tensorboard import SummaryWriter

        return SummaryWriter(str(output_dir / "tensorboard"))
    except Exception as exc:
        print(f"[specialist-tune] TensorBoard disabled: {exc}", flush=True)
        return None


def log_tensorboard_record(writer, record: dict) -> None:
    if writer is None:
        return
    step = int(finite_float(record.get("update", 0), 0.0))
    mode = str(getattr(writer, "_wayffusion_metric_mode", "core"))
    for key, value in record.items():
        if should_log_tensorboard_metric(key, value, mode=mode):
            writer.add_scalar(str(key), float(value), step)
    writer.flush()


def run_trial(task: str, stage: str, spec: TrialSpec, args: argparse.Namespace, timestamp: str) -> dict[str, Any]:
    base_env_config = load_yaml(args.env_config)
    config_path = SPECIALIST_CONFIGS.get(task, args.config)
    train_config = load_yaml(config_path)
    train_config, env_config = apply_trial(train_config, base_env_config, task, spec, args, stage)
    name = run_name(task, train_config, env_config, spec, stage)
    root = Path(args.debug_output_root if stage == "debug" else args.training_output_root)
    if not root.is_absolute():
        root = ROOT / root
    base_dir = root if bool(getattr(args, "flat_phase_layout", False)) else root / timestamp
    output_dir = base_dir / task / name
    output_dir.mkdir(parents=True, exist_ok=True)
    write_trial_snapshot(output_dir, train_config, env_config, spec, args)
    record_eval_episodes = int(args.debug_record_eval_episodes if stage == "debug" else args.training_record_eval_episodes)
    eval_episodes = int(args.debug_eval_episodes if stage == "debug" else args.training_eval_episodes)
    if args.dry_run:
        return {
            "task": task,
            "stage": stage,
            "run_name": name,
            "output_dir": str(output_dir),
            "status": "DRY_RUN",
            "config_path": config_path,
        }
    try:
        env_batch = make_waypoint_env_batch(env_config, int(train_config["num_envs"]), backend=args.env_backend, max_workers=args.env_workers)
        policy = build_policy(train_config, env_batch.envs[0].global_observation_space, env_batch.envs[0].action_space_n)
        trainer = MAPPOWaypointTrainer(env_batch, policy, train_config)
        writer = build_tensorboard_writer(output_dir, enabled=bool(getattr(args, "tensorboard", True)))
        if writer is not None:
            writer._wayffusion_metric_mode = str(train_config.get("tensorboard_metric_mode", "core"))
        history: list[dict] = []

        def on_record(record: dict) -> None:
            history.append(dict(record))
            write_metrics_csv(history, output_dir / "training_metrics.csv")
            log_tensorboard_record(writer, record)

        final_history = trainer.train(
            output_dir,
            env_config=env_config,
            task_names=[task],
            eval_episodes=eval_episodes,
            headless=bool(args.headless),
            record_eval_episodes=record_eval_episodes,
            record_format=args.record_format,
            record_fps=int(args.record_fps),
            record_interval=int(args.record_interval),
            log_callback=on_record,
        )
        if writer is not None:
            writer.close()
        env_batch.close()
        if not history:
            history = final_history
        write_metrics_csv(history, output_dir / "training_metrics.csv")
        records = read_csv_records(output_dir / "training_metrics.csv")
        episode_eval_records = read_csv_records(output_dir / "eval_metrics.csv")
        status, best, reason = evaluate_status(task, records)
        return {
            "task": task,
            "stage": stage,
            "run_name": name,
            "output_dir": str(output_dir),
            "status": status,
            "reason": reason,
            "config_path": config_path,
            "training_metrics_csv": str(output_dir / "training_metrics.csv"),
            "eval_metrics_csv": str(output_dir / "eval_metrics.csv"),
            "eval_episode_records": len(episode_eval_records),
            "best_eval_success_rate": finite_float(best.get("eval_success_rate", 0.0)),
            "best_eval_reward": finite_float(best.get("eval_reward", 0.0)),
            "best_eval_action_validity_rate": finite_float(best.get("eval_action_validity_rate", best.get("action_validity_rate", 1.0)), 1.0),
            "best_task_progress": task_progress(best, task),
            "positive_trend": positive_trend(records, task),
            "next_suggestion": next_suggestion(task, records, reason),
            "trial": spec.__dict__,
        }
    except Exception as exc:
        error_path = output_dir / "runtime_error.txt"
        error_path.write_text(traceback.format_exc(), encoding="utf-8")
        return {
            "task": task,
            "stage": stage,
            "run_name": name,
            "output_dir": str(output_dir),
            "status": "FAILED_RUNTIME",
            "reason": f"{type(exc).__name__}: {exc}",
            "config_path": config_path,
            "runtime_error_path": str(error_path),
            "trial": spec.__dict__,
        }


def write_task_report(task_dir: Path, task: str, runs: list[dict]) -> dict:
    task_dir.mkdir(parents=True, exist_ok=True)
    completed = [run for run in runs if run.get("status") != "FAILED_RUNTIME"]
    best = max(completed, key=lambda run: (finite_float(run.get("best_eval_success_rate", 0.0)), finite_float(run.get("best_eval_reward", -1e9))), default=(runs[-1] if runs else {}))
    status = "CONVERGED" if any(run.get("status") == "CONVERGED" for run in runs) else ("FAILED_RUNTIME" if runs and all(run.get("status") == "FAILED_RUNTIME" for run in runs) else "NEED_MORE_TUNING")
    summary = {
        "task": task,
        "status": status,
        "best_run": best,
        "runs": runs,
        "next_suggestion": best.get("next_suggestion", "Run the next scheduled tuning trial.") if status != "CONVERGED" else "No immediate action required; verify with more seeds before calling the specialist robust.",
    }
    (task_dir / "tuning_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        f"# MAPPO Specialist Tuning: {task}",
        "",
        f"- Status: `{status}`",
        f"- Best run: `{best.get('run_name', '')}`",
        f"- Best eval_success_rate: `{best.get('best_eval_success_rate', 0.0)}`",
        f"- Best eval_reward: `{best.get('best_eval_reward', 0.0)}`",
        f"- Best task progress: `{best.get('best_task_progress', 0.0)}`",
        f"- Next suggestion: {summary['next_suggestion']}",
        "",
        "## Runs",
        "",
        "| run | stage | status | success | reward | progress | output |",
        "| --- | --- | --- | ---: | ---: | ---: | --- |",
    ]
    for run in runs:
        lines.append(
            f"| `{run.get('run_name', '')}` | `{run.get('stage', '')}` | `{run.get('status', '')}` | "
            f"{finite_float(run.get('best_eval_success_rate', 0.0)):.3f} | "
            f"{finite_float(run.get('best_eval_reward', 0.0)):.3f} | "
            f"{finite_float(run.get('best_task_progress', 0.0)):.3f} | `{run.get('output_dir', '')}` |"
        )
    (task_dir / "tuning_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune waypoint MAPPO task-specialist policies.")
    parser.add_argument("--stage", choices=["debug", "training", "both"], default="debug")
    parser.add_argument("--tasks", nargs="+", default=TASKS)
    parser.add_argument("--config", default="configs/policy/mappo_waypoint.yaml")
    parser.add_argument("--env-config", default="configs/env/waypoint_missions.yaml")
    parser.add_argument("--debug-output-root", default="outputs/debug")
    parser.add_argument("--training-output-root", default="outputs/training")
    parser.add_argument("--phase-name", default="phase_change_on_mappo_specialists")
    parser.add_argument("--timestamp", default=None)
    parser.add_argument("--num-agents", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--debug-max-trials-per-task", type=int, default=6)
    parser.add_argument("--training-max-trials-per-task", type=int, default=3)
    parser.add_argument("--debug-total-updates", type=int, default=10)
    parser.add_argument("--training-total-updates", type=int, default=1000)
    parser.add_argument("--debug-num-envs", type=int, default=4)
    parser.add_argument("--training-num-envs", type=int, default=8)
    parser.add_argument("--debug-rollout-steps", type=int, default=64)
    parser.add_argument("--debug-eval-interval", type=int, default=5)
    parser.add_argument("--training-eval-interval", type=int, default=50)
    parser.add_argument("--debug-eval-episodes", type=int, default=2)
    parser.add_argument("--training-eval-episodes", type=int, default=20)
    parser.add_argument("--debug-record-eval-episodes", type=int, default=1)
    parser.add_argument("--training-record-eval-episodes", type=int, default=2)
    parser.add_argument("--record-format", choices=["gif", "mp4"], default="gif")
    parser.add_argument("--record-fps", type=int, default=8)
    parser.add_argument("--record-interval", type=int, default=1)
    parser.add_argument("--env-backend", choices=["sync", "thread"], default="sync")
    parser.add_argument("--env-workers", type=int, default=None)
    parser.add_argument("--tensorboard", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--trial-tags", nargs="*", default=None, help="Optional subset of trial tags to run after task-specific ordering.")
    args = parser.parse_args()

    timestamp = args.timestamp or minute_timestamp()
    args.debug_output_root = str(resolve_output_root(args.debug_output_root, args.phase_name, timestamp, ROOT))
    args.training_output_root = str(resolve_output_root(args.training_output_root, args.phase_name, timestamp, ROOT))
    args.flat_phase_layout = True
    stages = ["debug", "training"] if args.stage == "both" else [args.stage]
    global_summaries = []
    for task in [str(item) for item in args.tasks]:
        if task not in TASKS:
            raise ValueError(f"Unsupported first-round specialist task: {task}")
        task_runs = []
        debug_positive = False
        for stage in stages:
            if stage == "training" and args.stage == "both" and not debug_positive:
                task_runs.append(
                    {
                        "task": task,
                        "stage": stage,
                        "status": "NEED_MORE_TUNING",
                        "reason": "Skipped formal training because debug trials did not show positive trend.",
                    }
                )
                continue
            limit = int(args.debug_max_trials_per_task if stage == "debug" else args.training_max_trials_per_task)
            specs = trial_specs(task, stage, int(args.debug_rollout_steps))
            if args.trial_tags:
                wanted = set(str(item) for item in args.trial_tags)
                specs = [spec for spec in specs if spec.tag in wanted]
            for spec in specs[:limit]:
                print(f"[specialist-tune] task={task} stage={stage} trial={spec.tag}", flush=True)
                run = run_trial(task, stage, spec, args, timestamp)
                task_runs.append(run)
                if stage == "debug" and bool(run.get("positive_trend", False)):
                    debug_positive = True
                if run.get("status") == "CONVERGED":
                    break
        root = Path(args.debug_output_root if args.stage != "training" else args.training_output_root)
        if not root.is_absolute():
            root = ROOT / root
        task_summary = write_task_report(root / task, task, task_runs)
        global_summaries.append(task_summary)

    global_root = Path(args.debug_output_root if args.stage != "training" else args.training_output_root)
    if not global_root.is_absolute():
        global_root = ROOT / global_root
    global_root = global_root
    global_root.mkdir(parents=True, exist_ok=True)
    (global_root / "tuning_summary.json").write_text(json.dumps(global_summaries, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = ["# MAPPO Specialist Tuning Summary", ""]
    for item in global_summaries:
        best = item.get("best_run", {})
        lines.append(
            f"- `{item['task']}`: `{item['status']}`, best=`{best.get('run_name', '')}`, "
            f"success=`{best.get('best_eval_success_rate', 0.0)}`, reward=`{best.get('best_eval_reward', 0.0)}`"
        )
    (global_root / "tuning_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[specialist-tune] summary_dir={global_root}", flush=True)


if __name__ == "__main__":
    main()
