from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.debug import tune_mappo_specialists as common


BASE_NEXT_SUGGESTION = common.next_suggestion
TASKS = [
    "area_coverage",
    "belief_search",
    "priority_inspection",
    "connectivity_expansion",
]
DIRECT_SPECIALIST_CONFIGS = {
    "area_coverage": "configs/policy/direct_specialists/direct_area_coverage.yaml",
    "belief_search": "configs/policy/direct_specialists/direct_belief_search.yaml",
    "priority_inspection": "configs/policy/direct_specialists/direct_priority_inspection.yaml",
    "connectivity_expansion": "configs/policy/direct_specialists/direct_connectivity_expansion.yaml",
}


@dataclass(frozen=True)
class DirectTrialSpec:
    tag: str
    learning_rate: float
    rollout_steps: int
    epochs: int
    minibatch_size: int
    ent_coef: float
    vf_coef: float
    target_kl: float
    reward_norm: bool
    max_delta: float
    log_std_init: float
    log_std_min: float
    log_std_max: float
    env_overrides: dict[str, Any] | None = None


def _fmt_delta(value: float) -> str:
    return f"{float(value):.2f}".replace(".", "p")


def _sync_action_scale(train_config: dict, env_config: dict, max_delta: float) -> None:
    train_config["policy_class"] = "direct_waypoint"
    train_config["max_delta"] = float(max_delta)
    env_config["max_waypoint_distance"] = float(max_delta)
    adapter = dict(env_config.get("action_adapter", {}))
    adapter["max_command_distance"] = float(max_delta)
    env_config["action_adapter"] = adapter
    if "max_speed" in env_config:
        adapter.setdefault("max_speed", env_config["max_speed"])
    if "dynamics_backend" in env_config and "max_speed" in env_config:
        env_config["dynamics_backend"]["max_speed"] = env_config["max_speed"]


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
    ]:
        if key in env_config:
            train_config[key] = env_config[key]


def direct_trial_specs(task: str, stage: str, debug_rollout_steps: int) -> list[DirectTrialSpec]:
    base_rollout = int(debug_rollout_steps) if stage == "debug" else 256
    common_trials = [
        DirectTrialSpec("md020_std08", 2e-4, base_rollout, 4, 256, 0.01, 0.50, 0.03, True, 0.20, -0.8, -1.8, 0.1),
        DirectTrialSpec("md015_std10", 2e-4, 128 if stage == "debug" else 256, 4, 256, 0.01, 0.50, 0.03, True, 0.15, -1.0, -2.0, 0.0),
        DirectTrialSpec("md010_safe", 1e-4, 128 if stage == "debug" else 256, 4, 256, 0.003, 0.75, 0.02, True, 0.10, -1.5, -3.0, 0.0),
        DirectTrialSpec("md025_explore", 2e-4, 128 if stage == "debug" else 256, 4, 256, 0.02, 0.50, 0.05, True, 0.25, -0.5, -2.0, 0.5),
        DirectTrialSpec("lr3e-4", 3e-4, 128 if stage == "debug" else 256, 4, 256, 0.01, 0.50, 0.05, True, 0.20, -0.8, -1.8, 0.1),
        DirectTrialSpec("vf075", 1e-4, 256, 6, 512, 0.01, 0.75, 0.02, True, 0.15, -1.0, -2.0, 0.0),
        DirectTrialSpec("no_rnorm", 1e-4, 256, 4, 512, 0.01, 0.75, 0.02, False, 0.15, -1.0, -2.0, 0.0),
    ]
    if task == "area_coverage":
        return [
            common_trials[0],
            common_trials[1],
            common_trials[3],
            common_trials[4],
            common_trials[2],
            common_trials[5],
            common_trials[6],
        ]
    if task == "belief_search":
        return [
            DirectTrialSpec("belief_md020_ent", 2e-4, base_rollout, 4, 256, 0.02, 0.50, 0.03, True, 0.20, -0.8, -1.8, 0.1),
            DirectTrialSpec("belief_md025_explore", 2e-4, 128 if stage == "debug" else 256, 4, 256, 0.02, 0.50, 0.05, True, 0.25, -0.5, -2.0, 0.5),
            common_trials[1],
            common_trials[4],
            common_trials[5],
            common_trials[6],
        ]
    if task == "priority_inspection":
        return [
            DirectTrialSpec("poi_md020_vf", 2e-4, base_rollout, 4, 256, 0.01, 0.75, 0.03, True, 0.20, -0.8, -1.8, 0.1),
            common_trials[1],
            common_trials[3],
            common_trials[4],
            common_trials[5],
            common_trials[6],
        ]
    if task == "connectivity_expansion":
        safe_env = {"connectivity_action_filter": True}
        easy_comm = {"connectivity_action_filter": True, "comm_radius": 0.42, "base_comm_radius": 0.50}
        return [
            DirectTrialSpec("conn_md010_safe", 1e-4, base_rollout, 4, 256, 0.01, 0.75, 0.02, True, 0.10, -1.5, -3.0, 0.0, safe_env),
            DirectTrialSpec("conn_md015", 1e-4, 128 if stage == "debug" else 256, 4, 256, 0.02, 0.75, 0.02, True, 0.15, -1.0, -2.0, 0.0, safe_env),
            DirectTrialSpec("conn_md020_ent", 2e-4, 128 if stage == "debug" else 256, 4, 256, 0.02, 0.75, 0.03, True, 0.20, -0.8, -1.8, 0.1, safe_env),
            DirectTrialSpec("conn_easy_comm", 2e-4, 128 if stage == "debug" else 256, 4, 256, 0.02, 0.75, 0.03, True, 0.20, -0.8, -1.8, 0.1, easy_comm),
            DirectTrialSpec("conn_lowent", 3e-4, 128 if stage == "debug" else 256, 6, 256, 0.003, 0.75, 0.05, True, 0.15, -1.0, -2.0, 0.0, safe_env),
            DirectTrialSpec("conn_no_rnorm", 1e-4, 256, 4, 512, 0.01, 0.75, 0.02, False, 0.15, -1.0, -2.0, 0.0, safe_env),
        ]
    return common_trials


def apply_direct_trial(train_config: dict, env_config: dict, task: str, spec: DirectTrialSpec, args, stage: str) -> tuple[dict, dict]:
    train = deepcopy(train_config)
    env = deepcopy(env_config)
    env["task_name"] = task
    env["task_names"] = [task]
    env["task_sampling_probs"] = {task: 1.0}
    env["num_agents"] = int(args.num_agents)
    env["seed"] = int(args.seed)
    if spec.env_overrides:
        env = common.deep_update(env, spec.env_overrides)
    _sync_action_scale(train, env, spec.max_delta)
    train["learning_rate"] = float(spec.learning_rate)
    train["rollout_steps"] = int(spec.rollout_steps)
    train["epochs"] = int(spec.epochs)
    train["minibatch_size"] = int(spec.minibatch_size)
    train["ent_coef"] = float(spec.ent_coef)
    train["vf_coef"] = float(spec.vf_coef)
    train["target_kl"] = float(spec.target_kl)
    train["reward_norm"] = bool(spec.reward_norm)
    train["log_std_init"] = float(spec.log_std_init)
    train["log_std_min"] = float(spec.log_std_min)
    train["log_std_max"] = float(spec.log_std_max)
    train["num_envs"] = int(args.debug_num_envs if stage == "debug" else args.training_num_envs)
    train["total_updates"] = int(args.debug_total_updates if stage == "debug" else args.training_total_updates)
    train["eval_interval"] = int(args.debug_eval_interval if stage == "debug" else args.training_eval_interval)
    sync_policy_env_config(train, env)
    return train, env


def direct_run_name(task: str, train_config: dict, env_config: dict, spec: DirectTrialSpec, stage: str) -> str:
    return common.safe_name(
        f"{task}__direct"
        f"__N{env_config['num_agents']}__E{train_config['num_envs']}"
        f"__rs{train_config['rollout_steps']}__lr{common.format_lr(train_config['learning_rate'])}"
        f"__md{_fmt_delta(train_config['max_delta'])}"
        f"__seed{env_config.get('seed', 0)}__{spec.tag if stage == 'debug' else 'train_' + spec.tag}"
    )


def next_suggestion(task: str, records: list[dict], status_reason: str) -> str:
    if not records:
        return "Check runtime errors and snapshot configs before launching another direct trial."
    last = records[-1]
    action_validity = common.finite_float(last.get("action_validity_rate", last.get("eval_action_validity_rate", 1.0)), 1.0)
    mean_speed = common.finite_float(last.get("mean_speed", 0.0))
    mean_control = common.finite_float(last.get("mean_control_norm", 0.0))
    train_std = common.finite_float(last.get("train_action_std", 0.0))
    kl = common.finite_float(last.get("approx_kl", 0.0))
    clip = common.finite_float(last.get("clip_frac", 0.0))
    entropy = common.finite_float(last.get("entropy", 0.0))
    if action_validity < 0.90:
        return "Direct actions are being rejected/corrected; reduce max_delta or log_std_init before tuning PPO LR."
    if action_validity < 0.95:
        return "Direct action validity is marginal; try max_delta=0.10/0.15 or lower log_std before changing task rewards."
    if mean_speed < 1e-3 and mean_control < 1e-3:
        return "Direct policy is barely moving; increase max_delta/log_std_init or use a higher entropy trial."
    if train_std > 0.25 and common.task_progress(common.best_record(records, task), task) <= 0.0:
        return "Direct policy is exploring widely without task progress; lower log_std_max/log_std_init and strengthen dense reward."
    if kl > 0.05 or clip > 0.35:
        return "PPO update is too aggressive; reduce learning_rate/epochs or increase minibatch_size."
    if entropy < 0.2 and "success_rate" in status_reason:
        return "Policy became too deterministic before solving the task; raise ent_coef or log_std_init."
    return BASE_NEXT_SUGGESTION(task, records, status_reason)


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune Direct MAPPO waypoint task-specialist policies.")
    parser.add_argument("--stage", choices=["debug", "training", "both"], default="debug")
    parser.add_argument("--tasks", nargs="+", default=TASKS)
    parser.add_argument("--config", default="configs/policy/mappo_waypoint.yaml")
    parser.add_argument("--env-config", default="configs/env/waypoint_missions.yaml")
    parser.add_argument("--debug-output-root", default="outputs/debug/mappo_direct_specialists")
    parser.add_argument("--training-output-root", default="outputs/training/mappo_direct_specialists")
    parser.add_argument("--timestamp", default=None)
    parser.add_argument("--num-agents", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--debug-max-trials-per-task", type=int, default=8)
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
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--trial-tags", nargs="*", default=None)
    args = parser.parse_args()

    common.SPECIALIST_CONFIGS = DIRECT_SPECIALIST_CONFIGS
    common.apply_trial = apply_direct_trial
    common.run_name = direct_run_name
    common.trial_specs = direct_trial_specs
    common.next_suggestion = next_suggestion

    timestamp = args.timestamp or datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    stages = ["debug", "training"] if args.stage == "both" else [args.stage]
    global_summaries = []
    for task in [str(item) for item in args.tasks]:
        if task not in TASKS:
            raise ValueError(f"Unsupported direct specialist task: {task}")
        task_runs = []
        debug_positive = False
        for stage in stages:
            if stage == "training" and args.stage == "both" and not debug_positive:
                task_runs.append({"task": task, "stage": stage, "status": "NEED_MORE_TUNING", "reason": "Skipped formal training because debug trials did not show positive trend."})
                continue
            limit = int(args.debug_max_trials_per_task if stage == "debug" else args.training_max_trials_per_task)
            specs = direct_trial_specs(task, stage, int(args.debug_rollout_steps))
            if args.trial_tags:
                wanted = set(str(item) for item in args.trial_tags)
                specs = [spec for spec in specs if spec.tag in wanted]
            for spec in specs[:limit]:
                print(f"[direct-specialist-tune] task={task} stage={stage} trial={spec.tag}", flush=True)
                run = common.run_trial(task, stage, spec, args, timestamp)
                task_runs.append(run)
                if stage == "debug" and bool(run.get("positive_trend", False)):
                    debug_positive = True
                if run.get("status") == "CONVERGED":
                    break
        root = Path(args.debug_output_root if args.stage != "training" else args.training_output_root)
        if not root.is_absolute():
            root = ROOT / root
        task_summary = common.write_task_report(root / timestamp / task, task, task_runs)
        global_summaries.append(task_summary)

    global_root = Path(args.debug_output_root if args.stage != "training" else args.training_output_root)
    if not global_root.is_absolute():
        global_root = ROOT / global_root
    global_root = global_root / timestamp
    global_root.mkdir(parents=True, exist_ok=True)
    (global_root / "tuning_summary.json").write_text(json.dumps(global_summaries, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = ["# Direct MAPPO Specialist Tuning Summary", ""]
    for item in global_summaries:
        best = item.get("best_run", {})
        lines.append(
            f"- `{item['task']}`: `{item['status']}`, best=`{best.get('run_name', '')}`, "
            f"success=`{best.get('best_eval_success_rate', 0.0)}`, reward=`{best.get('best_eval_reward', 0.0)}`"
        )
    (global_root / "tuning_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[direct-specialist-tune] summary_dir={global_root}", flush=True)


if __name__ == "__main__":
    main()
