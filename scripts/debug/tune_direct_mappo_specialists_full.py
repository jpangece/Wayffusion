from __future__ import annotations

import argparse
import json
import math
import sys
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.debug import tune_direct_mappo_specialists as direct
from scripts.debug import tune_mappo_specialists as common
from utils.email_notify import send_tuning_email
from utils.output_layout import minute_timestamp, resolve_output_root, short_run_name


TASKS = [
    "area_coverage",
    "belief_search",
    "priority_inspection",
    "connectivity_expansion",
]
V2_CONFIG_FILENAMES = {
    "area_coverage": ("direct_area_coverage_v2.yaml", "direct_area_coverage.yaml"),
    "belief_search": ("direct_belief_search_v2.yaml", "direct_belief_search.yaml"),
    "priority_inspection": ("direct_priority_inspection_v2.yaml", "direct_priority_inspection.yaml"),
    "connectivity_expansion": ("direct_connectivity_expansion_v2.yaml", "direct_connectivity_expansion.yaml"),
}


def _fmt_float(value: float) -> str:
    return f"{float(value):.2f}".replace("-", "m").replace(".", "p")


def _formal_updates(args: argparse.Namespace, task: str) -> int:
    if task == "belief_search":
        return int(args.formal_updates_belief)
    if task == "area_coverage":
        return int(args.formal_updates_area)
    if task == "priority_inspection":
        return int(args.formal_updates_priority)
    if task == "connectivity_expansion":
        return int(args.formal_updates_connectivity)
    raise ValueError(f"Unsupported task: {task}")


def _smoke_run_name(task: str, train_config: dict, env_config: dict, spec: direct.DirectTrialSpec, stage: str) -> str:
    del task, train_config, env_config, stage
    return short_run_name("smoke", float(spec.max_delta), int(getattr(_ACTIVE_ARGS, "seed", 0)), tag=spec.tag)


def _formal_run_name(task: str, train_config: dict, env_config: dict, spec: direct.DirectTrialSpec, stage: str) -> str:
    del task, stage
    return short_run_name(
        "formal",
        float(train_config["max_delta"]),
        int(env_config.get("seed", 0)),
        updates=int(train_config["total_updates"]),
        tag=spec.tag,
    )


_ACTIVE_ARGS: argparse.Namespace | None = None


def _task_metric_name(task: str) -> str:
    if task == "area_coverage":
        return "coverage_ratio"
    if task == "belief_search":
        return "searched_probability_mass"
    if task == "priority_inspection":
        return "weighted_poi_completion"
    if task == "connectivity_expansion":
        return "coverage_or_effective_radius"
    return "task_progress"


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def _last_record(run: dict) -> dict:
    csv_path = run.get("training_metrics_csv")
    if not csv_path:
        return {}
    records = common.read_csv_records(Path(csv_path))
    return records[-1] if records else {}


def _best_record(run: dict, task: str) -> dict:
    csv_path = run.get("training_metrics_csv")
    if not csv_path:
        return {}
    records = common.read_csv_records(Path(csv_path))
    return common.best_record(records, task)


def _enrich_run(run: dict, task: str) -> dict:
    last = _last_record(run)
    best = _best_record(run, task)
    for key in [
        "delta_clip_fraction",
        "delta_raw_norm_mean",
        "delta_raw_norm_max",
        "clipped_delta_norm_mean",
        "clipped_delta_norm_max",
        "approx_kl",
        "clip_frac",
        "entropy",
        "value_loss",
        "policy_std_mean",
        "log_std_mean",
        "train_action_std",
        "env_action_std",
        "action_validity_rate",
    ]:
        if key in last:
            run[key] = _finite(last.get(key))
    for key in ["eval_success_rate", "eval_reward", "eval_action_validity_rate"]:
        if key in best:
            run[f"best_{key}"] = _finite(best.get(key))
    run["best_task_metric_name"] = _task_metric_name(task)
    run["best_task_metric"] = common.task_progress(best, task) if best else _finite(run.get("best_task_progress", 0.0))
    return run


def _debug_passes(task: str, run: dict) -> tuple[bool, str]:
    if run.get("status") == "FAILED_RUNTIME":
        return False, str(run.get("reason", "runtime failure"))
    run = _enrich_run(run, task)
    action_validity = _finite(run.get("best_eval_action_validity_rate", run.get("action_validity_rate", 1.0)), 1.0)
    clip_fraction = _finite(run.get("delta_clip_fraction", 0.0))
    train_std = _finite(run.get("train_action_std", 0.0))
    max_delta = _finite((run.get("trial") or {}).get("max_delta", 0.2), 0.2)
    if action_validity < 0.90:
        return False, f"action_validity_rate {action_validity:.3f} < 0.90"
    if clip_fraction >= 0.90:
        return False, f"delta_clip_fraction {clip_fraction:.3f} is too high"
    if train_std > max_delta * 4.0:
        return False, f"train_action_std {train_std:.3f} is far larger than max_delta {max_delta:.3f}"
    success = _finite(run.get("best_eval_success_rate", 0.0))
    progress = _finite(run.get("best_task_metric", run.get("best_task_progress", 0.0)))
    tag = str((run.get("trial") or {}).get("tag", ""))
    if task == "connectivity_expansion" and "easy" in tag:
        return False, f"easy curriculum diagnostic only: success={success:.3f}, progress={progress:.3f}"
    if task == "belief_search":
        passed = success >= 0.75 or progress >= 0.50
        return passed, f"belief success={success:.3f}, searched_mass={progress:.3f}"
    if task == "area_coverage":
        passed = progress >= 0.40
        return passed, f"coverage_ratio={progress:.3f}"
    if task == "priority_inspection":
        passed = progress >= 0.40
        return passed, f"weighted_poi_completion={progress:.3f}"
    if task == "connectivity_expansion":
        passed = progress >= 0.45 and action_validity >= 0.90
        return passed, f"connectivity progress={progress:.3f}, action_validity={action_validity:.3f}"
    return False, "unsupported task"


def _formal_status(task: str, runs: list[dict]) -> tuple[str, str, dict]:
    completed = [run for run in runs if run.get("status") != "FAILED_RUNTIME"]
    if not completed:
        return "FAILED_RUNTIME", "All formal runs failed at runtime.", {}
    best = max(
        completed,
        key=lambda run: (
            _finite(run.get("best_eval_success_rate", 0.0)),
            _finite(run.get("best_task_metric", run.get("best_task_progress", 0.0))),
            _finite(run.get("best_eval_reward", -1e9)),
        ),
    )
    success_values = [_finite(run.get("best_eval_success_rate", 0.0)) for run in completed]
    mean_success = float(sum(success_values) / max(len(success_values), 1))
    action_validity = float(
        sum(_finite(run.get("best_eval_action_validity_rate", run.get("action_validity_rate", 1.0)), 1.0) for run in completed) / max(len(completed), 1)
    )
    metric_values = [_finite(run.get("best_task_metric", run.get("best_task_progress", 0.0))) for run in completed]
    mean_metric = float(sum(metric_values) / max(len(metric_values), 1))
    reason = []
    if task == "belief_search":
        if mean_success < 0.80:
            reason.append(f"mean_eval_success_rate {mean_success:.3f} < 0.80")
        if mean_metric < 0.55:
            reason.append(f"searched_probability_mass {mean_metric:.3f} < 0.55")
        if action_validity < 0.95:
            reason.append(f"action_validity_rate {action_validity:.3f} < 0.95")
    elif task == "area_coverage":
        if mean_success < 0.75:
            reason.append(f"mean_eval_success_rate {mean_success:.3f} < 0.75")
        if mean_metric < 0.55:
            reason.append(f"coverage_ratio {mean_metric:.3f} < 0.55")
        if action_validity < 0.95:
            reason.append(f"action_validity_rate {action_validity:.3f} < 0.95")
    elif task == "priority_inspection":
        if mean_success < 0.65:
            reason.append(f"mean_eval_success_rate {mean_success:.3f} < 0.65")
        if mean_metric < 0.70:
            reason.append(f"weighted_poi_completion {mean_metric:.3f} < 0.70")
        if action_validity < 0.95:
            reason.append(f"action_validity_rate {action_validity:.3f} < 0.95")
    elif task == "connectivity_expansion":
        if mean_success < 0.60:
            reason.append(f"mean_eval_success_rate {mean_success:.3f} < 0.60")
        if action_validity < 0.90:
            reason.append(f"action_validity_rate {action_validity:.3f} < 0.90")
        violation = max(_finite(run.get("best_connectivity_violation_rate", 0.0)) for run in completed)
        if violation > 0.10:
            reason.append(f"connectivity_violation_rate {violation:.3f} > 0.10")
    status = "CONVERGED" if not reason else "NEED_MORE_TUNING"
    details = {
        "mean_eval_success_rate": mean_success,
        "mean_task_metric": mean_metric,
        "mean_action_validity_rate": action_validity,
        "best_run": best,
    }
    return status, "; ".join(reason) if reason else "Success criteria met.", details


def _next_action(task: str, runs: list[dict], reason: str) -> str:
    if not runs:
        return "Fix runtime errors before more tuning."
    last = runs[-1]
    clip_fraction = _finite(last.get("delta_clip_fraction", 0.0))
    kl = _finite(last.get("approx_kl", 0.0))
    metric = _finite(last.get("best_task_metric", last.get("best_task_progress", 0.0)))
    if clip_fraction > 0.50:
        return "Lower log_std_init/log_std_max or max_delta; do not raise learning_rate until clipping is controlled."
    if kl < 1e-6 and metric <= 0.0:
        return "Inspect advantage_std/reward scale and then try learning_rate=3e-4 only if action distribution is healthy."
    if task == "area_coverage" and metric < 0.55:
        return "Next round can adjust coverage reward: increase new coverage reward/overlap penalty and lower distance penalty."
    if task == "priority_inspection" and metric < 0.70:
        return "Next round can add curriculum over num_pois and strengthen first-visit/repeat-penalty reward shaping."
    if task == "connectivity_expansion":
        return "Use curriculum comm_radius 0.42 -> 0.39 -> 0.36; do not count easy-comm as target success."
    return f"Continue Direct MAPPO tuning; current blocker: {reason}"


def _configure_common(run_name_fn) -> None:
    common.SPECIALIST_CONFIGS = direct.DIRECT_SPECIALIST_CONFIGS
    common.apply_trial = direct.apply_direct_trial
    common.run_name = run_name_fn
    common.trial_specs = direct.direct_trial_specs
    common.next_suggestion = direct.next_suggestion


def _use_specialist_config_dir(config_dir: str | None) -> None:
    if not config_dir:
        return
    root = Path(config_dir)
    mapping: dict[str, str] = {}
    for task, candidates in V2_CONFIG_FILENAMES.items():
        for filename in candidates:
            path = root / filename
            if path.exists():
                mapping[task] = str(path)
                break
        if task not in mapping:
            raise FileNotFoundError(f"No specialist config for task={task} under {root}; tried {candidates}")
    direct.DIRECT_SPECIALIST_CONFIGS = mapping


def _base_args(args: argparse.Namespace, timestamp: str, stage: str, seed: int, output_root: str, total_updates: int) -> argparse.Namespace:
    return SimpleNamespace(
        stage=stage,
        tasks=args.tasks,
        config=args.config,
        env_config=args.env_config,
        debug_output_root=output_root,
        training_output_root=output_root,
        timestamp=timestamp,
        num_agents=int(args.num_agents),
        seed=int(seed),
        debug_max_trials_per_task=int(args.max_debug_trials),
        training_max_trials_per_task=int(args.max_formal_trials),
        debug_total_updates=int(args.debug_total_updates),
        training_total_updates=int(total_updates),
        debug_num_envs=int(args.debug_num_envs),
        training_num_envs=int(args.formal_num_envs),
        debug_rollout_steps=int(args.debug_rollout_steps),
        debug_eval_interval=int(args.debug_eval_interval),
        training_eval_interval=int(args.formal_eval_interval),
        debug_eval_episodes=int(args.debug_eval_episodes),
        training_eval_episodes=int(args.formal_eval_episodes),
        debug_record_eval_episodes=int(args.debug_record_eval_episodes),
        training_record_eval_episodes=int(args.formal_record_eval_episodes),
        record_format=args.record_format,
        record_fps=int(args.record_fps),
        record_interval=int(args.record_interval),
        env_backend=args.env_backend,
        env_workers=args.env_workers,
        tensorboard=bool(args.tensorboard),
        flat_phase_layout=True,
        headless=bool(args.headless),
        dry_run=bool(args.dry_run),
        trial_tags=None,
    )


def _run_smoke_trials(task: str, args: argparse.Namespace, timestamp: str) -> tuple[list[dict], dict | None, str]:
    global _ACTIVE_ARGS
    smoke_root = str(Path(args.debug_output_root))
    smoke_args = _base_args(args, timestamp, "debug", int(args.seeds[0]), smoke_root, int(args.debug_total_updates))
    _ACTIVE_ARGS = smoke_args
    _configure_common(_smoke_run_name)
    runs: list[dict] = []
    passing_run: dict | None = None
    specs = direct.direct_trial_specs(task, "debug", int(args.debug_rollout_steps))[: int(args.max_debug_trials)]
    for spec in specs:
        print(f"[direct-full] smoke task={task} trial={spec.tag}", flush=True)
        run = common.run_trial(task, "debug", spec, smoke_args, timestamp)
        run = _enrich_run(run, task)
        passed, reason = _debug_passes(task, run)
        run["smoke_passed"] = passed
        run["smoke_reason"] = reason
        runs.append(run)
        if passed:
            passing_run = run
            break
    if passing_run is None:
        return runs, None, runs[-1].get("smoke_reason", "no smoke trials passed") if runs else "no smoke trials ran"
    return runs, passing_run, str(passing_run.get("smoke_reason", "smoke passed"))


def _run_formal(task: str, passing_run: dict, args: argparse.Namespace, timestamp: str) -> list[dict]:
    _configure_common(_formal_run_name)
    formal_root = str(Path(args.training_output_root))
    runs: list[dict] = []
    selected = dict(passing_run.get("trial") or {})
    selected["rollout_steps"] = 256
    selected["minibatch_size"] = 512
    specs = [direct.DirectTrialSpec(**selected)]
    total_updates = _formal_updates(args, task)
    for spec in specs[: int(args.max_formal_trials)]:
        for seed in [int(item) for item in args.seeds]:
            formal_args = _base_args(args, timestamp, "training", seed, formal_root, total_updates)
            print(f"[direct-full] formal task={task} seed={seed} trial={spec.tag}", flush=True)
            run = common.run_trial(task, "training", spec, formal_args, timestamp)
            run = _enrich_run(run, task)
            best_summary = Path(run.get("output_dir", "")) / "best_eval_summary.json"
            if best_summary.exists():
                try:
                    run["best_checkpoint_path"] = json.loads(best_summary.read_text(encoding="utf-8")).get("checkpoint_path", "")
                except Exception:
                    run["best_checkpoint_path"] = ""
            runs.append(run)
    return runs


def _write_report(summary_dir: Path, results: list[dict], timestamp: str) -> tuple[Path, Path]:
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_path = summary_dir / "summary.json"
    report_path = summary_dir / "report.md"
    summary = {
        "timestamp": timestamp,
        "gae_truncation_fix": True,
        "direct_action_clipping_diagnostics": True,
        "tasks": results,
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# Direct MAPPO Specialist Full Tuning Report",
        "",
        f"- Timestamp: `{timestamp}`",
        "- GAE truncation fix: `true`",
        "- Direct action clipping diagnostics: `true`",
        "",
        "## Task Summary",
        "",
        "| task | status | entered formal | best success | best metric | delta_clip_fraction | approx_kl | clip_frac | best run |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for item in results:
        best = item.get("best_run", {}) or {}
        lines.append(
            f"| `{item['task']}` | `{item['status']}` | `{item.get('entered_formal', False)}` | "
            f"{_finite(best.get('best_eval_success_rate', 0.0)):.3f} | "
            f"{_finite(best.get('best_task_metric', best.get('best_task_progress', 0.0))):.3f} | "
            f"{_finite(best.get('delta_clip_fraction', 0.0)):.3f} | "
            f"{_finite(best.get('approx_kl', 0.0)):.6g} | "
            f"{_finite(best.get('clip_frac', 0.0)):.3f} | `{best.get('output_dir', '')}` |"
        )
    lines.extend(["", "## Per-Task Details", ""])
    for item in results:
        lines.extend(
            [
                f"### {item['task']}",
                "",
                f"- Status: `{item['status']}`",
                f"- Reason: {item.get('reason', '')}",
                f"- Next action: {item.get('next_action', '')}",
                f"- Smoke runs: `{len(item.get('smoke_runs', []))}`",
                f"- Formal runs: `{len(item.get('formal_runs', []))}`",
                "",
            ]
        )
        best = item.get("best_run", {}) or {}
        if best:
            lines.extend(
                [
                    f"- Best run: `{best.get('run_name', '')}`",
                    f"- Best checkpoint: `{best.get('best_checkpoint_path', '')}`",
                    f"- Best eval_success_rate: `{best.get('best_eval_success_rate', 0.0)}`",
                    f"- Best task metric: `{best.get('best_task_metric', best.get('best_task_progress', 0.0))}`",
                    f"- Output: `{best.get('output_dir', '')}`",
                    "",
                ]
            )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary_path, report_path


def _email_subject(results: list[dict]) -> str:
    statuses = {item.get("status") for item in results}
    if "FAILED_RUNTIME" in statuses:
        return "[Wayffusion] Direct MAPPO specialists finished: RUNTIME FAILURE"
    if statuses == {"CONVERGED"}:
        return "[Wayffusion] Direct MAPPO specialists finished: ALL CONVERGED"
    return "[Wayffusion] Direct MAPPO specialists finished: NEED MORE TUNING"


def _email_body(results: list[dict], report_path: Path, timestamp: str) -> str:
    converged = [item["task"] for item in results if item.get("status") == "CONVERGED"]
    need = [item["task"] for item in results if item.get("status") == "NEED_MORE_TUNING"]
    failed = [item["task"] for item in results if item.get("status") == "FAILED_RUNTIME"]
    lines = [
        f"timestamp: {timestamp}",
        f"repo_path: {ROOT}",
        f"total_tasks: {len(results)}",
        f"CONVERGED: {', '.join(converged) or 'none'}",
        f"NEED_MORE_TUNING: {', '.join(need) or 'none'}",
        f"FAILED_RUNTIME: {', '.join(failed) or 'none'}",
        f"report_path: {report_path}",
        "",
        "Best runs:",
    ]
    for item in results:
        best = item.get("best_run", {}) or {}
        lines.append(
            f"- {item['task']}: status={item['status']}, success={best.get('best_eval_success_rate', 0.0)}, "
            f"run={best.get('output_dir', '')}, next={item.get('next_action', '')}"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Full Direct MAPPO specialist smoke-to-formal tuning flow.")
    parser.add_argument("--tasks", nargs="+", default=TASKS)
    parser.add_argument("--mode", choices=["full"], default="full")
    parser.add_argument("--config", default="configs/policy/mappo_waypoint.yaml")
    parser.add_argument("--env-config", default="configs/env/waypoint_missions.yaml")
    parser.add_argument("--specialist-config-dir", default=None)
    parser.add_argument("--debug-output-root", default="outputs/debug")
    parser.add_argument("--training-output-root", default="outputs/training")
    parser.add_argument("--phase-name", default="phase_change_on_direct_mappo_specialists")
    parser.add_argument("--timestamp", default=None)
    parser.add_argument("--num-agents", type=int, default=4)
    parser.add_argument("--max-debug-trials", type=int, default=6)
    parser.add_argument("--max-formal-trials", type=int, default=3)
    parser.add_argument("--formal-updates-belief", type=int, default=1000)
    parser.add_argument("--formal-updates-area", type=int, default=1500)
    parser.add_argument("--formal-updates-priority", type=int, default=1500)
    parser.add_argument("--formal-updates-connectivity", type=int, default=2000)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--debug-total-updates", type=int, default=20)
    parser.add_argument("--debug-num-envs", type=int, default=4)
    parser.add_argument("--debug-rollout-steps", type=int, default=64)
    parser.add_argument("--debug-eval-interval", type=int, default=5)
    parser.add_argument("--debug-eval-episodes", type=int, default=4)
    parser.add_argument("--debug-record-eval-episodes", type=int, default=1)
    parser.add_argument("--formal-num-envs", type=int, default=8)
    parser.add_argument("--formal-eval-interval", type=int, default=50)
    parser.add_argument("--formal-eval-episodes", type=int, default=20)
    parser.add_argument("--formal-record-eval-episodes", type=int, default=2)
    parser.add_argument("--record-format", choices=["gif", "mp4"], default="gif")
    parser.add_argument("--record-fps", type=int, default=8)
    parser.add_argument("--record-interval", type=int, default=1)
    parser.add_argument("--env-backend", choices=["sync", "thread"], default="sync")
    parser.add_argument("--env-workers", type=int, default=None)
    parser.add_argument("--tensorboard", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--send-email", action="store_true")
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    _use_specialist_config_dir(args.specialist_config_dir)

    timestamp = args.timestamp or minute_timestamp()
    args.debug_output_root = str(resolve_output_root(args.debug_output_root, args.phase_name, timestamp, ROOT))
    args.training_output_root = str(resolve_output_root(args.training_output_root, args.phase_name, timestamp, ROOT))
    staged: list[dict] = []
    for task in [str(item) for item in args.tasks]:
        if task not in TASKS:
            raise ValueError(f"Unsupported first-round Direct specialist task: {task}")
        smoke_runs, passing_run, smoke_reason = _run_smoke_trials(task, args, timestamp)
        staged.append(
            {
                "task": task,
                "smoke_runs": smoke_runs,
                "passing_run": passing_run,
                "smoke_reason": smoke_reason,
            }
        )

    results: list[dict] = []
    for item in staged:
        task = str(item["task"])
        smoke_runs = list(item["smoke_runs"])
        passing_run = item["passing_run"]
        smoke_reason = str(item["smoke_reason"])
        formal_runs: list[dict] = []
        if passing_run is not None and not bool(args.dry_run):
            formal_runs = _run_formal(task, passing_run, args, timestamp)
            status, reason, formal_details = _formal_status(task, formal_runs)
            best_run = formal_details.get("best_run", passing_run)
            entered_formal = True
        else:
            status = "FAILED_RUNTIME" if smoke_runs and all(run.get("status") == "FAILED_RUNTIME" for run in smoke_runs) else "NEED_MORE_TUNING"
            reason = f"Smoke did not pass: {smoke_reason}"
            best_run = max(
                smoke_runs,
                key=lambda run: (
                    _finite(run.get("best_eval_success_rate", 0.0)),
                    _finite(run.get("best_task_metric", run.get("best_task_progress", 0.0))),
                    _finite(run.get("best_eval_reward", -1e9)),
                ),
                default={},
            )
            entered_formal = False
        all_runs = smoke_runs + formal_runs
        results.append(
            {
                "task": task,
                "status": status,
                "reason": reason,
                "entered_formal": entered_formal,
                "best_run": best_run,
                "smoke_runs": smoke_runs,
                "formal_runs": formal_runs,
                "next_action": _next_action(task, all_runs, reason) if status != "CONVERGED" else "Run robustness validation on held-out seeds before final release.",
            }
        )

    summary_dir = Path(args.debug_output_root)
    summary_path, report_path = _write_report(summary_dir, results, timestamp)
    if args.send_email:
        result = send_tuning_email(
            _email_subject(results),
            _email_body(results, report_path, timestamp),
            [str(report_path), str(summary_path)],
        )
        (summary_dir / "email_result.json").write_text(json.dumps(asdict(result), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[direct-full] report={report_path}", flush=True)


if __name__ == "__main__":
    main()
