from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
PHASE_NAME = "phase10_mappo_specialist_protocol_check"
DEFAULT_OUTPUT_ROOT = ROOT / "outputs/debug/mappo_direct_specialists"
TASK_CONFIGS = {
    "belief_search": "configs/policy/direct_specialists/direct_belief_search.yaml",
    "area_coverage": "configs/policy/direct_specialists/direct_area_coverage.yaml",
}
TASK_METRICS = {
    "belief_search": ["searched_probability_mass"],
    "area_coverage": ["coverage_ratio", "overlap_ratio"],
}


def _runtime() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _finite_float(value: Any, default: float | None = None) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _latest_eval_record(records: list[dict[str, str]]) -> dict[str, str]:
    for record in reversed(records):
        if record.get("eval_fixed_success_rate", "") and record.get("eval_random_success_rate", ""):
            return record
    return {}


def _metric(record: dict[str, str], key: str, default: float | None = None) -> float | None:
    return _finite_float(record.get(key), default)


def _task_eval_metric(record: dict[str, str], mode: str, task: str, metric: str) -> float | None:
    return _metric(record, f"eval_{mode}_{task}_{metric}_mean")


def _has_media(run_dir: Path, mode: str) -> tuple[bool, list[str]]:
    media_root = run_dir / "media" / f"eval_{mode}"
    paths = sorted(str(path) for path in media_root.rglob("*.gif")) if media_root.exists() else []
    return bool(paths), paths


def _clip_is_healthy(records: list[dict[str, str]]) -> tuple[bool, float | None, float | None]:
    values = [
        value
        for record in records
        if (value := _metric(record, "delta_clip_fraction")) is not None
    ]
    if not values:
        return False, None, None
    recent = values[len(values) // 2 :]
    recent_mean = float(sum(recent) / len(recent))
    high_fraction = float(sum(value >= 0.95 for value in recent) / len(recent))
    return recent_mean < 0.95 and high_fraction < 0.5, recent_mean, high_fraction


def _build_command(task: str, output_dir: Path, args: argparse.Namespace) -> list[str]:
    return [
        sys.executable,
        str(ROOT / "scripts/train_mappo_waypoint.py"),
        "--config",
        TASK_CONFIGS[task],
        "--env-config",
        "configs/env/waypoint_missions.yaml",
        "--eval-config",
        "configs/eval/waypoint_eval.yaml",
        "--tasks",
        task,
        "--seed",
        "0",
        "--run_name",
        "s0",
        "--output-dir",
        str(output_dir),
        "--train-randomization",
        "random",
        "--eval-randomization",
        "both",
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
        str(args.record_eval_episodes),
        "--record_format",
        "gif",
        "--record_fps",
        "8",
        "--headless",
        "--tensorboard",
    ]


def _run_command(command: list[str], output_dir: Path) -> tuple[int, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "phase10_command.json").write_text(
        json.dumps(command, indent=2) + "\n",
        encoding="utf-8",
    )
    log_path = output_dir / "phase10_stdout.log"
    environment = dict(os.environ)
    environment["PYTHONUNBUFFERED"] = "1"
    with log_path.open("w", encoding="utf-8") as log_handle:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log_handle.write(line)
            log_handle.flush()
        return_code = process.wait()
    return return_code, str(log_path)


def _inspect_run(task: str, output_dir: Path, return_code: int, stdout_log: str) -> dict[str, Any]:
    metrics_path = output_dir / "training_metrics.csv"
    eval_path = output_dir / "eval_metrics.csv"
    records = _read_csv(metrics_path)
    latest = _latest_eval_record(records)
    fields = set(records[0]) if records else set()
    failures: list[str] = []

    if return_code != 0:
        failures.append(f"training command exited with code {return_code}")
    if not metrics_path.exists() or not records:
        failures.append("training_metrics.csv missing or empty")
    required_prefixes = ("eval_fixed_", "eval_random_")
    for prefix in required_prefixes:
        if not any(field.startswith(prefix) for field in fields):
            failures.append(f"training_metrics.csv has no {prefix} fields")
    if not latest:
        failures.append("no row contains both fixed and random evaluation metrics")

    domain_values = [
        value
        for record in records
        if (value := _metric(record, "domain_randomization_enabled")) is not None
    ]
    if not domain_values:
        failures.append("domain_randomization_enabled is missing")
    elif min(domain_values) < 0.999:
        failures.append(f"training domain_randomization_enabled dropped below 1: min={min(domain_values):.3f}")

    validity_values = [
        value
        for record in records
        if (value := _metric(record, "action_validity_rate")) is not None
    ]
    action_validity = validity_values[-1] if validity_values else None
    if action_validity is None:
        failures.append("action_validity_rate is missing or non-finite")
    elif action_validity < 0.90:
        failures.append(f"action_validity_rate {action_validity:.3f} < 0.90")

    clip_healthy, clip_recent_mean, clip_high_fraction = _clip_is_healthy(records)
    if not clip_healthy:
        failures.append(
            "delta_clip_fraction is missing or remains near 1 "
            f"(recent_mean={clip_recent_mean}, high_fraction={clip_high_fraction})"
        )

    finite_fields = ["policy_std_mean", "approx_kl", "clip_frac"]
    for field in finite_fields:
        value = _metric(records[-1], field) if records else None
        if value is None:
            failures.append(f"{field} is missing or non-finite")

    fixed_media_ok, fixed_media = _has_media(output_dir, "fixed")
    random_media_ok, random_media = _has_media(output_dir, "random")
    if not fixed_media_ok:
        failures.append("fixed eval GIF directory is missing or empty")
    if not random_media_ok:
        failures.append("random eval GIF directory is missing or empty")

    result: dict[str, Any] = {
        "task": task,
        "completed": return_code == 0,
        "status": "READY_FOR_TUNING" if not failures else "NEED_FIX",
        "output_dir": str(output_dir),
        "return_code": return_code,
        "stdout_log": stdout_log,
        "training_metrics_csv": str(metrics_path),
        "eval_metrics_csv": str(eval_path),
        "eval_fixed_success_rate": _metric(latest, "eval_fixed_success_rate"),
        "eval_random_success_rate": _metric(latest, "eval_random_success_rate"),
        "eval_generalization_gap": _metric(latest, "eval_generalization_gap"),
        "eval_fixed_reward": _metric(latest, "eval_fixed_reward"),
        "eval_random_reward": _metric(latest, "eval_random_reward"),
        "delta_clip_fraction": _metric(records[-1], "delta_clip_fraction") if records else None,
        "delta_clip_fraction_recent_mean": clip_recent_mean,
        "delta_clip_fraction_recent_high_fraction": clip_high_fraction,
        "policy_std_mean": _metric(records[-1], "policy_std_mean") if records else None,
        "approx_kl": _metric(records[-1], "approx_kl") if records else None,
        "clip_frac": _metric(records[-1], "clip_frac") if records else None,
        "action_validity_rate": action_validity,
        "domain_randomization_enabled": domain_values[-1] if domain_values else None,
        "no_fly_zone_count": _metric(records[-1], "mean_no_fly_zone_count") if records else None,
        "fixed_media": fixed_media,
        "random_media": random_media,
        "failures": failures,
    }
    for metric_name in TASK_METRICS[task]:
        result[metric_name] = _task_eval_metric(latest, "random", task, metric_name)
        result[f"fixed_{metric_name}"] = _task_eval_metric(latest, "fixed", task, metric_name)
    return result


def _write_report(
    root: Path,
    runtime: str,
    task_results: list[dict[str, Any]],
    args: argparse.Namespace,
) -> tuple[Path, Path, str]:
    failures = [
        f"{result['task']}: {failure}"
        for result in task_results
        for failure in result["failures"]
    ]
    status = "READY_FOR_TUNING" if not failures else "NEED_FIX"
    summary = {
        "phase_name": PHASE_NAME,
        "runtime": runtime,
        "status": status,
        "protocol": {
            "seed": 0,
            "total_updates": int(args.total_updates),
            "num_envs": int(args.num_envs),
            "rollout_steps": int(args.rollout_steps),
            "eval_interval": int(args.eval_interval),
            "eval_episodes": int(args.eval_episodes),
            "record_eval_episodes": int(args.record_eval_episodes),
            "train_randomization": "random",
            "eval_randomization": "both",
        },
        "tasks": task_results,
        "failures": failures,
    }
    summary_path = root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# Phase 10 MAPPO Specialist Protocol Check",
        "",
        f"- Runtime: `{runtime}`",
        f"- Status: `{status}`",
        "- This is protocol validation, not a convergence claim.",
        "- Formal success thresholds remain active; low short-run success is allowed.",
        "",
        "## Task Results",
        "",
        "| Task | Completed | Fixed success | Random success | Gap | Action valid | Delta clip | KL | Status |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for result in task_results:
        lines.append(
            f"| {result['task']} | {result['completed']} | "
            f"{result['eval_fixed_success_rate']} | {result['eval_random_success_rate']} | "
            f"{result['eval_generalization_gap']} | {result['action_validity_rate']} | "
            f"{result['delta_clip_fraction']} | {result['approx_kl']} | {result['status']} |"
        )
    lines.extend(["", "## Detailed Metrics", ""])
    for result in task_results:
        lines.extend(
            [
                f"### {result['task']}",
                "",
                f"- Output: `{result['output_dir']}`",
                f"- Fixed reward: `{result['eval_fixed_reward']}`",
                f"- Random reward: `{result['eval_random_reward']}`",
                f"- Policy std: `{result['policy_std_mean']}`",
                f"- Clip fraction: `{result['clip_frac']}`",
                f"- Domain randomization enabled: `{result['domain_randomization_enabled']}`",
                f"- Mean no-fly-zone count: `{result['no_fly_zone_count']}`",
            ]
        )
        for metric_name in TASK_METRICS[result["task"]]:
            lines.append(
                f"- Random {metric_name}: `{result.get(metric_name)}`; "
                f"fixed: `{result.get(f'fixed_{metric_name}')}`"
            )
        lines.extend(
            [
                f"- Fixed GIFs: `{len(result['fixed_media'])}`",
                f"- Random GIFs: `{len(result['random_media'])}`",
                "",
            ]
        )
    lines.extend(["## Failures", ""])
    lines.extend([f"- {failure}" for failure in failures] or ["- None."])
    report_path = root / "report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary_path, report_path, status


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run phase10 Direct MAPPO protocol checks.")
    parser.add_argument("--runtime", default=None)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--tasks", nargs="+", choices=sorted(TASK_CONFIGS), default=list(TASK_CONFIGS))
    parser.add_argument("--total-updates", type=int, default=100)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--rollout-steps", type=int, default=64)
    parser.add_argument("--eval-interval", type=int, default=20)
    parser.add_argument("--eval-episodes", type=int, default=6)
    parser.add_argument("--record-eval-episodes", type=int, default=1)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    runtime = args.runtime or _runtime()
    phase_root = Path(args.output_root)
    if not phase_root.is_absolute():
        phase_root = ROOT / phase_root
    run_root = phase_root / PHASE_NAME / runtime
    run_root.mkdir(parents=True, exist_ok=True)

    task_results = []
    for task in args.tasks:
        output_dir = run_root / task / "s0"
        command = _build_command(task, output_dir, args)
        print(f"[phase10] task={task} output_dir={output_dir}", flush=True)
        return_code, stdout_log = _run_command(command, output_dir)
        result = _inspect_run(task, output_dir, return_code, stdout_log)
        task_results.append(result)
        print(f"[phase10] task={task} status={result['status']}", flush=True)
        for failure in result["failures"]:
            print(f"[phase10] task={task} failure={failure}", flush=True)

    summary_path, report_path, status = _write_report(run_root, runtime, task_results, args)
    print(f"[phase10] summary={summary_path}", flush=True)
    print(f"[phase10] report={report_path}", flush=True)
    print(f"[phase10] final_status={status}", flush=True)
    if status == "NEED_FIX":
        for result in task_results:
            for failure in result["failures"]:
                print(f"[phase10] NEED_FIX {result['task']}: {failure}", flush=True)
    return 0 if status == "READY_FOR_TUNING" else 1


if __name__ == "__main__":
    raise SystemExit(main())
