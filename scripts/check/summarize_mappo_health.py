from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def as_float(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        value = row.get(key, default)
        if value in {"", None}:
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def is_finite(value: float) -> bool:
    return math.isfinite(float(value))


def load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def summarize_run(run_dir: Path) -> dict:
    metrics = read_csv(run_dir / "training_metrics.csv")
    eval_rows = read_csv(run_dir / "eval_metrics.csv")
    if not metrics:
        return {
            "run_name": run_dir.name,
            "output_dir": str(run_dir),
            "passed": False,
            "failure_reason": "missing training_metrics.csv",
        }
    final = metrics[-1]
    eval_candidates = [row for row in metrics if row.get("eval_reward", "") not in {"", None}]
    best_eval = max(eval_candidates, key=lambda row: as_float(row, "eval_reward", float("-inf"))) if eval_candidates else final
    train_config = load_yaml(run_dir / "snapshot" / "train_config.yaml")
    env_config = load_yaml(run_dir / "snapshot" / "env_config.yaml")
    policy_loss = as_float(final, "policy_loss", float("nan"))
    value_loss = as_float(final, "value_loss", float("nan"))
    final_eval_reward = as_float(final, "eval_reward", as_float(best_eval, "eval_reward", float("nan")))
    gif_paths = sorted(str(path) for path in run_dir.rglob("*.gif"))
    checkpoint_paths = sorted(str(path) for path in (run_dir / "checkpoints").glob("*.pt")) if (run_dir / "checkpoints").exists() else []
    mpe_source = str(final.get("eval_mpe_source") or final.get("mpe_source") or "")
    uses_real = as_float(final, "eval_uses_real_mpe_core", as_float(final, "uses_real_mpe_core", 0.0))
    step_calls = as_float(final, "eval_mpe_world_step_calls", as_float(final, "mpe_world_step_calls", 0.0))
    action_validity = as_float(final, "eval_action_validity_rate", as_float(final, "action_validity_rate", 0.0))
    row = {
        "run_name": run_dir.name,
        "output_dir": str(run_dir),
        "policy_class": str(train_config.get("policy_class", "")),
        "tasks": " ".join(env_config.get("task_names", [])),
        "num_agents": int(env_config.get("num_agents", 0)),
        "total_updates": int(train_config.get("total_updates", as_float(final, "update", 0))),
        "final_eval_reward": final_eval_reward,
        "best_eval_reward": as_float(best_eval, "eval_reward", final_eval_reward),
        "final_eval_success_rate": as_float(final, "eval_success_rate", 0.0),
        "best_eval_success_rate": as_float(best_eval, "eval_success_rate", 0.0),
        "policy_loss": policy_loss,
        "value_loss": value_loss,
        "entropy": as_float(final, "entropy", 0.0),
        "approx_kl": as_float(final, "approx_kl", 0.0),
        "action_validity_rate": action_validity,
        "candidate_mask_empty_count": as_float(final, "candidate_mask_empty_count", 0.0),
        "mean_speed": as_float(final, "eval_mean_speed", as_float(final, "mean_speed", 0.0)),
        "max_speed_observed": as_float(final, "eval_max_speed_observed", as_float(final, "max_speed_observed", 0.0)),
        "mean_control_norm": as_float(final, "eval_mean_control_norm", as_float(final, "mean_control_norm", 0.0)),
        "collision_count": as_float(final, "eval_collision_count", as_float(final, "collision_count", 0.0)),
        "no_fly_violation_count": as_float(final, "eval_no_fly_violation_count", as_float(final, "no_fly_violation_count", 0.0)),
        "geofence_violation_count": as_float(final, "eval_geofence_violation_count", as_float(final, "geofence_violation_count", 0.0)),
        "connectivity_violation_rate": as_float(final, "eval_connectivity_expansion_connectivity_violation_rate_mean", as_float(final, "connectivity_violation_rate", 0.0)),
        "mpe_source": mpe_source,
        "uses_real_mpe_core": uses_real,
        "mpe_world_step_calls": step_calls,
        "gif_paths": json.dumps(gif_paths, ensure_ascii=False),
        "checkpoint_paths": json.dumps(checkpoint_paths, ensure_ascii=False),
        "eval_rows": len(eval_rows),
    }
    failures = []
    if not is_finite(policy_loss):
        failures.append("policy_loss_nonfinite")
    if not is_finite(value_loss):
        failures.append("value_loss_nonfinite")
    if not is_finite(final_eval_reward):
        failures.append("eval_reward_nonfinite")
    if action_validity < 0.98:
        failures.append("low_action_validity")
    if uses_real < 1.0:
        failures.append("not_real_mpe_core")
    if mpe_source != "third_party_openai_mpe":
        failures.append("wrong_mpe_source")
    if step_calls <= 0.0:
        failures.append("no_mpe_step_calls")
    if not gif_paths:
        failures.append("missing_gif")
    for key in ["max_speed_observed", "no_fly_violation_count", "geofence_violation_count"]:
        if not is_finite(float(row[key])):
            failures.append(f"{key}_nonfinite")
    row["passed"] = not failures
    row["failure_reason"] = ",".join(failures)
    return row


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    keys = sorted({key for row in rows for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, rows: list[dict], summary: dict) -> None:
    lines = [
        "# MAPPO Health Check Report",
        "",
        f"- Passed: `{summary['passed']}`",
        f"- Runs: `{summary['num_runs']}`",
        f"- Output root: `{summary['output_dir']}`",
        "",
        "| run | passed | policy | tasks | final eval reward | action validity | MPE source | GIFs | failures |",
        "| --- | --- | --- | --- | ---: | ---: | --- | ---: | --- |",
    ]
    for row in rows:
        gif_count = len(json.loads(row.get("gif_paths", "[]") or "[]"))
        lines.append(
            "| {run} | {passed} | {policy} | {tasks} | {reward:.3f} | {valid:.3f} | {source} | {gifs} | {failures} |".format(
                run=row.get("run_name", ""),
                passed=row.get("passed", False),
                policy=row.get("policy_class", ""),
                tasks=row.get("tasks", ""),
                reward=float(row.get("final_eval_reward", 0.0)),
                valid=float(row.get("action_validity_rate", 0.0)),
                source=row.get("mpe_source", ""),
                gifs=gif_count,
                failures=row.get("failure_reason", ""),
            )
        )
    lines.append("")
    lines.append("## GIF Paths")
    lines.append("")
    for row in rows:
        lines.append(f"### {row.get('run_name', '')}")
        for path_item in json.loads(row.get("gif_paths", "[]") or "[]"):
            lines.append(f"- `{path_item}`")
        if not json.loads(row.get("gif_paths", "[]") or "[]"):
            lines.append("- none")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default=str(ROOT / "outputs" / "debug" / "mappo_health_check"))
    parser.add_argument("--output-dir", default=str(ROOT / "outputs" / "debug" / "mappo_health_check"))
    args = parser.parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    if not input_dir.is_absolute():
        input_dir = ROOT / input_dir
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    run_dirs = sorted(path for path in input_dir.iterdir() if path.is_dir() and (path / "training_metrics.csv").exists())
    rows = [summarize_run(path) for path in run_dirs]
    summary = {
        "passed": bool(rows and all(row.get("passed", False) for row in rows)),
        "num_runs": len(rows),
        "output_dir": str(output_dir),
        "runs": rows,
    }
    write_csv(output_dir / "mappo_health_summary.csv", rows)
    (output_dir / "mappo_health_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_report(output_dir / "mappo_health_report.md", rows, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not summary["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
