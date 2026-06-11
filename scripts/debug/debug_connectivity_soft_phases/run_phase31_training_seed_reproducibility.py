from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import subprocess
import sys
import time
from argparse import Namespace
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.waypoint_marl_env import WaypointMultiUAVEnv
from policies import build_policy
from scripts.debug_connectivity_soft_phases.run_phase17_20 import (
    PHASES,
    make_env_config,
    make_policy_config,
    summarize_training,
)
from utils.waypoint_evaluation import evaluate_waypoint_policy_episodes


PHASE_NAME = "phase31_connectivity_training_seed_reproducibility"
TASK = "connectivity_expansion"
STAGE_KEYS = ("phase24", "phase25", "phase26")
TOTAL_UPDATES = 2000
NUM_ENVS = 8
ROLLOUT_STEPS = 128
EVAL_INTERVAL = 100
TRAIN_EVAL_EPISODES = 20
ROBUST_EVAL_EPISODES = 100


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def minute_runtime() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for record in records for key in record})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(records)


def number(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
        return parsed if math.isfinite(parsed) else default
    except (TypeError, ValueError):
        return default


def git_metadata() -> dict[str, str]:
    def run(*args: str) -> str:
        return subprocess.check_output(args, cwd=ROOT, text=True).strip()

    return {
        "branch": run("git", "branch", "--show-current"),
        "commit": run("git", "rev-parse", "HEAD"),
    }


def stage_args() -> Namespace:
    return Namespace(
        num_envs=NUM_ENVS,
        rollout_steps=ROLLOUT_STEPS,
        total_updates=TOTAL_UPDATES,
        eval_interval=EVAL_INTERVAL,
        eval_episodes=TRAIN_EVAL_EPISODES,
        epochs=4,
        minibatch_size=512,
        trial=1,
        tuning="baseline",
    )


def prepare_stage_configs(stage_key: str, seed: int, seed_root: Path) -> tuple[Path, Path]:
    phase = PHASES[stage_key]
    config_dir = seed_root / "configs" / phase.name
    args = stage_args()
    policy_path = make_policy_config(phase, args, config_dir)
    env_path = make_env_config(phase, args, config_dir)

    policy_config = read_yaml(policy_path)
    env_config = read_yaml(env_path)
    policy_config["seed"] = int(seed)
    env_config["seed"] = int(seed)
    env_config["connectivity_action_filter"] = False
    env_config["connectivity_candidate_filter"] = False
    env_config.setdefault("domain_randomization", {})["enabled"] = True
    write_yaml(policy_path, policy_config)
    write_yaml(env_path, env_config)
    return policy_path, env_path


def stage_command(
    stage_key: str,
    seed: int,
    policy_path: Path,
    env_path: Path,
    output_dir: Path,
    init_checkpoint: Path | None,
) -> list[str]:
    command = [
        sys.executable,
        str(ROOT / "scripts/train_mappo_waypoint.py"),
        "--config",
        str(policy_path),
        "--env-config",
        str(env_path),
        "--eval-config",
        str(ROOT / "configs/eval/waypoint_eval.yaml"),
        "--tasks",
        TASK,
        "--num_agents",
        "4",
        "--num_envs",
        str(NUM_ENVS),
        "--rollout_steps",
        str(ROLLOUT_STEPS),
        "--total_updates",
        str(TOTAL_UPDATES),
        "--eval_interval",
        str(EVAL_INTERVAL),
        "--eval_episodes",
        str(TRAIN_EVAL_EPISODES),
        "--record_eval_episodes",
        "1",
        "--record_format",
        "gif",
        "--record_interval",
        "1",
        "--seed",
        str(seed),
        "--run_name",
        f"s{seed}",
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
        "--phase-name",
        PHASE_NAME,
    ]
    if init_checkpoint is not None:
        command.extend(["--init-checkpoint", str(init_checkpoint)])
    return command


def filter_audit(rows: list[dict[str, str]]) -> dict[str, Any]:
    enabled_keys = [key for key in rows[0] if "connectivity_action_filter_enabled" in key] if rows else []
    replacement_keys = [key for key in rows[0] if "connectivity_filter_replacement_count" in key] if rows else []
    enabled_max = max(
        (number(row.get(key)) for row in rows for key in enabled_keys if row.get(key) not in ("", None)),
        default=0.0,
    )
    replacement_max = max(
        (number(row.get(key)) for row in rows for key in replacement_keys if row.get(key) not in ("", None)),
        default=0.0,
    )
    return {
        "enabled_keys": enabled_keys,
        "replacement_keys": replacement_keys,
        "hard_filter_enabled_max": enabled_max,
        "replacement_count_max": replacement_max,
        "valid": enabled_max == 0.0 and replacement_max == 0.0,
    }


def validate_stage(output_dir: Path) -> dict[str, Any]:
    rows = read_csv(output_dir / "training_metrics.csv")
    final_update = int(number(rows[-1].get("update"), -1.0)) if rows else -1
    final_checkpoint = output_dir / "checkpoints" / f"checkpoint_{TOTAL_UPDATES:04d}.pt"
    best_checkpoint = output_dir / "checkpoints" / "checkpoint_best_eval.pt"
    audit = filter_audit(rows)
    valid = (
        len(rows) == TOTAL_UPDATES
        and final_update == TOTAL_UPDATES
        and final_checkpoint.exists()
        and best_checkpoint.exists()
        and audit["valid"]
    )
    return {
        "valid": valid,
        "training_rows": len(rows),
        "actual_final_update": final_update,
        "expected_final_update": TOTAL_UPDATES,
        "final_checkpoint": str(final_checkpoint),
        "best_checkpoint": str(best_checkpoint),
        "final_checkpoint_exists": final_checkpoint.exists(),
        "best_checkpoint_exists": best_checkpoint.exists(),
        "training_early_stop": False if final_update == TOTAL_UPDATES else True,
        "smoke_or_short_debug": False,
        "filter_audit": audit,
        "best_metrics": summarize_training(output_dir),
    }


def archive_invalid_stage(output_dir: Path) -> None:
    if not output_dir.exists():
        return
    suffix = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    archived = output_dir.with_name(f"{output_dir.name}_invalid_{suffix}")
    shutil.move(str(output_dir), str(archived))


def run_stage(
    stage_key: str,
    seed: int,
    seed_root: Path,
    init_checkpoint: Path | None,
) -> tuple[Path, dict[str, Any]]:
    phase = PHASES[stage_key]
    output_dir = seed_root / phase.name
    existing = validate_stage(output_dir) if output_dir.exists() else {"valid": False}
    if existing.get("valid"):
        return output_dir, existing
    archive_invalid_stage(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    policy_path, env_path = prepare_stage_configs(stage_key, seed, seed_root)
    command = stage_command(stage_key, seed, policy_path, env_path, output_dir, init_checkpoint)
    (output_dir / "command.json").write_text(json.dumps(command, indent=2) + "\n", encoding="utf-8")
    metadata = {
        "stage": phase.name,
        "train_seed": seed,
        "started_at": utc_now(),
        "init_checkpoint": str(init_checkpoint) if init_checkpoint is not None else None,
        "random_initialization": init_checkpoint is None,
        "required_total_updates": TOTAL_UPDATES,
        "early_stop_allowed": False,
        "smoke_or_short_debug": False,
    }
    (output_dir / "phase31_stage_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (output_dir / "stdout.log").open("a", encoding="utf-8") as handle:
        result = subprocess.run(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, text=True)
    validation = validate_stage(output_dir)
    validation["return_code"] = int(result.returncode)
    validation["init_checkpoint"] = str(init_checkpoint) if init_checkpoint is not None else None
    validation["policy_config"] = str(policy_path)
    validation["env_config"] = str(env_path)
    (output_dir / "phase31_stage_validation.json").write_text(
        json.dumps(validation, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if result.returncode != 0 or not validation["valid"]:
        raise RuntimeError(
            f"{phase.name} seed={seed} failed validation: "
            f"return_code={result.returncode}, final_update={validation['actual_final_update']}, "
            f"filter_audit={validation['filter_audit']}"
        )
    return output_dir, validation


def load_policy_for_eval(checkpoint_path: Path, output_dir: Path):
    policy_config = read_yaml(output_dir / "snapshot" / "train_config.yaml")
    env_config = read_yaml(output_dir / "snapshot" / "env_config.yaml")
    env_config["connectivity_action_filter"] = False
    env_config["connectivity_candidate_filter"] = False
    env_config.setdefault("domain_randomization", {})["enabled"] = True
    env = WaypointMultiUAVEnv(env_config)
    policy = build_policy(policy_config, env.global_observation_space, env.action_space_n)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    policy.load_state_dict(checkpoint["model_state_dict"], strict=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy.to(device)
    policy.eval()
    env.close()
    return policy, env_config, device


def aggregate_eval(records: list[dict[str, Any]], checkpoint: Path) -> dict[str, Any]:
    def mean(key: str) -> float:
        return float(np.mean([number(record.get(key)) for record in records]))

    replacement_total = float(sum(number(record.get("connectivity_filter_replacement_count")) for record in records))
    hard_filter_max = float(max((number(record.get("connectivity_action_filter_enabled")) for record in records), default=0.0))
    recordings = [str(record["recording_path"]) for record in records if record.get("recording_path")]
    return {
        "checkpoint": str(checkpoint),
        "episodes": len(records),
        "deterministic_policy": True,
        "eval_randomization": "random",
        "success_rate": mean("success_rate"),
        "return": mean("return"),
        "coverage_ratio": mean("coverage_ratio"),
        "effective_explored_radius": mean("effective_explored_radius"),
        "connectivity_violation_rate": mean("connectivity_violation_rate"),
        "connected_to_base_ratio": mean("connected_to_base_ratio"),
        "connected_new_coverage_ratio": mean("connected_new_coverage_ratio"),
        "disconnected_new_coverage_ratio": mean("disconnected_new_coverage_ratio"),
        "action_validity_rate": mean("action_validity_rate"),
        "command_reject_rate": mean("command_reject_rate"),
        "command_update_rate": mean("command_update_rate"),
        "connectivity_action_filter_enabled": hard_filter_max,
        "connectivity_filter_replacement_count": replacement_total,
        "recordings": recordings,
    }


def run_robust_eval(seed: int, phase26_dir: Path, seed_root: Path) -> dict[str, Any]:
    checkpoint = phase26_dir / "checkpoints" / "checkpoint_best_eval.pt"
    eval_dir = seed_root / "eval_phase26_best_random100"
    if (eval_dir / "summary.json").exists():
        existing = json.loads((eval_dir / "summary.json").read_text(encoding="utf-8"))
        if (
            int(existing.get("episodes", 0)) == ROBUST_EVAL_EPISODES
            and number(existing.get("connectivity_action_filter_enabled")) == 0.0
            and number(existing.get("connectivity_filter_replacement_count")) == 0.0
        ):
            return existing
    archive_invalid_stage(eval_dir)
    eval_dir.mkdir(parents=True, exist_ok=True)
    policy, env_config, device = load_policy_for_eval(checkpoint, phase26_dir)
    env_config["seed"] = int(seed)
    records = evaluate_waypoint_policy_episodes(
        env_config,
        policy,
        ROBUST_EVAL_EPISODES,
        device,
        task_name=TASK,
        deterministic=True,
        headless=True,
        record_dir=eval_dir / "media",
        record_episodes=2,
        record_format="gif",
        record_fps=8,
        record_prefix=f"seed{seed:02d}_phase26_random100",
    )
    write_csv(eval_dir / "eval_records.csv", records)
    summary = aggregate_eval(records, checkpoint)
    summary["train_seed"] = seed
    summary["completed_at"] = utc_now()
    summary["valid"] = (
        len(records) == ROBUST_EVAL_EPISODES
        and summary["connectivity_action_filter_enabled"] == 0.0
        and summary["connectivity_filter_replacement_count"] == 0.0
    )
    (eval_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if not summary["valid"]:
        raise RuntimeError(f"Robust eval seed={seed} failed hard-filter audit: {summary}")
    return summary


def best_metric(stage: dict[str, Any], key: str) -> float:
    return number(stage.get("best_metrics", {}).get(key))


def failure_taxonomy(summary: dict[str, Any]) -> str:
    failures = []
    if number(summary.get("coverage_ratio")) < 0.55:
        failures.append("coverage")
    if number(summary.get("effective_explored_radius")) < 0.45:
        failures.append("radius")
    if number(summary.get("connectivity_violation_rate"), 1.0) > 0.10:
        failures.append("violation")
    if not failures:
        return "PASS"
    if failures == ["coverage"]:
        return "A. coverage-limited but safe"
    if failures == ["violation"]:
        return "B. violation-limited"
    if failures == ["radius"]:
        return "C. radius-limited"
    return "D. mixed failure"


def write_seed_report(seed_root: Path, seed: int, stages: dict[str, Any], robust: dict[str, Any]) -> None:
    lines = [
        f"# Phase31 Training Seed {seed}",
        "",
        "## Protocol",
        "",
        f"- Train seed: `{seed}`",
        "- Phase24 starts from random initialization.",
        "- Phase25 initializes only from this seed's Phase24 best checkpoint.",
        "- Phase26 initializes only from this seed's Phase25 best checkpoint.",
        "- Every training stage contains exactly 2000 PPO updates.",
        "- Training-level early stop: `false`.",
        "- Smoke/short-debug substitution: `false`.",
        "- Connectivity hard action/candidate filters: `false`.",
        "",
        "## Training Chain",
        "",
        "| stage | output | init checkpoint | final update | best success | coverage | radius | violation |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for stage_key in STAGE_KEYS:
        stage = stages[stage_key]
        lines.append(
            f"| {stage_key} | `{stage['output_dir']}` | `{stage.get('init_checkpoint') or 'random init'}` "
            f"| {stage['actual_final_update']} | {best_metric(stage, 'best_success'):.6f} "
            f"| {best_metric(stage, 'best_coverage'):.6f} | {best_metric(stage, 'best_radius'):.6f} "
            f"| {best_metric(stage, 'best_violation'):.6f} |"
        )
    lines.extend(
        [
            "",
            "## Randomized Deterministic Eval",
            "",
            f"- Checkpoint: `{robust['checkpoint']}`",
            f"- Episodes: `{robust['episodes']}`",
            f"- Success rate: `{robust['success_rate']:.6f}`",
            f"- Return: `{robust['return']:.6f}`",
            f"- Coverage: `{robust['coverage_ratio']:.6f}`",
            f"- Radius: `{robust['effective_explored_radius']:.6f}`",
            f"- Violation: `{robust['connectivity_violation_rate']:.6f}`",
            f"- Connected-to-base ratio: `{robust['connected_to_base_ratio']:.6f}`",
            f"- Connected new coverage ratio: `{robust['connected_new_coverage_ratio']:.6f}`",
            f"- Disconnected new coverage ratio: `{robust['disconnected_new_coverage_ratio']:.6f}`",
            f"- Action validity: `{robust['action_validity_rate']:.6f}`",
            f"- Command reject/update rate: `{robust['command_reject_rate']:.6f}` / `{robust['command_update_rate']:.6f}`",
            f"- Hard filter enabled: `{robust['connectivity_action_filter_enabled']}`",
            f"- Connectivity replacements: `{robust['connectivity_filter_replacement_count']}`",
            f"- Formal proxy: `{failure_taxonomy(robust)}`",
            "",
            "## Media",
            "",
        ]
    )
    lines.extend(f"- `{path}`" for path in robust.get("recordings", []))
    (seed_root / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_seed(seed: int, runtime_root: Path) -> int:
    seed_root = runtime_root / f"seed{seed:02d}"
    seed_root.mkdir(parents=True, exist_ok=True)
    state_path = seed_root / "state.json"
    state = {
        "phase_name": PHASE_NAME,
        "train_seed": seed,
        "status": "RUNNING",
        "started_at": utc_now(),
        "git": git_metadata(),
    }
    state_path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    stages: dict[str, Any] = {}
    init_checkpoint: Path | None = None
    try:
        for stage_key in STAGE_KEYS:
            output_dir, validation = run_stage(stage_key, seed, seed_root, init_checkpoint)
            validation["output_dir"] = str(output_dir)
            validation["init_checkpoint"] = str(init_checkpoint) if init_checkpoint is not None else None
            stages[stage_key] = validation
            init_checkpoint = output_dir / "checkpoints" / "checkpoint_best_eval.pt"
            state["stages"] = stages
            state["current_stage"] = stage_key
            state_path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        phase26_dir = Path(stages["phase26"]["output_dir"])
        robust = run_robust_eval(seed, phase26_dir, seed_root)
        state.update(
            {
                "status": "COMPLETED",
                "completed_at": utc_now(),
                "stages": stages,
                "robust_eval": robust,
                "failure_taxonomy": failure_taxonomy(robust),
            }
        )
        state_path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        write_seed_report(seed_root, seed, stages, robust)
        return 0
    except Exception as exc:
        state.update({"status": "FAILED_RUNTIME", "failed_at": utc_now(), "error": repr(exc), "stages": stages})
        state_path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (seed_root / "error.txt").write_text(f"{type(exc).__name__}: {exc}\n", encoding="utf-8")
        raise


def aggregate_stat(values: list[float], higher_is_better: bool) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "worst": float(min(values) if higher_is_better else max(values)),
    }


def build_aggregate(runtime_root: Path, seeds: list[int]) -> dict[str, Any]:
    seed_states = {}
    for seed in seeds:
        path = runtime_root / f"seed{seed:02d}" / "state.json"
        if path.exists():
            seed_states[str(seed)] = json.loads(path.read_text(encoding="utf-8"))
    completed = [state for state in seed_states.values() if state.get("status") == "COMPLETED"]
    metrics = {
        "success_rate": True,
        "coverage_ratio": True,
        "effective_explored_radius": True,
        "connectivity_violation_rate": False,
        "connected_new_coverage_ratio": True,
        "disconnected_new_coverage_ratio": False,
        "action_validity_rate": True,
    }
    aggregate_metrics = {}
    for key, higher_is_better in metrics.items():
        values = [number(state["robust_eval"].get(key)) for state in completed]
        if values:
            aggregate_metrics[key] = aggregate_stat(values, higher_is_better)
    per_seed = {}
    for seed, state in seed_states.items():
        robust = state.get("robust_eval", {})
        per_seed[seed] = {
            "status": state.get("status"),
            "formal_proxy": failure_taxonomy(robust) if robust else "NOT_EVALUATED",
            "robust_eval": robust,
        }
    all_completed = len(completed) == len(seeds)
    all_formal = all(item["formal_proxy"] == "PASS" for item in per_seed.values())
    mean_success = aggregate_metrics.get("success_rate", {}).get("mean", 0.0)
    mean_violation = aggregate_metrics.get("connectivity_violation_rate", {}).get("mean", 1.0)
    worst_violation = aggregate_metrics.get("connectivity_violation_rate", {}).get("worst", 1.0)
    mean_disconnected = aggregate_metrics.get("disconnected_new_coverage_ratio", {}).get("mean", 1.0)
    hard_filter_valid = all(
        number(state.get("robust_eval", {}).get("connectivity_action_filter_enabled")) == 0.0
        and number(state.get("robust_eval", {}).get("connectivity_filter_replacement_count")) == 0.0
        for state in completed
    )
    passed = (
        all_completed
        and all_formal
        and mean_success >= 0.75
        and mean_violation <= 0.10
        and worst_violation <= 0.12
        and mean_disconnected <= 0.01
        and hard_filter_valid
    )
    return {
        "phase_name": PHASE_NAME,
        "created_at": utc_now(),
        "git": git_metadata(),
        "requested_seeds": seeds,
        "completed_seeds": len(completed),
        "all_completed": all_completed,
        "all_formal_proxy_passed": all_formal,
        "hard_filter_audit_passed": hard_filter_valid,
        "metrics": aggregate_metrics,
        "per_seed": per_seed,
        "robustness_criteria": {
            "average_success_rate_min": 0.75,
            "average_violation_max": 0.10,
            "worst_seed_violation_warning_max": 0.12,
            "disconnected_new_coverage_ratio_near_zero_max": 0.01,
        },
        "passed": passed,
        "next_decision": (
            "Freeze connectivity specialist and move to multi-seed evaluation of the other specialists."
            if passed
            else "Use the per-seed failure taxonomy to select Phase32A/B/C/D; do not add another static reward sweep."
        ),
    }


def write_aggregate_report(runtime_root: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Phase31 Connectivity Training-Seed Reproducibility",
        "",
        "## Purpose",
        "",
        "This experiment tests whether the Phase24 -> Phase25 -> Phase26 training curriculum is reproducible across independent training seeds. It does not introduce a new reward, graph encoder, action filter, or policy action space.",
        "",
        "## Per-Seed Final Results",
        "",
        "| seed | status | success | coverage | radius | violation | connected new | disconnected new | validity | formal proxy |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for seed in summary["requested_seeds"]:
        item = summary["per_seed"].get(str(seed), {})
        robust = item.get("robust_eval", {})
        lines.append(
            f"| {seed} | {item.get('status', 'MISSING')} | {number(robust.get('success_rate')):.6f} "
            f"| {number(robust.get('coverage_ratio')):.6f} | {number(robust.get('effective_explored_radius')):.6f} "
            f"| {number(robust.get('connectivity_violation_rate')):.6f} "
            f"| {number(robust.get('connected_new_coverage_ratio')):.6f} "
            f"| {number(robust.get('disconnected_new_coverage_ratio')):.6f} "
            f"| {number(robust.get('action_validity_rate')):.6f} | {item.get('formal_proxy', 'NOT_EVALUATED')} |"
        )
    lines.extend(
        [
            "",
            "## Mean / Std / Worst",
            "",
            "| metric | mean | std | worst |",
            "|---|---:|---:|---:|",
        ]
    )
    for key, values in summary["metrics"].items():
        lines.append(f"| {key} | {values['mean']:.6f} | {values['std']:.6f} | {values['worst']:.6f} |")
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- All seeds completed: `{summary['all_completed']}`",
            f"- All formal proxies passed: `{summary['all_formal_proxy_passed']}`",
            f"- Hard filter/replacement audit passed: `{summary['hard_filter_audit_passed']}`",
            f"- Phase31 reproducibility passed: `{summary['passed']}`",
            f"- Next decision: {summary['next_decision']}",
            "",
            "## Failure Taxonomy",
            "",
            "- A: coverage < 0.55 while radius >= 0.45 and violation <= 0.10.",
            "- B: coverage >= 0.55 and radius >= 0.45 but violation > 0.10.",
            "- C: radius < 0.45 as the only failed formal metric.",
            "- D: multiple formal metrics fail together.",
        ]
    )
    (runtime_root / "aggregate_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_parent(args: argparse.Namespace, runtime_root: Path) -> int:
    runtime_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "phase_name": PHASE_NAME,
        "runtime": args.runtime,
        "started_at": utc_now(),
        "seeds": args.seeds,
        "gpus": args.gpus,
        "parallel_seeds": True,
        "protocol": {
            "stages": [PHASES[key].name for key in STAGE_KEYS],
            "updates_per_stage": TOTAL_UPDATES,
            "updates_per_seed": TOTAL_UPDATES * len(STAGE_KEYS),
            "robust_eval_episodes": ROBUST_EVAL_EPISODES,
            "hard_filter": False,
            "early_stop": False,
            "smoke_or_short_debug": False,
        },
        "git": git_metadata(),
    }
    (runtime_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    processes: dict[int, subprocess.Popen] = {}
    logs = {}
    for index, seed in enumerate(args.seeds):
        seed_root = runtime_root / f"seed{seed:02d}"
        seed_root.mkdir(parents=True, exist_ok=True)
        gpu = args.gpus[index % len(args.gpus)]
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--runtime",
            args.runtime,
            "--output-root",
            args.output_root,
            "--worker-seed",
            str(seed),
            "--gpu",
            str(gpu),
        ]
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        env["PYTHONHASHSEED"] = str(seed)
        log_handle = (seed_root / "orchestrator.log").open("a", encoding="utf-8")
        logs[seed] = log_handle
        processes[seed] = subprocess.Popen(
            command,
            cwd=ROOT,
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        print(f"[phase31] seed={seed} gpu={gpu} pid={processes[seed].pid}", flush=True)
    return_codes = {}
    while processes:
        for seed, process in list(processes.items()):
            code = process.poll()
            if code is None:
                continue
            return_codes[seed] = int(code)
            logs[seed].close()
            del processes[seed]
            print(f"[phase31] seed={seed} return_code={code}", flush=True)
        if processes:
            time.sleep(30)
    summary = build_aggregate(runtime_root, args.seeds)
    summary["worker_return_codes"] = return_codes
    (runtime_root / "aggregate_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_aggregate_report(runtime_root, summary)
    return 0 if all(code == 0 for code in return_codes.values()) and summary["passed"] else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run strict Phase31 connectivity training-seed reproducibility.")
    parser.add_argument("--runtime", default=minute_runtime())
    parser.add_argument(
        "--output-root",
        default="outputs/debug/mappo_direct_specialists/phase31_connectivity_training_seed_reproducibility",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--gpus", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--worker-seed", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--gpu", type=int, default=None, help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if sorted(args.seeds) != [1, 2, 3] and args.worker_seed is None:
        raise ValueError("Phase31 protocol requires exactly training seeds 1, 2, and 3.")
    if not args.gpus:
        raise ValueError("At least one GPU id is required.")
    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    runtime_root = output_root / args.runtime
    if args.worker_seed is not None:
        return run_seed(int(args.worker_seed), runtime_root)
    return run_parent(args, runtime_root)


if __name__ == "__main__":
    raise SystemExit(main())
