from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def run_command(cmd: list[str], cwd: Path = ROOT) -> dict:
    print("[debug-suite] running:", " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, cwd=cwd, text=True)
    status = {"cmd": cmd, "returncode": int(proc.returncode), "passed": proc.returncode == 0}
    print(f"[debug-suite] returncode={proc.returncode}", flush=True)
    return status


def write_final_report(output_root: Path, command_status: list[dict]) -> Path:
    calibration_path = output_root / "adapter_calibration" / "local_core_velocity_tracker_v1" / "adapter_calibration_summary.json"
    health_path = output_root / "mappo_health_check" / "mappo_health_summary.json"
    source_path = output_root / "mpe_core_source_check" / "source_check_summary.json"
    calibration = json.loads(calibration_path.read_text(encoding="utf-8")) if calibration_path.exists() else {}
    health = json.loads(health_path.read_text(encoding="utf-8")) if health_path.exists() else {}
    source = json.loads(source_path.read_text(encoding="utf-8")) if source_path.exists() else {}
    lines = [
        "# Wayffusion Adapter Calibration and MAPPO Health Debug Report",
        "",
        f"- output_root: `{output_root}`",
        f"- source_check_passed: `{source.get('passed', False)}`",
        f"- mpe_source: `{source.get('mpe_source', '')}`",
        f"- uses_real_mpe_core: `{source.get('uses_real_mpe_core', False)}`",
        f"- adapter_calibration_passed: `{calibration.get('passed', False)}`",
        f"- mappo_health_passed: `{health.get('passed', False)}`",
        "",
        "## Recommended Adapter Profiles",
        "",
    ]
    profiles = calibration.get("selected_profiles", {})
    for name, profile in profiles.items():
        lines.append(f"### {name}")
        for key in [
            "max_command_distance",
            "slowdown_radius",
            "velocity_gain",
            "max_accel",
            "control_smoothing",
            "acceptance_radius",
            "score",
            "rank",
        ]:
            if key in profile:
                lines.append(f"- {key}: `{profile[key]}`")
        lines.append("")
    lines.extend(["## MAPPO Runs", ""])
    for run in health.get("runs", []):
        lines.extend(
            [
                f"### {run.get('run_name', 'unknown')}",
                f"- passed: `{run.get('passed', False)}`",
                f"- output_dir: `{run.get('output_dir', '')}`",
                f"- policy_class: `{run.get('policy_class', '')}`",
                f"- tasks: `{run.get('tasks', '')}`",
                f"- final_eval_reward: `{run.get('final_eval_reward', '')}`",
                f"- action_validity_rate: `{run.get('action_validity_rate', '')}`",
                f"- max_speed_observed: `{run.get('max_speed_observed', '')}`",
                f"- mean_control_norm: `{run.get('mean_control_norm', '')}`",
                f"- collision_count: `{run.get('collision_count', '')}`",
                f"- connectivity_violation_rate: `{run.get('connectivity_violation_rate', '')}`",
                f"- mpe_source: `{run.get('mpe_source', '')}`",
                f"- mpe_world_step_calls: `{run.get('mpe_world_step_calls', '')}`",
                f"- failures: `{run.get('failure_reason', '')}`",
                "",
            ]
        )
    lines.extend(["## Command Status", ""])
    for item in command_status:
        lines.append(f"- `{item['returncode']}` {' '.join(item['cmd'])}")
    failed = [item for item in command_status if item["returncode"] != 0]
    if failed:
        lines.extend(["", "## Failed Commands", ""])
        for item in failed:
            lines.append(f"- {' '.join(item['cmd'])}")
    report = output_root / "final_debug_report.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default=str(ROOT / "outputs" / "debug"))
    parser.add_argument("--max-settings", type=int, default=80)
    parser.add_argument("--skip-calibration", action="store_true")
    parser.add_argument("--skip-mappo", action="store_true")
    parser.add_argument("--skip-email", action="store_true")
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    output_root.mkdir(parents=True, exist_ok=True)

    py = sys.executable
    statuses: list[dict] = []
    statuses.append(
        run_command(
            [
                py,
                "scripts/check/check_mpe_core_source.py",
                "--env-config",
                "configs/env/waypoint_missions.yaml",
                "--output-dir",
                str(output_root / "mpe_core_source_check"),
            ]
        )
    )
    statuses.append(
        run_command(
            [
                py,
                "-m",
                "pytest",
                "-q",
                "tests/test_mpe_core_backend.py",
                "tests/test_vendor_mpe_integrity.py",
                "tests/test_action_adapters.py",
                "tests/test_waypoint_env_api.py",
                "tests/test_candidate_selection_policy.py",
                "tests/test_direct_waypoint_policy.py",
                "tests/test_mappo_waypoint_trainer.py",
                "tests/test_waypoint_rewards.py",
                "tests/test_waypoint_rendering.py",
            ]
        )
    )
    if not args.skip_calibration:
        statuses.append(
            run_command(
                [
                    py,
                    "scripts/check/calibrate_waypoint_adapter.py",
                    "--env-config",
                    "configs/env/waypoint_missions.yaml",
                    "--tasks",
                    "area_coverage",
                    "belief_search",
                    "priority_inspection",
                    "connectivity_expansion",
                    "--num-agents-list",
                    "3",
                    "4",
                    "--seeds",
                    "0",
                    "1",
                    "2",
                    "--episodes-per-setting",
                    "2",
                    "--search-mode",
                    "staged",
                    "--max-settings",
                    str(args.max_settings),
                    "--record-top-k",
                    "5",
                    "--record-format",
                    "gif",
                    "--output-dir",
                    str(output_root / "adapter_calibration" / "local_core_velocity_tracker_v1"),
                    "--headless" if args.headless else "--no-headless",
                ]
            )
        )
    tuned = ROOT / "configs" / "env" / "waypoint_missions_tuned.yaml"
    env_config = str(tuned if tuned.exists() else ROOT / "configs" / "env" / "waypoint_missions.yaml")
    if not args.skip_mappo:
        health_root = output_root / "mappo_health_check"
        runs = [
            [
                py,
                "scripts/train_mappo_waypoint.py",
                "--config",
                "configs/policy/mappo_waypoint_debug.yaml",
                "--env-config",
                env_config,
                "--tasks",
                "area_coverage",
                "belief_search",
                "priority_inspection",
                "connectivity_expansion",
                "--num_agents",
                "4",
                "--num_envs",
                "4",
                "--total_updates",
                "20",
                "--eval_interval",
                "5",
                "--eval_episodes",
                "2",
                "--record_eval_episodes",
                "1",
                "--record_format",
                "gif",
                "--headless",
                "--output-dir",
                str(health_root / "candidate_4task_tuned_seed0"),
            ],
            [
                py,
                "scripts/train_mappo_waypoint.py",
                "--config",
                "configs/policy/mappo_waypoint_debug.yaml",
                "--env-config",
                env_config,
                "--tasks",
                "connectivity_expansion",
                "--num_agents",
                "6",
                "--num_envs",
                "4",
                "--total_updates",
                "15",
                "--eval_interval",
                "5",
                "--eval_episodes",
                "2",
                "--record_eval_episodes",
                "1",
                "--record_format",
                "gif",
                "--headless",
                "--output-dir",
                str(health_root / "candidate_connectivity_tuned_seed1"),
            ],
            [
                py,
                "scripts/train_mappo_waypoint.py",
                "--config",
                "configs/policy/direct_waypoint_debug.yaml",
                "--env-config",
                env_config,
                "--tasks",
                "area_coverage",
                "belief_search",
                "--num_agents",
                "3",
                "--num_envs",
                "4",
                "--total_updates",
                "15",
                "--eval_interval",
                "5",
                "--eval_episodes",
                "2",
                "--record_eval_episodes",
                "1",
                "--record_format",
                "gif",
                "--headless",
                "--output-dir",
                str(health_root / "direct_2task_tuned_seed0"),
            ],
        ]
        for cmd in runs:
            statuses.append(run_command(cmd))
        statuses.append(
            run_command(
                [
                    py,
                    "scripts/check/summarize_mappo_health.py",
                    "--input-dir",
                    str(health_root),
                    "--output-dir",
                    str(health_root),
                ]
            )
        )
    status_path = output_root / "debug_suite_command_status.json"
    status_path.write_text(json.dumps(statuses, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report = write_final_report(output_root, statuses)
    if not args.skip_email:
        statuses.append(
            run_command(
                [
                    py,
                    "scripts/check/send_smtp_report.py",
                    "--report-md",
                    str(report),
                    "--summary-json",
                    str(output_root / "mappo_health_check" / "mappo_health_summary.json"),
                    "--no-fail-if-missing-env",
                ]
            )
        )
        status_path.write_text(json.dumps(statuses, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    failed = [item for item in statuses if item["returncode"] != 0]
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
