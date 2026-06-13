from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.formal_mappo_specialists.run_formal_specialists import load_env_file
from utils.email_notify import send_tuning_email
from utils.swarm30_benchmark import (
    SWARM30_TASKS,
    build_scaled_env_config,
    sync_policy_scale,
)


TRACK_ALIASES = {
    "standard_specialist": "standardized_specialists",
    "standardized_specialist": "standardized_specialists",
    "standardized_specialists": "standardized_specialists",
    "tuned_specialist": "tuned_specialists",
    "tuned_specialists": "tuned_specialists",
    "generalist": "generalist",
}

TRACK_OUTPUT_NAMES = {
    "standardized_specialists": "standard_specialist",
    "tuned_specialists": "tuned_specialist",
    "generalist": "generalist",
}


@dataclass
class Job:
    job_id: str
    stage: str
    track: str
    seed: int
    tasks: list[str]
    gpu: int
    output_dir: Path
    command: list[str]
    expected_updates: int = 0


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


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def canonical_track(track: str) -> str:
    if track not in TRACK_ALIASES:
        supported = ", ".join(sorted(TRACK_ALIASES))
        raise ValueError(f"Unsupported track {track!r}. Supported tracks: {supported}")
    return TRACK_ALIASES[track]


def track_output_name(track: str) -> str:
    return TRACK_OUTPUT_NAMES[canonical_track(track)]


def final_update(path: Path) -> int:
    metrics = path / "training_metrics.csv"
    if not metrics.exists():
        return 0
    with metrics.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return 0
    try:
        return int(float(rows[-1].get("update", 0)))
    except (TypeError, ValueError):
        return 0


class Swarm30Benchmark:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.runtime = args.runtime or minute_runtime()
        self.protocol = read_yaml(ROOT / args.benchmark_config)
        self._apply_protocol_overrides()
        self.base_env = read_yaml(ROOT / args.env_config)
        self.output_root = Path(args.output_root)
        if not self.output_root.is_absolute():
            self.output_root = ROOT / self.output_root
        self.suite_root = self.output_root / self.runtime
        self.suite_root.mkdir(parents=True, exist_ok=True)
        self.results: dict[str, dict[str, Any]] = {}
        self.email_events: list[dict[str, Any]] = []
        load_env_file(ROOT / args.smtp_env_file)

    def _apply_protocol_overrides(self) -> None:
        train_cfg = self.protocol.setdefault("train", {})
        seeds = getattr(self.args, "seeds", None)
        if seeds:
            train_cfg["seeds"] = [int(seed) for seed in seeds]
        for arg_name in (
            "num_envs",
            "rollout_steps",
            "specialist_updates",
            "generalist_updates",
            "eval_interval",
            "eval_episodes",
            "record_eval_episodes",
        ):
            value = getattr(self.args, arg_name, None)
            if value is not None:
                train_cfg[arg_name] = int(value)
        env_backend = getattr(self.args, "env_backend", None)
        if env_backend is not None:
            train_cfg["env_backend"] = str(env_backend)
        dynamics_backend = getattr(self.args, "dynamics_backend", None)
        if dynamics_backend is not None:
            train_cfg["dynamics_backend"] = str(dynamics_backend)

    def _apply_dynamics_backend(self, env_config: dict[str, Any]) -> None:
        name = str(
            self.protocol.get("train", {}).get(
                "dynamics_backend",
                getattr(self.args, "dynamics_backend", "waypoint_behavior_fast"),
            )
        )
        if name == "mpe_core":
            env_config.setdefault("dynamics_backend", {})["name"] = "mpe_core"
            env_config["dynamics_backend"].setdefault("source", "third_party_openai_mpe")
            return
        if name == "waypoint_behavior_fast":
            env_config["dynamics_backend"] = {
                "name": "waypoint_behavior_fast",
                "dt": float(getattr(self.args, "backend_dt", 1.0)),
                "max_speed": float(env_config.get("max_speed", 0.04)),
                "acceptance_radius": float(getattr(self.args, "backend_acceptance_radius", 0.025)),
                "agent_size": float(getattr(self.args, "backend_agent_size", 0.02)),
                "enable_boundary_projection": True,
                "enable_no_fly_projection": True,
            }
            return
        if name == "waypoint_behavior_realistic":
            env_config["dynamics_backend"] = {
                "name": "waypoint_behavior_realistic",
                "dt": float(getattr(self.args, "backend_dt", 1.0)),
                "max_speed": float(env_config.get("max_speed", 0.04)),
                "acceptance_radius": float(getattr(self.args, "backend_acceptance_radius", 0.025)),
                "agent_size": float(getattr(self.args, "backend_agent_size", 0.02)),
                "velocity_smoothing": 0.35,
                "speed_scale_range": [0.85, 1.15],
                "tracking_noise_std": 0.005,
                "command_latency_steps": 1,
                "enable_boundary_projection": True,
                "enable_no_fly_projection": True,
            }
            return
        raise ValueError(f"Unsupported dynamics backend: {name}")

    def notify(self, event: str, subject: str, body: str, attachments: list[str] | None = None) -> None:
        if self.args.skip_email:
            result = {"sent": False, "reason": "skip_email", "missing_env": []}
        else:
            result = asdict(send_tuning_email(subject, body, attachments))
        self.email_events.append({"event": event, "at": utc_now(), **result})
        write_json(self.suite_root / "email_events.json", self.email_events)

    def save_state(self, status: str) -> None:
        write_json(
            self.suite_root / "suite_state.json",
            {
                "benchmark": "swarm30_v1",
                "runtime": self.runtime,
                "status": status,
                "updated_at": utc_now(),
                "results": self.results,
                "email_events": self.email_events,
            },
        )

    def prepare_training_jobs(self) -> list[Job]:
        train_cfg = self.protocol["train"]
        tracks_cfg = self.protocol["tracks"]
        seeds = [int(seed) for seed in train_cfg["seeds"]]
        jobs: list[Job] = []
        gpu_index = 0

        requested_tracks = [canonical_track(track) for track in self.args.tracks]
        for track in requested_tracks:
            output_track = track_output_name(track)
            if track == "generalist":
                task_sets = [list(SWARM30_TASKS)]
            else:
                task_sets = [[task] for task in SWARM30_TASKS]
            for tasks in task_sets:
                for seed in seeds:
                    task_key = "all_tasks" if len(tasks) > 1 else tasks[0]
                    output_dir = self.suite_root / output_track / task_key / f"s{seed}"
                    if track == "standardized_specialists":
                        policy_path = ROOT / tracks_cfg[track]["policy_config"]
                        updates = int(train_cfg["specialist_updates"])
                    elif track == "tuned_specialists":
                        policy_path = ROOT / tracks_cfg[track]["policy_configs"][tasks[0]]
                        updates = int(train_cfg["specialist_updates"])
                    elif track == "generalist":
                        policy_path = ROOT / tracks_cfg[track]["policy_config"]
                        updates = int(train_cfg["generalist_updates"])
                    else:
                        raise ValueError(f"Unsupported track: {track}")

                    env_config = build_scaled_env_config(self.base_env, 30, tasks, randomization=True)
                    self._apply_dynamics_backend(env_config)
                    env_config.setdefault("rendering", {})["render_detail"] = "low"
                    env_config.setdefault("runtime_acceleration", {})["sensor_footprint_mode"] = "disk_stamp"
                    env_config["seed"] = seed
                    if track == "tuned_specialists":
                        env_config["tasks"][tasks[0]].update(
                            deepcopy(tracks_cfg[track].get("env_task_overrides", {}).get(tasks[0], {}))
                        )
                    policy_config = sync_policy_scale(read_yaml(policy_path), env_config)
                    if bool(policy_config.get("use_spatial_field_context", False)):
                        absolute_radii = [
                            float(radius)
                            for radius in policy_config.get("spatial_context_radii", [0.05, 0.10, 0.18])
                        ]
                        policy_config["benchmark_spatial_context_radii_absolute"] = absolute_radii
                        policy_config["spatial_context_radii"] = [
                            radius / float(env_config["map_size"]) for radius in absolute_radii
                        ]
                    policy_config.update(
                        {
                            "seed": seed,
                            "num_envs": int(train_cfg["num_envs"]),
                            "rollout_steps": int(train_cfg["rollout_steps"]),
                            "total_updates": updates,
                            "eval_interval": int(train_cfg["eval_interval"]),
                            "connectivity_candidate_filter": False,
                        }
                    )
                    max_delta = float(policy_config.get("max_delta", env_config["max_waypoint_distance"]))
                    env_config["max_waypoint_distance"] = max_delta
                    env_config.setdefault("action_adapter", {})["max_command_distance"] = max_delta
                    policy_config["max_waypoint_distance"] = max_delta
                    resolved_dir = output_dir / "benchmark_configs"
                    env_path = resolved_dir / "env.yaml"
                    policy_out = resolved_dir / "policy.yaml"
                    write_yaml(env_path, env_config)
                    write_yaml(policy_out, policy_config)
                    command = [
                        sys.executable,
                        str(ROOT / "scripts/train_mappo_waypoint.py"),
                        "--config",
                        str(policy_out),
                        "--env-config",
                        str(env_path),
                        "--eval-config",
                        str(ROOT / "configs/eval/waypoint_eval.yaml"),
                        "--tasks",
                        *tasks,
                        "--num_agents",
                        "30",
                        "--num_envs",
                        str(train_cfg["num_envs"]),
                        "--rollout_steps",
                        str(train_cfg["rollout_steps"]),
                        "--total_updates",
                        str(updates),
                        "--eval_interval",
                        str(train_cfg["eval_interval"]),
                        "--eval_episodes",
                        str(train_cfg["eval_episodes"]),
                        "--record_eval_episodes",
                        str(train_cfg["record_eval_episodes"]),
                        "--record_format",
                        str(train_cfg["record_format"]),
                        "--record_fps",
                        str(train_cfg["record_fps"]),
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
                        str(train_cfg["env_backend"]),
                        "--headless",
                        "--tensorboard",
                    ]
                    jobs.append(
                        Job(
                            job_id=f"train__{track}__{task_key}__s{seed}",
                            stage="train",
                            track=track,
                            seed=seed,
                            tasks=tasks,
                            gpu=self.args.gpus[gpu_index % len(self.args.gpus)],
                            output_dir=output_dir,
                            command=command,
                            expected_updates=updates,
                        )
                    )
                    gpu_index += 1
        return jobs

    def prepare_evaluation_jobs(self, training_jobs: list[Job]) -> list[Job]:
        eval_cfg = self.protocol["evaluation"]
        jobs = []
        gpu_index = 0
        for train_job in training_jobs:
            checkpoint = train_job.output_dir / "checkpoints" / "checkpoint_best_eval.pt"
            output_dir = self.suite_root / "evaluations" / track_output_name(train_job.track) / (
                "all_tasks" if len(train_job.tasks) > 1 else train_job.tasks[0]
            ) / f"s{train_job.seed}"
            command = [
                sys.executable,
                str(ROOT / "scripts/benchmarks/swarm30/evaluate.py"),
                "--checkpoint",
                str(checkpoint),
                "--track",
                train_job.track,
                "--tasks",
                *train_job.tasks,
                "--scales",
                *[str(value) for value in eval_cfg["scales"]],
                "--modes",
                "fixed",
                "random",
                "--random-episodes",
                str(eval_cfg["random_episodes"]),
                "--fixed-episodes",
                str(eval_cfg["fixed_episodes"]),
                "--record-episodes",
                str(eval_cfg["record_episodes"]),
                "--record-format",
                str(eval_cfg["record_format"]),
                "--record-fps",
                str(eval_cfg["record_fps"]),
                "--seed",
                str(train_job.seed),
                "--output-dir",
                str(output_dir),
            ]
            jobs.append(
                Job(
                    job_id=train_job.job_id.replace("train__", "eval__", 1),
                    stage="eval",
                    track=train_job.track,
                    seed=train_job.seed,
                    tasks=train_job.tasks,
                    gpu=self.args.gpus[gpu_index % len(self.args.gpus)],
                    output_dir=output_dir,
                    command=command,
                )
            )
            gpu_index += 1
        return jobs

    def run_pool(self, jobs: list[Job], max_parallel: int) -> bool:
        pending = list(jobs)
        active: dict[str, tuple[Job, subprocess.Popen, Any]] = {}
        all_ok = True
        while pending or active:
            while pending and len(active) < max_parallel:
                job = pending.pop(0)
                if job.stage == "train":
                    complete = (
                        final_update(job.output_dir) == job.expected_updates
                        and (job.output_dir / "checkpoints/checkpoint_best_eval.pt").exists()
                    )
                else:
                    complete = (job.output_dir / "evaluation_index.json").exists()
                if complete:
                    self.results[job.job_id] = {
                        "status": "SKIPPED_COMPLETE",
                        "output_dir": str(job.output_dir),
                        "gpu": job.gpu,
                    }
                    continue
                job.output_dir.mkdir(parents=True, exist_ok=True)
                write_json(job.output_dir / f"{job.stage}_command.json", job.command)
                log_handle = (job.output_dir / f"{job.stage}_stdout.log").open("a", encoding="utf-8")
                environment = dict(os.environ)
                environment.update(
                    {
                        "CUDA_VISIBLE_DEVICES": str(job.gpu),
                        "PYTHONUNBUFFERED": "1",
                        "OMP_NUM_THREADS": str(self.args.cpu_threads_per_job),
                        "MKL_NUM_THREADS": str(self.args.cpu_threads_per_job),
                    }
                )
                process = subprocess.Popen(
                    job.command,
                    cwd=ROOT,
                    env=environment,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                active[job.job_id] = (job, process, log_handle)
                self.results[job.job_id] = {
                    "status": "RUNNING",
                    "pid": process.pid,
                    "gpu": job.gpu,
                    "tasks": job.tasks,
                    "output_dir": str(job.output_dir),
                    "started_at": utc_now(),
                }
                self.notify(
                    f"{job.job_id}_started",
                    f"[Wayffusion] Swarm30 {job.stage} started: {job.job_id}",
                    f"Job: {job.job_id}\nGPU: {job.gpu}\nOutput: {job.output_dir}\n",
                )
                self.save_state(f"{job.stage.upper()}_RUNNING")
            for job_id, (job, process, log_handle) in list(active.items()):
                code = process.poll()
                if code is None:
                    continue
                log_handle.close()
                status = "COMPLETED" if code == 0 else "FAILED_RUNTIME"
                self.results[job_id].update(
                    {
                        "status": status,
                        "return_code": int(code),
                        "finished_at": utc_now(),
                        "final_update": final_update(job.output_dir) if job.stage == "train" else None,
                    }
                )
                if code != 0:
                    all_ok = False
                self.notify(
                    f"{job_id}_{status.lower()}",
                    f"[Wayffusion] Swarm30 {job.stage} {status}: {job_id}",
                    f"Job: {job_id}\nStatus: {status}\nReturn code: {code}\nOutput: {job.output_dir}\n",
                )
                del active[job_id]
                self.save_state(f"{job.stage.upper()}_RUNNING")
            if pending or active:
                time.sleep(self.args.poll_seconds)
        return all_ok

    def run_baselines(self) -> bool:
        eval_cfg = self.protocol["evaluation"]
        output_dir = self.suite_root / "evaluations" / "baselines"
        command = [
            sys.executable,
            str(ROOT / "scripts/benchmarks/swarm30/evaluate.py"),
            "--baselines",
            "hold",
            "random",
            "greedy",
            "--tasks",
            *SWARM30_TASKS,
            "--scales",
            *[str(value) for value in eval_cfg["scales"]],
            "--modes",
            "fixed",
            "random",
            "--random-episodes",
            str(eval_cfg["random_episodes"]),
            "--fixed-episodes",
            str(eval_cfg["fixed_episodes"]),
            "--record-episodes",
            str(eval_cfg["record_episodes"]),
            "--record-format",
            str(eval_cfg["record_format"]),
            "--record-fps",
            str(eval_cfg["record_fps"]),
            "--output-dir",
            str(output_dir),
        ]
        log_path = output_dir / "baseline_stdout.log"
        output_dir.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as handle:
            code = subprocess.call(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)
        self.results["eval__baselines"] = {
            "status": "COMPLETED" if code == 0 else "FAILED_RUNTIME",
            "return_code": int(code),
            "output_dir": str(output_dir),
        }
        return code == 0

    def build_report(self) -> bool:
        output_dir = self.suite_root / "report"
        command = [
            sys.executable,
            str(ROOT / "scripts/benchmarks/swarm30/build_report.py"),
            "--input-dir",
            str(self.suite_root / "evaluations"),
            "--output-dir",
            str(output_dir),
        ]
        return subprocess.call(command, cwd=ROOT) == 0

    def run(self) -> int:
        manifest = {
            "benchmark": "swarm30_v1",
            "runtime": self.runtime,
            "started_at": utc_now(),
            "tracks": self.args.tracks,
            "tasks": list(SWARM30_TASKS),
            "protocol": self.protocol,
            "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        }
        write_json(self.suite_root / "manifest.json", manifest)
        self.notify(
            "suite_started",
            "[Wayffusion] Swarm30 v1 benchmark started",
            f"Runtime: {self.runtime}\nTracks: {', '.join(self.args.tracks)}\nOutput: {self.suite_root}\n",
        )
        training_jobs = self.prepare_training_jobs()
        training_ok = self.run_pool(training_jobs, self.args.max_parallel)
        evaluation_ok = False
        baseline_ok = False
        report_ok = False
        if training_ok and not self.args.skip_evaluation:
            evaluation_ok = self.run_pool(self.prepare_evaluation_jobs(training_jobs), self.args.max_parallel)
            baseline_ok = self.run_baselines()
            report_ok = self.build_report()
        status = "COMPLETED" if training_ok and (self.args.skip_evaluation or (evaluation_ok and baseline_ok and report_ok)) else "FAILED_RUNTIME"
        self.save_state(status)
        report_path = self.suite_root / "report/report.md"
        self.notify(
            "suite_finished",
            f"[Wayffusion] Swarm30 v1 benchmark {status}",
            (
                f"Runtime: {self.runtime}\nStatus: {status}\nTraining OK: {training_ok}\n"
                f"Evaluation OK: {evaluation_ok}\nBaselines OK: {baseline_ok}\nReport OK: {report_ok}\n"
                f"Output: {self.suite_root}\n"
            ),
            [str(report_path)] if report_path.exists() else None,
        )
        return 0 if status == "COMPLETED" else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the swarm30_v1 training and evaluation benchmark.")
    parser.add_argument("--benchmark-config", default="configs/benchmarks/swarm30_v1.yaml")
    parser.add_argument("--env-config", default="configs/env/waypoint_missions.yaml")
    parser.add_argument("--output-root", default="outputs/training/benchmarks/swarm30_v1")
    parser.add_argument("--runtime", default=None)
    parser.add_argument(
        "--tracks",
        nargs="+",
        choices=sorted(TRACK_ALIASES),
        default=["standard_specialist", "tuned_specialist", "generalist"],
    )
    parser.add_argument(
        "--dynamics-backend",
        choices=["waypoint_behavior_fast", "waypoint_behavior_realistic", "mpe_core"],
        default="waypoint_behavior_fast",
    )
    parser.add_argument("--backend-dt", type=float, default=1.0)
    parser.add_argument("--backend-acceptance-radius", type=float, default=0.025)
    parser.add_argument("--backend-agent-size", type=float, default=0.02)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--num-envs", type=int, default=None)
    parser.add_argument("--rollout-steps", type=int, default=None)
    parser.add_argument("--specialist-updates", type=int, default=None)
    parser.add_argument("--generalist-updates", type=int, default=None)
    parser.add_argument("--eval-interval", type=int, default=None)
    parser.add_argument("--eval-episodes", type=int, default=None)
    parser.add_argument("--record-eval-episodes", type=int, default=None)
    parser.add_argument("--env-backend", choices=["sync", "thread", "process"], default=None)
    parser.add_argument("--gpus", nargs="+", type=int, default=[0, 1, 2, 3])
    parser.add_argument("--max-parallel", type=int, default=4)
    parser.add_argument("--cpu-threads-per-job", type=int, default=4)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--skip-evaluation", action="store_true")
    parser.add_argument("--skip-email", action="store_true")
    parser.add_argument("--smtp-env-file", default=".secrets/wayffusion_mail.env")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.gpus:
        raise ValueError("At least one GPU is required")
    return Swarm30Benchmark(args).run()


if __name__ == "__main__":
    raise SystemExit(main())
