from __future__ import annotations

import argparse
import csv
import json
import math
import os
import signal
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_OUTPUT_ROOT = ROOT / "outputs/training/mappo_direct_specialists"
REGISTRY_PATH = ROOT / "outputs/training/run_registry.yaml"
DEFAULT_ENV_FILE = ROOT / ".secrets/wayffusion_mail.env"
TASK_SPECS = {
    "area_coverage": {
        "config": "configs/policy/direct_specialists/direct_area_coverage.yaml",
        "gpu": 0,
        "total_updates": 2000,
    },
    "belief_search": {
        "config": "configs/policy/direct_specialists/direct_belief_search.yaml",
        "gpu": 1,
        "total_updates": 2000,
    },
    "priority_inspection": {
        "config": "configs/policy/direct_specialists/direct_priority_inspection.yaml",
        "gpu": 2,
        "total_updates": 2000,
    },
    "connectivity_expansion": {
        "config": "configs/policy/direct_specialists/direct_connectivity_expansion.yaml",
        "gpu": 3,
        "total_updates": 2500,
    },
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def default_runtime() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        if key and key not in os.environ:
            os.environ[key] = value
    if not os.environ.get("SMTP_TO") and os.environ.get("EMAIL_TO"):
        os.environ["SMTP_TO"] = os.environ["EMAIL_TO"]
    if not os.environ.get("SMTP_USE_SSL") and os.environ.get("SMTP_SSL") is not None:
        os.environ["SMTP_USE_SSL"] = os.environ["SMTP_SSL"]
    if not os.environ.get("SMTP_USE_TLS") and os.environ.get("SMTP_STARTTLS") is not None:
        os.environ["SMTP_USE_TLS"] = os.environ["SMTP_STARTTLS"]


def finite_float(value: Any, default: float | None = None) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def read_last_metrics(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return rows[-1] if rows else {}


def task_output_dir(output_root: Path, runtime: str, task: str) -> Path:
    return output_root / runtime / task / "s0"


def build_train_command(
    task: str,
    output_dir: Path,
    num_envs: int,
    rollout_steps: int,
    eval_interval: int,
    eval_episodes: int,
    record_eval_episodes: int,
    record_interval: int,
    env_backend: str,
) -> list[str]:
    spec = TASK_SPECS[task]
    return [
        sys.executable,
        str(ROOT / "scripts/train_mappo_waypoint.py"),
        "--config",
        str(spec["config"]),
        "--env-config",
        "configs/env/waypoint_missions.yaml",
        "--eval-config",
        "configs/eval/waypoint_eval.yaml",
        "--tasks",
        task,
        "--num_agents",
        "4",
        "--num_envs",
        str(num_envs),
        "--rollout_steps",
        str(rollout_steps),
        "--total_updates",
        str(spec["total_updates"]),
        "--eval_interval",
        str(eval_interval),
        "--eval_episodes",
        str(eval_episodes),
        "--record_eval_episodes",
        str(record_eval_episodes),
        "--record_interval",
        str(record_interval),
        "--record_format",
        "gif",
        "--record_fps",
        "8",
        "--seed",
        "0",
        "--run_name",
        "s0",
        "--output-dir",
        str(output_dir),
        "--train-randomization",
        "random",
        "--eval-randomization",
        "random",
        "--env_backend",
        env_backend,
        "--headless",
        "--tensorboard",
    ]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def update_registry(entries: dict[str, dict[str, Any]]) -> None:
    payload = {"schema_version": 1, "updated_at": utc_now(), "runs": []}
    if REGISTRY_PATH.exists():
        loaded = yaml.safe_load(REGISTRY_PATH.read_text(encoding="utf-8")) or {}
        if isinstance(loaded, dict):
            payload.update(loaded)
    existing = {
        str(item.get("run_id")): item
        for item in payload.get("runs", [])
        if isinstance(item, dict) and item.get("run_id")
    }
    existing.update(entries)
    payload["runs"] = sorted(existing.values(), key=lambda item: str(item.get("run_id", "")))
    payload["updated_at"] = utc_now()
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    REGISTRY_PATH.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


class FormalSuite:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.runtime = args.runtime or default_runtime()
        self.output_root = Path(args.output_root)
        if not self.output_root.is_absolute():
            self.output_root = ROOT / self.output_root
        self.suite_root = self.output_root / self.runtime
        self.suite_root.mkdir(parents=True, exist_ok=True)
        self.processes: dict[str, subprocess.Popen] = {}
        self.log_handles: dict[str, Any] = {}
        self.results: dict[str, dict[str, Any]] = {}
        self.email_events: list[dict[str, Any]] = []
        self.stop_requested = False
        self.registry_entries: dict[str, dict[str, Any]] = {}

    def send_email(self, event: str, subject: str, body: str, attachments: list[str] | None = None) -> None:
        if self.args.skip_email:
            result = {"sent": False, "reason": "skip_email", "missing_env": []}
        else:
            from utils.email_notify import send_tuning_email

            result = asdict(send_tuning_email(subject, body, attachments))
        self.email_events.append({"event": event, "at": utc_now(), **result})
        write_json(self.suite_root / "email_events.json", self.email_events)

    def registry_entry(self, task: str, status: str) -> dict[str, Any]:
        spec = TASK_SPECS[task]
        output_dir = task_output_dir(self.output_root, self.runtime, task)
        previous = self.registry_entries.get(task, {})
        return {
            "run_id": f"{self.runtime}__{task}_s0",
            "purpose": "Formal Direct MAPPO specialist training with randomized training and evaluation",
            "kind": "training",
            "tasks": [task],
            "config": str(spec["config"]),
            "env_config": "configs/env/waypoint_missions.yaml",
            "eval_config": "configs/eval/waypoint_eval.yaml",
            "seed": 0,
            "gpu": int(spec["gpu"]),
            "num_envs": int(self.args.num_envs),
            "rollout_steps": int(self.args.rollout_steps),
            "total_updates": int(spec["total_updates"]),
            "train_randomization": "random",
            "eval_randomization": "random",
            "output_dir": str(output_dir.relative_to(ROOT)),
            "status": status,
            "created_at": previous.get("created_at", utc_now()),
            "updated_at": utc_now(),
        }

    def save_state(self, suite_status: str) -> None:
        state = {
            "runtime": self.runtime,
            "status": suite_status,
            "updated_at": utc_now(),
            "tasks": self.results,
            "active_tasks": [task for task, process in self.processes.items() if process.poll() is None],
            "email_events": self.email_events,
        }
        write_json(self.suite_root / "suite_state.json", state)

    def start_task(self, task: str) -> None:
        spec = TASK_SPECS[task]
        output_dir = task_output_dir(self.output_root, self.runtime, task)
        output_dir.mkdir(parents=True, exist_ok=True)
        command = build_train_command(
            task,
            output_dir,
            self.args.num_envs,
            self.args.rollout_steps,
            self.args.eval_interval,
            self.args.eval_episodes,
            self.args.record_eval_episodes,
            self.args.record_interval,
            self.args.env_backend,
        )
        write_json(output_dir / "formal_command.json", command)
        log_path = output_dir / "formal_stdout.log"
        log_handle = log_path.open("a", encoding="utf-8")
        environment = dict(os.environ)
        environment.update(
            {
                "CUDA_VISIBLE_DEVICES": str(spec["gpu"]),
                "PYTHONUNBUFFERED": "1",
                "OMP_NUM_THREADS": str(self.args.cpu_threads_per_task),
                "MKL_NUM_THREADS": str(self.args.cpu_threads_per_task),
            }
        )
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=environment,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        self.processes[task] = process
        self.log_handles[task] = log_handle
        self.results[task] = {
            "task": task,
            "status": "RUNNING",
            "pid": process.pid,
            "gpu": int(spec["gpu"]),
            "output_dir": str(output_dir),
            "stdout_log": str(log_path),
            "started_at": utc_now(),
        }
        self.registry_entries[task] = self.registry_entry(task, "RUNNING")
        update_registry({entry["run_id"]: entry for entry in self.registry_entries.values()})
        self.send_email(
            f"{task}_started",
            f"[Wayffusion] Formal specialist started: {task}",
            (
                f"Task: {task}\nRuntime: {self.runtime}\nGPU: {spec['gpu']}\n"
                f"PID: {process.pid}\nUpdates: {spec['total_updates']}\n"
                f"num_envs: {self.args.num_envs}\nrollout_steps: {self.args.rollout_steps}\n"
                "train_randomization: random\neval_randomization: random\n"
                f"Output: {output_dir}\n"
            ),
        )

    def finish_task(self, task: str, return_code: int) -> None:
        handle = self.log_handles.pop(task, None)
        if handle is not None:
            handle.close()
        output_dir = Path(self.results[task]["output_dir"])
        metrics = read_last_metrics(output_dir / "training_metrics.csv")
        status = "COMPLETED" if return_code == 0 else "FAILED_RUNTIME"
        result = self.results[task]
        result.update(
            {
                "status": status,
                "return_code": int(return_code),
                "finished_at": utc_now(),
                "final_update": finite_float(metrics.get("update")),
                "eval_success_rate": finite_float(metrics.get("eval_success_rate")),
                "eval_reward": finite_float(metrics.get("eval_reward")),
                "action_validity_rate": finite_float(metrics.get("action_validity_rate")),
                "delta_clip_fraction": finite_float(metrics.get("delta_clip_fraction")),
                "approx_kl": finite_float(metrics.get("approx_kl")),
                "best_checkpoint": str(output_dir / "checkpoints/checkpoint_best_eval.pt"),
            }
        )
        self.registry_entries[task] = self.registry_entry(task, status)
        update_registry({entry["run_id"]: entry for entry in self.registry_entries.values()})
        event = f"{task}_{'completed' if return_code == 0 else 'abnormal'}"
        subject_status = "completed" if return_code == 0 else "ABNORMAL TERMINATION"
        self.send_email(
            event,
            f"[Wayffusion] Formal specialist {subject_status}: {task}",
            (
                f"Task: {task}\nRuntime: {self.runtime}\nStatus: {status}\n"
                f"Return code: {return_code}\nFinal update: {result['final_update']}\n"
                f"Eval success: {result['eval_success_rate']}\nEval reward: {result['eval_reward']}\n"
                f"Action validity: {result['action_validity_rate']}\n"
                f"Delta clip fraction: {result['delta_clip_fraction']}\n"
                f"Output: {output_dir}\nLog: {result['stdout_log']}\n"
            ),
        )

    def write_report(self, suite_status: str) -> tuple[Path, Path]:
        summary = {
            "runtime": self.runtime,
            "status": suite_status,
            "train_randomization": "random",
            "eval_randomization": "random",
            "num_envs": int(self.args.num_envs),
            "rollout_steps": int(self.args.rollout_steps),
            "tasks": self.results,
            "email_events": self.email_events,
        }
        summary_path = self.suite_root / "summary.json"
        write_json(summary_path, summary)
        lines = [
            "# Formal Direct MAPPO Specialists",
            "",
            f"- Runtime: `{self.runtime}`",
            f"- Status: `{suite_status}`",
            "- Train randomization: `random`",
            "- Eval randomization: `random`",
            f"- Parallel environments per task: `{self.args.num_envs}`",
            f"- Rollout steps: `{self.args.rollout_steps}`",
            "",
            "| Task | Status | GPU | Final update | Eval success | Eval reward | Output |",
            "|---|---|---:|---:|---:|---:|---|",
        ]
        for task in TASK_SPECS:
            result = self.results.get(task, {})
            lines.append(
                f"| {task} | {result.get('status', 'NOT_STARTED')} | {result.get('gpu', '')} | "
                f"{result.get('final_update', '')} | {result.get('eval_success_rate', '')} | "
                f"{result.get('eval_reward', '')} | `{result.get('output_dir', '')}` |"
            )
        report_path = self.suite_root / "report.md"
        report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return summary_path, report_path

    def request_stop(self, signum, _frame) -> None:
        if self.stop_requested:
            return
        self.stop_requested = True
        self.send_email(
            "suite_signal",
            "[Wayffusion] Formal specialist suite interrupted",
            f"Runtime: {self.runtime}\nSignal: {signum}\nActive tasks: {list(self.processes)}\n",
        )
        for process in self.processes.values():
            if process.poll() is None:
                process.terminate()

    def run(self) -> int:
        signal.signal(signal.SIGTERM, self.request_stop)
        signal.signal(signal.SIGINT, self.request_stop)
        self.send_email(
            "suite_started",
            "[Wayffusion] Formal Direct MAPPO specialist suite started",
            (
                f"Runtime: {self.runtime}\nTasks: {', '.join(TASK_SPECS)}\n"
                f"Output root: {self.suite_root}\nnum_envs per task: {self.args.num_envs}\n"
                "train_randomization: random\neval_randomization: random\n"
            ),
        )
        try:
            for task in TASK_SPECS:
                self.start_task(task)
            self.save_state("RUNNING")
            pending = set(self.processes)
            while pending:
                for task in list(pending):
                    return_code = self.processes[task].poll()
                    if return_code is None:
                        continue
                    self.finish_task(task, return_code)
                    pending.remove(task)
                    self.save_state("RUNNING" if pending else "FINALIZING")
                if pending:
                    time.sleep(self.args.poll_interval)
        except Exception as exc:
            self.send_email(
                "suite_exception",
                "[Wayffusion] Formal specialist suite ABNORMAL TERMINATION",
                f"Runtime: {self.runtime}\nException: {type(exc).__name__}: {exc}\n",
            )
            for process in self.processes.values():
                if process.poll() is None:
                    process.terminate()
            raise
        failed = [task for task, result in self.results.items() if result.get("status") != "COMPLETED"]
        suite_status = "COMPLETED" if not failed and not self.stop_requested else "FAILED_RUNTIME"
        summary_path, report_path = self.write_report(suite_status)
        self.save_state(suite_status)
        subject = (
            "[Wayffusion] Formal Direct MAPPO specialists completed"
            if suite_status == "COMPLETED"
            else "[Wayffusion] Formal Direct MAPPO specialists ABNORMAL TERMINATION"
        )
        self.send_email(
            "suite_finished",
            subject,
            (
                f"Runtime: {self.runtime}\nStatus: {suite_status}\n"
                f"Completed: {[task for task, result in self.results.items() if result.get('status') == 'COMPLETED']}\n"
                f"Failed: {failed}\nOutput: {self.suite_root}\n"
            ),
            [str(report_path), str(summary_path)],
        )
        return 0 if suite_status == "COMPLETED" else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run four formal Direct MAPPO specialists in parallel.")
    parser.add_argument("--runtime", default=None)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--env-file", default=str(DEFAULT_ENV_FILE))
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--rollout-steps", type=int, default=256)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--record-eval-episodes", type=int, default=2)
    parser.add_argument("--record-interval", type=int, default=5)
    parser.add_argument("--cpu-threads-per-task", type=int, default=8)
    parser.add_argument("--env-backend", choices=["sync", "process"], default="process")
    parser.add_argument("--poll-interval", type=float, default=30.0)
    parser.add_argument("--skip-email", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    load_env_file(Path(args.env_file))
    suite = FormalSuite(args)
    if args.dry_run:
        plan = {
            task: {
                "gpu": spec["gpu"],
                "output_dir": str(task_output_dir(suite.output_root, suite.runtime, task)),
                "command": build_train_command(
                    task,
                    task_output_dir(suite.output_root, suite.runtime, task),
                    args.num_envs,
                    args.rollout_steps,
                    args.eval_interval,
                    args.eval_episodes,
                    args.record_eval_episodes,
                    args.record_interval,
                    args.env_backend,
                ),
            }
            for task, spec in TASK_SPECS.items()
        }
        write_json(suite.suite_root / "launch_plan.json", plan)
        print(json.dumps(plan, indent=2), flush=True)
        return 0
    return suite.run()


if __name__ == "__main__":
    raise SystemExit(main())
