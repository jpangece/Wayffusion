from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.formal_mappo_specialists.run_formal_specialists import load_env_file
from utils.email_notify import send_tuning_email


DEFAULT_WAIT_SESSION_PREFIXES = (
    "wf_swarm30_backend_",
    "wf_swarm30_reward_",
    "wf_swarm30_",
)
SELF_PROCESS_MARKERS = (
    "wait_for_resources_and_launch_reward_training.py",
    "one_click_runner.sh",
    "one_click_command.sh",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def minute_runtime() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_command(command: list[str], timeout: int = 20) -> tuple[int, str]:
    try:
        result = subprocess.run(
            command,
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError:
        return 127, ""
    except subprocess.TimeoutExpired as exc:
        return 124, str(exc.stdout or "")
    return int(result.returncode), str(result.stdout or "")


def tmux_session_exists(session: str) -> bool:
    if not session:
        return False
    code, _ = run_command(["tmux", "has-session", "-t", session], timeout=5)
    return code == 0


def list_tmux_sessions() -> list[str]:
    code, out = run_command(["tmux", "list-sessions", "-F", "#{session_name}"], timeout=5)
    if code != 0:
        return []
    return sorted({line.strip() for line in out.splitlines() if line.strip()})


def autodetect_wait_session(
    prefixes: tuple[str, ...] = DEFAULT_WAIT_SESSION_PREFIXES,
    exclude_sessions: set[str] | None = None,
) -> str:
    excluded = exclude_sessions or set()
    sessions = list_tmux_sessions()
    candidates = [
        name
        for name in sessions
        if name not in excluded and any(name.startswith(prefix) for prefix in prefixes)
    ]
    return sorted(candidates)[-1] if candidates else ""


def process_count(pattern: str) -> int:
    if not pattern:
        return 0
    code, out = run_command(["pgrep", "-af", pattern], timeout=10)
    if code not in {0, 1}:
        return 0
    lines = [line for line in out.splitlines() if line.strip()]
    self_pid = os.getpid()
    filtered = []
    for line in lines:
        head = line.split(maxsplit=1)[0]
        try:
            pid = int(head)
        except ValueError:
            continue
        if pid == self_pid:
            continue
        if any(marker in line for marker in SELF_PROCESS_MARKERS):
            continue
        filtered.append(line)
    return len(filtered)


def load_average_1m() -> float | None:
    try:
        return float(os.getloadavg()[0])
    except (AttributeError, OSError):
        return None


def nvidia_smi_snapshot() -> dict[str, Any]:
    code, out = run_command(
        [
            "nvidia-smi",
            "--query-gpu=index,utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ],
        timeout=10,
    )
    if code != 0 or not out.strip():
        return {"available": False, "gpus": []}
    gpus = []
    for raw in out.splitlines():
        parts = [part.strip() for part in raw.split(",")]
        if len(parts) != 4:
            continue
        try:
            gpus.append(
                {
                    "index": int(parts[0]),
                    "utilization_gpu_pct": float(parts[1]),
                    "memory_used_mb": float(parts[2]),
                    "memory_total_mb": float(parts[3]),
                }
            )
        except ValueError:
            continue
    return {"available": bool(gpus), "gpus": gpus}


def resources_are_free(
    *,
    wait_session: str,
    wait_pattern: str,
    max_load_1m: float | None,
    max_gpu_util_pct: float | None,
    max_gpu_mem_used_mb: float | None,
) -> tuple[bool, dict[str, Any]]:
    session_alive = tmux_session_exists(wait_session) if wait_session else False
    matching_processes = process_count(wait_pattern) if wait_pattern else 0
    load_1m = load_average_1m()
    gpu = nvidia_smi_snapshot()

    gpu_busy = False
    if gpu.get("available"):
        for item in gpu.get("gpus", []):
            if max_gpu_util_pct is not None and float(item["utilization_gpu_pct"]) > float(max_gpu_util_pct):
                gpu_busy = True
            if max_gpu_mem_used_mb is not None and float(item["memory_used_mb"]) > float(max_gpu_mem_used_mb):
                gpu_busy = True

    load_busy = bool(max_load_1m is not None and load_1m is not None and load_1m > max_load_1m)
    ready = not session_alive and matching_processes == 0 and not load_busy and not gpu_busy
    return ready, {
        "ready": ready,
        "wait_session": wait_session,
        "wait_session_alive": session_alive,
        "wait_pattern": wait_pattern,
        "matching_process_count": matching_processes,
        "load_1m": load_1m,
        "max_load_1m": max_load_1m,
        "load_busy": load_busy,
        "gpu": gpu,
        "max_gpu_util_pct": max_gpu_util_pct,
        "max_gpu_mem_used_mb": max_gpu_mem_used_mb,
        "gpu_busy": gpu_busy,
        "checked_at": utc_now(),
    }


def latest_metric_rows(suite_root: Path, limit: int = 8) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for metrics_path in suite_root.rglob("training_metrics.csv"):
        try:
            with metrics_path.open("r", encoding="utf-8", newline="") as handle:
                entries = list(csv.DictReader(handle))
        except OSError:
            continue
        if not entries:
            continue
        last = dict(entries[-1])
        last["metrics_path"] = str(metrics_path)
        last["job_dir"] = str(metrics_path.parent)
        rows.append(last)
    rows.sort(key=lambda item: str(item.get("metrics_path", "")))
    return rows[:limit]


def build_launch_command(args: argparse.Namespace, runtime: str) -> list[str]:
    command = [
        sys.executable,
        str(ROOT / "scripts/benchmarks/swarm30/run_benchmark.py"),
        "--output-root",
        str(args.output_root),
        "--runtime",
        runtime,
        "--tracks",
        *args.tracks,
        "--seeds",
        *[str(seed) for seed in args.seeds],
        "--dynamics-backend",
        args.dynamics_backend,
        "--num-envs",
        str(args.num_envs),
        "--rollout-steps",
        str(args.rollout_steps),
        "--specialist-updates",
        str(args.specialist_updates),
        "--generalist-updates",
        str(args.generalist_updates),
        "--eval-interval",
        str(args.eval_interval),
        "--eval-episodes",
        str(args.eval_episodes),
        "--record-eval-episodes",
        str(args.record_eval_episodes),
        "--env-backend",
        args.env_backend,
        "--gpus",
        *[str(gpu) for gpu in args.gpus],
        "--max-parallel",
        str(args.max_parallel),
        "--cpu-threads-per-job",
        str(args.cpu_threads_per_job),
    ]
    if args.skip_evaluation:
        command.append("--skip-evaluation")
    if args.skip_email_for_benchmark:
        command.append("--skip-email")
    return command


def send_email(enabled: bool, subject: str, body: str, attachments: list[str] | None = None) -> dict[str, Any]:
    if not enabled:
        return {"sent": False, "reason": "disabled", "missing_env": []}
    return asdict(send_tuning_email(subject, body, attachments))


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--wait-session",
        default="auto",
        help=(
            "Old tmux session to wait for. Use 'auto' to detect latest wf_swarm30_* session, "
            "or 'none' to disable tmux-session waiting."
        ),
    )
    parser.add_argument(
        "--exclude-wait-session",
        default="",
        help="Tmux session name that must not be treated as the previous run, usually the new launcher session itself.",
    )
    parser.add_argument(
        "--wait-pattern",
        default="scripts/benchmarks/swarm30/run_benchmark.py",
        help="pgrep -af pattern that must disappear before launch. Use 'none' to disable process-pattern waiting.",
    )
    parser.add_argument("--poll-seconds", type=int, default=120)
    parser.add_argument("--max-wait-hours", type=float, default=72.0)
    parser.add_argument("--max-load-1m", type=float, default=18.0)
    parser.add_argument("--max-gpu-util-pct", type=float, default=None)
    parser.add_argument("--max-gpu-mem-used-mb", type=float, default=None)
    parser.add_argument("--smtp-env-file", default=".secrets/wayffusion_mail.env")
    parser.add_argument("--disable-email", action="store_true")

    parser.add_argument("--output-root", default="outputs/training/benchmarks/swarm30_2000_direct")
    parser.add_argument("--runtime", default=None)
    parser.add_argument("--tracks", nargs="+", default=["tuned_specialist", "generalist"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument(
        "--dynamics-backend",
        choices=["waypoint_behavior_fast", "waypoint_behavior_realistic", "mpe_core"],
        default="waypoint_behavior_fast",
    )
    parser.add_argument("--env-backend", choices=["sync", "thread", "process"], default="process")
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--rollout-steps", type=int, default=128)
    parser.add_argument("--specialist-updates", type=int, default=2000)
    parser.add_argument("--generalist-updates", type=int, default=2000)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--record-eval-episodes", type=int, default=0)
    parser.add_argument("--gpus", nargs="+", type=int, default=[0, 1, 2, 3])
    parser.add_argument("--max-parallel", type=int, default=3)
    parser.add_argument("--cpu-threads-per-job", type=int, default=3)
    parser.add_argument("--skip-evaluation", action="store_true", default=True)
    parser.add_argument("--run-final-evaluation", dest="skip_evaluation", action="store_false")
    parser.add_argument("--skip-email-for-benchmark", action="store_true", default=True)

    parser.add_argument("--monitor-seconds", type=int, default=300)
    parser.add_argument("--monitor-max-hours", type=float, default=240.0)
    parser.add_argument("--status-email-interval-hours", type=float, default=6.0)


def normalize_wait_inputs(args: argparse.Namespace) -> None:
    exclude_sessions = {args.exclude_wait_session} if args.exclude_wait_session else set()
    if str(args.wait_session).lower() in {"none", "off", "false", "0"}:
        args.wait_session = ""
    elif str(args.wait_session).lower() == "auto":
        args.wait_session = autodetect_wait_session(exclude_sessions=exclude_sessions)
    if str(args.wait_pattern).lower() in {"none", "off", "false", "0"}:
        args.wait_pattern = ""


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "One-command launcher: wait for current Swarm30 training resources to be released, "
            "then launch reward-optimized Swarm30 training and monitor it lightly."
        )
    )
    add_arguments(parser)
    args = parser.parse_args()
    normalize_wait_inputs(args)

    load_env_file(ROOT / args.smtp_env_file)
    email_enabled = not bool(args.disable_email)
    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    runtime = args.runtime or f"{minute_runtime()}_rewardopt"
    suite_root = output_root / runtime
    launcher_dir = suite_root / "launcher"
    launcher_dir.mkdir(parents=True, exist_ok=True)
    state_path = launcher_dir / "resource_wait_state.json"
    monitor_path = launcher_dir / "monitor_state.json"
    launch_command_path = launcher_dir / "launch_command.sh"
    log_path = launcher_dir / f"queue_{runtime}.log"

    start_wait = time.time()
    auto_note = ""
    if not args.wait_session:
        auto_note = "No matching wf_swarm30_* tmux session was detected; waiting only by process pattern/resource thresholds.\n"
    send_email(
        email_enabled,
        "[Wayffusion] Reward-optimized Swarm30 launcher waiting for resources",
        (
            f"Runtime: {runtime}\n"
            f"Output root: {suite_root}\n"
            f"Wait session: {args.wait_session or '<disabled>'}\n"
            f"Excluded session: {args.exclude_wait_session or '<none>'}\n"
            f"Wait pattern: {args.wait_pattern or '<disabled>'}\n"
            f"Poll seconds: {args.poll_seconds}\n"
            f"{auto_note}"
        ),
    )

    last_snapshot: dict[str, Any] = {}
    while True:
        ready, snapshot = resources_are_free(
            wait_session=args.wait_session,
            wait_pattern=args.wait_pattern,
            max_load_1m=args.max_load_1m,
            max_gpu_util_pct=args.max_gpu_util_pct,
            max_gpu_mem_used_mb=args.max_gpu_mem_used_mb,
        )
        last_snapshot = snapshot
        write_json(
            state_path,
            {
                "status": "READY_TO_LAUNCH" if ready else "WAITING_FOR_RESOURCES",
                "runtime": runtime,
                "suite_root": str(suite_root),
                "elapsed_wait_seconds": round(time.time() - start_wait, 1),
                "resource_snapshot": snapshot,
                "updated_at": utc_now(),
            },
        )
        if ready:
            break
        if time.time() - start_wait > args.max_wait_hours * 3600.0:
            send_email(
                email_enabled,
                "[Wayffusion] Reward-optimized Swarm30 launcher timed out waiting for resources",
                (
                    f"Runtime: {runtime}\n"
                    f"Waited hours: {args.max_wait_hours}\n"
                    f"Last resource snapshot:\n{json.dumps(last_snapshot, indent=2)}\n"
                ),
                [str(state_path)],
            )
            return 2
        time.sleep(max(int(args.poll_seconds), 10))

    command = build_launch_command(args, runtime)
    launch_command_path.write_text(" ".join(subprocess.list2cmdline([part]) for part in command) + "\n", encoding="utf-8")
    write_json(
        state_path,
        {
            "status": "STARTING_REWARD_OPTIMIZED_TRAINING",
            "runtime": runtime,
            "suite_root": str(suite_root),
            "command": command,
            "resource_snapshot": last_snapshot,
            "started_at": utc_now(),
        },
    )
    send_email(
        email_enabled,
        "[Wayffusion] Starting reward-optimized Swarm30 training",
        (
            f"Runtime: {runtime}\n"
            f"Suite root: {suite_root}\n"
            f"Command saved at: {launch_command_path}\n"
            f"Log: {log_path}\n"
            f"Tracks: {' '.join(args.tracks)}\n"
            f"Seeds: {' '.join(str(seed) for seed in args.seeds)}\n"
            f"Dynamics backend: {args.dynamics_backend}\n"
            f"Env backend: {args.env_backend}\n"
            f"Resource snapshot:\n{json.dumps(last_snapshot, indent=2)}\n"
        ),
        [str(launch_command_path), str(state_path)],
    )

    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    with log_path.open("a", encoding="utf-8") as log_handle:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
        )

    monitor_started = time.time()
    last_status_email = time.time()
    final_code: int | None = None
    while True:
        final_code = process.poll()
        metrics = latest_metric_rows(suite_root)
        snapshot = {
            "status": "RUNNING" if final_code is None else ("COMPLETED" if final_code == 0 else "FAILED_RUNTIME"),
            "runtime": runtime,
            "suite_root": str(suite_root),
            "pid": process.pid,
            "return_code": final_code,
            "elapsed_monitor_seconds": round(time.time() - monitor_started, 1),
            "resource_snapshot": nvidia_smi_snapshot(),
            "load_1m": load_average_1m(),
            "latest_metrics": metrics,
            "updated_at": utc_now(),
        }
        write_json(monitor_path, snapshot)
        if final_code is not None:
            break
        if args.status_email_interval_hours > 0 and time.time() - last_status_email >= args.status_email_interval_hours * 3600.0:
            last_status_email = time.time()
            send_email(
                email_enabled,
                "[Wayffusion] Reward-optimized Swarm30 training still running",
                (
                    f"Runtime: {runtime}\n"
                    f"Suite root: {suite_root}\n"
                    f"Elapsed hours: {(time.time() - monitor_started) / 3600.0:.2f}\n"
                    f"Latest metrics sample:\n{json.dumps(metrics[:4], indent=2)}\n"
                ),
                [str(monitor_path)],
            )
        if time.time() - monitor_started > args.monitor_max_hours * 3600.0:
            send_email(
                email_enabled,
                "[Wayffusion] Reward-optimized Swarm30 monitor reached max monitor time",
                (
                    f"Runtime: {runtime}\n"
                    f"Suite root: {suite_root}\n"
                    f"Training process pid: {process.pid}\n"
                    "The process is left running. This monitor exits without killing training.\n"
                ),
                [str(monitor_path)],
            )
            return 3
        time.sleep(max(int(args.monitor_seconds), 30))

    send_email(
        email_enabled,
        "[Wayffusion] Reward-optimized Swarm30 training finished" if final_code == 0 else "[Wayffusion] Reward-optimized Swarm30 training failed",
        (
            f"Runtime: {runtime}\n"
            f"Suite root: {suite_root}\n"
            f"Return code: {final_code}\n"
            f"Elapsed hours: {(time.time() - monitor_started) / 3600.0:.2f}\n"
            f"Log: {log_path}\n"
            f"Latest metrics sample:\n{json.dumps(latest_metric_rows(suite_root)[:6], indent=2)}\n"
        ),
        [str(monitor_path), str(log_path)] if log_path.exists() else [str(monitor_path)],
    )
    return int(final_code or 0)


if __name__ == "__main__":
    raise SystemExit(main())
