from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.formal_mappo_specialists.run_formal_specialists import load_env_file
from utils.email_notify import send_tuning_email


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase31-root",
        default="outputs/debug/mappo_direct_specialists/phase31_connectivity_training_seed_reproducibility/20260610_0954",
    )
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--smtp-env-file", default=".secrets/wayffusion_mail.env")
    parser.add_argument("--benchmark-runtime", default=None)
    parser.add_argument("--gpus", nargs="+", type=int, default=[0, 1, 2, 3])
    parser.add_argument("--max-parallel", type=int, default=4)
    args = parser.parse_args()

    phase_root = Path(args.phase31_root)
    if not phase_root.is_absolute():
        phase_root = ROOT / phase_root
    state_path = ROOT / "outputs/training/benchmarks/swarm30_v1/queue_state.json"
    load_env_file(ROOT / args.smtp_env_file)
    write_json(
        state_path,
        {
            "status": "WAITING_FOR_PHASE31",
            "phase31_root": str(phase_root),
            "started_at": utc_now(),
        },
    )
    while not (phase_root / "aggregate_summary.json").exists():
        time.sleep(max(args.poll_seconds, 5))

    summary = json.loads((phase_root / "aggregate_summary.json").read_text(encoding="utf-8"))
    worker_codes = summary.get("worker_return_codes", {})
    structurally_complete = bool(worker_codes) and all(int(code) == 0 for code in worker_codes.values())
    if not structurally_complete:
        result = asdict(
            send_tuning_email(
                "[Wayffusion] Swarm30 benchmark not started: Phase31 runtime failure",
                (
                    f"Phase31 root: {phase_root}\n"
                    f"Worker return codes: {worker_codes}\n"
                    "The benchmark queue requires all Phase31 workers to finish without runtime errors.\n"
                ),
            )
        )
        write_json(
            state_path,
            {
                "status": "BLOCKED_PHASE31_RUNTIME",
                "phase31_root": str(phase_root),
                "worker_return_codes": worker_codes,
                "email": result,
                "finished_at": utc_now(),
            },
        )
        return 1

    runtime = args.benchmark_runtime or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
    command = [
        sys.executable,
        str(ROOT / "scripts/benchmarks/swarm30/run_benchmark.py"),
        "--runtime",
        runtime,
        "--gpus",
        *[str(gpu) for gpu in args.gpus],
        "--max-parallel",
        str(args.max_parallel),
    ]
    write_json(
        state_path,
        {
            "status": "STARTING_BENCHMARK",
            "phase31_root": str(phase_root),
            "phase31_passed": bool(summary.get("passed", False)),
            "benchmark_runtime": runtime,
            "command": command,
            "started_at": utc_now(),
        },
    )
    environment = dict(os.environ)
    environment["PYTHONUNBUFFERED"] = "1"
    code = subprocess.call(command, cwd=ROOT, env=environment)
    write_json(
        state_path,
        {
            "status": "COMPLETED" if code == 0 else "FAILED_RUNTIME",
            "phase31_root": str(phase_root),
            "benchmark_runtime": runtime,
            "return_code": int(code),
            "finished_at": utc_now(),
        },
    )
    return int(code)


if __name__ == "__main__":
    raise SystemExit(main())
