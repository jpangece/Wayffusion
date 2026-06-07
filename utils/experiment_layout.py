from __future__ import annotations

import re
from pathlib import Path


PHASE_NAME_PATTERN = re.compile(r"^phase\d{2}_[a-z0-9][a-z0-9_]*$")
RUNTIME_PATTERN = re.compile(r"^\d{8}_\d{4}$")


def validate_phase_name(phase_name: str) -> str:
    value = str(phase_name)
    if not PHASE_NAME_PATTERN.fullmatch(value):
        raise ValueError(
            f"Invalid phase_name {value!r}; expected phaseXX_<purpose>, "
            "for example phase01_action_scale"
        )
    return value


def validate_runtime(runtime: str) -> str:
    value = str(runtime)
    if not RUNTIME_PATTERN.fullmatch(value):
        raise ValueError(f"Invalid runtime {value!r}; expected YYYYMMDD_HHMM")
    return value


def trial_dir_name(trial: int) -> str:
    number = int(trial)
    if number <= 0:
        raise ValueError("trial must be a positive integer")
    return f"trial{number:02d}"


def seed_dir_name(seed: int) -> str:
    return f"s{int(seed)}"


def make_run_name(seed: int, trial: int | None = None) -> str:
    seed_name = seed_dir_name(seed)
    if trial is None:
        return seed_name
    return f"{trial_dir_name(trial)}_{seed_name}"


def make_debug_output_dir(
    root: str | Path,
    phase_name: str,
    runtime: str,
    task_name: str,
    trial: int,
    seed: int,
) -> Path:
    phase = validate_phase_name(phase_name)
    stamp = validate_runtime(runtime)
    task = _validate_task_name(task_name)
    return Path(root) / phase / stamp / task / trial_dir_name(trial) / seed_dir_name(seed)


def make_training_output_dir(
    root: str | Path,
    runtime: str,
    task_name: str,
    seed: int,
    trial: int | None = None,
) -> Path:
    stamp = validate_runtime(runtime)
    task = _validate_task_name(task_name)
    base = Path(root) / stamp / task
    if trial is not None:
        base = base / trial_dir_name(trial)
    return base / seed_dir_name(seed)


def debug_summary_dir(root: str | Path, phase_name: str, runtime: str) -> Path:
    return Path(root) / validate_phase_name(phase_name) / validate_runtime(runtime)


def training_summary_dir(root: str | Path, runtime: str) -> Path:
    return Path(root) / validate_runtime(runtime)


def _validate_task_name(task_name: str) -> str:
    value = str(task_name)
    if not re.fullmatch(r"[a-z][a-z0-9_]*", value):
        raise ValueError(f"Invalid task_name {value!r}; use the full snake_case task name")
    return value


__all__ = [
    "debug_summary_dir",
    "make_debug_output_dir",
    "make_run_name",
    "make_training_output_dir",
    "seed_dir_name",
    "training_summary_dir",
    "trial_dir_name",
    "validate_phase_name",
    "validate_runtime",
]
