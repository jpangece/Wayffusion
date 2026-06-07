from __future__ import annotations

from datetime import datetime
from pathlib import Path


def safe_slug(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(value)).strip("_") or "run"


def minute_timestamp() -> str:
    return datetime.now().astimezone().strftime("%Y%m%d_%H%M")


def phase_folder(phase_name: str, timestamp: str | None = None) -> str:
    stamp = timestamp or minute_timestamp()
    name = safe_slug(phase_name)
    return f"{stamp}_{name}"


def resolve_output_root(root: str | Path, phase_name: str, timestamp: str | None = None, workspace_root: Path | None = None) -> Path:
    base = Path(root)
    if workspace_root is not None and not base.is_absolute():
        base = workspace_root / base
    return base / phase_folder(phase_name, timestamp)


def short_float(value: float) -> str:
    return f"{float(value):.2f}".replace("-", "m").replace(".", "p")


def short_run_name(kind: str, max_delta: float, seed: int, updates: int | None = None, tag: str | None = None) -> str:
    parts = [safe_slug(kind)]
    if updates is not None:
        parts.append(f"u{int(updates)}")
    parts.append(f"md{short_float(max_delta)}")
    parts.append(f"s{int(seed)}")
    if tag:
        parts.append(safe_slug(tag))
    return "__".join(parts)


__all__ = [
    "minute_timestamp",
    "phase_folder",
    "resolve_output_root",
    "safe_slug",
    "short_float",
    "short_run_name",
]
