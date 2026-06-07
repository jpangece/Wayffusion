from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.tensorboard_metrics import should_log_tensorboard_metric

def _load_state(path: Path) -> dict[str, int]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return {str(key): int(value) for key, value in data.items()}


def _save_state(path: Path, state: dict[str, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _float_or_none(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def _iter_metric_csvs(roots: list[Path]) -> list[Path]:
    files: list[Path] = []
    for root in roots:
        if root.is_file() and root.name.endswith(".csv"):
            files.append(root)
            continue
        if root.exists():
            files.extend(root.rglob("training_metrics.csv"))
            files.extend(root.rglob("eval_metrics.csv"))
    return sorted(set(files))


def export_once(roots: list[Path], output_subdir: str, state_path: Path, metric_mode: str = "core") -> dict[str, int]:
    from torch.utils.tensorboard import SummaryWriter

    state = _load_state(state_path)
    exported: dict[str, int] = {}
    for csv_path in _iter_metric_csvs(roots):
        key = str(csv_path.resolve())
        rows = _read_csv(csv_path)
        start = max(0, int(state.get(key, 0)))
        if start >= len(rows):
            exported[key] = 0
            continue
        prefix = "train" if csv_path.name == "training_metrics.csv" else "eval_episode"
        writer = SummaryWriter(str(csv_path.parent / output_subdir))
        count = 0
        for idx, row in enumerate(rows[start:], start=start):
            step_value = _float_or_none(row.get("update"))
            step = int(step_value) if step_value is not None else idx
            if prefix == "eval_episode":
                episode = _float_or_none(row.get("episode"))
                if episode is not None:
                    step = int(step * 100000 + episode)
            for metric, raw in row.items():
                if metric in {"update", "episode"}:
                    continue
                value = _float_or_none(raw)
                if value is None:
                    continue
                if not should_log_tensorboard_metric(metric, value, mode=metric_mode):
                    continue
                writer.add_scalar(f"{prefix}/{metric}", value, step)
            count += 1
        writer.flush()
        writer.close()
        state[key] = len(rows)
        exported[key] = count
    _save_state(state_path, state)
    return exported


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Wayffusion CSV metrics to TensorBoard event files.")
    parser.add_argument("--roots", nargs="+", required=True)
    parser.add_argument("--output-subdir", default="tensorboard_csv")
    parser.add_argument("--metric-mode", choices=["core", "all"], default="core")
    parser.add_argument("--state-path", default=None)
    parser.add_argument("--watch-interval", type=float, default=0.0)
    args = parser.parse_args()

    roots = [Path(item if Path(item).is_absolute() else ROOT / item) for item in args.roots]
    if args.state_path:
        state_path = Path(args.state_path)
        if not state_path.is_absolute():
            state_path = ROOT / state_path
    else:
        first = roots[0] if roots else ROOT
        state_path = (first if first.is_dir() else first.parent) / ".csv_to_tensorboard_state.json"

    while True:
        exported = export_once(roots, str(args.output_subdir), state_path, metric_mode=str(args.metric_mode))
        total = sum(exported.values())
        print(f"[csv-to-tensorboard] exported_rows={total} state={state_path}", flush=True)
        if float(args.watch_interval) <= 0.0:
            break
        time.sleep(float(args.watch_interval))


if __name__ == "__main__":
    main()
