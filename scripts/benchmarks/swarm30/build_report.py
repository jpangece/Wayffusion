from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def number(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def load_summaries(root: Path) -> list[dict]:
    rows = []
    for path in root.rglob("summary.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if payload.get("benchmark") != "swarm30_v1":
            continue
        payload["summary_path"] = str(path)
        rows.append(payload)
    return rows


def aggregate(rows: list[dict]) -> dict:
    metrics = {
        "success_rate": "success_rate_mean",
        "normalized_progress": "normalized_progress_mean",
        "return": "return_mean",
        "action_validity_rate": "action_validity_rate_mean",
        "collision_rate": "collision_rate_mean",
        "connectivity_violation_rate": "connectivity_violation_rate_mean",
    }
    result = {"seed_count": len({row.get("seed") for row in rows})}
    for output_name, source_name in metrics.items():
        values = [number(row.get(source_name)) for row in rows if source_name in row]
        result[f"{output_name}_mean"] = float(np.mean(values)) if values else 0.0
        result[f"{output_name}_std"] = float(np.std(values)) if values else 0.0
        result[f"{output_name}_worst"] = float(np.min(values)) if values else 0.0
    result["hard_filter_valid"] = all(bool(row.get("hard_filter_valid", True)) for row in rows)
    return result


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    keys = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def build_report(input_dir: Path, output_dir: Path) -> tuple[list[dict], list[dict]]:
    raw = load_summaries(input_dir)
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in raw:
        key = (
            str(row.get("track", "unknown")),
            int(row.get("num_agents", 0)),
            str(row.get("randomization_mode", "unknown")),
            str(row.get("task_name", "unknown")),
        )
        grouped[key].append(row)

    task_rows = []
    for (track, scale, mode, task), rows in sorted(grouped.items()):
        task_rows.append(
            {
                "track": track,
                "num_agents": scale,
                "randomization_mode": mode,
                "task_name": task,
                **aggregate(rows),
            }
        )

    macro_groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in task_rows:
        macro_groups[(row["track"], row["num_agents"], row["randomization_mode"])].append(row)
    leaderboard = []
    for (track, scale, mode), rows in sorted(macro_groups.items()):
        leaderboard.append(
            {
                "track": track,
                "num_agents": scale,
                "randomization_mode": mode,
                "task_count": len(rows),
                "macro_success_rate": float(np.mean([number(row["success_rate_mean"]) for row in rows])),
                "macro_normalized_progress": float(np.mean([number(row["normalized_progress_mean"]) for row in rows])),
                "action_validity_rate": float(np.mean([number(row["action_validity_rate_mean"]) for row in rows])),
                "collision_rate": float(np.mean([number(row["collision_rate_mean"]) for row in rows])),
                "connectivity_violation_rate": float(np.mean([number(row["connectivity_violation_rate_mean"]) for row in rows])),
                "hard_filter_valid": all(bool(row["hard_filter_valid"]) for row in rows),
            }
        )
    leaderboard.sort(
        key=lambda row: (
            int(row["num_agents"]),
            str(row["randomization_mode"]),
            -float(row["macro_success_rate"]),
            -float(row["macro_normalized_progress"]),
            -float(row["action_validity_rate"]),
        )
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "task_results.csv", task_rows)
    write_csv(output_dir / "leaderboard.csv", leaderboard)
    (output_dir / "report_data.json").write_text(
        json.dumps({"task_results": task_rows, "leaderboard": leaderboard}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Swarm30 v1 Benchmark Report",
        "",
        "Primary ranking is six-task macro success rate. Raw task rewards are not summed across tasks.",
        "",
        "## Leaderboard",
        "",
        "| Track | N | Eval | Macro success | Normalized progress | Validity | Collision | Connectivity violation | Hard filter valid |",
        "|---|---:|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in leaderboard:
        lines.append(
            f"| {row['track']} | {row['num_agents']} | {row['randomization_mode']} | "
            f"{row['macro_success_rate']:.3f} | {row['macro_normalized_progress']:.3f} | "
            f"{row['action_validity_rate']:.3f} | {row['collision_rate']:.3f} | "
            f"{row['connectivity_violation_rate']:.3f} | {row['hard_filter_valid']} |"
        )
    lines.extend(
        [
            "",
            "## Per-task Results",
            "",
            "| Track | N | Eval | Task | Success | Progress | Return | Validity |",
            "|---|---:|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in task_rows:
        lines.append(
            f"| {row['track']} | {row['num_agents']} | {row['randomization_mode']} | {row['task_name']} | "
            f"{row['success_rate_mean']:.3f} +/- {row['success_rate_std']:.3f} | "
            f"{row['normalized_progress_mean']:.3f} | {row['return_mean']:.3f} | "
            f"{row['action_validity_rate_mean']:.3f} |"
        )
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return task_rows, leaderboard


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    task_rows, leaderboard = build_report(Path(args.input_dir), Path(args.output_dir))
    print(f"[swarm30-report] task_rows={len(task_rows)} leaderboard_rows={len(leaderboard)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
