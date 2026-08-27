#!/usr/bin/env python3
"""Audit and plot the completed OFF/REAL/ZERO heatmap ablation runs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


CONDITIONS = ("OFF", "REAL", "ZERO")
COLORS = {"OFF": "#4C78A8", "REAL": "#E45756", "ZERO": "#72B7B2"}
SEEDS = tuple(range(5))
RUN_TEMPLATE = "priority_inspection_heatmap_{condition}_ablation_seed{seed}"
TIME_SERIES = (
    ("evaluation_reward", "Evaluation reward", "eval_reward", "eval_reward"),
    ("evaluation_success_rate", "Evaluation success rate", "eval_success_rate", "eval_success_rate"),
    ("mean_rollout_reward", "Mean rollout reward", "mean_rollout_reward", "mean_rollout_reward"),
    ("weighted_poi_completion", "Weighted POI completion", "weighted_poi_completion", "weighted_poi_completion"),
    ("goal_achieved", "Goal achieved", "goal_achieved", "goal_achieved"),
    ("arrival_rate", "Arrival rate", "arrival_rate", "arrival_rate"),
    ("policy_loss", "Policy loss", "policy_loss", "policy_loss"),
    ("value_loss", "Value loss", "value_loss", "value_loss"),
    ("entropy", "Entropy", "entropy", "entropy"),
    ("gradient_norm", "Gradient norm", "grad_norm", "grad_norm"),
    ("approximate_kl", "Approximate KL divergence", "approx_kl", "approx_kl"),
    ("clip_fraction", "Clip fraction", "clip_frac", "clip_frac"),
    ("explained_variance", "Explained variance", "explained_variance", "explained_variance"),
    ("learning_rate", "Learning rate", "learning_rate", "learning_rate"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path.home() / "heatmap_phd_evidence_inputs" / "outputs",
        help="External root containing the 15 raw run directories.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/results/heatmap_capacity_control/tensorboard_plots"),
        help="Repository directory for plots, plotted-data CSVs, and manifests.",
    )
    return parser.parse_args()


def finite_float(value: str | None) -> float | None:
    if value is None or not value.strip():
        return None
    try:
        number = float(value)
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def duplicate_values(values: Iterable[int]) -> list[int]:
    return sorted(value for value, count in Counter(values).items() if count > 1)


def load_csv(path: Path) -> tuple[list[dict[str, str]], dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        columns = list(reader.fieldnames or [])
    updates = [int(float(row["update"])) for row in rows]
    missing: dict[str, int] = {}
    non_finite: dict[str, int] = {}
    for column in columns:
        blank_count = sum(not (row.get(column) or "").strip() for row in rows)
        if blank_count:
            missing[column] = blank_count
        bad_count = 0
        for row in rows:
            raw = (row.get(column) or "").strip()
            if not raw:
                continue
            try:
                value = float(raw)
            except ValueError:
                continue
            bad_count += int(not math.isfinite(value))
        if bad_count:
            non_finite[column] = bad_count
    eval_updates = [
        int(float(row["update"]))
        for row in rows
        if finite_float(row.get("eval_reward")) is not None
    ]
    return rows, {
        "path": str(path),
        "columns": columns,
        "row_count": len(rows),
        "first_update": updates[0] if updates else None,
        "last_update": updates[-1] if updates else None,
        "evaluation_updates": eval_updates,
        "missing_values_by_column": missing,
        "non_finite_values_by_column": non_finite,
        "duplicate_updates": duplicate_values(updates),
        "non_monotonic_updates": any(b <= a for a, b in zip(updates, updates[1:])),
    }


def load_events(event_files: list[Path]) -> tuple[dict[str, dict[int, float]], dict[str, Any]]:
    merged: dict[str, list[tuple[int, float, float, str]]] = {}
    unreadable: list[dict[str, str]] = []
    file_inventory: list[dict[str, Any]] = []
    for event_file in event_files:
        try:
            accumulator = EventAccumulator(str(event_file), size_guidance={"scalars": 0})
            accumulator.Reload()
            tags = accumulator.Tags().get("scalars", [])
            file_inventory.append({"path": str(event_file), "scalar_tags": tags})
            for tag in tags:
                for event in accumulator.Scalars(tag):
                    merged.setdefault(tag, []).append(
                        (int(event.step), float(event.value), float(event.wall_time), str(event_file))
                    )
        except Exception as exc:  # event corruption must be recorded, not hidden
            unreadable.append({"path": str(event_file), "error": f"{type(exc).__name__}: {exc}"})
    series: dict[str, dict[int, float]] = {}
    tags_audit: dict[str, Any] = {}
    for tag, points in sorted(merged.items()):
        raw_steps = [point[0] for point in points]
        ordered = sorted(points, key=lambda point: (point[2], point[0], point[3]))
        by_step: dict[int, float] = {}
        for step, value, _, _ in ordered:
            by_step[step] = value
        series[tag] = dict(sorted(by_step.items()))
        tags_audit[tag] = {
            "point_count": len(points),
            "unique_step_count": len(by_step),
            "first_step": min(raw_steps) if raw_steps else None,
            "last_step": max(raw_steps) if raw_steps else None,
            "duplicate_steps": duplicate_values(raw_steps),
            "non_monotonic_or_restarted": any(b <= a for a, b in zip(raw_steps, raw_steps[1:])),
            "non_finite_count": sum(not math.isfinite(point[1]) for point in points),
        }
    return series, {
        "event_files": file_inventory,
        "scalar_tags": tags_audit,
        "unreadable_events": unreadable,
    }


def csv_series(rows: list[dict[str, str]], column: str) -> dict[int, float]:
    result: dict[int, float] = {}
    for row in rows:
        value = finite_float(row.get(column))
        if value is not None:
            result[int(float(row["update"]))] = value
    return result


def compare_series(tb: dict[int, float], csv_values: dict[int, float]) -> dict[str, Any]:
    shared = sorted(set(tb) & set(csv_values))
    diffs = [abs(tb[step] - csv_values[step]) for step in shared]
    max_diff = max(diffs, default=0.0)
    value_scale = max(
        [abs(tb[step]) for step in shared] + [abs(csv_values[step]) for step in shared],
        default=0.0,
    )
    pass_tolerance = max(1e-4, 1e-6 * value_scale)
    if not shared:
        status = "FAIL"
    elif max_diff <= pass_tolerance:
        status = "PASS"
    elif max_diff <= 1e-3:
        status = "WARNING"
    else:
        status = "FAIL"
    return {
        "status": status,
        "matching_step_count": len(shared),
        "maximum_absolute_difference": max_diff,
        "mean_absolute_difference": float(np.mean(diffs)) if diffs else None,
        "pass_tolerance": pass_tolerance,
        "tensorboard_steps_missing_from_csv": sorted(set(tb) - set(csv_values)),
        "csv_steps_missing_from_tensorboard": sorted(set(csv_values) - set(tb)),
    }


def aggregate_rows(
    run_series: dict[tuple[str, int], dict[int, float]], source: str
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for condition in CONDITIONS:
        updates = sorted({step for seed in SEEDS for step in run_series[(condition, seed)]})
        for update in updates:
            values = [run_series[(condition, seed)].get(update) for seed in SEEDS]
            valid = np.asarray([value for value in values if value is not None], dtype=np.float64)
            row: dict[str, Any] = {"update": update, "condition": condition}
            row.update({f"seed{seed}": values[seed] for seed in SEEDS})
            row["mean"] = float(valid.mean()) if len(valid) else None
            row["sample_std"] = float(valid.std(ddof=1)) if len(valid) > 1 else 0.0 if len(valid) == 1 else None
            row["valid_seed_count"] = int(len(valid))
            row["source"] = source
            rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty plotted-data CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_figure(fig: plt.Figure, stem: Path) -> tuple[str, str]:
    png = stem.with_suffix(".png")
    svg = stem.with_suffix(".svg")
    fig.savefig(png, dpi=240, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    plt.close(fig)
    return png.name, svg.name


def plot_time_series(
    out_dir: Path,
    slug: str,
    label: str,
    run_series: dict[tuple[str, int], dict[int, float]],
    source: str,
    horizontal_reference: float | None = None,
) -> dict[str, Any]:
    rows = aggregate_rows(run_series, source)
    data_name = f"{slug}_plotted_data.csv"
    write_csv(out_dir / data_name, rows)
    fig, ax = plt.subplots(figsize=(9.2, 5.5))
    for condition in CONDITIONS:
        for seed in SEEDS:
            values = run_series[(condition, seed)]
            ax.plot(list(values), list(values.values()), color=COLORS[condition], alpha=0.24, linewidth=0.9)
        condition_rows = [row for row in rows if row["condition"] == condition]
        x = np.asarray([row["update"] for row in condition_rows])
        mean = np.asarray([row["mean"] for row in condition_rows], dtype=float)
        std = np.asarray([row["sample_std"] for row in condition_rows], dtype=float)
        ax.plot(x, mean, color=COLORS[condition], linewidth=2.6, label=f"{condition} mean")
        ax.fill_between(x, mean - std, mean + std, color=COLORS[condition], alpha=0.15)
    ax.set(title=f"{label} across five training seeds", xlabel="Training update", ylabel=label)
    if horizontal_reference is not None:
        ax.axhline(
            horizontal_reference,
            color="#333333",
            linestyle="--",
            linewidth=1.4,
            label=f"Reference = {horizontal_reference:g}",
        )
    ax.grid(True, alpha=0.28)
    ax.legend(frameon=True)
    png, svg = save_figure(fig, out_dir / slug)
    return {"id": slug, "title": label, "kind": "time_series", "source": source, "png": png, "svg": svg, "data_csv": data_name}


def plot_seed_bars(
    out_dir: Path,
    slug: str,
    title: str,
    ylabel: str,
    values: dict[str, list[float]],
    source: str,
    diagnostic: bool = False,
) -> dict[str, Any]:
    rows = [{"condition": condition, "seed": seed, "value": values[condition][seed], "source": source} for condition in values for seed in SEEDS]
    write_csv(out_dir / f"{slug}_plotted_data.csv", rows)
    fig, ax = plt.subplots(figsize=(8.4, 5.4))
    x = np.arange(len(values))
    labels = list(values)
    means = [float(np.mean(values[label])) for label in labels]
    stds = [float(np.std(values[label], ddof=1)) for label in labels]
    ax.bar(x, means, yerr=stds, color=[COLORS.get(label, "#999999") for label in labels], alpha=0.38, capsize=5)
    for idx, label in enumerate(labels):
        jitter = np.linspace(-0.13, 0.13, len(SEEDS))
        ax.scatter(idx + jitter, values[label], color=COLORS.get(label, "#333333"), edgecolor="white", linewidth=0.6, s=48, zorder=3)
        for j, seed in enumerate(SEEDS):
            ax.annotate(str(seed), (idx + jitter[j], values[label][seed]), xytext=(0, 5), textcoords="offset points", ha="center", fontsize=7)
    ax.set_xticks(x, labels)
    ax.set(title=title, ylabel=ylabel, xlabel="Condition")
    ax.grid(True, axis="y", alpha=0.28)
    png, svg = save_figure(fig, out_dir / slug)
    return {"id": slug, "title": title, "kind": "diagnostic" if diagnostic else "paired_seed", "source": source, "png": png, "svg": svg, "data_csv": f"{slug}_plotted_data.csv"}


def plot_xy(
    out_dir: Path, slug: str, title: str, xlabel: str, ylabel: str,
    x_values: dict[str, list[float]], y_values: dict[str, list[float]], source: str,
) -> dict[str, Any]:
    rows = [{"condition": c, "seed": s, "x": x_values[c][s], "y": y_values[c][s], "source": source} for c in CONDITIONS for s in SEEDS]
    write_csv(out_dir / f"{slug}_plotted_data.csv", rows)
    fig, ax = plt.subplots(figsize=(7.2, 5.7))
    for c in CONDITIONS:
        ax.scatter(x_values[c], y_values[c], label=c, color=COLORS[c], s=55)
        for s, (x, y) in enumerate(zip(x_values[c], y_values[c])):
            ax.annotate(str(s), (x, y), xytext=(4, 3), textcoords="offset points", fontsize=7)
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
    ax.grid(True, alpha=0.28)
    ax.legend()
    png, svg = save_figure(fig, out_dir / slug)
    return {"id": slug, "title": title, "kind": "diagnostic", "source": source, "png": png, "svg": svg, "data_csv": f"{slug}_plotted_data.csv"}


def series_stat(values: dict[int, float], mode: str) -> float:
    ordered = [values[step] for step in sorted(values)]
    if not ordered:
        raise ValueError("Required paired metric has no values")
    return float(np.mean(ordered)) if mode == "mean" else float(ordered[-1])


def main() -> None:
    args = parse_args()
    raw_root = args.raw_root.expanduser().resolve()
    out_dir = args.output_dir.resolve()
    if not raw_root.is_dir():
        raise SystemExit(f"Raw-data root does not exist: {raw_root}")
    out_dir.mkdir(parents=True, exist_ok=True)

    runs: dict[tuple[str, int], dict[str, Any]] = {}
    audit_runs: list[dict[str, Any]] = []
    all_tags: Counter[str] = Counter()
    all_columns: Counter[str] = Counter()
    consistency: dict[str, dict[str, Any]] = {}
    for condition in CONDITIONS:
        for seed in SEEDS:
            run_name = RUN_TEMPLATE.format(condition=condition.lower(), seed=seed)
            run_dir = raw_root / run_name
            csv_path = run_dir / "training_metrics.csv"
            best_path = run_dir / "best_eval_summary.json"
            event_files = sorted(run_dir.rglob("events.out.tfevents.*")) if run_dir.is_dir() else []
            missing = [str(path) for path in (run_dir, csv_path, best_path) if not path.exists()]
            if missing or not event_files:
                raise SystemExit(f"Incomplete required run {run_name}: missing={missing}, event_files={len(event_files)}")
            csv_rows, csv_audit = load_csv(csv_path)
            tb_series, tb_audit = load_events(event_files)
            if tb_audit["unreadable_events"]:
                raise SystemExit(f"Unreadable TensorBoard event in {run_name}: {tb_audit['unreadable_events']}")
            all_tags.update(tb_series.keys())
            all_columns.update(csv_audit["columns"])
            runs[(condition, seed)] = {"csv_rows": csv_rows, "tb": tb_series, "best": json.loads(best_path.read_text(encoding="utf-8"))}
            for tag in set(tb_series) & set(csv_audit["columns"]):
                key = f"{condition}_seed{seed}:{tag}"
                consistency[key] = compare_series(tb_series[tag], csv_series(csv_rows, tag))
            audit_runs.append({"condition": condition, "seed": seed, "run_name": run_name, "snapshot_present": (run_dir / "snapshot").is_dir(), "tensorboard": tb_audit, "csv": csv_audit})

    selected: dict[str, dict[tuple[str, int], dict[int, float]]] = {}
    selected_source: dict[str, str] = {}
    skipped: list[dict[str, str]] = []
    plots: list[dict[str, Any]] = []
    for slug, label, tag, column in TIME_SERIES:
        tag_exists_everywhere = all(tag in runs[key]["tb"] for key in runs)
        column_exists_everywhere = all(column in runs[key]["csv_rows"][0] for key in runs)
        if tag_exists_everywhere:
            checks = [consistency[f"{condition}_seed{seed}:{tag}"] for condition in CONDITIONS for seed in SEEDS]
            if not all(item["status"] == "PASS" for item in checks):
                skipped.append({"metric": slug, "reason": "TensorBoard tag exists but CSV consistency did not PASS"})
                continue
            source = "TensorBoard"
            data = {key: runs[key]["tb"][tag] for key in runs}
        elif column_exists_everywhere:
            source = "CSV"
            data = {key: csv_series(runs[key]["csv_rows"], column) for key in runs}
            if not all(data[key] for key in data):
                skipped.append({"metric": slug, "reason": "CSV column exists but contains no finite points in at least one run"})
                continue
        else:
            skipped.append({"metric": slug, "reason": "Metric absent from both complete TensorBoard and CSV inventories"})
            continue
        selected[slug] = data
        selected_source[slug] = source
        plots.append(plot_time_series(out_dir, slug, label, data, source))

    for metric in ("evaluation_reward", "evaluation_success_rate"):
        if metric not in selected:
            raise SystemExit(f"Required paired metric was not available after audit: {metric}")
    reward = selected["evaluation_reward"]
    success = selected["evaluation_success_rate"]
    paired_specs = (
        ("real_minus_zero_mean_evaluation_reward_by_seed", "REAL − ZERO mean evaluation reward by seed", reward, "mean"),
        ("real_minus_zero_final_evaluation_reward_by_seed", "REAL − ZERO final evaluation reward by seed", reward, "final"),
        ("real_minus_zero_mean_evaluation_success_by_seed", "REAL − ZERO mean evaluation success by seed", success, "mean"),
        ("real_minus_zero_final_evaluation_success_by_seed", "REAL − ZERO final evaluation success by seed", success, "final"),
    )
    for slug, title, data, mode in paired_specs:
        values = {"REAL-ZERO": [series_stat(data[("REAL", s)], mode) - series_stat(data[("ZERO", s)], mode) for s in SEEDS]}
        plots.append(plot_seed_bars(out_dir, slug, title, "Paired difference", values, selected_source["evaluation_reward" if "reward" in slug else "evaluation_success_rate"]))
    final_reward = {c: [series_stat(reward[(c, s)], "final") for s in SEEDS] for c in CONDITIONS}
    final_success = {c: [series_stat(success[(c, s)], "final") for s in SEEDS] for c in CONDITIONS}
    plots.append(plot_seed_bars(out_dir, "final_evaluation_reward_by_condition", "Final evaluation reward by condition", "Final evaluation reward", final_reward, selected_source["evaluation_reward"]))
    plots.append(plot_seed_bars(out_dir, "final_evaluation_success_by_condition", "Final evaluation success by condition", "Final evaluation success rate", final_success, selected_source["evaluation_success_rate"]))

    best_updates = {c: [float(runs[(c, s)]["best"]["update"]) for s in SEEDS] for c in CONDITIONS}
    plots.append(plot_seed_bars(out_dir, "diagnostic_best_evaluation_update", "Diagnostic: best evaluation update", "Training update", best_updates, "best_eval_summary.json", diagnostic=True))
    if "policy_loss" in selected and "value_loss" in selected:
        x = {c: [float(np.mean(list(selected["policy_loss"][(c, s)].values()))) for s in SEEDS] for c in CONDITIONS}
        y = {c: [float(np.mean(list(selected["value_loss"][(c, s)].values()))) for s in SEEDS] for c in CONDITIONS}
        plots.append(plot_xy(out_dir, "diagnostic_policy_vs_value_loss", "Diagnostic: mean policy loss versus mean value loss", "Mean policy loss", "Mean value loss", x, y, "TensorBoard"))
    plots.append(plot_xy(out_dir, "diagnostic_final_reward_vs_success", "Diagnostic: final evaluation reward versus success", "Final evaluation reward", "Final evaluation success rate", final_reward, final_success, selected_source["evaluation_reward"]))
    if "gradient_norm" in selected:
        item = plot_time_series(
            out_dir,
            "diagnostic_gradient_norm_with_clip_threshold",
            "Gradient norm (clip threshold 0.5)",
            selected["gradient_norm"],
            selected_source["gradient_norm"],
            horizontal_reference=0.5,
        )
        item["diagnostic_reference"] = {"gradient_clip_threshold": 0.5}
        plots.append(item)
    if "entropy" in selected:
        entropy_values = {c: [] for c in CONDITIONS}
        rows = []
        for c in CONDITIONS:
            for s in SEEDS:
                values = selected["entropy"][(c, s)]
                first, last = values[min(values)], values[max(values)]
                entropy_values[c].append(last - first)
                rows.append({"condition": c, "seed": s, "update_1": first, "update_200": last, "change": last - first, "source": selected_source["entropy"]})
        slug = "diagnostic_entropy_update_1_vs_200"
        write_csv(out_dir / f"{slug}_plotted_data.csv", rows)
        fig, ax = plt.subplots(figsize=(8.4, 5.4))
        for idx, c in enumerate(CONDITIONS):
            for s in SEEDS:
                first, last = rows[idx * 5 + s]["update_1"], rows[idx * 5 + s]["update_200"]
                ax.plot([0, 1], [first, last], color=COLORS[c], alpha=0.5)
                ax.scatter([0, 1], [first, last], color=COLORS[c], s=25)
        ax.set(xticks=[0, 1], xticklabels=["Update 1", "Update 200"], title="Diagnostic: entropy at update 1 versus update 200", ylabel="Entropy")
        ax.grid(True, alpha=0.28)
        png, svg = save_figure(fig, out_dir / slug)
        plots.append({"id": slug, "title": "Diagnostic: entropy at update 1 versus update 200", "kind": "diagnostic", "source": selected_source["entropy"], "png": png, "svg": svg, "data_csv": f"{slug}_plotted_data.csv"})

    audit = {
        "raw_root": str(raw_root),
        "run_count": len(runs),
        "condition_counts": {c: sum(key[0] == c for key in runs) for c in CONDITIONS},
        "normalized_tensorboard_tag_inventory": dict(sorted(all_tags.items())),
        "normalized_csv_column_inventory": dict(sorted(all_columns.items())),
        "runs": audit_runs,
        "tensorboard_csv_consistency": consistency,
        "selected_sources": selected_source,
        "skipped_metrics": skipped,
    }
    (out_dir / "audit_manifest.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest = {"raw_root": str(raw_root), "output_dir": str(out_dir), "plots": plots, "skipped_metrics": skipped}
    (out_dir / "plot_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown = ["# Heatmap ablation plot manifest", "", "| Plot | Kind | Source | PNG | SVG | Data |", "|---|---|---|---|---|---|"]
    markdown.extend(f"| {p['title']} | {p['kind']} | {p['source']} | [{p['png']}]({p['png']}) | [{p['svg']}]({p['svg']}) | [{p['data_csv']}]({p['data_csv']}) |" for p in plots)
    if skipped:
        markdown.extend(["", "## Skipped metrics", ""] + [f"- `{item['metric']}`: {item['reason']}" for item in skipped])
    (out_dir / "plot_manifest.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    print(json.dumps({"runs": len(runs), "plots": len(plots), "sources": selected_source, "skipped": skipped}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
