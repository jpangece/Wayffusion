from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class MetricSpec:
    label: str
    candidates: tuple[str, ...]


METRICS = (
    MetricSpec("success_rate", ("eval_success_rate", "eval_coverage_success_rate", "eval_overall_success_rate", "success_rate_mean", "success_mean")),
    MetricSpec("eval_reward", ("eval_reward", "eval_coverage_return", "eval_overall_return", "return_mean")),
    MetricSpec("coverage_ratio", ("eval_coverage_coverage_ratio", "eval_overall_coverage_ratio", "coverage_ratio_mean")),
    MetricSpec(
        "collision_rate",
        ("eval_collision_rate", "eval_coverage_collision_rate", "eval_overall_collision_rate", "collision_rate_mean"),
    ),
    MetricSpec(
        "repeated_coverage_ratio",
        ("eval_coverage_repeated_coverage_ratio", "eval_overall_repeated_coverage_ratio", "repeated_coverage_ratio_mean"),
    ),
    MetricSpec(
        "demand_revisit_excess",
        ("eval_coverage_demand_revisit_excess", "eval_overall_demand_revisit_excess", "demand_revisit_excess_mean"),
    ),
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _to_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _first_float(row: dict[str, str], candidates: Iterable[str]) -> tuple[float | None, str | None]:
    for field in candidates:
        value = _to_float(row.get(field))
        if value is not None:
            return value, field
    return None, None


def _prefer_coverage_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    preferred = [
        row
        for row in rows
        if row.get("task_name") == "coverage" or row.get("eval_group") in {"coverage", "per_task"}
    ]
    return preferred or rows


def _best_training_row(rows: list[dict[str, str]]) -> dict[str, str] | None:
    scored = []
    for idx, row in enumerate(rows):
        success, _ = _first_float(row, METRICS[0].candidates)
        if success is None:
            continue
        reward, _ = _first_float(row, METRICS[1].candidates)
        update = _to_float(row.get("update")) or float(idx)
        scored.append((success, reward if reward is not None else float("-inf"), update, idx, row))
    if not scored:
        return rows[-1] if rows else None
    return max(scored, key=lambda item: (item[0], item[1], item[2]))[-1]


def _best_eval_row(rows: list[dict[str, str]]) -> dict[str, str] | None:
    rows = _prefer_coverage_rows(rows)
    scored = []
    for idx, row in enumerate(rows):
        success, _ = _first_float(row, METRICS[0].candidates)
        reward, _ = _first_float(row, METRICS[1].candidates)
        score = success if success is not None else float("-inf")
        scored.append((score, reward if reward is not None else float("-inf"), idx, row))
    if not scored:
        return rows[-1] if rows else None
    return max(scored, key=lambda item: (item[0], item[1], item[2]))[-1]


def _summarize_stage(name: str, row: dict[str, str] | None) -> tuple[dict[str, str], list[str]]:
    if row is None:
        return {"stage": name}, [spec.label for spec in METRICS]
    summary: dict[str, str] = {"stage": name}
    missing: list[str] = []
    if row.get("update"):
        summary["update"] = row["update"]
    elif row.get("epoch"):
        summary["update"] = row["epoch"]
    else:
        summary["update"] = "-"
    for spec in METRICS:
        value, source = _first_float(row, spec.candidates)
        if value is None:
            summary[spec.label] = "missing"
            missing.append(spec.label)
        else:
            summary[spec.label] = f"{value:.6g}"
            summary[f"{spec.label}_source"] = source or ""
    return summary, missing


def _numeric(summary: dict[str, str], key: str) -> float | None:
    return _to_float(summary.get(key))


def _conclusion(bc: dict[str, str], stage1: dict[str, str], stage2: dict[str, str]) -> str:
    bc_success = _numeric(bc, "success_rate")
    stage1_success = _numeric(stage1, "success_rate")
    stage2_success = _numeric(stage2, "success_rate")
    if stage2_success is None:
        return "Stage2 缺少 success_rate，无法判断 PPO-main 是否超过 BC/Stage1。"
    comparisons = []
    if bc_success is not None:
        comparisons.append(("BC", bc_success))
    if stage1_success is not None:
        comparisons.append(("Stage1", stage1_success))
    if not comparisons:
        return "BC 和 Stage1 都缺少 success_rate，只能查看 Stage2 曲线，无法做阶段间结论。"
    better_than = [name for name, value in comparisons if stage2_success > value]
    tied_with = [name for name, value in comparisons if stage2_success == value]
    worse_than = [name for name, value in comparisons if stage2_success < value]
    if not worse_than and better_than:
        return f"Stage2 success_rate 高于 {', '.join(better_than)}" + (
            f"，并与 {', '.join(tied_with)} 持平。" if tied_with else "。"
        )
    if worse_than:
        return f"Stage2 success_rate 未超过 {', '.join(worse_than)}；当前不能证明 PPO-main 带来净提升。"
    return "Stage2 success_rate 与 BC/Stage1 持平；需要结合 reward、coverage_ratio、重复覆盖和独立 seed 复验判断。"


def _markdown_table(rows: list[dict[str, str]]) -> str:
    columns = ["stage", "update", "success_rate", "eval_reward", "coverage_ratio", "collision_rate", "repeated_coverage_ratio", "demand_revisit_excess"]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row.get(column, "-") for column in columns) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bc-eval-csv", required=True)
    parser.add_argument("--stage1-training-csv", required=True)
    parser.add_argument("--stage2-training-csv", required=True)
    parser.add_argument("--output-md", required=True)
    args = parser.parse_args()

    bc_path = Path(args.bc_eval_csv)
    stage1_path = Path(args.stage1_training_csv)
    stage2_path = Path(args.stage2_training_csv)
    output_path = Path(args.output_md)

    bc_row = _best_eval_row(_read_csv(bc_path))
    stage1_row = _best_training_row(_read_csv(stage1_path))
    stage2_row = _best_training_row(_read_csv(stage2_path))

    bc_summary, bc_missing = _summarize_stage("BC-only eval", bc_row)
    stage1_summary, stage1_missing = _summarize_stage("Stage1 safe PPO", stage1_row)
    stage2_summary, stage2_missing = _summarize_stage("Stage2 PPO-main", stage2_row)
    rows = [bc_summary, stage1_summary, stage2_summary]

    missing_lines = []
    for name, missing in (
        ("BC-only eval", bc_missing),
        ("Stage1 safe PPO", stage1_missing),
        ("Stage2 PPO-main", stage2_missing),
    ):
        if missing:
            missing_lines.append(f"- {name}: {', '.join(missing)}")
    if not missing_lines:
        missing_lines.append("- none")

    source_lines = []
    for row in rows:
        stage = row["stage"]
        sources = []
        for spec in METRICS:
            source = row.get(f"{spec.label}_source")
            if source:
                sources.append(f"{spec.label}={source}")
        source_lines.append(f"- {stage}: " + (", ".join(sources) if sources else "none"))

    output = f"""# Coverage BC -> Safe PPO -> PPO-main Comparison

## Inputs

- BC eval CSV: `{bc_path}`
- Stage1 training CSV: `{stage1_path}`
- Stage2 training CSV: `{stage2_path}`

## Best Rows

{_markdown_table(rows)}

## Conclusion

{_conclusion(bc_summary, stage1_summary, stage2_summary)}

## Missing Fields

{chr(10).join(missing_lines)}

## Field Sources

{chr(10).join(source_lines)}
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(output, encoding="utf-8")
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
