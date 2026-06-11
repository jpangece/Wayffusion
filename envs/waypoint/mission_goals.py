from __future__ import annotations

from typing import Any

import numpy as np


GOAL_DIM = 3
GOAL_SPLIT_ALIASES = {"seen": "train"}
SUPPORTED_GOAL_SPLITS = {"fixed", "train", "seen", "interpolation", "formal"}

TASK_GOAL_MASKS = {
    "area_coverage": np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
    "belief_search": np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
    "priority_inspection": np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
    "connectivity_expansion": np.asarray([1.0, 1.0, 1.0], dtype=np.float32),
}

DEFAULT_GOAL_SETS = {
    "area_coverage": {
        "train": [[0.45, 0.0, 0.0], [0.60, 0.0, 0.0], [0.75, 0.0, 0.0]],
        "interpolation": [[0.525, 0.0, 0.0], [0.675, 0.0, 0.0]],
    },
    "belief_search": {
        "train": [[0.45, 0.0, 0.0], [0.60, 0.0, 0.0], [0.75, 0.0, 0.0]],
        "interpolation": [[0.525, 0.0, 0.0], [0.675, 0.0, 0.0]],
    },
    "priority_inspection": {
        "train": [[0.40, 0.0, 0.0], [0.55, 0.0, 0.0], [0.70, 0.0, 0.0]],
        "interpolation": [[0.475, 0.0, 0.0], [0.625, 0.0, 0.0]],
    },
    "connectivity_expansion": {
        "train": [
            [0.35, 0.30, 0.20],
            [0.45, 0.35, 0.15],
            [0.50, 0.40, 0.12],
        ],
        "interpolation": [
            [0.40, 0.325, 0.18],
            [0.475, 0.375, 0.135],
        ],
    },
}


def goal_mask_for_task(task_name: str) -> np.ndarray:
    return TASK_GOAL_MASKS.get(str(task_name), np.zeros(GOAL_DIM, dtype=np.float32)).copy()


def formal_goal_for_task(task_name: str, env_config: dict[str, Any]) -> np.ndarray:
    task_config = dict(env_config.get("tasks", {}).get(task_name, {}))
    if task_name == "area_coverage":
        values = [task_config.get("success_ratio", 0.85), 0.0, 0.0]
    elif task_name == "belief_search":
        values = [task_config.get("success_mass", 0.85), 0.0, 0.0]
    elif task_name == "priority_inspection":
        values = [task_config.get("success_weighted_completion", 0.80), 0.0, 0.0]
    elif task_name == "connectivity_expansion":
        values = [
            task_config.get("success_coverage", 0.55),
            task_config.get("success_radius", 0.45),
            task_config.get("max_violation_rate", 0.10),
        ]
    else:
        values = [0.0, 0.0, 0.0]
    return _goal_array(values)


def goal_values_for_split(task_name: str, env_config: dict[str, Any], split: str) -> list[np.ndarray]:
    split = _normalize_split(split)
    formal = formal_goal_for_task(task_name, env_config)
    if split in {"fixed", "formal"}:
        return [formal]

    configured = (
        env_config.get("mission_goals", {})
        .get("tasks", {})
        .get(task_name, {})
        .get(split)
    )
    source = configured if configured is not None else DEFAULT_GOAL_SETS.get(task_name, {}).get(split)
    if not source:
        return [formal]
    return [_goal_array(value) for value in source]


def resolve_episode_goal(
    task_name: str,
    env_config: dict[str, Any],
    rng: np.random.Generator,
    split: str = "fixed",
    forced_goal: np.ndarray | list[float] | None = None,
) -> tuple[np.ndarray, np.ndarray, str, str]:
    normalized_split = _normalize_split(split)
    goals = goal_values_for_split(task_name, env_config, normalized_split)
    if forced_goal is not None:
        goal = _goal_array(forced_goal)
        goal_id = f"{normalized_split}_forced"
    else:
        index = int(rng.integers(0, len(goals))) if len(goals) > 1 else 0
        goal = goals[index].copy()
        goal_id = f"{normalized_split}_{index:02d}"
    return goal, goal_mask_for_task(task_name), normalized_split, goal_id


def goal_progress(task_name: str, goal: np.ndarray, metrics: dict[str, Any]) -> float:
    goal = _goal_array(goal)
    if bool(metrics.get("success", False)):
        return 1.0
    if task_name == "area_coverage":
        return _ratio(metrics.get("coverage_ratio", 0.0), goal[0])
    if task_name == "belief_search":
        if bool(metrics.get("detection_rate", 0.0)):
            return 1.0
        return _ratio(metrics.get("searched_probability_mass", 0.0), goal[0])
    if task_name == "priority_inspection":
        return _ratio(metrics.get("weighted_poi_completion", 0.0), goal[0])
    if task_name == "connectivity_expansion":
        coverage = _ratio(metrics.get("coverage_ratio", 0.0), goal[0])
        radius = _ratio(metrics.get("effective_explored_radius", 0.0), goal[1])
        violation = float(metrics.get("connectivity_violation_rate", 0.0))
        budget = max(float(goal[2]), 1e-6)
        safety = float(np.clip(1.0 - max(violation - budget, 0.0) / budget, 0.0, 1.0))
        return float(np.mean([coverage, radius, safety]))
    return 0.0


def _normalize_split(split: str) -> str:
    value = str(split).lower()
    if value not in SUPPORTED_GOAL_SPLITS:
        raise ValueError(f"Unsupported mission goal split: {value}")
    return GOAL_SPLIT_ALIASES.get(value, value)


def _goal_array(value: np.ndarray | list[float]) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    if array.size != GOAL_DIM:
        raise ValueError(f"mission_goal must contain {GOAL_DIM} values, got shape {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError("mission_goal must contain only finite values")
    return array.astype(np.float32)


def _ratio(value: Any, target: float) -> float:
    return float(np.clip(float(value) / max(float(target), 1e-6), 0.0, 1.0))


__all__ = [
    "DEFAULT_GOAL_SETS",
    "GOAL_DIM",
    "SUPPORTED_GOAL_SPLITS",
    "formal_goal_for_task",
    "goal_mask_for_task",
    "goal_progress",
    "goal_values_for_split",
    "resolve_episode_goal",
]
