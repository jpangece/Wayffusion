from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


SWARM30_TASKS = (
    "area_coverage",
    "belief_search",
    "priority_inspection",
    "connectivity_expansion",
    "dynamic_target_escort",
    "target_interception",
)


@dataclass(frozen=True)
class ScaleProfile:
    num_agents: int
    map_size: float
    grid_size: int
    max_steps: int
    poi_count: int
    belief_peaks: int
    dynamic_targets: int
    no_fly_count: tuple[int, int]
    spawn_max_radius: float


SCALE_PROFILES = {
    4: ScaleProfile(4, 1.00, 32, 80, 8, 3, 1, (0, 1), 0.25),
    10: ScaleProfile(10, 1.58, 56, 128, 20, 8, 2, (1, 3), 0.28),
    20: ScaleProfile(20, 2.24, 72, 180, 40, 15, 4, (2, 5), 0.32),
    30: ScaleProfile(30, 2.74, 88, 220, 60, 23, 5, (3, 8), 0.35),
}


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def build_scaled_env_config(
    base_config: dict[str, Any],
    num_agents: int,
    tasks: list[str] | tuple[str, ...] | None = None,
    randomization: bool = True,
) -> dict[str, Any]:
    """Create a density-normalized benchmark environment.

    Map area and task load scale with swarm size. Physical capabilities remain
    absolute map units: sensor/communication radius, speed, and action scale do
    not grow with the map.
    """
    if int(num_agents) not in SCALE_PROFILES:
        raise ValueError(f"Unsupported swarm size {num_agents}; expected one of {sorted(SCALE_PROFILES)}")
    profile = SCALE_PROFILES[int(num_agents)]
    config = deepcopy(base_config)
    selected_tasks = list(tasks or SWARM30_TASKS)
    unknown = sorted(set(selected_tasks) - set(SWARM30_TASKS))
    if unknown:
        raise ValueError(f"Unsupported swarm30 benchmark tasks: {unknown}")

    config.update(
        {
            "benchmark_protocol": "swarm30_v1",
            "num_agents": profile.num_agents,
            "map_size": profile.map_size,
            "grid_size": profile.grid_size,
            "max_steps": profile.max_steps,
            "geofence": [0.0, profile.map_size, 0.0, profile.map_size],
            "task_names": selected_tasks,
            "task_name": selected_tasks[0] if len(selected_tasks) == 1 else None,
            "task_sampling_probs": {task: 1.0 for task in selected_tasks},
            "connectivity_action_filter": False,
            "connectivity_candidate_filter": False,
            "no_fly_zones": [],
        }
    )
    random_cfg = config.setdefault("domain_randomization", {})
    random_cfg["enabled"] = bool(randomization)
    spawn_cfg = random_cfg.setdefault("spawn", {})
    spawn_cfg.update(
        {
            "spatial_units": "absolute",
            "min_radius": 0.03,
            "max_radius": profile.spawn_max_radius,
        }
    )
    zone_cfg = random_cfg.setdefault("no_fly_zones", {})
    zone_cfg.update(
        {
            "enabled": True,
            "include_static": False,
            "spatial_units": "absolute",
            "min_count": profile.no_fly_count[0],
            "max_count": profile.no_fly_count[1],
        }
    )

    task_cfg = config.setdefault("tasks", {})
    task_cfg.setdefault("area_coverage", {}).update(
        {
            "spatial_units": "absolute",
            "uncovered_approach_scale": 0.20,
        }
    )
    task_cfg.setdefault("belief_search", {}).update(
        {
            "spatial_units": "absolute",
            "num_peaks": profile.belief_peaks,
            "num_peaks_range": [profile.belief_peaks, profile.belief_peaks],
            "peak_min_separation": 0.08,
            "peak_sigma_range": [0.08, 0.18],
        }
    )
    task_cfg.setdefault("priority_inspection", {}).update(
        {
            "spatial_units": "absolute",
            "num_pois": profile.poi_count,
            "num_pois_range": [profile.poi_count, profile.poi_count],
            "poi_min_separation": 0.05,
            "poi_target_sigma": 0.08,
            "approach_scale": 0.20,
        }
    )
    task_cfg.setdefault("connectivity_expansion", {}).update(
        {
            "spatial_units": "absolute",
            "uncovered_approach_scale": 0.24,
        }
    )
    task_cfg.setdefault("dynamic_target_escort", {}).update(
        {
            "spatial_units": "absolute",
            "num_targets": profile.dynamic_targets,
            "target_min_separation": 0.12,
            "target_speed": 0.035,
            "escort_radius": 0.18,
            "observation_radius": 0.20,
            "success_target_coverage": 0.70,
            "success_worst_target_coverage": 0.50,
        }
    )
    task_cfg.setdefault("target_interception", {}).update(
        {
            "spatial_units": "absolute",
            "num_targets": profile.dynamic_targets,
            "target_speed": 0.035,
            "interception_radius": 0.12,
            "containment_radius": 0.30,
            "success_interception_ratio": 0.80,
        }
    )
    return config


def sync_policy_scale(policy_config: dict[str, Any], env_config: dict[str, Any]) -> dict[str, Any]:
    config = deepcopy(policy_config)
    for key in (
        "map_size",
        "grid_size",
        "max_waypoint_distance",
        "min_uav_distance",
        "comm_radius",
        "base_comm_radius",
        "base_position",
        "no_fly_zones",
        "connectivity_candidate_filter",
    ):
        if key in env_config:
            config[key] = deepcopy(env_config[key])
    config["connectivity_candidate_filter"] = False
    absolute_radii = config.get("benchmark_spatial_context_radii_absolute")
    if absolute_radii is not None:
        config["spatial_context_radii"] = [
            float(radius) / float(env_config["map_size"]) for radius in absolute_radii
        ]
    return config


def normalized_task_progress(task_name: str, metrics: dict[str, Any]) -> float:
    def value(name: str, default: float = 0.0) -> float:
        try:
            return float(metrics.get(name, default))
        except (TypeError, ValueError):
            return default

    if task_name == "area_coverage":
        progress = value("coverage_ratio") / 0.85
    elif task_name == "belief_search":
        progress = value("searched_probability_mass") / 0.85
    elif task_name == "priority_inspection":
        progress = value("weighted_poi_completion") / 0.80
    elif task_name == "connectivity_expansion":
        coverage = value("coverage_ratio") / 0.55
        radius = value("effective_explored_radius") / 0.45
        violation = value("connectivity_violation_rate")
        safety = max(0.0, 1.0 - violation / 0.10)
        progress = min(coverage, radius, safety)
    elif task_name == "dynamic_target_escort":
        mean_progress = value("target_coverage_ratio") / 0.70
        worst_progress = value("worst_target_coverage_ratio") / 0.50
        progress = min(mean_progress, worst_progress)
    elif task_name == "target_interception":
        progress = value("intercepted_target_ratio", value("interception_success")) / 0.80
    else:
        raise ValueError(f"Unsupported benchmark task: {task_name}")
    return float(max(0.0, min(progress, 1.0)))


__all__ = [
    "SCALE_PROFILES",
    "SWARM30_TASKS",
    "ScaleProfile",
    "build_scaled_env_config",
    "load_yaml",
    "normalized_task_progress",
    "sync_policy_scale",
]
