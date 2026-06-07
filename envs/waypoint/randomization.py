from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np


def build_episode_config(
    base_config: dict[str, Any],
    rng: np.random.Generator,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build a deterministic-per-seed episode layout from the base config."""
    config = deepcopy(base_config)
    randomization = dict(config.get("domain_randomization", {}))
    enabled = bool(randomization.get("enabled", False))
    metadata: dict[str, Any] = {
        "enabled": enabled,
        "no_fly_zones_randomized": False,
    }
    if not enabled:
        metadata["no_fly_zone_count"] = len(config.get("no_fly_zones", []))
        return config, metadata

    zone_cfg = dict(randomization.get("no_fly_zones", {}))
    if bool(zone_cfg.get("enabled", True)):
        static_zones = list(config.get("no_fly_zones", [])) if bool(zone_cfg.get("include_static", False)) else []
        sampled_zones = _sample_no_fly_zones(config, zone_cfg, rng, static_zones)
        config["no_fly_zones"] = static_zones + sampled_zones
        metadata["no_fly_zones_randomized"] = True
        metadata["sampled_no_fly_zone_count"] = len(sampled_zones)

    spawn_cfg = dict(randomization.get("spawn", {}))
    if spawn_cfg:
        config["spawn_randomization"] = spawn_cfg

    metadata["no_fly_zone_count"] = len(config.get("no_fly_zones", []))
    metadata["no_fly_zones"] = deepcopy(config.get("no_fly_zones", []))
    return config, metadata


def _sample_no_fly_zones(
    config: dict[str, Any],
    zone_cfg: dict[str, Any],
    rng: np.random.Generator,
    existing: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    map_size = float(config.get("map_size", 1.0))
    geofence = tuple(config.get("geofence", [0.0, map_size, 0.0, map_size]))
    x_min, x_max, y_min, y_max = map(float, geofence)
    count_min = int(zone_cfg.get("min_count", zone_cfg.get("count", 1)))
    count_max = int(zone_cfg.get("max_count", zone_cfg.get("count", count_min)))
    count = int(rng.integers(count_min, count_max + 1))
    circle_probability = float(zone_cfg.get("circle_probability", 0.75))
    radius_range = _ordered_range(zone_cfg.get("radius_range", [0.04, 0.09]))
    rect_range = _ordered_range(zone_cfg.get("rectangle_size_range", [0.06, 0.14]))
    boundary_margin = float(zone_cfg.get("boundary_margin", 0.03)) * map_size
    base_clearance = float(zone_cfg.get("base_clearance", 0.08)) * map_size
    zone_clearance = float(zone_cfg.get("zone_clearance", 0.025)) * map_size
    max_attempts = int(zone_cfg.get("max_attempts", 300))
    base = np.asarray(config.get("base_position", [0.1, 0.1]), dtype=np.float32)

    zones: list[dict[str, Any]] = []
    occupied = list(existing)
    for _ in range(count):
        accepted = None
        for _attempt in range(max_attempts):
            if rng.random() < circle_probability:
                radius = float(rng.uniform(*radius_range)) * map_size
                low = np.asarray([x_min + boundary_margin + radius, y_min + boundary_margin + radius])
                high = np.asarray([x_max - boundary_margin - radius, y_max - boundary_margin - radius])
                if np.any(high <= low):
                    continue
                center = rng.uniform(low, high).astype(np.float32)
                candidate = {"center": center.tolist(), "radius": radius}
            else:
                width = float(rng.uniform(*rect_range)) * map_size
                height = float(rng.uniform(*rect_range)) * map_size
                low = np.asarray([x_min + boundary_margin + width / 2.0, y_min + boundary_margin + height / 2.0])
                high = np.asarray([x_max - boundary_margin - width / 2.0, y_max - boundary_margin - height / 2.0])
                if np.any(high <= low):
                    continue
                center = rng.uniform(low, high).astype(np.float32)
                candidate = {
                    "x_min": float(center[0] - width / 2.0),
                    "x_max": float(center[0] + width / 2.0),
                    "y_min": float(center[1] - height / 2.0),
                    "y_max": float(center[1] + height / 2.0),
                }
            if not _point_clear_of_zone(base, candidate, base_clearance):
                continue
            if any(not _zones_separated(candidate, other, zone_clearance) for other in occupied):
                continue
            accepted = candidate
            break
        if accepted is not None:
            zones.append(accepted)
            occupied.append(accepted)
    return zones


def _ordered_range(values: Any) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float32).reshape(-1)
    if len(array) != 2:
        raise ValueError(f"Expected a two-value range, got {values!r}")
    return float(min(array)), float(max(array))


def _zone_bounding_circle(zone: dict[str, Any]) -> tuple[np.ndarray, float]:
    if "center" in zone and "radius" in zone:
        return np.asarray(zone["center"], dtype=np.float32), float(zone["radius"])
    center = np.asarray(
        [
            (float(zone["x_min"]) + float(zone["x_max"])) / 2.0,
            (float(zone["y_min"]) + float(zone["y_max"])) / 2.0,
        ],
        dtype=np.float32,
    )
    half_width = (float(zone["x_max"]) - float(zone["x_min"])) / 2.0
    half_height = (float(zone["y_max"]) - float(zone["y_min"])) / 2.0
    return center, float(np.hypot(half_width, half_height))


def _point_clear_of_zone(point: np.ndarray, zone: dict[str, Any], clearance: float) -> bool:
    center, radius = _zone_bounding_circle(zone)
    return bool(np.linalg.norm(np.asarray(point, dtype=np.float32) - center) > radius + clearance)


def _zones_separated(first: dict[str, Any], second: dict[str, Any], clearance: float) -> bool:
    center_a, radius_a = _zone_bounding_circle(first)
    center_b, radius_b = _zone_bounding_circle(second)
    return bool(np.linalg.norm(center_a - center_b) > radius_a + radius_b + clearance)


__all__ = ["build_episode_config"]
