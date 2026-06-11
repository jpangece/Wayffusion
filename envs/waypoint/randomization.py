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
        "base_position_randomized": False,
        "no_fly_zones_randomized": False,
    }
    if not enabled:
        metadata["base_position"] = list(config.get("base_position", [0.1, 0.1]))
        metadata["no_fly_zone_count"] = len(config.get("no_fly_zones", []))
        metadata["no_fly_zones"] = deepcopy(config.get("no_fly_zones", []))
        return config, metadata

    zone_cfg = dict(randomization.get("no_fly_zones", {}))
    static_zones = list(config.get("no_fly_zones", [])) if bool(zone_cfg.get("include_static", False)) else []
    base_cfg = dict(randomization.get("base_position", {}))
    if bool(base_cfg.get("enabled", False)):
        config["base_position"] = _sample_base_position(config, base_cfg, zone_cfg, rng, static_zones).tolist()
        metadata["base_position_randomized"] = True
    metadata["base_position"] = list(config.get("base_position", [0.1, 0.1]))

    if bool(zone_cfg.get("enabled", True)):
        _validate_base_clear_of_zones(config, static_zones)
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


def _sample_base_position(
    config: dict[str, Any],
    base_cfg: dict[str, Any],
    zone_cfg: dict[str, Any],
    rng: np.random.Generator,
    static_zones: list[dict[str, Any]],
) -> np.ndarray:
    map_size = float(config.get("map_size", 1.0))
    geofence = tuple(config.get("geofence", [0.0, map_size, 0.0, map_size]))
    x_min, x_max, y_min, y_max = map(float, geofence)
    x_range = _ordered_range(base_cfg.get("x_range", [0.06, 0.16]))
    y_range = _ordered_range(base_cfg.get("y_range", [0.06, 0.16]))
    x_low = max(x_min, x_range[0] * map_size)
    x_high = min(x_max, x_range[1] * map_size)
    y_low = max(y_min, y_range[0] * map_size)
    y_high = min(y_max, y_range[1] * map_size)
    if x_high < x_low or y_high < y_low:
        raise ValueError("domain_randomization.base_position ranges do not intersect the geofence")

    clearance = float(zone_cfg.get("base_clearance", 0.0)) * _spatial_scale(config, zone_cfg)
    max_attempts = int(base_cfg.get("max_attempts", 300))
    for _ in range(max_attempts):
        candidate = np.asarray(
            [rng.uniform(x_low, x_high), rng.uniform(y_low, y_high)],
            dtype=np.float32,
        )
        if all(_point_clear_of_zone(candidate, zone, clearance) for zone in static_zones):
            return candidate
    raise RuntimeError("Unable to sample a base position clear of static no-fly zones")


def _validate_base_clear_of_zones(config: dict[str, Any], zones: list[dict[str, Any]]) -> None:
    base = np.asarray(config.get("base_position", [0.1, 0.1]), dtype=np.float32)
    if any(not _point_clear_of_zone(base, zone, 0.0) for zone in zones):
        raise ValueError("Configured static no-fly zone covers the episode base position")


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
    spatial_scale = _spatial_scale(config, zone_cfg)
    boundary_margin = float(zone_cfg.get("boundary_margin", 0.03)) * spatial_scale
    base_clearance = float(zone_cfg.get("base_clearance", 0.08)) * spatial_scale
    zone_clearance = float(zone_cfg.get("zone_clearance", 0.025)) * spatial_scale
    max_attempts = int(zone_cfg.get("max_attempts", 300))
    base = np.asarray(config.get("base_position", [0.1, 0.1]), dtype=np.float32)

    zones: list[dict[str, Any]] = []
    occupied = list(existing)
    for _ in range(count):
        accepted = None
        for _attempt in range(max_attempts):
            if rng.random() < circle_probability:
                radius = float(rng.uniform(*radius_range)) * spatial_scale
                low = np.asarray([x_min + boundary_margin + radius, y_min + boundary_margin + radius])
                high = np.asarray([x_max - boundary_margin - radius, y_max - boundary_margin - radius])
                if np.any(high <= low):
                    continue
                center = rng.uniform(low, high).astype(np.float32)
                candidate = {"center": center.tolist(), "radius": radius}
            else:
                width = float(rng.uniform(*rect_range)) * spatial_scale
                height = float(rng.uniform(*rect_range)) * spatial_scale
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


def _spatial_scale(config: dict[str, Any], section: dict[str, Any]) -> float:
    """Resolve whether randomization distances are map-relative or absolute.

    Existing configs remain map-relative. Scale benchmarks opt into absolute
    map units so a 0.06 no-fly radius remains six metres at every map size.
    """
    units = str(section.get("spatial_units", "relative")).lower()
    if units == "absolute":
        return 1.0
    if units != "relative":
        raise ValueError(f"Unsupported spatial_units={units!r}; expected 'relative' or 'absolute'")
    return float(config.get("map_size", 1.0))


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
