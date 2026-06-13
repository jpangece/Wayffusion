from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from envs.waypoint.entities import UAVState


@dataclass
class CoverageTransitionStats:
    positions: np.ndarray
    radii: np.ndarray
    sensor_masks: np.ndarray
    navigable_mask: np.ndarray
    before_covered: np.ndarray
    before_visited: np.ndarray
    newly_covered_mask: np.ndarray
    per_agent_new_masks: np.ndarray
    per_agent_new_coverage: np.ndarray
    repeat_footprint_ratios: np.ndarray


def _zone_center_radius(zone: dict[str, Any]) -> tuple[np.ndarray, float] | None:
    if "center" in zone and "radius" in zone:
        return np.asarray(zone["center"], dtype=np.float32), float(zone["radius"])
    return None


def _zone_rect(zone: dict[str, Any]) -> tuple[float, float, float, float] | None:
    if {"x_min", "x_max", "y_min", "y_max"}.issubset(zone):
        return float(zone["x_min"]), float(zone["x_max"]), float(zone["y_min"]), float(zone["y_max"])
    return None


@dataclass
class MissionWorld:
    map_size: float = 1.0
    world_dim: int = 2
    default_altitude: float = 30.0
    dt_decision: float = 1.0
    physics_dt: float = 0.05
    substeps: int = 20
    max_speed: float = 0.08
    mass: float = 1.0
    damping: float = 0.2
    radius: float = 0.02
    max_waypoint_distance: float = 0.2
    min_uav_distance: float = 0.03
    base_position: np.ndarray = field(default_factory=lambda: np.asarray([0.1, 0.1], dtype=np.float32))
    base_comm_radius: float = 0.45
    grid_size: int = 32
    max_steps: int = 100
    no_fly_zones: list[dict[str, Any]] = field(default_factory=list)
    geofence: tuple[float, float, float, float] | None = None
    spawn_randomization: dict[str, Any] = field(default_factory=dict)
    sensor_footprint_mode: str = "exact"
    geometry_backend: str = "numpy"
    geometry_device: str = "cpu"
    rng: np.random.Generator = field(default_factory=np.random.default_rng)

    uavs: list[UAVState] = field(default_factory=list)
    coverage_grid: np.ndarray = field(init=False)
    visit_count_grid: np.ndarray = field(init=False)
    probability_grid: np.ndarray = field(init=False)
    belief_grid: np.ndarray = field(init=False)
    communication_graph: np.ndarray = field(init=False)
    step_count: int = 0

    def __post_init__(self) -> None:
        self.base_position = np.asarray(self.base_position, dtype=np.float32)
        self.geofence = self.geofence or (0.0, float(self.map_size), 0.0, float(self.map_size))
        self.coverage_grid = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        self.visit_count_grid = np.zeros_like(self.coverage_grid)
        self.probability_grid = np.ones_like(self.coverage_grid) / float(self.grid_size * self.grid_size)
        self.belief_grid = self.probability_grid.copy()
        self.communication_graph = np.zeros((1, 1), dtype=bool)
        self._grid_centers_cache: np.ndarray | None = None
        self._navigable_grid_mask_cache: np.ndarray | None = None
        self._risk_grid_cache: np.ndarray | None = None
        self._communication_positions_cache: np.ndarray | None = None
        self._connected_mask_cache: np.ndarray | None = None
        self._pairwise_positions_cache: np.ndarray | None = None
        self._pairwise_distances_cache: np.ndarray | None = None
        self._sensor_mask_positions_cache: np.ndarray | None = None
        self._sensor_mask_radii_cache: np.ndarray | None = None
        self._sensor_masks_cache: np.ndarray | None = None
        self._exact_sensor_mask_positions_cache: np.ndarray | None = None
        self._exact_sensor_mask_radii_cache: np.ndarray | None = None
        self._exact_sensor_masks_cache: np.ndarray | None = None
        self._last_coverage_transition_stats: CoverageTransitionStats | None = None
        self._disk_offsets_cache: dict[float, np.ndarray] = {}

    @classmethod
    def from_config(cls, config: dict[str, Any], rng: np.random.Generator | None = None) -> "MissionWorld":
        dynamics_cfg = dict(config.get("dynamics_backend", {}))
        runtime_cfg = dict(config.get("runtime_acceleration", {}))
        geometry_backend = str(runtime_cfg.get("geometry_backend", "numpy"))
        geometry_device = str(runtime_cfg.get("geometry_device", "cuda" if geometry_backend in {"cuda", "torch_cuda"} else "cpu"))
        return cls(
            map_size=float(config.get("map_size", 1.0)),
            world_dim=int(config.get("world_dim", 2)),
            default_altitude=float(config.get("default_altitude", 30.0)),
            dt_decision=float(config.get("dt_decision", 1.0)),
            physics_dt=float(dynamics_cfg.get("dt", dynamics_cfg.get("physics_dt", 0.1))),
            substeps=int(dynamics_cfg.get("substeps", 10)),
            max_speed=float(dynamics_cfg.get("max_speed", config.get("max_speed", 0.08))),
            mass=float(dynamics_cfg.get("mass", 1.0)),
            damping=float(dynamics_cfg.get("damping", 0.25)),
            radius=float(dynamics_cfg.get("agent_size", dynamics_cfg.get("radius", 0.02))),
            max_waypoint_distance=float(config.get("max_waypoint_distance", 0.2)),
            min_uav_distance=float(config.get("min_uav_distance", 0.03)),
            base_position=np.asarray(config.get("base_position", [0.1, 0.1]), dtype=np.float32),
            base_comm_radius=float(config.get("base_comm_radius", 0.45)),
            grid_size=int(config.get("grid_size", 32)),
            max_steps=int(config.get("max_steps", 100)),
            no_fly_zones=list(config.get("no_fly_zones", [])),
            geofence=tuple(config.get("geofence", [0.0, config.get("map_size", 1.0), 0.0, config.get("map_size", 1.0)])),
            spawn_randomization=dict(config.get("spawn_randomization", {})),
            sensor_footprint_mode=str(runtime_cfg.get("sensor_footprint_mode", "exact")),
            geometry_backend=geometry_backend,
            geometry_device=geometry_device,
            rng=rng or np.random.default_rng(int(config.get("seed", 0))),
        )

    def reset_common(self, num_agents: int, rng: np.random.Generator | None = None, initial_positions=None) -> None:
        self.rng = rng or self.rng
        self._invalidate_communication_cache()
        if not bool(self.check_geofence(self.base_position[None, :])[0]):
            raise ValueError(f"Base position {self.base_position.tolist()} lies outside the geofence")
        if not bool(self.check_no_fly(self.base_position[None, :])[0]):
            raise ValueError(f"Base position {self.base_position.tolist()} lies inside a no-fly zone")
        self.step_count = 0
        self._last_coverage_transition_stats = None
        self.coverage_grid.fill(0.0)
        self.visit_count_grid.fill(0.0)
        safe_mask = self.navigable_grid_mask()
        self.probability_grid.fill(0.0)
        self.probability_grid[safe_mask] = 1.0 / max(float(safe_mask.sum()), 1.0)
        self.belief_grid = self.probability_grid.copy()
        self.uavs = []
        if initial_positions is None:
            initial_positions = self._sample_safe_initial_positions(num_agents)
        for idx in range(num_agents):
            pos = np.asarray(initial_positions[idx], dtype=np.float32)
            self.uavs.append(
                UAVState(
                    position=pos.copy(),
                    velocity=np.zeros(2, dtype=np.float32),
                    current_waypoint=pos.copy(),
                    role_id=idx,
                    mass=float(self.mass),
                    radius=float(self.radius),
                    sensor_radius=float(getattr(self, "sensor_radius", 0.12)),
                    comm_radius=float(getattr(self, "comm_radius", 0.35)),
                )
            )
        self.update_communication_graph()
        self.accumulate_sensor_footprints()

    def _invalidate_communication_cache(self) -> None:
        self._communication_positions_cache = None
        self._connected_mask_cache = None
        self._pairwise_positions_cache = None
        self._pairwise_distances_cache = None

    def _invalidate_sensor_mask_cache(self) -> None:
        self._sensor_mask_positions_cache = None
        self._sensor_mask_radii_cache = None
        self._sensor_masks_cache = None
        self._exact_sensor_mask_positions_cache = None
        self._exact_sensor_mask_radii_cache = None
        self._exact_sensor_masks_cache = None

    def _sample_safe_initial_positions(self, num_agents: int) -> np.ndarray:
        cfg = self.spawn_randomization
        units = str(cfg.get("spatial_units", "relative")).lower()
        if units not in {"relative", "absolute"}:
            raise ValueError(f"Unsupported spawn spatial_units={units!r}")
        scale = 1.0 if units == "absolute" else self.map_size
        min_radius = float(cfg.get("min_radius", 0.02)) * scale
        max_radius = float(cfg.get("max_radius", 0.22)) * scale
        min_dist_scale = float(cfg.get("min_separation_scale", 1.8))
        min_dist = max(float(self.min_uav_distance) * min_dist_scale, float(self.radius) * 2.5)
        positions: list[np.ndarray] = []
        for idx in range(num_agents):
            accepted = None
            for _ in range(int(cfg.get("max_attempts", 300))):
                radius = self.rng.uniform(min_radius, max_radius)
                angle = self.rng.uniform(0.0, 2.0 * np.pi)
                candidate = self.base_position + radius * np.asarray([np.cos(angle), np.sin(angle)], dtype=np.float32)
                if not self.check_geofence(candidate[None, :])[0] or not self.check_no_fly(candidate[None, :])[0]:
                    continue
                if all(np.linalg.norm(candidate - prev) >= min_dist for prev in positions):
                    accepted = candidate.astype(np.float32)
                    break
            if accepted is None:
                accepted = self._fallback_spawn_position(idx, num_agents, positions, min_dist)
            positions.append(accepted)
        return np.asarray(positions, dtype=np.float32)

    def _fallback_spawn_position(
        self,
        index: int,
        num_agents: int,
        positions: list[np.ndarray],
        min_distance: float,
    ) -> np.ndarray:
        for ring in range(1, 12):
            radius = min_distance * ring
            for offset in range(max(num_agents, 8)):
                angle = 2.0 * np.pi * (index + offset / max(num_agents, 1)) / max(num_agents, 1)
                candidate = self.base_position + radius * np.asarray([np.cos(angle), np.sin(angle)], dtype=np.float32)
                if not self.check_geofence(candidate[None, :])[0] or not self.check_no_fly(candidate[None, :])[0]:
                    continue
                if all(np.linalg.norm(candidate - prev) >= min_distance for prev in positions):
                    return candidate.astype(np.float32)
        if self.check_geofence(self.base_position[None, :])[0] and self.check_no_fly(self.base_position[None, :])[0]:
            return self.base_position.copy()
        return self.sample_safe_points(1)[0]

    def clone(self, include_trajectory: bool = True) -> "MissionWorld":
        new = MissionWorld(
            map_size=self.map_size,
            world_dim=self.world_dim,
            default_altitude=self.default_altitude,
            dt_decision=self.dt_decision,
            physics_dt=self.physics_dt,
            substeps=self.substeps,
            max_speed=self.max_speed,
            mass=self.mass,
            damping=self.damping,
            radius=self.radius,
            max_waypoint_distance=self.max_waypoint_distance,
            min_uav_distance=self.min_uav_distance,
            base_position=self.base_position.copy(),
            base_comm_radius=self.base_comm_radius,
            grid_size=self.grid_size,
            max_steps=self.max_steps,
            no_fly_zones=[dict(zone) for zone in self.no_fly_zones],
            geofence=self.geofence,
            spawn_randomization=dict(self.spawn_randomization),
            sensor_footprint_mode=str(self.sensor_footprint_mode),
            geometry_backend=str(self.geometry_backend),
            geometry_device=str(self.geometry_device),
            rng=self.rng,
        )
        new.uavs = [uav.snapshot(include_trajectory=include_trajectory) for uav in self.uavs]
        new.coverage_grid = self.coverage_grid.copy()
        new.visit_count_grid = self.visit_count_grid.copy()
        new.probability_grid = self.probability_grid.copy()
        new.belief_grid = self.belief_grid.copy()
        new.communication_graph = self.communication_graph.copy()
        new.step_count = int(self.step_count)
        new._grid_centers_cache = self._grid_centers_cache
        new._navigable_grid_mask_cache = self._navigable_grid_mask_cache
        new._risk_grid_cache = self._risk_grid_cache
        new._disk_offsets_cache = dict(self._disk_offsets_cache)
        return new

    snapshot = clone

    def snapshot_for_reward(self) -> "MissionWorld":
        return self.clone(include_trajectory=False)

    def check_geofence(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=np.float32)
        x_min, x_max, y_min, y_max = self.geofence or (0.0, self.map_size, 0.0, self.map_size)
        return (points[..., 0] >= x_min) & (points[..., 0] <= x_max) & (points[..., 1] >= y_min) & (points[..., 1] <= y_max)

    def project_to_geofence(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=np.float32).copy()
        x_min, x_max, y_min, y_max = self.geofence or (0.0, self.map_size, 0.0, self.map_size)
        points[..., 0] = np.clip(points[..., 0], x_min, x_max)
        points[..., 1] = np.clip(points[..., 1], y_min, y_max)
        return points

    def check_no_fly(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=np.float32)
        valid = np.ones(points.shape[:-1], dtype=bool)
        for zone in self.no_fly_zones:
            circle = _zone_center_radius(zone)
            rect = _zone_rect(zone)
            if circle is not None:
                center, radius = circle
                valid &= np.linalg.norm(points[..., :2] - center[:2], axis=-1) > radius
            elif rect is not None:
                x_min, x_max, y_min, y_max = rect
                inside = (
                    (points[..., 0] >= x_min)
                    & (points[..., 0] <= x_max)
                    & (points[..., 1] >= y_min)
                    & (points[..., 1] <= y_max)
                )
                valid &= ~inside
        return valid

    def world_to_grid(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=np.float32)
        scaled = np.clip(points[..., :2] / max(self.map_size, 1e-6), 0.0, 0.999999)
        return np.floor(scaled * self.grid_size).astype(np.int64)

    def grid_to_world(self, indices: np.ndarray) -> np.ndarray:
        indices = np.asarray(indices, dtype=np.float32)
        return (indices[..., ::-1] + 0.5) / float(self.grid_size) * self.map_size

    def grid_cell_centers(self) -> np.ndarray:
        if self._grid_centers_cache is None:
            yy, xx = np.mgrid[0 : self.grid_size, 0 : self.grid_size]
            self._grid_centers_cache = (
                np.stack(
                    [(xx + 0.5) / self.grid_size, (yy + 0.5) / self.grid_size],
                    axis=-1,
                ).astype(np.float32)
                * self.map_size
            )
        return self._grid_centers_cache

    def navigable_grid_mask(self) -> np.ndarray:
        if self._navigable_grid_mask_cache is None:
            centers = self.grid_cell_centers()
            self._navigable_grid_mask_cache = self.check_geofence(centers) & self.check_no_fly(centers)
        return self._navigable_grid_mask_cache

    def risk_grid(self) -> np.ndarray:
        if self._risk_grid_cache is None:
            risk = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
            centers = self.grid_cell_centers()
            for zone in self.no_fly_zones:
                circle = _zone_center_radius(zone)
                rect = _zone_rect(zone)
                if circle is not None:
                    center, radius = circle
                    distance = np.linalg.norm(centers - center[:2], axis=-1)
                    risk = np.maximum(risk, (distance <= radius).astype(np.float32))
                elif rect is not None:
                    x_min, x_max, y_min, y_max = rect
                    inside = (
                        (centers[..., 0] >= x_min)
                        & (centers[..., 0] <= x_max)
                        & (centers[..., 1] >= y_min)
                        & (centers[..., 1] <= y_max)
                    )
                    risk = np.maximum(risk, inside.astype(np.float32))
            self._risk_grid_cache = risk
        return self._risk_grid_cache

    def coverage_ratio(self) -> float:
        navigable = self.navigable_grid_mask()
        return float(np.count_nonzero((self.coverage_grid > 0.0) & navigable) / max(float(navigable.sum()), 1.0))

    def sample_safe_points(
        self,
        count: int,
        lower_fraction: float = 0.08,
        upper_fraction: float = 0.92,
        min_separation: float = 0.0,
        avoid_points: np.ndarray | None = None,
        max_attempts: int = 2000,
    ) -> np.ndarray:
        x_min, x_max, y_min, y_max = self.geofence or (0.0, self.map_size, 0.0, self.map_size)
        low = np.asarray(
            [
                x_min + lower_fraction * (x_max - x_min),
                y_min + lower_fraction * (y_max - y_min),
            ],
            dtype=np.float32,
        )
        high = np.asarray(
            [
                x_min + upper_fraction * (x_max - x_min),
                y_min + upper_fraction * (y_max - y_min),
            ],
            dtype=np.float32,
        )
        accepted: list[np.ndarray] = []
        avoid = [] if avoid_points is None else [point for point in np.asarray(avoid_points, dtype=np.float32)]
        for _ in range(max_attempts):
            if len(accepted) >= int(count):
                break
            candidate = self.rng.uniform(low, high).astype(np.float32)
            if not self.check_no_fly(candidate[None, :])[0]:
                continue
            if any(np.linalg.norm(candidate - point) < min_separation for point in accepted + avoid):
                continue
            accepted.append(candidate)
        if len(accepted) < int(count):
            safe_indices = np.argwhere(self.navigable_grid_mask())
            self.rng.shuffle(safe_indices)
            for row, col in safe_indices:
                candidate = self.grid_to_world(np.asarray([[row, col]], dtype=np.float32))[0]
                if any(np.linalg.norm(candidate - point) < min_separation for point in accepted + avoid):
                    continue
                accepted.append(candidate.astype(np.float32))
                if len(accepted) >= int(count):
                    break
        if len(accepted) < int(count):
            raise RuntimeError(f"Unable to sample {count} safe points; sampled {len(accepted)}")
        return np.asarray(accepted[:count], dtype=np.float32)

    def sensor_mask_for_position(self, position: np.ndarray, radius: float | None = None) -> np.ndarray:
        radius = float(radius if radius is not None else (self.uavs[0].sensor_radius if self.uavs else 0.12))
        centers = self.grid_cell_centers()
        dist = np.linalg.norm(centers - np.asarray(position, dtype=np.float32), axis=-1)
        return dist <= radius

    def exact_sensor_masks_for_positions(
        self,
        positions: np.ndarray,
        radii: np.ndarray | float | None = None,
    ) -> np.ndarray:
        positions = np.asarray(positions, dtype=np.float32)
        if positions.size == 0:
            return np.zeros((0, self.grid_size, self.grid_size), dtype=bool)
        if radii is None:
            radii_arr = np.asarray(
                [uav.sensor_radius for uav in self.uavs[: len(positions)]],
                dtype=np.float32,
            )
            if len(radii_arr) != len(positions):
                default_radius = self.uavs[0].sensor_radius if self.uavs else 0.12
                radii_arr = np.full(len(positions), default_radius, dtype=np.float32)
        else:
            radii_arr = np.asarray(radii, dtype=np.float32)
            if radii_arr.ndim == 0:
                radii_arr = np.full(len(positions), float(radii_arr), dtype=np.float32)
        if (
            self._exact_sensor_masks_cache is not None
            and self._exact_sensor_mask_positions_cache is not None
            and self._exact_sensor_mask_radii_cache is not None
            and self._exact_sensor_mask_positions_cache.shape == positions.shape
            and self._exact_sensor_mask_radii_cache.shape == radii_arr.shape
            and np.array_equal(self._exact_sensor_mask_positions_cache, positions)
            and np.array_equal(self._exact_sensor_mask_radii_cache, radii_arr)
        ):
            return self._exact_sensor_masks_cache
        centers = self.grid_cell_centers()
        distances = np.linalg.norm(
            positions[:, None, None, :] - centers[None, :, :, :],
            axis=-1,
        )
        masks = distances <= radii_arr[:, None, None]
        self._exact_sensor_mask_positions_cache = positions.copy()
        self._exact_sensor_mask_radii_cache = radii_arr.copy()
        self._exact_sensor_masks_cache = masks
        return masks

    def sensor_masks_for_positions(self, positions: np.ndarray, radii: np.ndarray | float | None = None) -> np.ndarray:
        positions = np.asarray(positions, dtype=np.float32)
        if positions.size == 0:
            return np.zeros((0, self.grid_size, self.grid_size), dtype=bool)
        if radii is None:
            radii_arr = np.asarray([uav.sensor_radius for uav in self.uavs[: len(positions)]], dtype=np.float32)
            if len(radii_arr) != len(positions):
                radii_arr = np.full(len(positions), self.uavs[0].sensor_radius if self.uavs else 0.12, dtype=np.float32)
        else:
            radii_arr = np.asarray(radii, dtype=np.float32)
            if radii_arr.ndim == 0:
                radii_arr = np.full(len(positions), float(radii_arr), dtype=np.float32)
        if (
            self._sensor_masks_cache is not None
            and self._sensor_mask_positions_cache is not None
            and self._sensor_mask_radii_cache is not None
            and self._sensor_mask_positions_cache.shape == positions.shape
            and self._sensor_mask_radii_cache.shape == radii_arr.shape
            and np.array_equal(self._sensor_mask_positions_cache, positions)
            and np.array_equal(self._sensor_mask_radii_cache, radii_arr)
        ):
            return self._sensor_masks_cache
        if str(self.sensor_footprint_mode).lower() == "disk_stamp":
            masks = self._disk_stamp_sensor_masks(positions, radii_arr)
            self._sensor_mask_positions_cache = positions.copy()
            self._sensor_mask_radii_cache = radii_arr.copy()
            self._sensor_masks_cache = masks
            return masks
        if self._use_torch_geometry():
            masks = self._torch_exact_sensor_masks(positions, radii_arr)
            self._sensor_mask_positions_cache = positions.copy()
            self._sensor_mask_radii_cache = radii_arr.copy()
            self._sensor_masks_cache = masks
            return masks
        centers = self.grid_cell_centers()
        dist = np.linalg.norm(positions[:, None, None, :] - centers[None, :, :, :], axis=-1)
        masks = dist <= radii_arr[:, None, None]
        self._sensor_mask_positions_cache = positions.copy()
        self._sensor_mask_radii_cache = radii_arr.copy()
        self._sensor_masks_cache = masks
        return masks

    def _use_torch_geometry(self) -> bool:
        return str(self.geometry_backend).lower() in {"torch", "torch_cpu", "torch_cuda", "cuda"}

    def _torch_device(self):
        import torch

        backend = str(self.geometry_backend).lower()
        requested = "cuda" if backend in {"torch_cuda", "cuda"} else str(self.geometry_device)
        device = torch.device(requested)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("runtime_acceleration.geometry_backend requested CUDA, but torch.cuda.is_available() is false")
        return device

    def _torch_exact_sensor_masks(self, positions: np.ndarray, radii: np.ndarray) -> np.ndarray:
        import torch

        device = self._torch_device()
        pos_t = torch.as_tensor(positions, dtype=torch.float32, device=device)
        radii_t = torch.as_tensor(radii, dtype=torch.float32, device=device)
        centers_t = torch.as_tensor(self.grid_cell_centers(), dtype=torch.float32, device=device)
        dist_t = torch.linalg.norm(pos_t[:, None, None, :] - centers_t[None, :, :, :], dim=-1)
        return (dist_t <= radii_t[:, None, None]).detach().cpu().numpy()

    def _disk_offsets_for_radius(self, radius: float) -> np.ndarray:
        key = float(np.round(radius, 6))
        if key in self._disk_offsets_cache:
            return self._disk_offsets_cache[key]
        cell = float(self.map_size) / max(int(self.grid_size), 1)
        max_cells = int(np.ceil(float(radius) / max(cell, 1e-8))) + 1
        offsets = []
        for dy in range(-max_cells, max_cells + 1):
            for dx in range(-max_cells, max_cells + 1):
                distance = np.sqrt((dx * cell) ** 2 + (dy * cell) ** 2)
                if distance <= float(radius) + 0.5 * cell:
                    offsets.append((dy, dx))
        out = np.asarray(offsets, dtype=np.int64)
        self._disk_offsets_cache[key] = out
        return out

    def _disk_stamp_sensor_masks(self, positions: np.ndarray, radii: np.ndarray) -> np.ndarray:
        positions = np.asarray(positions, dtype=np.float32)
        radii = np.asarray(radii, dtype=np.float32)
        masks = np.zeros((len(positions), self.grid_size, self.grid_size), dtype=bool)
        centers = self.world_to_grid(positions)
        for idx, center in enumerate(centers):
            offsets = self._disk_offsets_for_radius(float(radii[idx]))
            rows = int(center[1]) + offsets[:, 0]
            cols = int(center[0]) + offsets[:, 1]
            valid = (rows >= 0) & (rows < self.grid_size) & (cols >= 0) & (cols < self.grid_size)
            masks[idx, rows[valid], cols[valid]] = True
        return masks

    def _build_coverage_transition_stats(
        self,
        before_covered: np.ndarray,
        before_visited: np.ndarray,
        masks: np.ndarray,
        navigable: np.ndarray,
        positions: np.ndarray | None = None,
        radii: np.ndarray | None = None,
    ) -> CoverageTransitionStats:
        before_covered = np.asarray(before_covered, dtype=bool)
        before_visited = np.asarray(before_visited, dtype=bool)
        masks = np.asarray(masks, dtype=bool)
        navigable = np.asarray(navigable, dtype=bool)
        per_agent_new_masks = masks & ~before_covered[None, :, :]
        denominator = max(float(navigable.sum()), 1.0)
        per_agent_new_coverage = (
            per_agent_new_masks.sum(axis=(1, 2)).astype(np.float32) / denominator
            if len(masks)
            else np.zeros(0, dtype=np.float32)
        )
        footprint_sizes = (
            np.maximum(masks.sum(axis=(1, 2)).astype(np.float32), 1.0)
            if len(masks)
            else np.zeros(0, dtype=np.float32)
        )
        repeat_ratios = (
            (masks & before_visited[None, :, :]).sum(axis=(1, 2)) / footprint_sizes
            if len(masks)
            else np.zeros(0, dtype=np.float64)
        )
        newly_covered = masks.any(axis=0) & ~before_covered if len(masks) else np.zeros_like(navigable)
        return CoverageTransitionStats(
            positions=np.asarray(
                positions if positions is not None else np.zeros((len(masks), 2)),
                dtype=np.float32,
            ).copy(),
            radii=np.asarray(
                radii if radii is not None else np.zeros(len(masks)),
                dtype=np.float32,
            ).copy(),
            sensor_masks=masks,
            navigable_mask=navigable,
            before_covered=before_covered,
            before_visited=before_visited,
            newly_covered_mask=newly_covered,
            per_agent_new_masks=per_agent_new_masks,
            per_agent_new_coverage=np.asarray(per_agent_new_coverage, dtype=np.float32),
            repeat_footprint_ratios=np.asarray(repeat_ratios),
        )

    def coverage_transition_stats(self, prev_world: "MissionWorld") -> CoverageTransitionStats:
        before_covered = np.asarray(prev_world.coverage_grid > 0.0, dtype=bool)
        before_visited = np.asarray(prev_world.visit_count_grid > 0.0, dtype=bool)
        positions = self.get_uav_positions()
        radii = np.asarray([uav.sensor_radius for uav in self.uavs], dtype=np.float32)
        cached = self._last_coverage_transition_stats
        if (
            cached is not None
            and np.array_equal(cached.before_covered, before_covered)
            and np.array_equal(cached.before_visited, before_visited)
            and np.array_equal(cached.positions, positions)
            and np.array_equal(cached.radii, radii)
        ):
            return cached
        navigable = self.navigable_grid_mask()
        if self.uavs:
            masks = self.sensor_masks_for_positions(positions, radii) & navigable[None, :, :]
        else:
            masks = np.zeros((0, self.grid_size, self.grid_size), dtype=bool)
        return self._build_coverage_transition_stats(
            before_covered,
            before_visited,
            masks,
            navigable,
            positions,
            radii,
        )

    def accumulate_sensor_footprints(self) -> dict[str, float]:
        before_covered = self.coverage_grid > 0.0
        before_visited = self.visit_count_grid > 0.0
        navigable = self.navigable_grid_mask()
        if self.uavs:
            positions = self.get_uav_positions()
            radii = np.asarray([uav.sensor_radius for uav in self.uavs], dtype=np.float32)
            masks = self.sensor_masks_for_positions(positions, radii) & navigable[None, :, :]
            footprint = masks.any(axis=0)
            self.coverage_grid[footprint] = 1.0
            self.visit_count_grid += masks.sum(axis=0).astype(np.float32)
        else:
            masks = np.zeros((0, self.grid_size, self.grid_size), dtype=bool)
        self._last_coverage_transition_stats = self._build_coverage_transition_stats(
            before_covered,
            before_visited,
            masks,
            navigable,
            positions if self.uavs else np.zeros((0, 2), dtype=np.float32),
            radii if self.uavs else np.zeros(0, dtype=np.float32),
        )
        new_cells = self._last_coverage_transition_stats.newly_covered_mask
        repeated_cells = self.visit_count_grid > 1.0
        return {
            "new_coverage_cells": float(new_cells.sum()),
            "coverage_ratio": self.coverage_ratio(),
            "overlap_ratio": float(repeated_cells.sum() / max(float((self.visit_count_grid > 0.0).sum()), 1.0)),
        }

    def compute_pairwise_distances(self, positions: np.ndarray | None = None) -> np.ndarray:
        positions = self.get_uav_positions() if positions is None else np.asarray(positions, dtype=np.float32)
        if len(positions) == 0:
            return np.zeros((0, 0), dtype=np.float32)
        if (
            self._pairwise_positions_cache is not None
            and self._pairwise_distances_cache is not None
            and self._pairwise_positions_cache.shape == positions.shape
            and np.array_equal(self._pairwise_positions_cache, positions)
        ):
            return self._pairwise_distances_cache
        if self._use_torch_geometry():
            import torch

            pos_t = torch.as_tensor(positions, dtype=torch.float32, device=self._torch_device())
            diff_t = pos_t[:, None, :] - pos_t[None, :, :]
            distances = torch.linalg.norm(diff_t, dim=-1).detach().cpu().numpy().astype(np.float32)
        else:
            diff = positions[:, None, :] - positions[None, :, :]
            distances = np.linalg.norm(diff, axis=-1).astype(np.float32)
        self._pairwise_positions_cache = positions.copy()
        self._pairwise_distances_cache = distances
        return distances

    def get_uav_positions(self) -> np.ndarray:
        return np.asarray([uav.position for uav in self.uavs], dtype=np.float32)

    def update_communication_graph(self) -> np.ndarray:
        positions = self.get_uav_positions()
        graph = self.communication_graph_for_positions(positions)
        visited = self.base_connected_nodes_for_graph(graph)
        connected = visited[1:].astype(bool)
        for i, uav in enumerate(self.uavs):
            uav.connected_to_base = bool(connected[i])
        self.communication_graph = graph
        self._communication_positions_cache = positions.copy()
        self._connected_mask_cache = connected.copy()
        return graph

    def communication_graph_for_positions(self, positions: np.ndarray) -> np.ndarray:
        """Build the base+UAV communication graph for arbitrary 2D positions."""
        positions = np.asarray(positions, dtype=np.float32)
        n = int(len(positions))
        graph = np.zeros((n + 1, n + 1), dtype=bool)
        graph[0, 0] = True
        if n:
            base_links = np.linalg.norm(positions - self.base_position[None, :], axis=-1) <= self.base_comm_radius
            graph[0, 1:] = base_links
            graph[1:, 0] = base_links
            dist = self.compute_pairwise_distances(positions)
            if n == len(self.uavs):
                radii = np.asarray([uav.comm_radius for uav in self.uavs], dtype=np.float32)
            else:
                radii = np.full(n, float(getattr(self, "comm_radius", self.base_comm_radius)), dtype=np.float32)
            agent_links = dist <= np.minimum(radii[:, None], radii[None, :])
            np.fill_diagonal(agent_links, False)
            graph[1:, 1:] = agent_links
        return graph

    def base_connected_nodes_for_graph(self, graph: np.ndarray) -> np.ndarray:
        graph = np.asarray(graph, dtype=bool)
        visited = np.zeros(graph.shape[0], dtype=bool)
        if graph.shape[0] == 0:
            return visited
        visited[0] = True
        stack = [0]
        while stack:
            node = stack.pop()
            for nxt in np.where(graph[node])[0]:
                if not visited[nxt]:
                    visited[nxt] = True
                    stack.append(int(nxt))
        return visited

    def connected_to_base_mask_for_positions(self, positions: np.ndarray) -> np.ndarray:
        graph = self.communication_graph_for_positions(positions)
        return self.base_connected_nodes_for_graph(graph)[1:].astype(bool)

    def connectivity_margin_stats(self, positions: np.ndarray | None = None, threshold: float = 0.05) -> dict[str, np.ndarray | float]:
        """Distance-to-disconnection diagnostics for soft connectivity rewards.

        For each UAV, the margin is the best communication slack to the current
        base-connected component, excluding the UAV itself. Base links use
        base_comm_radius; UAV-UAV links use the pair's smaller comm radius.
        Disconnected UAVs therefore receive negative margins.
        """
        positions = self.get_uav_positions() if positions is None else np.asarray(positions, dtype=np.float32)
        n = int(len(positions))
        if n == 0:
            empty = np.zeros(0, dtype=np.float32)
            return {
                "connectivity_margins": empty,
                "connectivity_margin_penalties": empty,
                "connectivity_margin_mean": 0.0,
                "connectivity_margin_min": 0.0,
                "connectivity_margin_penalty": 0.0,
            }
        graph = self.communication_graph_for_positions(positions)
        connected_nodes = self.base_connected_nodes_for_graph(graph)
        base_margins = self.base_comm_radius - np.linalg.norm(
            positions - self.base_position[None, :],
            axis=-1,
        )
        radii = np.asarray(
            [
                self.uavs[idx].comm_radius
                if idx < len(self.uavs)
                else float(getattr(self, "comm_radius", self.base_comm_radius))
                for idx in range(n)
            ],
            dtype=np.float32,
        )
        pairwise = self.compute_pairwise_distances(positions)
        peer_margins = np.minimum(radii[:, None], radii[None, :]) - pairwise
        valid_peers = np.broadcast_to(connected_nodes[None, 1:], (n, n)).copy()
        np.fill_diagonal(valid_peers, False)
        peer_margins = np.where(valid_peers, peer_margins, -np.inf)
        best_peer_margins = peer_margins.max(axis=1)
        margins = np.maximum(base_margins, best_peer_margins).astype(np.float32)
        threshold = max(float(threshold), 1e-6)
        penalties = np.maximum(0.0, threshold - margins) / threshold
        return {
            "connectivity_margins": margins.astype(np.float32),
            "connectivity_margin_penalties": penalties.astype(np.float32),
            "connectivity_margin_mean": float(margins.mean()),
            "connectivity_margin_min": float(margins.min()),
            "connectivity_margin_penalty": float(penalties.mean()),
        }

    def connected_to_base_mask(self) -> np.ndarray:
        positions = self.get_uav_positions()
        if (
            self._communication_positions_cache is None
            or self._connected_mask_cache is None
            or self._communication_positions_cache.shape != positions.shape
            or not np.array_equal(self._communication_positions_cache, positions)
        ):
            self.update_communication_graph()
        return np.asarray(self._connected_mask_cache, dtype=bool).copy()

    def get_global_summary(self) -> np.ndarray:
        positions = self.get_uav_positions()
        dist = self.compute_pairwise_distances(positions)
        if len(positions) > 1:
            min_dist = float(dist[np.triu_indices(len(positions), k=1)].min())
        else:
            min_dist = float(self.map_size)
        connected = self.connected_to_base_mask()
        total_path = float(sum(uav.path_length for uav in self.uavs))
        return np.asarray(
            [
                self.step_count / max(self.max_steps, 1),
                self.coverage_ratio(),
                float((self.visit_count_grid > 1.0).mean()),
                float(np.mean(connected)) if len(connected) else 1.0,
                min_dist / max(self.map_size, 1e-6),
                total_path / max(len(self.uavs) * self.map_size, 1e-6),
                float(self.belief_grid.sum()),
                float(len(self.no_fly_zones)),
            ],
            dtype=np.float32,
        )
