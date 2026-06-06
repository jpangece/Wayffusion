from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from envs.waypoint.entities import UAVState


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

    @classmethod
    def from_config(cls, config: dict[str, Any], rng: np.random.Generator | None = None) -> "MissionWorld":
        dynamics_cfg = dict(config.get("dynamics_backend", {}))
        return cls(
            map_size=float(config.get("map_size", 1.0)),
            world_dim=int(config.get("world_dim", 2)),
            default_altitude=float(config.get("default_altitude", 30.0)),
            dt_decision=float(dynamics_cfg.get("decision_dt", config.get("dt_decision", 1.0))),
            physics_dt=float(dynamics_cfg.get("physics_dt", 0.05)),
            substeps=int(dynamics_cfg.get("substeps", 20)),
            max_speed=float(dynamics_cfg.get("max_speed", config.get("max_speed", 0.08))),
            mass=float(dynamics_cfg.get("mass", 1.0)),
            damping=float(dynamics_cfg.get("damping", 0.2)),
            radius=float(dynamics_cfg.get("radius", 0.02)),
            max_waypoint_distance=float(config.get("max_waypoint_distance", 0.2)),
            min_uav_distance=float(config.get("min_uav_distance", 0.03)),
            base_position=np.asarray(config.get("base_position", [0.1, 0.1]), dtype=np.float32),
            base_comm_radius=float(config.get("base_comm_radius", 0.45)),
            grid_size=int(config.get("grid_size", 32)),
            max_steps=int(config.get("max_steps", 100)),
            no_fly_zones=list(config.get("no_fly_zones", [])),
            geofence=tuple(config.get("geofence", [0.0, config.get("map_size", 1.0), 0.0, config.get("map_size", 1.0)])),
            rng=rng or np.random.default_rng(int(config.get("seed", 0))),
        )

    def reset_common(self, num_agents: int, rng: np.random.Generator | None = None, initial_positions=None) -> None:
        self.rng = rng or self.rng
        self.step_count = 0
        self.coverage_grid.fill(0.0)
        self.visit_count_grid.fill(0.0)
        self.probability_grid.fill(1.0 / float(self.grid_size * self.grid_size))
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

    def _sample_safe_initial_positions(self, num_agents: int) -> np.ndarray:
        positions: list[np.ndarray] = []
        min_dist = max(float(self.min_uav_distance) * 1.8, float(self.radius) * 2.5)
        for idx in range(num_agents):
            accepted = None
            for _ in range(200):
                radius = self.rng.uniform(0.02 * self.map_size, 0.22 * self.map_size)
                angle = self.rng.uniform(0.0, 2.0 * np.pi)
                candidate = self.base_position + radius * np.asarray([np.cos(angle), np.sin(angle)], dtype=np.float32)
                candidate = self.project_to_geofence(candidate[None, :])[0]
                if not self.check_no_fly(candidate[None, :])[0]:
                    continue
                if all(np.linalg.norm(candidate - prev) >= min_dist for prev in positions):
                    accepted = candidate.astype(np.float32)
                    break
            if accepted is None:
                angle = 2.0 * np.pi * idx / max(num_agents, 1)
                radius = min_dist * (1.0 + idx // 8)
                accepted = self.project_to_geofence((self.base_position + radius * np.asarray([np.cos(angle), np.sin(angle)], dtype=np.float32))[None, :])[0]
            positions.append(accepted.astype(np.float32))
        return np.asarray(positions, dtype=np.float32)

    def clone(self) -> "MissionWorld":
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
            rng=self.rng,
        )
        new.uavs = [uav.snapshot() for uav in self.uavs]
        new.coverage_grid = self.coverage_grid.copy()
        new.visit_count_grid = self.visit_count_grid.copy()
        new.probability_grid = self.probability_grid.copy()
        new.belief_grid = self.belief_grid.copy()
        new.communication_graph = self.communication_graph.copy()
        new.step_count = int(self.step_count)
        return new

    snapshot = clone

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

    def sensor_mask_for_position(self, position: np.ndarray, radius: float | None = None) -> np.ndarray:
        radius = float(radius if radius is not None else (self.uavs[0].sensor_radius if self.uavs else 0.12))
        yy, xx = np.mgrid[0 : self.grid_size, 0 : self.grid_size]
        centers = np.stack([(xx + 0.5) / self.grid_size, (yy + 0.5) / self.grid_size], axis=-1) * self.map_size
        dist = np.linalg.norm(centers - np.asarray(position, dtype=np.float32), axis=-1)
        return dist <= radius

    def accumulate_sensor_footprints(self) -> dict[str, float]:
        before = self.coverage_grid.copy()
        for uav in self.uavs:
            mask = self.sensor_mask_for_position(uav.position, uav.sensor_radius)
            self.coverage_grid[mask] = 1.0
            self.visit_count_grid[mask] += 1.0
        new_cells = np.logical_and(self.coverage_grid > 0.0, before <= 0.0)
        repeated_cells = self.visit_count_grid > 1.0
        return {
            "new_coverage_cells": float(new_cells.sum()),
            "coverage_ratio": float(self.coverage_grid.mean()),
            "overlap_ratio": float(repeated_cells.sum() / max(float((self.visit_count_grid > 0.0).sum()), 1.0)),
        }

    def compute_pairwise_distances(self, positions: np.ndarray | None = None) -> np.ndarray:
        positions = self.get_uav_positions() if positions is None else np.asarray(positions, dtype=np.float32)
        if len(positions) == 0:
            return np.zeros((0, 0), dtype=np.float32)
        diff = positions[:, None, :] - positions[None, :, :]
        return np.linalg.norm(diff, axis=-1).astype(np.float32)

    def get_uav_positions(self) -> np.ndarray:
        return np.asarray([uav.position for uav in self.uavs], dtype=np.float32)

    def update_communication_graph(self) -> np.ndarray:
        n = len(self.uavs)
        graph = np.zeros((n + 1, n + 1), dtype=bool)
        graph[0, 0] = True
        positions = self.get_uav_positions()
        for idx, pos in enumerate(positions, start=1):
            base_link = np.linalg.norm(pos - self.base_position) <= self.base_comm_radius
            graph[0, idx] = graph[idx, 0] = bool(base_link)
        if n:
            dist = self.compute_pairwise_distances(positions)
            for i in range(n):
                for j in range(i + 1, n):
                    radius = min(self.uavs[i].comm_radius, self.uavs[j].comm_radius)
                    if dist[i, j] <= radius:
                        graph[i + 1, j + 1] = graph[j + 1, i + 1] = True
        visited = np.zeros(n + 1, dtype=bool)
        stack = [0]
        visited[0] = True
        while stack:
            node = stack.pop()
            for nxt in np.where(graph[node])[0]:
                if not visited[nxt]:
                    visited[nxt] = True
                    stack.append(int(nxt))
        for i, uav in enumerate(self.uavs):
            uav.connected_to_base = bool(visited[i + 1])
        self.communication_graph = graph
        return graph

    def connected_to_base_mask(self) -> np.ndarray:
        self.update_communication_graph()
        return np.asarray([uav.connected_to_base for uav in self.uavs], dtype=bool)

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
                float(self.coverage_grid.mean()),
                float((self.visit_count_grid > 1.0).mean()),
                float(np.mean(connected)) if len(connected) else 1.0,
                min_dist / max(self.map_size, 1e-6),
                total_path / max(len(self.uavs) * self.map_size, 1e-6),
                float(self.belief_grid.sum()),
                float(len(self.no_fly_zones)),
            ],
            dtype=np.float32,
        )
