from __future__ import annotations

from typing import Any

import numpy as np

from envs.waypoint.control.action_adapters import AdapterOutput
from envs.waypoint.control.dynamics_base import DynamicsBackend, DynamicsInfo
from envs.waypoint.world import MissionWorld


def _limit_norm(vectors: np.ndarray, max_norm: float) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    scale = np.minimum(1.0, float(max_norm) / np.maximum(norms, 1e-8))
    return (vectors * scale).astype(np.float32)


def _target_waypoints(world: MissionWorld, controls: np.ndarray, adapter_output: AdapterOutput | None) -> np.ndarray:
    if adapter_output is not None:
        return np.asarray(adapter_output.clipped_waypoints, dtype=np.float32).copy()
    return (world.get_uav_positions() + np.asarray(controls, dtype=np.float32)).astype(np.float32)


class WaypointBehaviorFastBackend(DynamicsBackend):
    """Fast mission-level waypoint behavior backend for large-scale training.

    This backend intentionally does not model low-level particle contacts. It
    moves each UAV toward its selected waypoint with a bounded speed and then
    updates the same Wayffusion world state fields as the MPE backend.
    """

    name = "waypoint_behavior_fast"

    def __init__(self, config: dict[str, Any] | None = None):
        cfg = dict(config or {})
        self.dt = float(cfg.get("dt", cfg.get("decision_dt", 1.0)))
        self.max_speed = float(cfg.get("max_speed", 0.04))
        self.acceptance_radius = float(cfg.get("acceptance_radius", 0.025))
        self.agent_size = float(cfg.get("agent_size", cfg.get("radius", 0.02)))
        self.enable_boundary_projection = bool(cfg.get("enable_boundary_projection", True))
        self.enable_no_fly_projection = bool(cfg.get("enable_no_fly_projection", True))
        self.last_positions: np.ndarray | None = None
        self.last_velocities: np.ndarray | None = None

    def reset(self, world: MissionWorld) -> None:
        self.last_positions = world.get_uav_positions().copy()
        self.last_velocities = np.asarray([uav.velocity for uav in world.uavs], dtype=np.float32).copy()
        for uav in world.uavs:
            uav.radius = float(getattr(uav, "radius", self.agent_size) or self.agent_size)

    def _apply_safety_projection(
        self,
        world: MissionWorld,
        proposed: np.ndarray,
        previous: np.ndarray,
        velocities: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
        positions = np.asarray(proposed, dtype=np.float32).copy()
        velocities = np.asarray(velocities, dtype=np.float32).copy()
        rejected = np.zeros(len(positions), dtype=bool)
        geofence_count = 0
        if self.enable_boundary_projection:
            projected = world.project_to_geofence(positions)
            geofence_mask = np.linalg.norm(projected - positions, axis=-1) > 1e-8
            geofence_count = int(geofence_mask.sum())
            positions = projected.astype(np.float32)
            velocities[geofence_mask] = 0.0
            rejected |= geofence_mask
        no_fly_count = 0
        if self.enable_no_fly_projection:
            no_fly_mask = ~world.check_no_fly(positions)
            no_fly_count = int(no_fly_mask.sum())
            positions[no_fly_mask] = previous[no_fly_mask]
            velocities[no_fly_mask] = 0.0
            rejected |= no_fly_mask
        return positions, velocities, rejected, geofence_count, no_fly_count

    def _collision_stats(self, world: MissionWorld, positions: np.ndarray) -> tuple[int, float]:
        if len(positions) <= 1:
            return 0, float(world.map_size)
        pairwise = world.compute_pairwise_distances(positions)
        upper = pairwise[np.triu_indices(len(positions), k=1)]
        min_pairwise = float(upper.min()) if upper.size else float(world.map_size)
        threshold = max(float(world.min_uav_distance), 2.0 * float(self.agent_size))
        collision_count = int(np.count_nonzero(upper < threshold)) if upper.size else 0
        return collision_count, min_pairwise

    def step(
        self,
        world: MissionWorld,
        controls: np.ndarray,
        adapter_output: AdapterOutput | None = None,
        safety_result=None,
    ) -> DynamicsInfo:
        controls = np.asarray(controls, dtype=np.float32)
        previous = world.get_uav_positions().astype(np.float32)
        if controls.shape != previous.shape:
            raise ValueError(f"controls must have shape {previous.shape}, got {controls.shape}")
        target_waypoints = _target_waypoints(world, controls, adapter_output)
        target_waypoints = world.project_to_geofence(target_waypoints)

        rejected = np.zeros(len(world.uavs), dtype=bool)
        if safety_result is not None and hasattr(safety_result, "rejected_mask"):
            rejected |= np.asarray(safety_result.rejected_mask, dtype=bool)
        distance = np.linalg.norm(target_waypoints - previous, axis=-1)
        arrived = distance <= float(self.acceptance_radius)
        direction = np.divide(
            target_waypoints - previous,
            distance[:, None] + 1e-8,
            out=np.zeros_like(previous),
            where=distance[:, None] > 1e-8,
        )
        max_step = max(float(self.max_speed) * max(float(self.dt), 1e-8), 0.0)
        step_distance = np.minimum(distance, max_step)
        step_distance[arrived] = 0.0
        displacement = direction * step_distance[:, None]
        proposed = previous + displacement
        velocities = displacement / max(float(self.dt), 1e-8)
        velocities = _limit_norm(velocities, self.max_speed)
        positions, velocities, safety_rejected, geofence_count, no_fly_count = self._apply_safety_projection(
            world,
            proposed,
            previous,
            velocities,
        )
        rejected |= safety_rejected
        path_delta = np.linalg.norm(positions - previous, axis=-1).astype(np.float32)
        collision_count, min_pairwise = self._collision_stats(world, positions)
        if collision_count:
            pairwise = world.compute_pairwise_distances(positions)
            threshold = max(float(world.min_uav_distance), 2.0 * float(self.agent_size))
            for i in range(len(positions)):
                for j in range(i + 1, len(positions)):
                    if float(pairwise[i, j]) < threshold:
                        rejected[i] = True
                        rejected[j] = True

        for idx, uav in enumerate(world.uavs):
            uav.position = positions[idx].astype(np.float32)
            uav.velocity = velocities[idx].astype(np.float32)
            uav.current_waypoint = target_waypoints[idx].astype(np.float32)
            uav.path_length += float(path_delta[idx])
            uav.last_control = controls[idx].astype(np.float32)
            uav.last_action_valid = not bool(rejected[idx])
            uav.battery = max(0.0, float(uav.battery) - float(path_delta[idx]) * 0.01)
            uav.trajectory.append(uav.position.copy())

        world.step_count += 1
        world.update_communication_graph()
        footprint_info = world.accumulate_sensor_footprints()
        speeds = np.linalg.norm(velocities, axis=-1)
        control_norm = np.linalg.norm(controls, axis=-1)
        finite = bool(np.isfinite(positions).all() and np.isfinite(velocities).all() and np.isfinite(path_delta).all())
        self.last_positions = positions.copy()
        self.last_velocities = velocities.copy()
        return DynamicsInfo(
            path_length_delta=path_delta.astype(np.float32),
            arrived_mask=arrived.astype(bool),
            rejected_mask=rejected.astype(bool),
            safety_violation_count=int(rejected.sum()),
            min_pairwise_distance=float(min_pairwise),
            collision_count=int(collision_count),
            no_fly_violation_count=int(no_fly_count),
            geofence_violation_count=int(geofence_count),
            mean_speed=float(speeds.mean()) if len(speeds) else 0.0,
            max_speed_observed=float(speeds.max()) if len(speeds) else 0.0,
            mean_control_norm=float(control_norm.mean()) if len(control_norm) else 0.0,
            disturbance_norm=0.0,
            substeps=1,
            info={
                "dynamics_backend": self.name,
                "dynamics_finite": finite,
                "uses_real_mpe_core": False,
                "mpe_source": "none",
                "mpe_world_step_calls": 0,
                "mpe_world_step_calls_total": 0,
                "dt": float(self.dt),
                "acceptance_radius": float(self.acceptance_radius),
                **footprint_info,
            },
        )
