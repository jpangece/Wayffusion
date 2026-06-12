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


class WaypointBehaviorRealisticBackend(DynamicsBackend):
    """Mission-level waypoint behavior backend with lightweight execution noise.

    This backend is intentionally independent from the fast backend. It keeps
    the same external interface but adds command latency, velocity smoothing,
    bounded speed variability, and tracking diagnostics for validation runs.
    """

    name = "waypoint_behavior_realistic"

    def __init__(self, config: dict[str, Any] | None = None):
        cfg = dict(config or {})
        self.dt = float(cfg.get("dt", cfg.get("decision_dt", 1.0)))
        self.max_speed = float(cfg.get("max_speed", 0.04))
        self.acceptance_radius = float(cfg.get("acceptance_radius", 0.025))
        self.agent_size = float(cfg.get("agent_size", cfg.get("radius", 0.02)))
        self.velocity_smoothing = float(cfg.get("velocity_smoothing", 0.35))
        speed_range = cfg.get("speed_scale_range", [0.85, 1.15])
        self.speed_scale_range = (
            float(speed_range[0]),
            float(speed_range[1]),
        )
        self.tracking_noise_std = float(cfg.get("tracking_noise_std", 0.005))
        self.command_latency_steps = int(cfg.get("command_latency_steps", 1))
        self.gps_noise_std = float(cfg.get("gps_noise_std", 0.0))
        self.enable_boundary_projection = bool(cfg.get("enable_boundary_projection", True))
        self.enable_no_fly_projection = bool(cfg.get("enable_no_fly_projection", True))
        self.previous_velocity: np.ndarray | None = None
        self.command_buffer: list[np.ndarray] = []
        self.last_positions: np.ndarray | None = None
        self.last_velocities: np.ndarray | None = None

    def reset(self, world: MissionWorld) -> None:
        n = len(world.uavs)
        self.previous_velocity = np.asarray([uav.velocity for uav in world.uavs], dtype=np.float32).copy()
        if self.previous_velocity.shape != (n, 2):
            self.previous_velocity = np.zeros((n, 2), dtype=np.float32)
        self.command_buffer = []
        self.last_positions = world.get_uav_positions().copy()
        self.last_velocities = self.previous_velocity.copy()
        for uav in world.uavs:
            uav.radius = float(getattr(uav, "radius", self.agent_size) or self.agent_size)

    def _target_waypoints(self, world: MissionWorld, controls: np.ndarray, adapter_output: AdapterOutput | None) -> np.ndarray:
        if adapter_output is not None:
            target = np.asarray(adapter_output.clipped_waypoints, dtype=np.float32).copy()
        else:
            target = (world.get_uav_positions() + np.asarray(controls, dtype=np.float32)).astype(np.float32)
        if self.command_latency_steps <= 0:
            return target
        self.command_buffer.append(target.copy())
        if len(self.command_buffer) <= self.command_latency_steps:
            return self.command_buffer[0].copy()
        return self.command_buffer.pop(0)

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

    def _collision_stats(self, world: MissionWorld, positions: np.ndarray) -> tuple[np.ndarray, int, float]:
        rejected = np.zeros(len(positions), dtype=bool)
        if len(positions) <= 1:
            return rejected, 0, float(world.map_size)
        pairwise = world.compute_pairwise_distances(positions)
        upper = pairwise[np.triu_indices(len(positions), k=1)]
        min_pairwise = float(upper.min()) if upper.size else float(world.map_size)
        threshold = max(float(world.min_uav_distance), 2.0 * float(self.agent_size))
        collision_count = 0
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                if float(pairwise[i, j]) < threshold:
                    collision_count += 1
                    rejected[i] = True
                    rejected[j] = True
        return rejected, int(collision_count), min_pairwise

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
        if self.previous_velocity is None or self.previous_velocity.shape != previous.shape:
            self.reset(world)
        target_waypoints = self._target_waypoints(world, controls, adapter_output)
        target_waypoints = world.project_to_geofence(target_waypoints)

        rejected = np.zeros(len(world.uavs), dtype=bool)
        if safety_result is not None and hasattr(safety_result, "rejected_mask"):
            rejected |= np.asarray(safety_result.rejected_mask, dtype=bool)
        error = target_waypoints - previous
        distance = np.linalg.norm(error, axis=-1)
        arrived = distance <= float(self.acceptance_radius)
        direction = np.divide(
            error,
            distance[:, None] + 1e-8,
            out=np.zeros_like(previous),
            where=distance[:, None] > 1e-8,
        )
        low, high = self.speed_scale_range
        if high < low:
            low, high = high, low
        speed_scale = world.rng.uniform(low, high, size=(len(world.uavs), 1)).astype(np.float32)
        desired_velocity = direction * (float(self.max_speed) * speed_scale)
        desired_velocity[arrived] = 0.0
        disturbance = np.zeros_like(desired_velocity, dtype=np.float32)
        if self.tracking_noise_std > 0.0:
            disturbance = world.rng.normal(0.0, self.tracking_noise_std, size=desired_velocity.shape).astype(np.float32)
            desired_velocity = desired_velocity + disturbance
        desired_velocity = _limit_norm(desired_velocity, self.max_speed * max(high, 1.0))
        smoothing = float(np.clip(self.velocity_smoothing, 0.0, 1.0))
        velocities = (1.0 - smoothing) * self.previous_velocity + smoothing * desired_velocity
        velocities = _limit_norm(velocities, self.max_speed * max(high, 1.0))
        displacement = velocities * max(float(self.dt), 1e-8)
        max_step = np.minimum(distance, np.linalg.norm(displacement, axis=-1))
        disp_norm = np.linalg.norm(displacement, axis=-1, keepdims=True)
        displacement = displacement * (max_step[:, None] / np.maximum(disp_norm, 1e-8))
        displacement[arrived] = 0.0
        proposed = previous + displacement
        velocities = displacement / max(float(self.dt), 1e-8)
        positions, velocities, safety_rejected, geofence_count, no_fly_count = self._apply_safety_projection(
            world,
            proposed,
            previous,
            velocities,
        )
        rejected |= safety_rejected
        collision_rejected, collision_count, min_pairwise = self._collision_stats(world, positions)
        rejected |= collision_rejected
        path_delta = np.linalg.norm(positions - previous, axis=-1).astype(np.float32)
        gps_noise_norm = 0.0
        if self.gps_noise_std > 0.0:
            gps_noise = world.rng.normal(0.0, self.gps_noise_std, size=positions.shape).astype(np.float32)
            gps_noise_norm = float(np.linalg.norm(gps_noise, axis=-1).mean()) if len(gps_noise) else 0.0

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
        disturbance_norm = float(np.linalg.norm(disturbance, axis=-1).mean()) if len(disturbance) else 0.0
        finite = bool(np.isfinite(positions).all() and np.isfinite(velocities).all() and np.isfinite(path_delta).all())
        self.previous_velocity = velocities.copy()
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
            disturbance_norm=float(disturbance_norm),
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
                "velocity_smoothing": float(self.velocity_smoothing),
                "speed_scale_min": float(low),
                "speed_scale_max": float(high),
                "tracking_noise_std": float(self.tracking_noise_std),
                "command_latency_steps": int(self.command_latency_steps),
                "gps_noise_norm": float(gps_noise_norm),
                **footprint_info,
            },
        )
