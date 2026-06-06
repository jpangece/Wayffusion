from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from envs.waypoint.control.action_adapters import AdapterOutput
from envs.waypoint.world import MissionWorld


def _limit_norm(vectors: np.ndarray, max_norm: float) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    scale = np.minimum(1.0, float(max_norm) / np.maximum(norms, 1e-8))
    return (vectors * scale).astype(np.float32)


@dataclass
class DynamicsInfo:
    path_length_delta: np.ndarray
    arrived_mask: np.ndarray
    rejected_mask: np.ndarray
    safety_violation_count: int
    min_pairwise_distance: float
    collision_count: int
    no_fly_violation_count: int
    geofence_violation_count: int
    mean_speed: float
    max_speed_observed: float
    mean_control_norm: float
    disturbance_norm: float
    substeps: int
    info: dict[str, Any] = field(default_factory=dict)

    def to_transition_info(self) -> dict[str, Any]:
        return {
            "path_length_delta": self.path_length_delta.astype(np.float32),
            "arrived_mask": self.arrived_mask.astype(bool),
            "rejected_mask": self.rejected_mask.astype(bool),
            "safety_violation_count": int(self.safety_violation_count),
            "min_pairwise_distance": float(self.min_pairwise_distance),
            "collision_count": int(self.collision_count),
            "no_fly_violation_count": int(self.no_fly_violation_count),
            "geofence_violation_count": int(self.geofence_violation_count),
            "mean_speed": float(self.mean_speed),
            "max_speed_observed": float(self.max_speed_observed),
            "mean_control_norm": float(self.mean_control_norm),
            "disturbance_norm": float(self.disturbance_norm),
            "substeps": int(self.substeps),
            **dict(self.info),
        }


class DynamicsBackend:
    name = "base"

    def reset(self, world: MissionWorld) -> None:
        del world

    def step(self, world: MissionWorld, controls: np.ndarray, adapter_output: AdapterOutput | None = None, safety_result=None) -> DynamicsInfo:
        raise NotImplementedError


class KinematicPointBackend(DynamicsBackend):
    """Debug backend preserving the old bounded geometric waypoint step."""

    name = "kinematic_point"

    def __init__(self, config: dict[str, Any] | None = None):
        cfg = dict(config or {})
        self.decision_dt = float(cfg.get("decision_dt", cfg.get("dt_decision", 1.0)))
        self.max_speed = float(cfg.get("max_speed", 0.08))

    def step(self, world: MissionWorld, controls: np.ndarray, adapter_output: AdapterOutput | None = None, safety_result=None) -> DynamicsInfo:
        del controls
        old_positions = world.get_uav_positions()
        selected = old_positions.copy() if adapter_output is None else np.asarray(adapter_output.clipped_waypoints, dtype=np.float32).copy()
        n = len(world.uavs)
        rejected = np.zeros(n, dtype=bool)
        if safety_result is not None and hasattr(safety_result, "rejected_mask"):
            rejected |= np.asarray(safety_result.rejected_mask, dtype=bool)
        no_fly_rejected = ~world.check_no_fly(selected)
        rejected |= no_fly_rejected
        selected[no_fly_rejected] = old_positions[no_fly_rejected]

        max_step = float(self.max_speed * self.decision_dt)
        delta = selected - old_positions
        distance = np.linalg.norm(delta, axis=-1)
        clipped_distance = np.minimum(distance, max_step)
        direction = np.divide(delta, distance[:, None], out=np.zeros_like(delta), where=distance[:, None] > 1e-8)
        proposed = world.project_to_geofence(old_positions + direction * clipped_distance[:, None])

        geofence_rejected = ~world.check_geofence(proposed)
        no_fly_step_rejected = ~world.check_no_fly(proposed)
        rejected |= geofence_rejected | no_fly_step_rejected
        proposed[geofence_rejected | no_fly_step_rejected] = old_positions[geofence_rejected | no_fly_step_rejected]

        collision_count = 0
        pairwise = world.compute_pairwise_distances(proposed)
        min_dist = float(world.map_size)
        if n > 1:
            upper = pairwise[np.triu_indices(n, k=1)]
            min_dist = float(upper.min()) if upper.size else float(world.map_size)
            conflict_pairs = np.argwhere((pairwise < world.min_uav_distance) & (pairwise > 0.0))
            collision_count = int(len(conflict_pairs) // 2)
            for i, j in conflict_pairs:
                rejected[int(i)] = True
                rejected[int(j)] = True
            proposed[rejected] = old_positions[rejected]
            pairwise = world.compute_pairwise_distances(proposed)
            upper = pairwise[np.triu_indices(n, k=1)]
            if upper.size:
                min_dist = float(upper.min())

        path_delta = np.linalg.norm(proposed - old_positions, axis=-1).astype(np.float32)
        speeds = path_delta / max(self.decision_dt, 1e-8)
        arrived = np.linalg.norm(proposed - selected, axis=-1) <= max(1e-4, max_step * 0.05)
        for idx, uav in enumerate(world.uavs):
            uav.velocity = ((proposed[idx] - old_positions[idx]) / max(self.decision_dt, 1e-6)).astype(np.float32)
            uav.position = proposed[idx].astype(np.float32)
            uav.current_waypoint = selected[idx].astype(np.float32)
            uav.path_length += float(path_delta[idx])
            uav.last_control = np.zeros(2, dtype=np.float32) if adapter_output is None else adapter_output.controls[idx].astype(np.float32)
            uav.last_action_valid = not bool(rejected[idx])
            uav.battery = max(0.0, float(uav.battery) - float(path_delta[idx]) * 0.01)
            uav.trajectory.append(uav.position.copy())

        world.step_count += 1
        world.update_communication_graph()
        footprint_info = world.accumulate_sensor_footprints()
        control_norm = 0.0 if adapter_output is None else float(np.mean(adapter_output.control_norms))
        return DynamicsInfo(
            path_length_delta=path_delta,
            arrived_mask=arrived.astype(bool),
            rejected_mask=rejected.astype(bool),
            safety_violation_count=int(rejected.sum()),
            min_pairwise_distance=float(min_dist),
            collision_count=int(collision_count),
            no_fly_violation_count=int((no_fly_rejected | no_fly_step_rejected).sum()),
            geofence_violation_count=int(geofence_rejected.sum()),
            mean_speed=float(speeds.mean()) if len(speeds) else 0.0,
            max_speed_observed=float(speeds.max()) if len(speeds) else 0.0,
            mean_control_norm=control_norm,
            disturbance_norm=0.0,
            substeps=1,
            info={"dynamics_backend": self.name, "dynamics_finite": True, **footprint_info},
        )


class MPEParticleBackend(DynamicsBackend):
    """MPE-like particle dynamics backend for Wayffusion waypoint tasks."""

    name = "mpe_particle"

    def __init__(self, config: dict[str, Any] | None = None):
        cfg = dict(config or {})
        self.decision_dt = float(cfg.get("decision_dt", 1.0))
        self.physics_dt = float(cfg.get("physics_dt", 0.05))
        self.substeps = int(cfg.get("substeps", max(1, round(self.decision_dt / max(self.physics_dt, 1e-8)))))
        self.mass = float(cfg.get("mass", 1.0))
        self.damping = float(cfg.get("damping", 0.20))
        self.max_speed = float(cfg.get("max_speed", 0.08))
        self.radius = float(cfg.get("radius", 0.02))
        self.contact_force = float(cfg.get("contact_force", 100.0))
        self.contact_margin = float(cfg.get("contact_margin", 0.01))
        self.enable_collision_force = bool(cfg.get("enable_collision_force", True))
        self.enable_boundary_projection = bool(cfg.get("enable_boundary_projection", True))
        self.enable_no_fly_projection = bool(cfg.get("enable_no_fly_projection", True))
        self.wind_std = float(cfg.get("wind_std", 0.0))
        self.action_noise_std = float(cfg.get("action_noise_std", 0.0))
        self.position_noise_std = float(cfg.get("position_noise_std", 0.0))

    def reset(self, world: MissionWorld) -> None:
        for uav in world.uavs:
            uav.mass = float(getattr(uav, "mass", self.mass) or self.mass)
            uav.radius = float(getattr(uav, "radius", self.radius) or self.radius)
            uav.last_control = np.zeros(2, dtype=np.float32)

    def _collision_forces(self, world: MissionWorld, positions: np.ndarray) -> tuple[np.ndarray, int]:
        forces = np.zeros_like(positions, dtype=np.float32)
        collision_count = 0
        n = len(positions)
        threshold = max(float(world.min_uav_distance), 2.0 * self.radius) + self.contact_margin
        for i in range(n):
            for j in range(i + 1, n):
                delta = positions[i] - positions[j]
                distance = float(np.linalg.norm(delta))
                if distance >= threshold:
                    continue
                collision_count += 1
                if distance < 1e-8:
                    direction = np.asarray([1.0, 0.0], dtype=np.float32)
                else:
                    direction = delta / distance
                penetration = threshold - distance
                force = self.contact_force * penetration * direction
                forces[i] += force
                forces[j] -= force
        return forces.astype(np.float32), collision_count

    def _project_no_fly(self, world: MissionWorld, proposed: np.ndarray, previous: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        valid = world.check_no_fly(proposed)
        if valid.all() or not self.enable_no_fly_projection:
            return proposed, ~valid
        proposed = proposed.copy()
        proposed[~valid] = previous[~valid]
        return proposed, ~valid

    def _separate_min_distance(self, world: MissionWorld, positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        n = len(positions)
        adjusted = positions.copy()
        violated = np.zeros(n, dtype=bool)
        min_dist = float(world.min_uav_distance)
        for i in range(n):
            for j in range(i + 1, n):
                delta = adjusted[i] - adjusted[j]
                distance = float(np.linalg.norm(delta))
                if distance >= min_dist or distance < 1e-8:
                    continue
                direction = delta / max(distance, 1e-8)
                correction = 0.5 * (min_dist - distance) * direction
                adjusted[i] += correction
                adjusted[j] -= correction
                violated[i] = True
                violated[j] = True
        if self.enable_boundary_projection:
            adjusted = world.project_to_geofence(adjusted)
        return adjusted.astype(np.float32), violated

    def step(self, world: MissionWorld, controls: np.ndarray, adapter_output: AdapterOutput | None = None, safety_result=None) -> DynamicsInfo:
        controls = np.asarray(controls, dtype=np.float32)
        positions = world.get_uav_positions().astype(np.float32)
        velocities = np.asarray([uav.velocity for uav in world.uavs], dtype=np.float32)
        old_positions = positions.copy()
        n = len(world.uavs)
        if controls.shape != positions.shape:
            raise ValueError(f"controls must have shape {positions.shape}, got {controls.shape}")

        rejected = np.zeros(n, dtype=bool)
        if safety_result is not None and hasattr(safety_result, "rejected_mask"):
            rejected |= np.asarray(safety_result.rejected_mask, dtype=bool)
        no_fly_count = 0
        geofence_count = 0
        collision_count = 0
        disturbance_norms = []
        path_delta = np.zeros(n, dtype=np.float32)
        max_speed_observed = 0.0
        target_waypoints = positions if adapter_output is None else np.asarray(adapter_output.clipped_waypoints, dtype=np.float32)
        control_norm = float(np.mean(np.linalg.norm(controls, axis=-1))) if len(controls) else 0.0

        for _ in range(self.substeps):
            step_controls = controls.copy()
            if self.action_noise_std > 0.0:
                step_controls += world.rng.normal(0.0, self.action_noise_std, size=step_controls.shape).astype(np.float32)
            env_force = np.zeros_like(step_controls, dtype=np.float32)
            if self.enable_collision_force:
                contact_force, contacts = self._collision_forces(world, positions)
                env_force += contact_force
                collision_count += int(contacts)
            if self.wind_std > 0.0:
                wind = world.rng.normal(0.0, self.wind_std, size=step_controls.shape).astype(np.float32)
                env_force += wind
                disturbance_norms.append(float(np.linalg.norm(wind, axis=-1).mean()))

            mass = np.asarray([max(float(getattr(uav, "mass", self.mass)), 1e-6) for uav in world.uavs], dtype=np.float32)[:, None]
            velocities = (1.0 - self.damping) * velocities + (step_controls + env_force) / mass * self.physics_dt
            velocities = _limit_norm(velocities, self.max_speed)
            max_speed_observed = max(max_speed_observed, float(np.linalg.norm(velocities, axis=-1).max()) if len(velocities) else 0.0)
            proposed = positions + velocities * self.physics_dt
            if self.position_noise_std > 0.0:
                noise = world.rng.normal(0.0, self.position_noise_std, size=proposed.shape).astype(np.float32)
                proposed += noise
                disturbance_norms.append(float(np.linalg.norm(noise, axis=-1).mean()))
            previous = positions.copy()
            if self.enable_boundary_projection:
                projected = world.project_to_geofence(proposed)
                clipped = np.linalg.norm(projected - proposed, axis=-1) > 1e-8
                geofence_count += int(clipped.sum())
                velocities[clipped] = 0.0
                proposed = projected
            proposed, no_fly_mask = self._project_no_fly(world, proposed, previous)
            if no_fly_mask.any():
                velocities[no_fly_mask] = 0.0
                no_fly_count += int(no_fly_mask.sum())
                rejected |= no_fly_mask
            proposed, min_distance_mask = self._separate_min_distance(world, proposed)
            if min_distance_mask.any():
                velocities[min_distance_mask] = 0.0
                rejected |= min_distance_mask
            path_delta += np.linalg.norm(proposed - positions, axis=-1).astype(np.float32)
            positions = proposed.astype(np.float32)

        pairwise = world.compute_pairwise_distances(positions)
        if n > 1:
            upper = pairwise[np.triu_indices(n, k=1)]
            min_pairwise = float(upper.min()) if upper.size else float(world.map_size)
        else:
            min_pairwise = float(world.map_size)
        arrived = np.linalg.norm(positions - target_waypoints, axis=-1) <= 0.025

        for idx, uav in enumerate(world.uavs):
            uav.velocity = velocities[idx].astype(np.float32)
            uav.position = positions[idx].astype(np.float32)
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
        disturbance_norm = float(np.mean(disturbance_norms)) if disturbance_norms else 0.0
        finite = bool(np.isfinite(positions).all() and np.isfinite(velocities).all() and np.isfinite(path_delta).all())
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
            max_speed_observed=float(max_speed_observed),
            mean_control_norm=float(control_norm),
            disturbance_norm=float(disturbance_norm),
            substeps=int(self.substeps),
            info={"dynamics_backend": self.name, "dynamics_finite": finite, **footprint_info},
        )


def build_dynamics_backend(config: dict[str, Any] | None):
    cfg = dict(config or {})
    name = str(cfg.get("name", "mpe_particle"))
    if name == "mpe_particle":
        return MPEParticleBackend(cfg)
    if name == "kinematic_point":
        return KinematicPointBackend(cfg)
    raise ValueError(f"Unsupported dynamics backend: {name}")
