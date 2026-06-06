from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from envs.waypoint.world import MissionWorld


def _limit_norm(vectors: np.ndarray, max_norm: float) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    scale = np.minimum(1.0, float(max_norm) / np.maximum(norms, 1e-8))
    return (vectors * scale).astype(np.float32)


@dataclass
class AdapterOutput:
    controls: np.ndarray
    desired_velocities: np.ndarray
    accepted_mask: np.ndarray
    clipped_waypoints: np.ndarray
    control_norms: np.ndarray
    info: dict[str, Any] = field(default_factory=dict)

    def to_info(self) -> dict[str, Any]:
        return {
            "controls": self.controls.astype(np.float32),
            "desired_velocities": self.desired_velocities.astype(np.float32),
            "accepted_mask": self.accepted_mask.astype(bool),
            "clipped_waypoints": self.clipped_waypoints.astype(np.float32),
            "control_norms": self.control_norms.astype(np.float32),
            **dict(self.info),
        }


class WaypointVelocityTracker:
    """Waypoint-to-acceleration adapter using a velocity tracking loop."""

    def __init__(self, config: dict[str, Any] | None = None):
        cfg = dict(config or {})
        self.name = "waypoint_velocity_tracker"
        self.max_command_distance = float(cfg.get("max_command_distance", 0.22))
        self.max_speed = float(cfg.get("max_speed", 0.08))
        self.slowdown_radius = float(cfg.get("slowdown_radius", 0.12))
        self.velocity_gain = float(cfg.get("velocity_gain", 4.0))
        self.max_accel = float(cfg.get("max_accel", 0.25))
        self.control_smoothing = float(cfg.get("control_smoothing", 0.35))
        self.acceptance_radius = float(cfg.get("acceptance_radius", 0.025))
        self.hover_when_arrived = bool(cfg.get("hover_when_arrived", True))
        self.command_latency_steps = int(cfg.get("command_latency_steps", 0))
        self.tracking_noise_std = float(cfg.get("tracking_noise_std", 0.0))
        self.clip_to_geofence = bool(cfg.get("clip_to_geofence", True))
        self.reject_no_fly_waypoints = bool(cfg.get("reject_no_fly_waypoints", True))
        self.previous_control: np.ndarray | None = None
        self.latency_buffer: list[np.ndarray] = []

    def reset(self, world: MissionWorld) -> None:
        n = len(world.uavs)
        self.previous_control = np.zeros((n, 2), dtype=np.float32)
        self.latency_buffer = []

    def _sanitize_waypoints(self, world: MissionWorld, waypoint_actions: np.ndarray) -> np.ndarray:
        waypoints = np.asarray(waypoint_actions, dtype=np.float32).copy()
        if self.clip_to_geofence:
            waypoints = world.project_to_geofence(waypoints)
        if self.reject_no_fly_waypoints:
            no_fly_ok = world.check_no_fly(waypoints)
            current = world.get_uav_positions()
            waypoints[~no_fly_ok] = current[~no_fly_ok]
        current = world.get_uav_positions()
        delta = waypoints - current
        distance = np.linalg.norm(delta, axis=-1, keepdims=True)
        scale = np.minimum(1.0, self.max_command_distance / np.maximum(distance, 1e-8))
        return (current + delta * scale).astype(np.float32)

    def _apply_latency(self, clipped_waypoints: np.ndarray) -> np.ndarray:
        if self.command_latency_steps <= 0:
            return clipped_waypoints
        self.latency_buffer.append(clipped_waypoints.copy())
        if len(self.latency_buffer) <= self.command_latency_steps:
            return self.latency_buffer[0].copy()
        return self.latency_buffer.pop(0)

    def adapt(self, world: MissionWorld, waypoint_actions: np.ndarray, safety_info=None) -> AdapterOutput:
        clipped = self._sanitize_waypoints(world, waypoint_actions)
        clipped = self._apply_latency(clipped)
        positions = world.get_uav_positions()
        velocities = np.asarray([uav.velocity for uav in world.uavs], dtype=np.float32)
        error = clipped - positions
        distance = np.linalg.norm(error, axis=-1)
        direction = np.divide(error, distance[:, None] + 1e-8, out=np.zeros_like(error), where=distance[:, None] > 1e-8)
        speed_scale = np.minimum(1.0, distance / max(self.slowdown_radius, 1e-8))
        desired = self.max_speed * speed_scale[:, None] * direction
        if self.hover_when_arrived:
            desired[distance < self.acceptance_radius] = 0.0

        raw_control = self.velocity_gain * (desired - velocities)
        raw_control = _limit_norm(raw_control, self.max_accel)
        if self.tracking_noise_std > 0.0:
            raw_control += world.rng.normal(0.0, self.tracking_noise_std, size=raw_control.shape).astype(np.float32)
            raw_control = _limit_norm(raw_control, self.max_accel)
        if self.previous_control is None or self.previous_control.shape != raw_control.shape:
            self.reset(world)
        smoothing = float(np.clip(self.control_smoothing, 0.0, 1.0))
        controls = (1.0 - smoothing) * self.previous_control + smoothing * raw_control
        controls = _limit_norm(controls, self.max_accel)
        self.previous_control = controls.astype(np.float32)

        accepted = np.ones(len(world.uavs), dtype=bool)
        if safety_info is not None and hasattr(safety_info, "valid_mask"):
            accepted = np.asarray(safety_info.valid_mask, dtype=bool).copy()
        for idx, uav in enumerate(world.uavs):
            uav.current_waypoint = clipped[idx].copy()
            uav.last_desired_velocity = desired[idx].copy()
        control_norms = np.linalg.norm(controls, axis=-1).astype(np.float32)
        return AdapterOutput(
            controls=controls.astype(np.float32),
            desired_velocities=desired.astype(np.float32),
            accepted_mask=accepted,
            clipped_waypoints=clipped.astype(np.float32),
            control_norms=control_norms,
            info={
                "adapter_name": self.name,
                "adapter_finite": bool(np.isfinite(controls).all() and np.isfinite(desired).all()),
                "mean_control_norm": float(control_norms.mean()) if len(control_norms) else 0.0,
                "max_control_norm": float(control_norms.max()) if len(control_norms) else 0.0,
                "mean_desired_speed": float(np.linalg.norm(desired, axis=-1).mean()) if len(desired) else 0.0,
                "acceptance_radius": float(self.acceptance_radius),
            },
        )


class WaypointPDTracker(WaypointVelocityTracker):
    """Waypoint-to-acceleration adapter using a direct PD controller."""

    def __init__(self, config: dict[str, Any] | None = None):
        cfg = dict(config or {})
        super().__init__(cfg)
        self.name = "waypoint_pd_tracker"
        self.kp = float(cfg.get("kp", 2.0))
        self.kd = float(cfg.get("kd", 2.8))

    def adapt(self, world: MissionWorld, waypoint_actions: np.ndarray, safety_info=None) -> AdapterOutput:
        clipped = self._sanitize_waypoints(world, waypoint_actions)
        clipped = self._apply_latency(clipped)
        positions = world.get_uav_positions()
        velocities = np.asarray([uav.velocity for uav in world.uavs], dtype=np.float32)
        error = clipped - positions
        distance = np.linalg.norm(error, axis=-1)
        desired = np.zeros_like(velocities, dtype=np.float32)
        raw_control = self.kp * error - self.kd * velocities
        raw_control[distance < self.acceptance_radius] = -self.kd * velocities[distance < self.acceptance_radius]
        raw_control = _limit_norm(raw_control, self.max_accel)
        if self.previous_control is None or self.previous_control.shape != raw_control.shape:
            self.reset(world)
        smoothing = float(np.clip(self.control_smoothing, 0.0, 1.0))
        controls = (1.0 - smoothing) * self.previous_control + smoothing * raw_control
        controls = _limit_norm(controls, self.max_accel)
        self.previous_control = controls.astype(np.float32)
        accepted = np.ones(len(world.uavs), dtype=bool)
        if safety_info is not None and hasattr(safety_info, "valid_mask"):
            accepted = np.asarray(safety_info.valid_mask, dtype=bool).copy()
        for idx, uav in enumerate(world.uavs):
            uav.current_waypoint = clipped[idx].copy()
            uav.last_desired_velocity = desired[idx].copy()
        control_norms = np.linalg.norm(controls, axis=-1).astype(np.float32)
        return AdapterOutput(
            controls=controls.astype(np.float32),
            desired_velocities=desired.astype(np.float32),
            accepted_mask=accepted,
            clipped_waypoints=clipped.astype(np.float32),
            control_norms=control_norms,
            info={
                "adapter_name": self.name,
                "adapter_finite": bool(np.isfinite(controls).all()),
                "mean_control_norm": float(control_norms.mean()) if len(control_norms) else 0.0,
                "max_control_norm": float(control_norms.max()) if len(control_norms) else 0.0,
                "kp": float(self.kp),
                "kd": float(self.kd),
            },
        )


class VelocityToForceAdapter(WaypointVelocityTracker):
    """Smoke adapter for future velocity action interfaces."""

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self.name = "velocity_to_force"
        self.velocity_gain = float((config or {}).get("velocity_gain", self.velocity_gain))

    def adapt(self, world: MissionWorld, velocity_actions: np.ndarray, safety_info=None) -> AdapterOutput:
        desired = np.asarray(velocity_actions, dtype=np.float32)
        desired = _limit_norm(desired, self.max_speed)
        velocities = np.asarray([uav.velocity for uav in world.uavs], dtype=np.float32)
        controls = _limit_norm(self.velocity_gain * (desired - velocities), self.max_accel)
        control_norms = np.linalg.norm(controls, axis=-1).astype(np.float32)
        return AdapterOutput(
            controls=controls.astype(np.float32),
            desired_velocities=desired.astype(np.float32),
            accepted_mask=np.ones(len(world.uavs), dtype=bool),
            clipped_waypoints=world.get_uav_positions().astype(np.float32),
            control_norms=control_norms,
            info={"adapter_name": self.name, "adapter_finite": bool(np.isfinite(controls).all())},
        )


class DirectAccelerationAdapter(WaypointVelocityTracker):
    """Smoke adapter for future low-level acceleration/force action interfaces."""

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self.name = "direct_acceleration"

    def adapt(self, world: MissionWorld, acceleration_actions: np.ndarray, safety_info=None) -> AdapterOutput:
        controls = _limit_norm(np.asarray(acceleration_actions, dtype=np.float32), self.max_accel)
        control_norms = np.linalg.norm(controls, axis=-1).astype(np.float32)
        return AdapterOutput(
            controls=controls.astype(np.float32),
            desired_velocities=np.zeros_like(controls, dtype=np.float32),
            accepted_mask=np.ones(len(world.uavs), dtype=bool),
            clipped_waypoints=world.get_uav_positions().astype(np.float32),
            control_norms=control_norms,
            info={"adapter_name": self.name, "adapter_finite": bool(np.isfinite(controls).all())},
        )


def build_action_adapter(config: dict[str, Any] | None):
    cfg = dict(config or {})
    name = str(cfg.get("name", "waypoint_velocity_tracker"))
    if name == "waypoint_velocity_tracker":
        return WaypointVelocityTracker(cfg)
    if name == "waypoint_pd_tracker":
        return WaypointPDTracker(cfg)
    if name == "velocity_to_force":
        return VelocityToForceAdapter(cfg)
    if name in {"direct_acceleration", "direct_force"}:
        return DirectAccelerationAdapter(cfg)
    raise ValueError(f"Unsupported action adapter: {name}")
