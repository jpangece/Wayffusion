from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class UAVState:
    """Runtime state for one waypoint-level UAV agent."""

    position: np.ndarray
    velocity: np.ndarray
    battery: float = 1.0
    current_waypoint: np.ndarray | None = None
    status: str = "active"
    role_id: int = 0
    mass: float = 1.0
    radius: float = 0.02
    sensor_radius: float = 0.12
    comm_radius: float = 0.35
    last_control: np.ndarray | None = None
    last_desired_velocity: np.ndarray | None = None
    path_length: float = 0.0
    connected_to_base: bool = True
    last_action_valid: bool = True
    trajectory: list[np.ndarray] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.position = np.asarray(self.position, dtype=np.float32)
        self.velocity = np.asarray(self.velocity, dtype=np.float32)
        if self.current_waypoint is None:
            self.current_waypoint = self.position.copy()
        else:
            self.current_waypoint = np.asarray(self.current_waypoint, dtype=np.float32)
        if self.last_control is None:
            self.last_control = np.zeros_like(self.velocity, dtype=np.float32)
        else:
            self.last_control = np.asarray(self.last_control, dtype=np.float32)
        if self.last_desired_velocity is None:
            self.last_desired_velocity = np.zeros_like(self.velocity, dtype=np.float32)
        else:
            self.last_desired_velocity = np.asarray(self.last_desired_velocity, dtype=np.float32)
        if not self.trajectory:
            self.trajectory.append(self.position.copy())

    def snapshot(self, include_trajectory: bool = True) -> "UAVState":
        return UAVState(
            position=self.position.copy(),
            velocity=self.velocity.copy(),
            battery=float(self.battery),
            current_waypoint=self.current_waypoint.copy(),
            status=str(self.status),
            role_id=int(self.role_id),
            mass=float(self.mass),
            radius=float(self.radius),
            sensor_radius=float(self.sensor_radius),
            comm_radius=float(self.comm_radius),
            last_control=self.last_control.copy(),
            last_desired_velocity=self.last_desired_velocity.copy(),
            path_length=float(self.path_length),
            connected_to_base=bool(self.connected_to_base),
            last_action_valid=bool(self.last_action_valid),
            trajectory=[point.copy() for point in self.trajectory] if include_trajectory else [self.position.copy()],
        )
