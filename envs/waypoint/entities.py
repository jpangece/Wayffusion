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
    sensor_radius: float = 0.12
    comm_radius: float = 0.35
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
        if not self.trajectory:
            self.trajectory.append(self.position.copy())

    def snapshot(self) -> "UAVState":
        return UAVState(
            position=self.position.copy(),
            velocity=self.velocity.copy(),
            battery=float(self.battery),
            current_waypoint=self.current_waypoint.copy(),
            status=str(self.status),
            role_id=int(self.role_id),
            sensor_radius=float(self.sensor_radius),
            comm_radius=float(self.comm_radius),
            path_length=float(self.path_length),
            connected_to_base=bool(self.connected_to_base),
            last_action_valid=bool(self.last_action_valid),
            trajectory=[point.copy() for point in self.trajectory],
        )
