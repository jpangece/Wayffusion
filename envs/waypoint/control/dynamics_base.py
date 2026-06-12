from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from envs.waypoint.control.action_adapters import AdapterOutput
from envs.waypoint.world import MissionWorld


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

    def step(
        self,
        world: MissionWorld,
        controls: np.ndarray,
        adapter_output: AdapterOutput | None = None,
        safety_result=None,
    ) -> DynamicsInfo:
        raise NotImplementedError
