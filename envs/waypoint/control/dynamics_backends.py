from __future__ import annotations

from typing import Any

from envs.waypoint.control.dynamics_base import DynamicsBackend, DynamicsInfo
from envs.waypoint.control.dynamics_mpe_core import MPECoreBackend
from envs.waypoint.control.dynamics_waypoint_fast import WaypointBehaviorFastBackend
from envs.waypoint.control.dynamics_waypoint_realistic import WaypointBehaviorRealisticBackend


REMOVED_BACKENDS = {"mpe_particle", "mpe_like_particle", "kinematic_point"}


def build_dynamics_backend(config: dict[str, Any] | None):
    cfg = dict(config or {})
    name = str(cfg.get("name", "mpe_core"))
    if name == "mpe_core":
        return MPECoreBackend(cfg)
    if name == "waypoint_behavior_fast":
        return WaypointBehaviorFastBackend(cfg)
    if name == "waypoint_behavior_realistic":
        return WaypointBehaviorRealisticBackend(cfg)
    if name in REMOVED_BACKENDS:
        raise ValueError(
            f"{name} was removed as a runnable dynamics backend. For large-scale training use "
            "waypoint_behavior_fast; for medium-realism validation use waypoint_behavior_realistic; "
            "for high-fidelity pre-deployment checks use mpe_core."
        )
    raise ValueError(
        "Unsupported dynamics backend: "
        f"{name}. Supported backends: mpe_core, waypoint_behavior_fast, waypoint_behavior_realistic."
    )


__all__ = [
    "DynamicsBackend",
    "DynamicsInfo",
    "MPECoreBackend",
    "WaypointBehaviorFastBackend",
    "WaypointBehaviorRealisticBackend",
    "build_dynamics_backend",
]
