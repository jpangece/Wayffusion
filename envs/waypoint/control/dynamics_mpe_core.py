from __future__ import annotations

from importlib import import_module
from typing import Any

import numpy as np

from envs.waypoint.control.action_adapters import AdapterOutput
from envs.waypoint.control.dynamics_base import DynamicsBackend, DynamicsInfo
from envs.waypoint.world import MissionWorld


DEFAULT_MPE_SOURCE = "third_party_openai_mpe"
MPE_CORE_SOURCES = {
    "third_party_openai_mpe": "third_party.openai_mpe.core",
    "mpe2": "mpe2._mpe_utils.core",
    "pettingzoo_mpe": "pettingzoo.mpe._mpe_utils.core",
}
MPE_SOURCE_ALIASES = {
    "local": "third_party_openai_mpe",
    "third_party": "third_party_openai_mpe",
    "openai_mpe": "third_party_openai_mpe",
    "vendored_openai_mpe": "third_party_openai_mpe",
}


def _canonical_mpe_source(source: str | None) -> str:
    requested = str(source or DEFAULT_MPE_SOURCE)
    return MPE_SOURCE_ALIASES.get(requested, requested)


def _load_mpe_core(source: str | None = None):
    """Load the explicitly requested MPE core source.

    This intentionally does not fall back across sources. Reproducible dynamics
    require selecting the source in the env config rather than depending on
    whichever optional package happens to be installed.
    """

    canonical = _canonical_mpe_source(source)
    if canonical not in MPE_CORE_SOURCES:
        supported = ", ".join(sorted(MPE_CORE_SOURCES))
        raise ValueError(f"Unsupported MPE core source: {source!r}. Supported sources: {supported}.")
    module_name = MPE_CORE_SOURCES[canonical]
    try:
        module = import_module(module_name)
    except Exception as exc:
        raise ImportError(
            f"Could not import configured MPE core source {canonical!r} from {module_name!r}. "
            "Edit dynamics_backend.source manually instead of relying on automatic fallback."
        ) from exc
    if not (hasattr(module, "World") and hasattr(module, "Agent")):
        raise ImportError(
            f"Configured MPE core source {canonical!r} ({module_name}) does not expose World and Agent."
        )
    return module, canonical


class MPECoreBackend(DynamicsBackend):
    """Wrapper around real MPE core World.step() dynamics."""

    name = "mpe_core"

    def __init__(self, config: dict[str, Any] | None = None):
        cfg = dict(config or {})
        self.requested_source = str(cfg.get("source", DEFAULT_MPE_SOURCE))
        self.mpe_core_module, self.source = _load_mpe_core(self.requested_source)
        self.mpe_world_cls = self.mpe_core_module.World
        self.mpe_agent_cls = self.mpe_core_module.Agent
        self.dt = float(cfg.get("dt", cfg.get("physics_dt", 0.1)))
        self.substeps = int(cfg.get("substeps", 10))
        self.damping = float(cfg.get("damping", 0.25))
        self.contact_force = float(cfg.get("contact_force", 100.0))
        self.contact_margin = float(cfg.get("contact_margin", 0.01))
        self.agent_size = float(cfg.get("agent_size", cfg.get("radius", 0.02)))
        self.agent_accel = float(cfg.get("agent_accel", 1.0))
        self.max_speed = float(cfg.get("max_speed", 0.08))
        self.mass = float(cfg.get("mass", 1.0))
        self.action_scale = float(cfg.get("action_scale", 1.0))
        self.enable_boundary_projection = bool(cfg.get("enable_boundary_projection", True))
        self.enable_no_fly_projection = bool(cfg.get("enable_no_fly_projection", True))
        self.mpe_world = None
        self.mpe_agents: list[Any] = []
        self.agent_name_to_mpe_agent: dict[str, Any] = {}
        self.last_positions: np.ndarray | None = None
        self.last_velocities: np.ndarray | None = None
        self.mpe_world_step_calls = 0

    def reset(self, world: MissionWorld) -> None:
        self.mpe_world = self.mpe_world_cls()
        self.mpe_world.dim_p = 2
        self.mpe_world.dim_c = 0
        self.mpe_world.dt = float(self.dt)
        self.mpe_world.damping = float(self.damping)
        self.mpe_world.contact_force = float(self.contact_force)
        self.mpe_world.contact_margin = float(self.contact_margin)
        self.mpe_agents = []
        self.agent_name_to_mpe_agent = {}
        for idx, uav in enumerate(world.uavs):
            agent = self.mpe_agent_cls()
            agent.name = f"uav_{idx}"
            agent.movable = True
            agent.collide = True
            agent.silent = True
            agent.blind = False
            agent.size = float(getattr(uav, "radius", self.agent_size) or self.agent_size)
            agent.accel = float(self.agent_accel)
            agent.max_speed = float(self.max_speed)
            agent.initial_mass = float(getattr(uav, "mass", self.mass) or self.mass)
            agent.u_noise = None
            agent.c_noise = None
            agent.action.u = np.zeros(2, dtype=np.float32)
            agent.action.c = np.zeros(0, dtype=np.float32)
            agent.state.p_pos = np.asarray(uav.position, dtype=np.float32).copy()
            agent.state.p_vel = np.asarray(uav.velocity, dtype=np.float32).copy()
            self.mpe_agents.append(agent)
            self.agent_name_to_mpe_agent[agent.name] = agent
            uav.mass = float(agent.initial_mass)
            uav.radius = float(agent.size)
        self.mpe_world.agents = self.mpe_agents
        self.last_positions = world.get_uav_positions().copy()
        self.last_velocities = np.asarray([uav.velocity for uav in world.uavs], dtype=np.float32).copy()
        self.mpe_world_step_calls = 0

    def _ensure_world(self, world: MissionWorld) -> None:
        if self.mpe_world is None or len(self.mpe_agents) != len(world.uavs):
            self.reset(world)

    def _sync_wayffusion_to_mpe(self, world: MissionWorld) -> None:
        for agent, uav in zip(self.mpe_agents, world.uavs):
            agent.state.p_pos = np.asarray(uav.position, dtype=np.float32).copy()
            agent.state.p_vel = np.asarray(uav.velocity, dtype=np.float32).copy()

    def _assign_mpe_actions(self, physical_actions: np.ndarray) -> None:
        for idx, agent in enumerate(self.mpe_agents):
            agent.action.u = physical_actions[idx].astype(np.float32)
            agent.action.c = np.zeros(0, dtype=np.float32)

    def _set_mpe_actions(self, controls: np.ndarray) -> np.ndarray:
        physical_actions = np.asarray(controls, dtype=np.float32) * float(self.action_scale)
        self._assign_mpe_actions(physical_actions)
        return physical_actions

    def _apply_wayffusion_safety_corrections(self, world: MissionWorld, previous: np.ndarray) -> tuple[np.ndarray, np.ndarray, int, int]:
        positions = np.asarray([agent.state.p_pos for agent in self.mpe_agents], dtype=np.float32)
        velocities = np.asarray([agent.state.p_vel for agent in self.mpe_agents], dtype=np.float32)
        geofence_mask = np.zeros(len(positions), dtype=bool)
        if self.enable_boundary_projection:
            projected = world.project_to_geofence(positions)
            geofence_mask = np.linalg.norm(projected - positions, axis=-1) > 1e-8
            positions = projected.astype(np.float32)
            velocities[geofence_mask] = 0.0
        no_fly_mask = np.zeros(len(positions), dtype=bool)
        if self.enable_no_fly_projection:
            no_fly_mask = ~world.check_no_fly(positions)
            positions[no_fly_mask] = previous[no_fly_mask]
            velocities[no_fly_mask] = 0.0
        for idx, agent in enumerate(self.mpe_agents):
            agent.state.p_pos = positions[idx].astype(np.float32)
            agent.state.p_vel = velocities[idx].astype(np.float32)
        return positions, velocities, int(geofence_mask.sum()), int(no_fly_mask.sum())

    def _count_overlaps(self, positions: np.ndarray) -> int:
        count = 0
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                min_dist = float(self.mpe_agents[i].size + self.mpe_agents[j].size)
                if float(np.linalg.norm(positions[i] - positions[j])) < min_dist:
                    count += 1
        return int(count)

    def step(
        self,
        world: MissionWorld,
        controls: np.ndarray,
        adapter_output: AdapterOutput | None = None,
        safety_result=None,
    ) -> DynamicsInfo:
        self._ensure_world(world)
        controls = np.asarray(controls, dtype=np.float32)
        old_positions = world.get_uav_positions().astype(np.float32)
        if controls.shape != old_positions.shape:
            raise ValueError(f"controls must have shape {old_positions.shape}, got {controls.shape}")

        self._sync_wayffusion_to_mpe(world)
        physical_actions = self._set_mpe_actions(controls)
        rejected = np.zeros(len(world.uavs), dtype=bool)
        if safety_result is not None and hasattr(safety_result, "rejected_mask"):
            rejected |= np.asarray(safety_result.rejected_mask, dtype=bool)

        path_delta = np.zeros(len(world.uavs), dtype=np.float32)
        geofence_count = 0
        no_fly_count = 0
        collision_count = 0
        step_calls_before = int(self.mpe_world_step_calls)
        for _ in range(max(int(self.substeps), 1)):
            previous = np.asarray([agent.state.p_pos for agent in self.mpe_agents], dtype=np.float32)
            self._assign_mpe_actions(physical_actions)
            self.mpe_world.step()
            self.mpe_world_step_calls += 1
            positions, _, geofence_hits, no_fly_hits = self._apply_wayffusion_safety_corrections(world, previous)
            geofence_count += geofence_hits
            no_fly_count += no_fly_hits
            collision_count += self._count_overlaps(positions)
            rejected |= geofence_hits > 0 and ~world.check_geofence(positions)
            rejected |= no_fly_hits > 0 and ~world.check_no_fly(positions)
            path_delta += np.linalg.norm(positions - previous, axis=-1).astype(np.float32)

        positions = np.asarray([agent.state.p_pos for agent in self.mpe_agents], dtype=np.float32)
        velocities = np.asarray([agent.state.p_vel for agent in self.mpe_agents], dtype=np.float32)
        target_waypoints = positions if adapter_output is None else np.asarray(adapter_output.clipped_waypoints, dtype=np.float32)
        arrived = np.linalg.norm(positions - target_waypoints, axis=-1) <= 0.025
        pairwise = world.compute_pairwise_distances(positions)
        if len(world.uavs) > 1:
            upper = pairwise[np.triu_indices(len(world.uavs), k=1)]
            min_pairwise = float(upper.min()) if upper.size else float(world.map_size)
        else:
            min_pairwise = float(world.map_size)

        for idx, uav in enumerate(world.uavs):
            uav.position = positions[idx].astype(np.float32)
            uav.velocity = velocities[idx].astype(np.float32)
            uav.current_waypoint = target_waypoints[idx].astype(np.float32)
            uav.path_length += float(path_delta[idx])
            uav.last_control = physical_actions[idx].astype(np.float32)
            uav.last_action_valid = not bool(rejected[idx])
            uav.battery = max(0.0, float(uav.battery) - float(path_delta[idx]) * 0.01)
            uav.trajectory.append(uav.position.copy())

        world.step_count += 1
        world.update_communication_graph()
        footprint_info = world.accumulate_sensor_footprints()
        speeds = np.linalg.norm(velocities, axis=-1)
        control_norm = np.linalg.norm(physical_actions, axis=-1)
        finite = bool(np.isfinite(positions).all() and np.isfinite(velocities).all() and np.isfinite(path_delta).all())
        self.last_positions = positions.copy()
        self.last_velocities = velocities.copy()
        last_step_calls = int(self.mpe_world_step_calls - step_calls_before)
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
            substeps=last_step_calls,
            info={
                "dynamics_backend": self.name,
                "dynamics_finite": finite,
                "mpe_source": self.source,
                "requested_mpe_source": self.requested_source,
                "uses_real_mpe_core": True,
                "mpe_world_step_calls": int(last_step_calls),
                "mpe_world_step_calls_total": int(self.mpe_world_step_calls),
                "mpe_world_class": f"{self.mpe_world.__class__.__module__}.{self.mpe_world.__class__.__name__}",
                **footprint_info,
            },
        )
