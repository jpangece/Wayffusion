from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from envs.waypoint.world import MissionWorld


WAYPOINT_TASKS = [
    "area_coverage",
    "belief_search",
    "priority_inspection",
    "connectivity_expansion",
    "dynamic_target_escort",
    "target_interception",
]


@dataclass
class WaypointScenario:
    name: str
    task_id: int
    config: dict[str, Any]

    def reset(self, rng: np.random.Generator, world: MissionWorld) -> dict:
        return {}

    def build_task_field(self, world: MissionWorld, task_state: dict) -> np.ndarray:
        coverage = world.coverage_grid.astype(np.float32)
        visits = np.clip(world.visit_count_grid / 3.0, 0.0, 1.0).astype(np.float32)
        belief = np.asarray(world.belief_grid, dtype=np.float32)
        if belief.max() > 1e-8:
            belief = belief / float(belief.max())
        risk = world.risk_grid()
        target = np.zeros_like(coverage, dtype=np.float32)
        target_positions = task_state.get("target_positions")
        if target_positions is None and "target_position" in task_state:
            target_positions = np.asarray(task_state["target_position"], dtype=np.float32)[None, :]
        if target_positions is not None:
            for position in np.asarray(target_positions, dtype=np.float32).reshape(-1, 2):
                idx = world.world_to_grid(position[None, :])[0]
                target[idx[1], idx[0]] = 1.0
        if "pois" in task_state:
            for point, visited in zip(task_state["pois"], task_state.get("visited", np.zeros(len(task_state["pois"]), dtype=bool))):
                if not visited:
                    idx = world.world_to_grid(np.asarray(point, dtype=np.float32)[None, :])[0]
                    target[idx[1], idx[0]] = 1.0
        return np.stack([coverage, visits, belief, risk, target], axis=0).astype(np.float32)

    def generate_candidate_waypoints(self, world: MissionWorld, task_state: dict) -> tuple[np.ndarray, np.ndarray]:
        """Optional planner helper kept out of the environment action API."""
        del task_state
        positions = world.get_uav_positions()
        return positions[:, None, :].astype(np.float32), np.ones((len(positions), 1), dtype=bool)

    def step_update(self, world: MissionWorld, task_state: dict) -> None:
        del world, task_state

    def compute_rewards(
        self,
        prev_world: MissionWorld,
        world: MissionWorld,
        task_state: dict,
        transition_info: dict,
    ) -> dict:
        raise NotImplementedError

    def get_metrics(self, world: MissionWorld, task_state: dict) -> dict:
        return {"success": False}

    def task_one_hot(self) -> np.ndarray:
        out = np.zeros(len(WAYPOINT_TASKS), dtype=np.float32)
        out[int(self.task_id)] = 1.0
        return out

    def mission_goal(self, task_state: dict) -> np.ndarray:
        return np.asarray(task_state.get("_mission_goal", np.zeros(3)), dtype=np.float32)

    def goal_value(self, task_state: dict, index: int, config_key: str, default: float) -> float:
        goal = self.mission_goal(task_state)
        mask = np.asarray(task_state.get("_goal_mask", np.zeros(3)), dtype=np.float32)
        if int(index) < len(goal) and int(index) < len(mask) and mask[int(index)] > 0.5:
            return float(goal[int(index)])
        return float(self.cfg(config_key, default))

    def cfg(self, key: str, default):
        return self.config.get(self.name, {}).get(key, self.config.get(key, default))

    def distance_cfg(self, key: str, default: float, world: MissionWorld) -> float:
        """Read a task distance in map-relative or absolute map units."""
        task_config = self.config.get(self.name, {})
        units = str(task_config.get("spatial_units", self.config.get("spatial_units", "relative"))).lower()
        value = float(self.cfg(key, default))
        if units == "absolute":
            return value
        if units != "relative":
            raise ValueError(f"Unsupported task spatial_units={units!r} for {self.name}")
        return value * world.map_size


def per_agent_new_coverage(prev_world: MissionWorld, world: MissionWorld) -> np.ndarray:
    return world.coverage_transition_stats(prev_world).per_agent_new_coverage.copy()


def uncovered_approach_gain(
    prev_world: MissionWorld,
    world: MissionWorld,
    scale: float,
) -> float:
    navigable = world.navigable_grid_mask()
    uncovered = (prev_world.coverage_grid <= 0.0) & navigable
    if not bool(uncovered.any()) or not world.uavs:
        return 0.0
    points = world.grid_cell_centers()[uncovered]
    scale = max(float(scale), 1e-6)

    def score(positions: np.ndarray) -> float:
        nearest_distances, _ = cKDTree(np.asarray(positions, dtype=np.float32)).query(
            points,
            k=1,
            workers=1,
        )
        nearest_distances = np.asarray(nearest_distances, dtype=np.float32)
        return float(np.exp(-nearest_distances / scale).mean())

    before = score(prev_world.get_uav_positions())
    after = score(world.get_uav_positions())
    return max(after - before, 0.0)


def path_length_delta(transition_info: dict) -> float:
    return float(np.asarray(transition_info.get("path_length_delta", 0.0), dtype=np.float32).sum())


def safety_penalty_count(transition_info: dict) -> float:
    return float(transition_info.get("safety_violation_count", 0)) + float(transition_info.get("no_fly_violation_count", 0))


def normalized_positions(world: MissionWorld) -> np.ndarray:
    positions = world.get_uav_positions()
    velocity = np.asarray([uav.velocity for uav in world.uavs], dtype=np.float32)
    extras = np.asarray(
        [
            [
                uav.battery,
                float(uav.role_id) / max(len(world.uavs) - 1, 1),
                float(uav.connected_to_base),
                float(uav.last_action_valid),
            ]
            for uav in world.uavs
        ],
        dtype=np.float32,
    )
    return np.concatenate([positions / max(world.map_size, 1e-6), velocity / max(world.max_speed, 1e-6), extras], axis=-1)
