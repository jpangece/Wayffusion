from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

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
        risk = np.zeros_like(coverage, dtype=np.float32)
        for zone in world.no_fly_zones:
            if "center" in zone and "radius" in zone:
                yy, xx = np.mgrid[0 : world.grid_size, 0 : world.grid_size]
                centers = np.stack([(xx + 0.5) / world.grid_size, (yy + 0.5) / world.grid_size], axis=-1) * world.map_size
                dist = np.linalg.norm(centers - np.asarray(zone["center"], dtype=np.float32), axis=-1)
                risk = np.maximum(risk, (dist <= float(zone["radius"])).astype(np.float32))
        target = np.zeros_like(coverage, dtype=np.float32)
        if "target_position" in task_state:
            idx = world.world_to_grid(np.asarray(task_state["target_position"], dtype=np.float32)[None, :])[0]
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

    def cfg(self, key: str, default):
        return self.config.get(self.name, {}).get(key, self.config.get(key, default))


def per_agent_new_coverage(prev_world: MissionWorld, world: MissionWorld) -> np.ndarray:
    before = prev_world.coverage_grid > 0.0
    rewards = []
    for uav in world.uavs:
        mask = world.sensor_mask_for_position(uav.position, uav.sensor_radius)
        rewards.append(float(np.logical_and(mask, ~before).mean()))
    return np.asarray(rewards, dtype=np.float32)


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
