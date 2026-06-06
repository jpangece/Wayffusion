from __future__ import annotations

import numpy as np

from envs.waypoint.candidates import coverage_candidates
from envs.waypoint.world import MissionWorld
from tasks.waypoint.base import WaypointScenario, path_length_delta, per_agent_new_coverage, safety_penalty_count


class AreaCoverageScenario(WaypointScenario):
    def __init__(self, config: dict):
        super().__init__("area_coverage", 0, config)

    def generate_candidate_waypoints(self, world: MissionWorld, task_state: dict) -> tuple[np.ndarray, np.ndarray]:
        del task_state
        return coverage_candidates(world, int(self.cfg("candidate_count", 12)))

    def compute_rewards(self, prev_world: MissionWorld, world: MissionWorld, task_state: dict, transition_info: dict) -> dict:
        prev_ratio = float(prev_world.coverage_grid.mean())
        coverage_ratio = float(world.coverage_grid.mean())
        new_ratio = max(coverage_ratio - prev_ratio, 0.0)
        overlap_ratio = float((world.visit_count_grid > 1.0).mean())
        distance = path_length_delta(transition_info)
        safety = safety_penalty_count(transition_info)
        reward = (
            float(self.cfg("w_new", 160.0)) * new_ratio
            - float(self.cfg("w_overlap", 0.25)) * overlap_ratio
            - float(self.cfg("w_distance", 0.2)) * distance
            - float(self.cfg("w_safety", 2.0)) * safety
        )
        per_agent = float(self.cfg("w_new", 160.0)) * per_agent_new_coverage(prev_world, world)
        per_agent -= float(self.cfg("w_distance", 0.2)) * np.asarray(transition_info["path_length_delta"], dtype=np.float32)
        metrics = self.get_metrics(world, {})
        metrics.update(
            {
                "new_coverage_ratio": new_ratio,
                "overlap_ratio": overlap_ratio,
                "path_length": float(sum(uav.path_length for uav in world.uavs)),
                "min_pairwise_distance": float(transition_info.get("min_pairwise_distance", world.map_size)),
                "safety_violation_count": safety,
            }
        )
        if metrics["coverage_ratio"] >= 0.8 and task_state.get("time_to_80_coverage") is None:
            task_state["time_to_80_coverage"] = world.step_count
        metrics["time_to_80_coverage"] = float(task_state.get("time_to_80_coverage") or 0.0)
        return {
            "team_reward": float(reward),
            "per_agent_rewards": per_agent.astype(np.float32),
            "components": {"new_coverage": new_ratio, "overlap": -overlap_ratio, "distance": -distance, "safety": -safety},
            "metrics": metrics,
            "success": bool(metrics["success"]),
        }

    def reset(self, rng: np.random.Generator, world: MissionWorld) -> dict:
        del rng, world
        return {"time_to_80_coverage": None}

    def get_metrics(self, world: MissionWorld, task_state: dict) -> dict:
        del task_state
        ratio = float(world.coverage_grid.mean())
        return {"coverage_ratio": ratio, "success": ratio >= float(self.cfg("success_ratio", 0.55))}
