from __future__ import annotations

import numpy as np

from envs.waypoint.world import MissionWorld
from planners.waypoint_candidates import coverage_candidates
from tasks.waypoint.base import WaypointScenario, path_length_delta, per_agent_new_coverage, safety_penalty_count


class AreaCoverageScenario(WaypointScenario):
    def __init__(self, config: dict):
        super().__init__("area_coverage", 0, config)

    def generate_candidate_waypoints(self, world: MissionWorld, task_state: dict) -> tuple[np.ndarray, np.ndarray]:
        del task_state
        return coverage_candidates(world, int(self.cfg("candidate_count", 12)))

    def compute_rewards(self, prev_world: MissionWorld, world: MissionWorld, task_state: dict, transition_info: dict) -> dict:
        prev_ratio = prev_world.coverage_ratio()
        coverage_ratio = world.coverage_ratio()
        new_ratio = max(coverage_ratio - prev_ratio, 0.0)
        overlap_ratio = float((world.visit_count_grid > 1.0).mean())
        uncovered_approach = self._uncovered_approach_gain(prev_world, world)
        repeat_footprint_ratio = self._repeat_footprint_ratio(prev_world, world)
        distance = path_length_delta(transition_info)
        safety = safety_penalty_count(transition_info)
        reward = (
            float(self.cfg("w_new", 160.0)) * new_ratio
            + float(self.cfg("w_uncovered_approach", 0.0)) * uncovered_approach
            - float(self.cfg("w_overlap", 0.25)) * overlap_ratio
            - float(self.cfg("w_repeat_footprint", 0.0)) * repeat_footprint_ratio
            - float(self.cfg("w_distance", 0.2)) * distance
            - float(self.cfg("w_safety", 2.0)) * safety
        )
        per_agent = float(self.cfg("w_new", 160.0)) * per_agent_new_coverage(prev_world, world)
        per_agent -= float(self.cfg("w_distance", 0.2)) * np.asarray(transition_info["path_length_delta"], dtype=np.float32)
        metrics = self.get_metrics(world, task_state)
        metrics.update(
            {
                "new_coverage_ratio": new_ratio,
                "overlap_ratio": overlap_ratio,
                "uncovered_approach_gain": uncovered_approach,
                "repeat_footprint_ratio": repeat_footprint_ratio,
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
            "components": {
                "new_coverage": new_ratio,
                "uncovered_approach": uncovered_approach,
                "overlap": -overlap_ratio,
                "repeat_footprint": -repeat_footprint_ratio,
                "distance": -distance,
                "safety": -safety,
            },
            "metrics": metrics,
            "success": bool(metrics["success"]),
        }

    def reset(self, rng: np.random.Generator, world: MissionWorld) -> dict:
        del rng, world
        return {"time_to_80_coverage": None}

    def _uncovered_approach_gain(self, prev_world: MissionWorld, world: MissionWorld) -> float:
        """Dense potential: reward moving closer to still-uncovered navigable cells."""
        navigable = world.navigable_grid_mask()
        uncovered = (prev_world.coverage_grid <= 0.0) & navigable
        if not bool(uncovered.any()) or not world.uavs:
            return 0.0
        points = world.grid_cell_centers()[uncovered]
        scale = max(self.distance_cfg("uncovered_approach_scale", 0.20, world), 1e-6)

        def score(positions: np.ndarray) -> float:
            distances = np.linalg.norm(positions[:, None, :] - points[None, :, :], axis=-1)
            proximity = np.exp(-distances / scale).max(axis=0)
            return float(proximity.mean())

        before = score(prev_world.get_uav_positions())
        after = score(world.get_uav_positions())
        return max(after - before, 0.0)

    def _repeat_footprint_ratio(self, prev_world: MissionWorld, world: MissionWorld) -> float:
        if not world.uavs:
            return 0.0
        ratios = []
        navigable = world.navigable_grid_mask()
        previously_seen = prev_world.visit_count_grid > 0.0
        for uav in world.uavs:
            mask = world.sensor_mask_for_position(uav.position, uav.sensor_radius) & navigable
            denom = max(float(mask.sum()), 1.0)
            ratios.append(float((mask & previously_seen).sum() / denom))
        return float(np.mean(ratios)) if ratios else 0.0

    def get_metrics(self, world: MissionWorld, task_state: dict) -> dict:
        ratio = world.coverage_ratio()
        threshold = self.goal_value(task_state, 0, "success_ratio", 0.55)
        return {
            "coverage_ratio": ratio,
            "goal_coverage_target": threshold,
            "success": ratio >= threshold,
        }
