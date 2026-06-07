from __future__ import annotations

import numpy as np

from envs.waypoint.world import MissionWorld
from planners.waypoint_candidates import belief_candidates
from tasks.waypoint.base import WaypointScenario, path_length_delta, safety_penalty_count


class BeliefSearchScenario(WaypointScenario):
    def __init__(self, config: dict):
        super().__init__("belief_search", 1, config)

    def reset(self, rng: np.random.Generator, world: MissionWorld) -> dict:
        centers = world.grid_cell_centers()
        num_peaks_range = self.cfg("num_peaks_range", None)
        num_peaks = int(rng.integers(int(num_peaks_range[0]), int(num_peaks_range[1]) + 1)) if num_peaks_range else int(self.cfg("num_peaks", 3))
        grid = np.zeros((world.grid_size, world.grid_size), dtype=np.float32)
        peak_centers = world.sample_safe_points(num_peaks, lower_fraction=0.12, upper_fraction=0.88, min_separation=0.08 * world.map_size)
        for center in peak_centers:
            sigma = float(rng.uniform(0.08, 0.18)) * world.map_size
            grid += np.exp(-0.5 * (np.linalg.norm(centers - center, axis=-1) / max(sigma, 1e-6)) ** 2).astype(np.float32)
        grid[~world.navigable_grid_mask()] = 0.0
        grid /= max(float(grid.sum()), 1e-8)
        world.probability_grid = grid.copy()
        world.belief_grid = grid.copy()
        flat = grid.reshape(-1)
        target_idx = int(rng.choice(np.arange(flat.size), p=flat / flat.sum()))
        target = np.asarray([(target_idx % world.grid_size + 0.5), (target_idx // world.grid_size + 0.5)], dtype=np.float32)
        target = target / float(world.grid_size) * world.map_size
        return {"target_position": target, "peak_centers": peak_centers, "detected": False, "time_to_detection": 0}

    def generate_candidate_waypoints(self, world: MissionWorld, task_state: dict) -> tuple[np.ndarray, np.ndarray]:
        del task_state
        return belief_candidates(world, int(self.cfg("candidate_count", 12)))

    def compute_rewards(self, prev_world: MissionWorld, world: MissionWorld, task_state: dict, transition_info: dict) -> dict:
        prev_belief = prev_world.belief_grid.copy()
        newly_mass = 0.0
        repeated_mass = 0.0
        for uav in world.uavs:
            mask = world.sensor_mask_for_position(uav.position, uav.sensor_radius)
            newly_mass += float(prev_belief[mask].sum())
            repeated_mass += float((prev_world.visit_count_grid[mask] > 0.0).mean()) * 0.01
            world.belief_grid[mask] = 0.0
        remaining = float(world.belief_grid.sum())
        target = np.asarray(task_state["target_position"], dtype=np.float32)
        detected_now = any(np.linalg.norm(uav.position - target) <= uav.sensor_radius for uav in world.uavs)
        detect_bonus = 0.0
        if detected_now and not task_state.get("detected", False):
            task_state["detected"] = True
            task_state["time_to_detection"] = world.step_count
            detect_bonus = float(self.cfg("detection_bonus", 8.0))
        distance = path_length_delta(transition_info)
        safety = safety_penalty_count(transition_info)
        reward = (
            float(self.cfg("w_prob", 120.0)) * newly_mass
            + detect_bonus
            - float(self.cfg("w_repeat", 1.0)) * repeated_mass
            - float(self.cfg("w_distance", 0.15)) * distance
            - float(self.cfg("w_safety", 2.0)) * safety
        )
        per_agent = np.zeros(len(world.uavs), dtype=np.float32)
        for i, uav in enumerate(world.uavs):
            mask = prev_world.sensor_mask_for_position(uav.position, uav.sensor_radius)
            per_agent[i] = float(self.cfg("w_prob", 120.0)) * float(prev_belief[mask].sum())
        metrics = {
            "searched_probability_mass": float(1.0 - remaining),
            "detection_rate": float(task_state.get("detected", False)),
            "time_to_detection": float(task_state.get("time_to_detection", 0)),
            "repeated_search_mass": float(repeated_mass),
            "remaining_belief_mass": remaining,
            "success": bool(task_state.get("detected", False) or (1.0 - remaining) >= float(self.cfg("success_mass", 0.55))),
        }
        return {
            "team_reward": float(reward),
            "per_agent_rewards": per_agent,
            "components": {"prob_mass": newly_mass, "detect": detect_bonus, "repeat": -repeated_mass, "distance": -distance, "safety": -safety},
            "metrics": metrics,
            "success": bool(metrics["success"]),
        }

    def get_metrics(self, world: MissionWorld, task_state: dict) -> dict:
        searched = float(1.0 - world.belief_grid.sum())
        return {
            "searched_probability_mass": searched,
            "remaining_belief_mass": float(world.belief_grid.sum()),
            "detection_rate": float(task_state.get("detected", False)),
            "success": bool(task_state.get("detected", False) or searched >= float(self.cfg("success_mass", 0.55))),
        }
