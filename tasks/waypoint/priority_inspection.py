from __future__ import annotations

import numpy as np

from envs.waypoint.world import MissionWorld
from planners.waypoint_candidates import poi_candidates
from tasks.waypoint.base import WaypointScenario, path_length_delta, safety_penalty_count


class PriorityInspectionScenario(WaypointScenario):
    def __init__(self, config: dict):
        super().__init__("priority_inspection", 2, config)

    def reset(self, rng: np.random.Generator, world: MissionWorld) -> dict:
        count_range = self.cfg("num_pois_range", None)
        count = int(rng.integers(int(count_range[0]), int(count_range[1]) + 1)) if count_range else int(self.cfg("num_pois", 8))
        pois = world.sample_safe_points(
            count,
            lower_fraction=0.10,
            upper_fraction=0.92,
            min_separation=self.distance_cfg("poi_min_separation", 0.05, world),
            avoid_points=world.get_uav_positions(),
        )
        weights = rng.uniform(0.5, 2.0, size=count).astype(np.float32)
        deadlines = rng.integers(max(8, world.max_steps // 4), world.max_steps + 1, size=count)
        return {"pois": pois, "weights": weights, "deadlines": deadlines, "visited": np.zeros(count, dtype=bool), "repeat_visit_count": 0}

    def generate_candidate_waypoints(self, world: MissionWorld, task_state: dict) -> tuple[np.ndarray, np.ndarray]:
        return poi_candidates(world, np.asarray(task_state["pois"], dtype=np.float32), int(self.cfg("candidate_count", 12)))

    def build_task_field(self, world: MissionWorld, task_state: dict) -> np.ndarray:
        field = super().build_task_field(world, task_state)
        if str(self.cfg("poi_target_field", "binary")) != "gaussian":
            return field
        pois = np.asarray(task_state.get("pois", []), dtype=np.float32)
        visited = np.asarray(task_state.get("visited", np.zeros(len(pois), dtype=bool)), dtype=bool)
        weights = np.asarray(task_state.get("weights", np.ones(len(pois), dtype=np.float32)), dtype=np.float32)
        target = np.zeros_like(field[4], dtype=np.float32)
        if len(pois):
            centers = world.grid_cell_centers()
            sigma = max(self.distance_cfg("poi_target_sigma", 0.08, world), 1e-6)
            for point, weight, is_visited in zip(pois, weights, visited):
                if is_visited:
                    continue
                distance = np.linalg.norm(centers - point[None, None, :], axis=-1)
                target += float(weight) * np.exp(-0.5 * (distance / sigma) ** 2).astype(np.float32)
        target[~world.navigable_grid_mask()] = 0.0
        if target.max() > 1e-8:
            target /= float(target.max())
        field[4] = target
        return field

    def compute_rewards(self, prev_world: MissionWorld, world: MissionWorld, task_state: dict, transition_info: dict) -> dict:
        pois = np.asarray(task_state["pois"], dtype=np.float32)
        weights = np.asarray(task_state["weights"], dtype=np.float32)
        visited = np.asarray(task_state["visited"], dtype=bool)
        approach_gain = self._approach_gain(prev_world, world, pois, weights, visited)
        newly = np.zeros(len(pois), dtype=bool)
        repeats = 0
        per_agent = np.zeros(len(world.uavs), dtype=np.float32)
        for agent_idx, uav in enumerate(world.uavs):
            dist = np.linalg.norm(pois - uav.position[None, :], axis=-1)
            hit = dist <= uav.sensor_radius
            for poi_idx in np.where(hit)[0]:
                if not visited[poi_idx]:
                    newly[poi_idx] = True
                    per_agent[agent_idx] += weights[poi_idx]
                else:
                    repeats += 1
        task_state["visited"][newly] = True
        task_state["repeat_visit_count"] += repeats
        late = newly & (world.step_count > np.asarray(task_state["deadlines"]))
        newly_weight = float(weights[newly].sum())
        late_penalty = float(weights[late].sum())
        distance = path_length_delta(transition_info)
        safety = safety_penalty_count(transition_info)
        reward = (
            float(self.cfg("w_visit", 6.0)) * newly_weight
            + float(self.cfg("w_approach", 0.0)) * approach_gain
            - float(self.cfg("w_repeat", 0.4)) * repeats
            - float(self.cfg("w_late", 1.0)) * late_penalty
            - float(self.cfg("w_distance", 0.15)) * distance
            - float(self.cfg("w_safety", 2.0)) * safety
        )
        per_agent *= float(self.cfg("w_visit", 6.0))
        metrics = self.get_metrics(world, task_state)
        metrics.update({"repeat_visit_count": float(task_state["repeat_visit_count"]), "deadline_satisfaction": float(1.0 - late.mean()) if len(late) else 1.0})
        return {
            "team_reward": float(reward),
            "per_agent_rewards": per_agent.astype(np.float32),
            "components": {"visited_weight": newly_weight, "approach": approach_gain, "repeat": -float(repeats), "late": -late_penalty, "distance": -distance, "safety": -safety},
            "metrics": metrics,
            "success": bool(metrics["success"]),
        }

    def _approach_gain(self, prev_world: MissionWorld, world: MissionWorld, pois: np.ndarray, weights: np.ndarray, visited: np.ndarray) -> float:
        if not len(pois) or bool(np.all(visited)):
            return 0.0
        scale = self.distance_cfg("approach_scale", 0.20, world)
        remaining = ~visited
        points = pois[remaining]
        point_weights = weights[remaining]

        def proximity(positions: np.ndarray) -> float:
            dist = np.linalg.norm(positions[:, None, :] - points[None, :, :], axis=-1)
            score = np.exp(-dist / max(scale, 1e-6)).max(axis=0)
            return float(np.sum(score * point_weights) / max(float(point_weights.sum()), 1e-8))

        before = proximity(prev_world.get_uav_positions())
        after = proximity(world.get_uav_positions())
        return max(after - before, 0.0)

    def get_metrics(self, world: MissionWorld, task_state: dict) -> dict:
        del world
        visited = np.asarray(task_state.get("visited", []), dtype=bool)
        weights = np.asarray(task_state.get("weights", np.ones(len(visited))), dtype=np.float32)
        weighted = float(weights[visited].sum() / max(float(weights.sum()), 1e-8)) if len(weights) else 0.0
        ratio = float(visited.mean()) if len(visited) else 0.0
        return {
            "weighted_poi_completion": weighted,
            "visited_poi_ratio": ratio,
            "repeat_visit_count": float(task_state.get("repeat_visit_count", 0)),
            "success": weighted >= float(self.cfg("success_weighted_completion", 0.7)),
        }
