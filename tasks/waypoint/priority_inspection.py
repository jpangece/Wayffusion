from __future__ import annotations

import numpy as np

from envs.waypoint.candidates import poi_candidates
from envs.waypoint.world import MissionWorld
from tasks.waypoint.base import WaypointScenario, path_length_delta, safety_penalty_count


class PriorityInspectionScenario(WaypointScenario):
    def __init__(self, config: dict):
        super().__init__("priority_inspection", 2, config)

    def reset(self, rng: np.random.Generator, world: MissionWorld) -> dict:
        count = int(self.cfg("num_pois", 8))
        pois = rng.uniform(0.12, 0.92, size=(count, 2)).astype(np.float32) * world.map_size
        weights = rng.uniform(0.5, 2.0, size=count).astype(np.float32)
        deadlines = rng.integers(max(8, world.max_steps // 4), world.max_steps + 1, size=count)
        return {"pois": pois, "weights": weights, "deadlines": deadlines, "visited": np.zeros(count, dtype=bool), "repeat_visit_count": 0}

    def generate_candidate_waypoints(self, world: MissionWorld, task_state: dict) -> tuple[np.ndarray, np.ndarray]:
        return poi_candidates(world, np.asarray(task_state["pois"], dtype=np.float32), int(self.cfg("candidate_count", 12)))

    def compute_rewards(self, prev_world: MissionWorld, world: MissionWorld, task_state: dict, transition_info: dict) -> dict:
        del prev_world
        pois = np.asarray(task_state["pois"], dtype=np.float32)
        weights = np.asarray(task_state["weights"], dtype=np.float32)
        visited = np.asarray(task_state["visited"], dtype=bool)
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
            "components": {"visited_weight": newly_weight, "repeat": -float(repeats), "late": -late_penalty, "distance": -distance, "safety": -safety},
            "metrics": metrics,
            "success": bool(metrics["success"]),
        }

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
