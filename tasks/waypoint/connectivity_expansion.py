from __future__ import annotations

import numpy as np

from envs.waypoint.world import MissionWorld
from planners.waypoint_candidates import connectivity_boundary_candidates
from tasks.waypoint.base import WaypointScenario, path_length_delta, per_agent_new_coverage, safety_penalty_count


class ConnectivityExpansionScenario(WaypointScenario):
    def __init__(self, config: dict):
        super().__init__("connectivity_expansion", 3, config)

    def reset(self, rng: np.random.Generator, world: MissionWorld) -> dict:
        del rng
        return {"disconnect_steps": 0, "max_disconnect_run": 0, "current_disconnect_run": 0, "effective_explored_radius": self._effective_radius(world)}

    def generate_candidate_waypoints(self, world: MissionWorld, task_state: dict) -> tuple[np.ndarray, np.ndarray]:
        del task_state
        return connectivity_boundary_candidates(world, int(self.cfg("candidate_count", 12)))

    def compute_rewards(self, prev_world: MissionWorld, world: MissionWorld, task_state: dict, transition_info: dict) -> dict:
        prev_radius = self._effective_radius(prev_world)
        radius = self._effective_radius(world)
        radius_gain = max(radius - prev_radius, 0.0)
        prev_cov = prev_world.coverage_ratio()
        coverage_gain = max(world.coverage_ratio() - prev_cov, 0.0)
        connected = world.connected_to_base_mask()
        violation = 1.0 - float(np.mean(connected)) if len(connected) else 0.0
        if violation > 1e-6:
            task_state["disconnect_steps"] += 1
            task_state["current_disconnect_run"] += 1
        else:
            task_state["current_disconnect_run"] = 0
        task_state["max_disconnect_run"] = max(int(task_state["max_disconnect_run"]), int(task_state["current_disconnect_run"]))
        relay_credit = self._relay_credit(world)
        distance = path_length_delta(transition_info)
        safety = safety_penalty_count(transition_info)
        reward = (
            float(self.cfg("w_expand", 5.0)) * radius_gain
            + float(self.cfg("w_coverage", 120.0)) * coverage_gain
            + float(self.cfg("w_relay", 0.5)) * relay_credit
            - float(self.cfg("w_disconnect", 4.0)) * violation
            - float(self.cfg("w_distance", 0.15)) * distance
            - float(self.cfg("w_safety", 2.0)) * safety
        )
        per_agent = float(self.cfg("w_coverage", 120.0)) * per_agent_new_coverage(prev_world, world)
        per_agent += float(self.cfg("w_expand", 5.0)) * radius_gain / max(len(world.uavs), 1)
        metrics = self.get_metrics(world, task_state)
        metrics.update(
            {
                "connectivity_violation_rate": float(task_state["disconnect_steps"] / max(world.step_count, 1)),
                "longest_disconnected_duration": float(task_state["max_disconnect_run"]),
                "relay_utilization_ratio": float(relay_credit / max(len(world.uavs), 1)),
                "effective_explored_radius": float(radius),
                "coverage_ratio": world.coverage_ratio(),
            }
        )
        return {
            "team_reward": float(reward),
            "per_agent_rewards": per_agent.astype(np.float32),
            "components": {"expand": radius_gain, "coverage": coverage_gain, "relay": relay_credit, "disconnect": -violation, "distance": -distance, "safety": -safety},
            "metrics": metrics,
            "success": bool(metrics["success"]),
        }

    def _effective_radius(self, world: MissionWorld) -> float:
        connected = world.connected_to_base_mask()
        positions = world.get_uav_positions()
        if len(positions) == 0 or not connected.any():
            return 0.0
        dist = np.linalg.norm(positions[connected] - world.base_position[None, :], axis=-1)
        return float(dist.max() / max(world.map_size, 1e-6))

    def _relay_credit(self, world: MissionWorld) -> float:
        graph = world.communication_graph.copy()
        n = len(world.uavs)
        credit = 0.0
        for remove_idx in range(1, n + 1):
            reduced = graph.copy()
            reduced[remove_idx, :] = False
            reduced[:, remove_idx] = False
            visited = np.zeros(n + 1, dtype=bool)
            visited[0] = True
            stack = [0]
            while stack:
                node = stack.pop()
                for nxt in np.where(reduced[node])[0]:
                    if not visited[nxt]:
                        visited[nxt] = True
                        stack.append(int(nxt))
            if np.count_nonzero(world.connected_to_base_mask()) > np.count_nonzero(visited[1:]):
                credit += 1.0
        return credit

    def get_metrics(self, world: MissionWorld, task_state: dict) -> dict:
        connected_ratio = float(np.mean(world.connected_to_base_mask())) if world.uavs else 1.0
        coverage = world.coverage_ratio()
        radius = self._effective_radius(world)
        violation_rate = float(task_state.get("disconnect_steps", 0) / max(world.step_count, 1))
        return {
            "connected_to_base_ratio": connected_ratio,
            "connectivity_violation_rate": violation_rate,
            "effective_explored_radius": radius,
            "coverage_ratio": coverage,
            "success": bool(coverage >= float(self.cfg("success_coverage", 0.35)) and radius >= float(self.cfg("success_radius", 0.45)) and violation_rate <= float(self.cfg("max_violation_rate", 0.1))),
        }
