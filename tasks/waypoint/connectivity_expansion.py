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
        w_expand = float(self.cfg("w_expand", 5.0))
        w_coverage = float(self.cfg("w_coverage", 120.0))
        w_connected_new_coverage = float(self.cfg("w_connected_new_coverage", 0.0))
        w_connected_radius_hold = float(self.cfg("w_connected_radius_hold", 0.0))
        w_connected_spread = float(self.cfg("w_connected_spread", 0.0))
        w_uncovered_approach = float(self.cfg("w_uncovered_approach", 0.0))
        w_relay = float(self.cfg("w_relay", 0.5))
        w_disconnect = float(self.cfg("w_disconnect", 4.0))
        w_connectivity_margin = float(self.cfg("w_connectivity_margin", 0.0))
        w_action_disconnect_risk = float(self.cfg("w_action_disconnect_risk", 0.0))
        w_disconnected_new_coverage = float(self.cfg("w_disconnected_new_coverage", 0.0))
        w_repeat_footprint = float(self.cfg("w_repeat_footprint", 0.0))
        w_distance = float(self.cfg("w_distance", 0.15))
        w_safety = float(self.cfg("w_safety", 2.0))
        prev_radius = self._effective_radius(prev_world)
        radius = self._effective_radius(world)
        radius_gain = max(radius - prev_radius, 0.0)
        prev_cov = prev_world.coverage_ratio()
        coverage = world.coverage_ratio()
        coverage_gain = max(coverage - prev_cov, 0.0)
        connected = world.connected_to_base_mask()
        connected_coverage = self._connected_coverage_breakdown(prev_world, world, connected)
        connected_new_coverage_gain = float(connected_coverage["connected_new_coverage_gain"])
        disconnected_new_coverage_gain = float(connected_coverage["disconnected_new_coverage_gain"])
        connected_new_coverage_cells = float(connected_coverage["connected_new_coverage_cells"])
        disconnected_new_coverage_cells = float(connected_coverage["disconnected_new_coverage_cells"])
        violation = 1.0 - float(np.mean(connected)) if len(connected) else 0.0
        compute_margin = bool(w_connectivity_margin != 0.0 or self.cfg("log_connectivity_margin", False))
        margin_stats = (
            world.connectivity_margin_stats(threshold=float(self.cfg("connectivity_margin_threshold", 0.05)))
            if compute_margin
            else {
                "connectivity_margin_penalty": 0.0,
                "connectivity_margin_penalties": np.zeros(len(world.uavs), dtype=np.float32),
                "connectivity_margin_mean": 0.0,
                "connectivity_margin_min": 0.0,
            }
        )
        margin_penalty = float(margin_stats["connectivity_margin_penalty"])
        margin_penalties = np.asarray(margin_stats["connectivity_margin_penalties"], dtype=np.float32)
        action_disconnect_risk = float(transition_info.get("action_disconnect_risk", 0.0))
        predicted_connected_ratio = float(transition_info.get("predicted_connected_to_base_ratio", 1.0))
        predicted_violation = float(transition_info.get("predicted_connectivity_violation", action_disconnect_risk))
        unsafe_action_ratio = float(transition_info.get("unsafe_action_ratio", action_disconnect_risk))
        predicted_disconnected = np.asarray(
            transition_info.get("predicted_disconnected_mask", np.zeros(len(world.uavs), dtype=bool)),
            dtype=bool,
        )
        connected_ratio = float(np.mean(connected)) if len(connected) else 1.0
        connected_radius_hold = radius * connected_ratio
        uncovered_approach = (
            self._uncovered_approach_gain(prev_world, world)
            if bool(w_uncovered_approach != 0.0 or self.cfg("log_uncovered_approach", False))
            else 0.0
        )
        repeat_footprint_ratio = (
            self._repeat_footprint_ratio(prev_world, world)
            if bool(w_repeat_footprint != 0.0 or self.cfg("log_repeat_footprint", False))
            else 0.0
        )
        if violation > 1e-6:
            task_state["disconnect_steps"] += 1
            task_state["current_disconnect_run"] += 1
        else:
            task_state["current_disconnect_run"] = 0
        task_state["max_disconnect_run"] = max(int(task_state["max_disconnect_run"]), int(task_state["current_disconnect_run"]))
        relay_credit = self._relay_credit(world) if bool(w_relay != 0.0 or self.cfg("log_relay_credit", False)) else 0.0
        connected_spread = (
            self._connected_angular_spread(world, connected)
            if bool(w_connected_spread != 0.0 or self.cfg("log_connected_spread", False))
            else 0.0
        )
        distance = path_length_delta(transition_info)
        safety = safety_penalty_count(transition_info)
        connectivity_cost = self._connectivity_cost(
            violation=violation,
            action_disconnect_risk=action_disconnect_risk,
            margin_penalty=margin_penalty,
        )
        reward = (
            w_expand * radius_gain
            + w_coverage * coverage_gain
            + w_connected_new_coverage * connected_new_coverage_gain
            + w_connected_radius_hold * connected_radius_hold
            + w_connected_spread * connected_spread
            + w_uncovered_approach * uncovered_approach
            + w_relay * relay_credit
            - w_disconnect * violation
            - w_connectivity_margin * margin_penalty
            - w_action_disconnect_risk * action_disconnect_risk
            - w_disconnected_new_coverage * disconnected_new_coverage_gain
            - w_repeat_footprint * repeat_footprint_ratio
            - w_distance * distance
            - w_safety * safety
        )
        per_agent = w_coverage * per_agent_new_coverage(prev_world, world)
        if len(connected_coverage["per_agent_connected_new_coverage"]) == len(per_agent):
            per_agent += w_connected_new_coverage * connected_coverage["per_agent_connected_new_coverage"]
            per_agent -= w_disconnected_new_coverage * connected_coverage["per_agent_disconnected_new_coverage"]
        per_agent += w_expand * radius_gain / max(len(world.uavs), 1)
        if len(margin_penalties) == len(per_agent):
            per_agent -= w_connectivity_margin * margin_penalties
        if len(predicted_disconnected) == len(per_agent):
            per_agent -= w_action_disconnect_risk * predicted_disconnected.astype(np.float32)
        metrics = self._metrics_from_values(
            world=world,
            task_state=task_state,
            connected_ratio=connected_ratio,
            coverage=coverage,
            radius=radius,
            violation_rate=float(task_state["disconnect_steps"] / max(world.step_count, 1)),
        )
        metrics.update(
            {
                "connectivity_violation": float(violation),
                "connectivity_violation_rate": float(task_state["disconnect_steps"] / max(world.step_count, 1)),
                "longest_disconnected_duration": float(task_state["max_disconnect_run"]),
                "relay_utilization_ratio": float(relay_credit / max(len(world.uavs), 1)),
                "effective_explored_radius": float(radius),
                "coverage_gain": float(coverage_gain),
                "connected_new_coverage_gain": float(connected_new_coverage_gain),
                "disconnected_new_coverage_gain": float(disconnected_new_coverage_gain),
                "connected_new_coverage_cells": float(connected_new_coverage_cells),
                "disconnected_new_coverage_cells": float(disconnected_new_coverage_cells),
                "connected_new_coverage_ratio": float(connected_coverage["connected_new_coverage_ratio"]),
                "disconnected_new_coverage_ratio": float(connected_coverage["disconnected_new_coverage_ratio"]),
                "connected_coverage_reward": w_connected_new_coverage * connected_new_coverage_gain,
                "disconnected_coverage_penalty": w_disconnected_new_coverage * disconnected_new_coverage_gain,
                "connected_radius_hold": float(connected_radius_hold),
                "connected_angular_spread": float(connected_spread),
                "connected_to_base_ratio": connected_ratio,
                "connectivity_margin_mean": float(margin_stats["connectivity_margin_mean"]),
                "connectivity_margin_min": float(margin_stats["connectivity_margin_min"]),
                "connectivity_margin_penalty": float(margin_penalty),
                "action_disconnect_risk": float(action_disconnect_risk),
                "action_disconnect_risk_mean": float(action_disconnect_risk),
                "predicted_connected_to_base_ratio": float(predicted_connected_ratio),
                "predicted_connectivity_violation": float(predicted_violation),
                "unsafe_action_ratio": float(unsafe_action_ratio),
                "action_risk_penalty": w_action_disconnect_risk * action_disconnect_risk,
                "connectivity_cost": float(connectivity_cost),
                "raw_task_reward": float(reward),
                "penalized_reward": float(reward),
                "lagrangian_penalty": 0.0,
                "uncovered_approach_gain": float(uncovered_approach),
                "repeat_footprint_ratio": float(repeat_footprint_ratio),
                "coverage_ratio": coverage,
            }
        )
        return {
            "team_reward": float(reward),
            "per_agent_rewards": per_agent.astype(np.float32),
            "components": {
                "expand": radius_gain,
                "coverage": coverage_gain,
                "connected_new_coverage": connected_new_coverage_gain,
                "disconnected_new_coverage": -disconnected_new_coverage_gain,
                "connected_radius_hold": connected_radius_hold,
                "connected_spread": connected_spread,
                "uncovered_approach": uncovered_approach,
                "relay": relay_credit,
                "disconnect": -violation,
                "connectivity_margin": -margin_penalty,
                "action_disconnect_risk": -action_disconnect_risk,
                "repeat_footprint": -repeat_footprint_ratio,
                "distance": -distance,
                "safety": -safety,
            },
            "metrics": metrics,
            "success": bool(metrics["success"]),
        }

    def _connected_coverage_breakdown(
        self,
        prev_world: MissionWorld,
        world: MissionWorld,
        connected: np.ndarray,
    ) -> dict[str, np.ndarray | float]:
        navigable = world.navigable_grid_mask()
        before = prev_world.coverage_grid > 0.0
        denominator = max(float(navigable.sum()), 1.0)
        connected = np.asarray(connected, dtype=bool)
        if not world.uavs:
            empty = np.zeros(0, dtype=np.float32)
            return {
                "connected_new_coverage_cells": 0.0,
                "disconnected_new_coverage_cells": 0.0,
                "connected_new_coverage_gain": 0.0,
                "disconnected_new_coverage_gain": 0.0,
                "connected_new_coverage_ratio": 0.0,
                "disconnected_new_coverage_ratio": 0.0,
                "per_agent_connected_new_coverage": empty,
                "per_agent_disconnected_new_coverage": empty,
            }
        positions = world.get_uav_positions()
        radii = np.asarray([uav.sensor_radius for uav in world.uavs], dtype=np.float32)
        footprints = world.sensor_masks_for_positions(positions, radii) & navigable[None, :, :] & ~before[None, :, :]
        connected_agent = np.zeros(len(world.uavs), dtype=bool)
        connected_agent[: min(len(connected), len(world.uavs))] = connected[: min(len(connected), len(world.uavs))]
        connected_mask = footprints[connected_agent].any(axis=0) if bool(connected_agent.any()) else np.zeros_like(navigable, dtype=bool)
        disconnected_mask = (
            footprints[~connected_agent].any(axis=0)
            if bool((~connected_agent).any())
            else np.zeros_like(navigable, dtype=bool)
        )
        connected_new = float(connected_mask.sum())
        disconnected_only = disconnected_mask & ~connected_mask
        disconnected_new = float(disconnected_only.sum())
        per_agent_footprints = footprints.sum(axis=(1, 2)).astype(np.float32) / denominator
        per_agent_disconnected_cells = (footprints & disconnected_only[None, :, :]).sum(axis=(1, 2)).astype(np.float32) / denominator
        per_agent_connected = np.where(connected_agent, per_agent_footprints, 0.0).astype(np.float32)
        per_agent_disconnected = np.where(connected_agent, 0.0, per_agent_disconnected_cells).astype(np.float32)
        all_new = connected_mask | disconnected_mask
        total_new = max(float(all_new.sum()), 1.0)
        return {
            "connected_new_coverage_cells": connected_new,
            "disconnected_new_coverage_cells": disconnected_new,
            "connected_new_coverage_gain": connected_new / denominator,
            "disconnected_new_coverage_gain": disconnected_new / denominator,
            "connected_new_coverage_ratio": connected_new / total_new,
            "disconnected_new_coverage_ratio": disconnected_new / total_new,
            "per_agent_connected_new_coverage": per_agent_connected,
            "per_agent_disconnected_new_coverage": per_agent_disconnected,
        }

    def _connectivity_cost(
        self,
        violation: float,
        action_disconnect_risk: float,
        margin_penalty: float,
    ) -> float:
        signal = str(self.cfg("cost_signal", self.cfg("connectivity_cost_signal", "connectivity_violation_rate")))
        if signal == "action_disconnect_risk":
            return float(action_disconnect_risk)
        if signal == "connectivity_margin_penalty":
            return float(margin_penalty)
        if signal in {"action_risk_margin", "action_disconnect_risk_plus_margin", "combined"}:
            return float(action_disconnect_risk + margin_penalty)
        return float(violation)

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
        original_connected = world.base_connected_nodes_for_graph(graph)[1:]
        original_count = int(np.count_nonzero(original_connected))
        if original_count <= 1:
            return 0.0
        credit = 0.0
        for remove_idx in range(1, n + 1):
            if not bool(original_connected[remove_idx - 1]):
                continue
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
            if original_count > np.count_nonzero(visited[1:]):
                credit += 1.0
        return credit

    def _connected_angular_spread(self, world: MissionWorld, connected: np.ndarray | None = None) -> float:
        connected = world.connected_to_base_mask() if connected is None else np.asarray(connected, dtype=bool)
        positions = world.get_uav_positions()
        if len(positions) < 2 or np.count_nonzero(connected) < 2:
            return 0.0
        rel = positions[connected] - world.base_position[None, :]
        dist = np.linalg.norm(rel, axis=-1)
        rel = rel[dist > 1e-6]
        if len(rel) < 2:
            return 0.0
        angles = np.arctan2(rel[:, 1], rel[:, 0])
        unit = np.stack([np.cos(angles), np.sin(angles)], axis=-1)
        resultant = np.linalg.norm(unit.mean(axis=0))
        return float(np.clip(1.0 - resultant, 0.0, 1.0))

    def _uncovered_approach_gain(self, prev_world: MissionWorld, world: MissionWorld) -> float:
        navigable = world.navigable_grid_mask()
        uncovered = (prev_world.coverage_grid <= 0.0) & navigable
        if not bool(uncovered.any()) or not world.uavs:
            return 0.0
        points = world.grid_cell_centers()[uncovered]
        scale = max(float(self.cfg("uncovered_approach_scale", 0.24)) * world.map_size, 1e-6)

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
        navigable = world.navigable_grid_mask()
        previously_seen = prev_world.visit_count_grid > 0.0
        positions = world.get_uav_positions()
        radii = np.asarray([uav.sensor_radius for uav in world.uavs], dtype=np.float32)
        masks = world.sensor_masks_for_positions(positions, radii) & navigable[None, :, :]
        denom = np.maximum(masks.sum(axis=(1, 2)).astype(np.float32), 1.0)
        ratios = (masks & previously_seen[None, :, :]).sum(axis=(1, 2)) / denom
        return float(np.mean(ratios)) if ratios.size else 0.0

    def _metrics_from_values(
        self,
        world: MissionWorld,
        task_state: dict,
        connected_ratio: float,
        coverage: float,
        radius: float,
        violation_rate: float,
    ) -> dict:
        success_coverage = self.goal_value(task_state, 0, "success_coverage", 0.35)
        success_radius = self.goal_value(task_state, 1, "success_radius", 0.45)
        max_violation_rate = self.goal_value(task_state, 2, "max_violation_rate", 0.1)
        relaxed_success_coverage = float(self.cfg("relaxed_success_coverage", max(success_coverage - 0.10, 0.0)))
        relaxed_success_radius = float(self.cfg("relaxed_success_radius", max(success_radius - 0.10, 0.0)))
        relaxed_max_violation_rate = float(self.cfg("relaxed_max_violation_rate", min(max_violation_rate + 0.10, 1.0)))
        return {
            "connected_to_base_ratio": float(connected_ratio),
            "connectivity_violation_rate": float(violation_rate),
            "effective_explored_radius": float(radius),
            "coverage_ratio": float(coverage),
            "goal_coverage_target": success_coverage,
            "goal_radius_target": success_radius,
            "goal_violation_budget": max_violation_rate,
            "relaxed_success": bool(
                coverage >= relaxed_success_coverage
                and radius >= relaxed_success_radius
                and violation_rate <= relaxed_max_violation_rate
            ),
            "relaxed_success_coverage": relaxed_success_coverage,
            "relaxed_success_radius": relaxed_success_radius,
            "relaxed_max_violation_rate": relaxed_max_violation_rate,
            "success": bool(
                coverage >= success_coverage
                and radius >= success_radius
                and violation_rate <= max_violation_rate
            ),
        }

    def get_metrics(self, world: MissionWorld, task_state: dict) -> dict:
        connected_ratio = float(np.mean(world.connected_to_base_mask())) if world.uavs else 1.0
        coverage = world.coverage_ratio()
        radius = self._effective_radius(world)
        violation_rate = float(task_state.get("disconnect_steps", 0) / max(world.step_count, 1))
        success_coverage = self.goal_value(task_state, 0, "success_coverage", 0.35)
        success_radius = self.goal_value(task_state, 1, "success_radius", 0.45)
        max_violation_rate = self.goal_value(task_state, 2, "max_violation_rate", 0.1)
        relaxed_success_coverage = float(self.cfg("relaxed_success_coverage", max(success_coverage - 0.10, 0.0)))
        relaxed_success_radius = float(self.cfg("relaxed_success_radius", max(success_radius - 0.10, 0.0)))
        relaxed_max_violation_rate = float(self.cfg("relaxed_max_violation_rate", min(max_violation_rate + 0.10, 1.0)))
        return {
            "connected_to_base_ratio": connected_ratio,
            "connectivity_violation_rate": violation_rate,
            "effective_explored_radius": radius,
            "coverage_ratio": coverage,
            "goal_coverage_target": success_coverage,
            "goal_radius_target": success_radius,
            "goal_violation_budget": max_violation_rate,
            "relaxed_success": bool(
                coverage >= relaxed_success_coverage
                and radius >= relaxed_success_radius
                and violation_rate <= relaxed_max_violation_rate
            ),
            "relaxed_success_coverage": relaxed_success_coverage,
            "relaxed_success_radius": relaxed_success_radius,
            "relaxed_max_violation_rate": relaxed_max_violation_rate,
            "success": bool(
                coverage >= success_coverage
                and radius >= success_radius
                and violation_rate <= max_violation_rate
            ),
        }
