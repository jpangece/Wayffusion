from __future__ import annotations

from copy import deepcopy
from typing import Any
import warnings

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from gymnasium import spaces
from scipy.optimize import linear_sum_assignment

from envs.collision import obstacle_collision_mask, pairwise_collision_pairs
from envs.dynamics import waypoint_controller
from envs.metrics import compute_intrinsic_score
from envs.rewards import common_reward
from fields.field_utils import (
    accumulate_disks,
    gaussian_map,
    sample_obstacle_map,
    sample_points,
    sample_risk_map,
    world_to_grid,
)
from fields.task_field import CHANNEL_NAMES, adapt_task_field, build_task_field
from tasks import TASK_NAME_TO_ID, TASK_ORDER, TaskSampler


class CentralizedMultiUAVEnv(gym.Env):
    """Centralized multi-UAV benchmark environment.

    The whole swarm is modeled as one Gymnasium agent. A policy observes the
    global task field plus all UAV states, then outputs one joint waypoint delta
    for every UAV. This class intentionally keeps the dynamics numerical and
    lightweight: it is a benchmark environment, not a flight-control simulator.
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, config: dict):
        super().__init__()
        self.config = deepcopy(config)
        self.grid_size = int(self.config["grid_size"])
        self.num_agents = int(self.config["num_agents"])
        self.reference_num_agents = int(self.config.get("reference_num_agents", 4))
        self.scaling_mode = str(self.config.get("scaling_mode", "fixed_map"))
        self.observation_mode = str(self.config.get("observation_mode", "multi_channel_field"))
        self.include_task_id = bool(self.config.get("include_task_id", True))
        self.include_agent_density = bool(self.config.get("include_agent_density", True))
        self.drop_channels = set(self.config.get("drop_channels", []))
        self.task_sampler = TaskSampler(self.config)
        self.rng = np.random.default_rng(int(self.config.get("seed", 0)))
        self.forced_task_name = self.config.get("task_name")
        self.current_task = None
        self.current_task_state: dict[str, Any] | None = None
        self.last_info: dict[str, Any] = {}
        self.last_observation: dict[str, np.ndarray] | None = None
        self.trajectory_history: list[list[np.ndarray]] = [[] for _ in range(self.num_agents)]
        self.state: dict[str, Any] = {}

        # Runtime parameters fold the selected scaling mode into all physical
        # distances and speeds. The public config remains task-agnostic, while
        # the environment state uses the derived map size and radii.
        self.runtime_params = self._derive_runtime_params()
        task_field_shape = self._task_field_shape()
        agent_low = np.tile(
            np.array(
                [0.0, 0.0, -self.runtime_params["max_speed"], -self.runtime_params["max_speed"], 0.0, -1.0],
                dtype=np.float32,
            ),
            (self.num_agents, 1),
        )
        agent_high = np.tile(
            np.array(
                [
                    self.runtime_params["map_size"],
                    self.runtime_params["map_size"],
                    self.runtime_params["max_speed"],
                    self.runtime_params["max_speed"],
                    1.0,
                    1.0,
                ],
                dtype=np.float32,
            ),
            (self.num_agents, 1),
        )
        if self._include_agent_task_targets():
            extra_low = np.tile(np.array([-1.0, -1.0, 0.0, 0.0], dtype=np.float32), (self.num_agents, 1))
            extra_high = np.tile(np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32), (self.num_agents, 1))
            agent_low = np.concatenate([agent_low, extra_low], axis=-1)
            agent_high = np.concatenate([agent_high, extra_high], axis=-1)
        # Observation contract used by all policies:
        # - task_field: spatial task channels [C, H, W]
        # - agents: per-UAV state [x, y, vx, vy, battery, role_id]
        #   optionally followed by task target delta/position hints
        # - task_id: one-hot task family id, optionally zeroed for ablations
        # - global_info: compact scalar progress / safety summary
        self.observation_space = spaces.Dict(
            {
                "task_field": spaces.Box(low=0.0, high=1.0, shape=task_field_shape, dtype=np.float32),
                "agents": spaces.Box(
                    low=agent_low,
                    high=agent_high,
                    shape=agent_low.shape,
                    dtype=np.float32,
                ),
                "task_id": spaces.Box(low=0.0, high=1.0, shape=(len(TASK_ORDER),), dtype=np.float32),
                "global_info": spaces.Box(low=-10.0, high=10.0, shape=(5,), dtype=np.float32),
            }
        )
        # Action is a normalized joint waypoint delta for the centralized
        # controller. The environment rescales it by max_waypoint_step.
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.num_agents, 2), dtype=np.float32)

    def _task_field_shape(self) -> tuple[int, int, int]:
        mode = self._canonical_observation_mode()
        if mode == "single_channel_field":
            return (1, self.grid_size, self.grid_size)
        return (len(CHANNEL_NAMES), self.grid_size, self.grid_size)

    def _canonical_observation_mode(self) -> str:
        if self.observation_mode == "multi_channel":
            return "multi_channel_field"
        if self.observation_mode == "single_channel":
            return "single_channel_field"
        if self.observation_mode == "task_id_only":
            warnings.warn(
                "'task_id_only' is a deprecated alias; use 'no_spatial_field' instead.",
                FutureWarning,
                stacklevel=2,
            )
            return "no_spatial_field"
        return self.observation_mode

    def _include_agent_task_targets(self) -> bool:
        return bool(
            self.config.get("include_task_targets_in_agents", False)
            or self.config.get("include_route_targets_in_agents", False)
        )

    def _derive_runtime_params(self) -> dict[str, float]:
        """Derive scale-dependent simulation constants from the base config.

        `fixed_map` keeps the map size fixed while increasing task density with
        N. `density_preserving` expands the map with sqrt(N/reference_N), so the
        average spatial density stays closer to the reference setting.
        """

        task_count_scale = float(max(self.num_agents, 1) / max(self.reference_num_agents, 1))
        if self.scaling_mode == "density_preserving":
            spatial_scale = float(np.sqrt(max(self.num_agents, 1) / max(self.reference_num_agents, 1)))
        else:
            spatial_scale = 1.0
        map_size = float(self.config["map_size"]) * spatial_scale
        return {
            "spatial_scale": spatial_scale,
            "task_count_scale": task_count_scale,
            "map_size": map_size,
            "max_steps": int(
                round(
                    float(self.config["max_steps"])
                    * (spatial_scale if self.config.get("scale_max_steps_with_map", True) else 1.0)
                )
            ),
            "max_speed": float(self.config["max_speed"]) * spatial_scale,
            "max_waypoint_step": float(self.config["max_waypoint_step"]) * spatial_scale,
            "collision_radius": float(self.config["collision_radius"]) * spatial_scale,
            "goal_radius": float(self.config["goal_radius"]) * spatial_scale,
            "coverage_radius": float(self.config["coverage_radius"]) * spatial_scale,
            "formation_radius": float(self.config["formation_radius"]) * spatial_scale,
            "risk_blob_sigma": float(self.config["risk_blob_sigma"]) * spatial_scale,
        }

    def set_task(self, task_name: str) -> None:
        self.forced_task_name = task_name

    def sample_task(self) -> str:
        if self.forced_task_name:
            return str(self.forced_task_name)
        return self.task_sampler.sample(self.rng).name

    def reset(self, seed: int | None = None, options: dict | None = None):
        """Sample a task, initialize maps/UAV states, and build the first obs."""

        if seed is not None:
            self.rng = np.random.default_rng(seed)
        if options and "task_name" in options:
            self.forced_task_name = options["task_name"]
        self.current_task = self.task_sampler.get(self.forced_task_name) if self.forced_task_name else self.task_sampler.sample(self.rng)
        self.state = self._build_initial_state()
        self.current_task_state = self.current_task.reset(self.rng, self.state)
        self.state["visit_requirement"] = float(self.current_task_state.get("required_visits", 1.0))
        self.trajectory_history = [[pos.copy()] for pos in self.state["positions"]]
        obs = self._build_observation()
        info = self._build_info(
            task_success=False,
            task_metrics=self.current_task.get_metrics(self.current_task_state, self.state),
            reward_components={},
            reward=0.0,
        )
        self.last_observation = obs
        self.last_info = info
        return obs, info

    def _build_initial_state(self) -> dict:
        """Create the mutable episode state shared by dynamics and tasks.

        Task-specific objects such as goals, coverage peaks, or formation slots
        are added by the selected task's `reset(...)`; this method only creates
        common maps, UAV kinematics, and counters.
        """

        obstacle_size_range = [float(x) * self.runtime_params["spatial_scale"] for x in self.config["obstacle_size_range"]]
        obstacle_map = sample_obstacle_map(
            self.rng,
            self.grid_size,
            density=float(self.config["obstacle_density"]),
            size_range=obstacle_size_range,
            map_size=self.runtime_params["map_size"],
            area_scale=self.runtime_params["spatial_scale"] ** 2,
        )
        risk_map = sample_risk_map(
            self.rng,
            self.grid_size,
            blob_count=max(1, int(round(int(self.config["risk_blob_count"]) * self.runtime_params["spatial_scale"]))),
            sigma=self.runtime_params["risk_blob_sigma"],
            map_size=self.runtime_params["map_size"],
        )
        positions = self._sample_free_positions(obstacle_map, self.num_agents)
        visited_map = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        visit_count_map = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        return {
            "num_agents": self.num_agents,
            "grid_size": self.grid_size,
            "dt": float(self.config["dt"]),
            "map_size": self.runtime_params["map_size"],
            "spatial_scale": self.runtime_params["spatial_scale"],
            "task_count_scale": self.runtime_params["task_count_scale"],
            "scaling_mode": self.scaling_mode,
            "max_steps": self.runtime_params["max_steps"],
            "positions": positions.astype(np.float32),
            "velocities": np.zeros((self.num_agents, 2), dtype=np.float32),
            "battery": np.ones((self.num_agents, 1), dtype=np.float32),
            "roles": np.zeros((self.num_agents, 1), dtype=np.float32),
            "waypoints": positions.astype(np.float32).copy(),
            "obstacle_map": obstacle_map.astype(np.float32),
            "risk_map": risk_map.astype(np.float32),
            "visited_map": visited_map.astype(np.float32),
            "visit_count_map": visit_count_map.astype(np.float32),
            "step_coverage_mask": np.zeros_like(visited_map, dtype=np.float32),
            "path_length": 0.0,
            "collision_count": 0,
            "obstacle_collision_count": 0,
            "safety_violation_count": 0,
            "risk_exposure": 0.0,
            "step_count": 0,
            "formation_error_history": [],
            "visit_requirement": 1.0,
        }

    def _sample_free_positions(self, obstacle_map: np.ndarray, num_agents: int) -> np.ndarray:
        """Sample initial UAV positions outside obstacles with relaxed spacing.

        Dense large-N cases may not fit the ideal separation. The staged
        relaxation avoids failing early while still preferring non-overlapping
        starts when the map allows it.
        """

        positions = []
        trials = 0
        base_separation = self.runtime_params["collision_radius"] * 1.8
        while len(positions) < num_agents and trials < 10000:
            candidate = sample_points(self.rng, 1, margin=0.08, map_size=self.runtime_params["map_size"])[0]
            if obstacle_collision_mask(candidate[None, :], obstacle_map, map_size=self.runtime_params["map_size"])[0]:
                trials += 1
                continue
            if trials < 3000:
                separation = base_separation
            elif trials < 6000:
                separation = self.runtime_params["collision_radius"] * 1.1
            elif trials < 8500:
                separation = self.runtime_params["collision_radius"] * 0.7
            else:
                separation = self.runtime_params["collision_radius"] * 0.35
            if any(
                np.linalg.norm(candidate - existing) < separation
                for existing in positions
            ):
                trials += 1
                continue
            positions.append(candidate.astype(np.float32))
        if len(positions) < num_agents:
            while len(positions) < num_agents and trials < 20000:
                candidate = sample_points(self.rng, 1, margin=0.04, map_size=self.runtime_params["map_size"])[0]
                if not obstacle_collision_mask(candidate[None, :], obstacle_map, map_size=self.runtime_params["map_size"])[0]:
                    positions.append(candidate.astype(np.float32))
                trials += 1
        if len(positions) < num_agents:
            raise RuntimeError("Failed to sample free initial positions.")
        return np.stack(positions, axis=0)

    def _snapshot_state(self) -> dict:
        """Copy state before transition so reward functions can compare deltas."""

        snapshot = {}
        for key, value in self.state.items():
            if isinstance(value, np.ndarray):
                snapshot[key] = value.copy()
            elif isinstance(value, list):
                snapshot[key] = list(value)
            else:
                snapshot[key] = value
        return snapshot

    def _compose_channel_maps(self) -> dict[str, np.ndarray]:
        """Build named spatial maps before packing them into fixed channels."""

        positions = self.state["positions"]
        centroid = positions.mean(axis=0, keepdims=True)
        channel_maps = {
            "obstacle": self.state["obstacle_map"],
            "risk": self.state["risk_map"],
            "visited": np.clip(self.state["visited_map"], 0.0, 1.0),
            "agent_density": gaussian_map(
                positions,
                self.grid_size,
                sigma=0.05 * self.runtime_params["spatial_scale"],
                map_size=self.runtime_params["map_size"],
            ),
            "communication_quality": np.clip(
                gaussian_map(
                    centroid,
                    self.grid_size,
                    sigma=0.18 * self.runtime_params["spatial_scale"],
                    map_size=self.runtime_params["map_size"],
                ),
                0.0,
                1.0,
            ),
        }
        # Task modules inject their own semantic channels. For example,
        # goal_nav contributes goal_reward / target_probability, coverage
        # contributes desired_occupancy, and formation contributes templates.
        channel_maps.update(self.current_task.build_field(self.current_task_state, self.state))
        if not self.include_agent_density:
            channel_maps["agent_density"] = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        for channel_name in self.drop_channels:
            if channel_name in channel_maps:
                channel_maps[channel_name] = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        return channel_maps

    def _build_observation(self) -> dict[str, np.ndarray]:
        """Convert mutable simulator state into the policy observation dict."""

        full_task_field = build_task_field(self._compose_channel_maps(), self.grid_size)
        mode = self._canonical_observation_mode()
        task_field = adapt_task_field(
            full_task_field,
            "no_spatial_field" if mode == "no_spatial_field" else ("single_channel" if mode == "single_channel_field" else "multi_channel"),
            weights=self.config.get("single_channel_weights"),
        )
        agents = np.concatenate(
            [self.state["positions"], self.state["velocities"], self.state["battery"], self.state["roles"]],
            axis=-1,
        ).astype(np.float32)
        if self._include_agent_task_targets():
            route_targets = self._agent_task_targets()
            map_size = max(float(self.runtime_params["map_size"]), 1e-6)
            target_delta = np.clip((route_targets - self.state["positions"]) / map_size, -1.0, 1.0)
            target_norm = np.clip(route_targets / map_size, 0.0, 1.0)
            agents = np.concatenate([agents, target_delta, target_norm], axis=-1).astype(np.float32)
        task_id = np.zeros((len(TASK_ORDER),), dtype=np.float32)
        if self.include_task_id:
            task_id[TASK_NAME_TO_ID[self.current_task.name]] = 1.0
        completed_ratio = self._completed_ratio()
        # Keep scalar features normalized where possible. map_size remains raw
        # because policies need to know the coordinate scale in density-preserved
        # evaluation where maps can be larger than the unit square.
        global_info = np.array(
            [
                self.state["step_count"] / max(int(self.state["max_steps"]), 1),
                self.state["collision_count"] / max(self.state["step_count"], 1),
                self.state["risk_exposure"] / max(self.state["step_count"], 1),
                completed_ratio,
                self.runtime_params["map_size"],
            ],
            dtype=np.float32,
        )
        self.full_task_field = full_task_field
        return {
            "task_field": task_field.astype(np.float32),
            "agents": agents,
            "task_id": task_id,
            "global_info": global_info,
        }

    def _assign_targets_to_agents(self, targets: np.ndarray) -> np.ndarray:
        positions = np.asarray(self.state["positions"], dtype=np.float32)
        targets = np.asarray(targets, dtype=np.float32)
        if len(positions) == 0 or len(targets) == 0:
            return positions.copy()
        distances = np.linalg.norm(positions[:, None, :] - targets[None, :, :], axis=-1)
        assigned = positions.copy()
        if len(targets) >= len(positions):
            rows, cols = linear_sum_assignment(distances)
            assigned[rows] = targets[cols]
        else:
            target_rows, agent_cols = linear_sum_assignment(distances.T)
            assigned[agent_cols] = targets[target_rows]
        return assigned.astype(np.float32)

    def _agent_task_targets(self) -> np.ndarray:
        positions = np.asarray(self.state["positions"], dtype=np.float32)
        if not isinstance(self.current_task_state, dict) or self.current_task is None:
            return positions.copy()
        task_name = self.current_task.name
        if task_name == "coverage":
            route_targets = self.current_task_state.get("route_hint_targets")
            if route_targets is not None and np.asarray(route_targets).shape == positions.shape:
                return np.asarray(route_targets, dtype=np.float32)
            return positions.copy()
        if task_name in {"goal_nav", "risk_nav"}:
            goals = np.asarray(self.current_task_state.get("goals", np.zeros((0, 2), dtype=np.float32)), dtype=np.float32)
            reached = np.asarray(self.current_task_state.get("goal_reached", np.zeros((len(goals),), dtype=bool)), dtype=bool)
            remaining = goals[~reached] if len(reached) == len(goals) else goals
            return self._assign_targets_to_agents(remaining)
        if task_name == "formation":
            slots = np.asarray(self.current_task_state.get("slots", np.zeros((0, 2), dtype=np.float32)), dtype=np.float32)
            return self._assign_targets_to_agents(slots)
        return positions.copy()

    def _completed_ratio(self) -> float:
        metrics = self.current_task.get_metrics(self.current_task_state, self.state)
        for key in ("goal_coverage_ratio", "coverage_ratio", "success", "task_success_rate"):
            if key in metrics:
                return float(metrics[key])
        if "formation_error" in metrics:
            return float(np.clip(1.0 - metrics["formation_error"] / max(0.3 * self.runtime_params["spatial_scale"], 1e-6), 0.0, 1.0))
        return 0.0

    def step(self, action: np.ndarray):
        """Advance one centralized-control environment step.

        The policy proposes waypoint deltas for all UAVs at once. The simulator
        then applies a simple proportional waypoint controller, resolves
        collisions by reverting unsafe moves, updates shared task maps/counters,
        and finally combines task-specific reward with common safety costs.
        """

        prev_state = self._snapshot_state()
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, self.action_space.low, self.action_space.high)
        deltas = action * self.runtime_params["max_waypoint_step"]
        waypoints = np.clip(self.state["positions"] + deltas, 0.0, self.runtime_params["map_size"])

        # Let task logic update moving targets or other task-side state before
        # dynamics and reward evaluation for this transition.
        self.current_task.step_update(self.current_task_state, self.state)
        proposed_positions, proposed_velocities = waypoint_controller(
            self.state["positions"],
            waypoints,
            kp=float(self.config["kp"]),
            max_speed=self.runtime_params["max_speed"],
            dt=float(self.config["dt"]),
            map_size=self.runtime_params["map_size"],
        )
        # Obstacle and inter-agent collisions are handled conservatively: any
        # unsafe proposed move is rejected for the affected UAV(s), and the
        # corresponding counters feed common reward penalties and metrics.
        obstacle_mask = obstacle_collision_mask(
            proposed_positions,
            self.state["obstacle_map"],
            map_size=self.runtime_params["map_size"],
        )
        resolved_positions = proposed_positions.copy()
        resolved_velocities = proposed_velocities.copy()
        resolved_positions[obstacle_mask] = self.state["positions"][obstacle_mask]
        resolved_velocities[obstacle_mask] = 0.0

        pairs = pairwise_collision_pairs(resolved_positions, self.runtime_params["collision_radius"])
        colliding_agents = sorted({idx for pair in pairs for idx in pair})
        if colliding_agents:
            resolved_positions[colliding_agents] = self.state["positions"][colliding_agents]
            resolved_velocities[colliding_agents] = 0.0

        movement = np.linalg.norm(resolved_positions - self.state["positions"], axis=-1)
        self.state["positions"] = resolved_positions.astype(np.float32)
        self.state["velocities"] = resolved_velocities.astype(np.float32)
        self.state["waypoints"] = waypoints.astype(np.float32)
        self.state["path_length"] += float(movement.sum())
        self.state["collision_count"] += len(pairs)
        self.state["obstacle_collision_count"] += int(obstacle_mask.sum())
        self.state["step_count"] += 1

        # Coverage-style state is maintained for all tasks because it is both a
        # task signal and a useful observation channel for multi-task policies.
        step_coverage_mask = accumulate_disks(
            self.state["positions"],
            self.runtime_params["coverage_radius"],
            self.grid_size,
            map_size=self.runtime_params["map_size"],
        )
        self.state["step_coverage_mask"] = step_coverage_mask.astype(np.float32)
        self.state["visit_count_map"] += step_coverage_mask
        visit_requirement = max(float(self.state.get("visit_requirement", 1.0)), 1.0)
        self.state["visited_map"] = np.clip(self.state["visit_count_map"] / visit_requirement, 0.0, 1.0)

        # Risk exposure is sampled at the resolved positions, so rejected unsafe
        # moves do not accidentally accrue exposure at unreachable positions.
        risk_samples = self._sample_map_values(self.state["risk_map"], self.state["positions"])
        step_risk_exposure = float(risk_samples.sum())
        step_safety_violations = int((risk_samples >= float(self.config["no_fly_threshold"])).sum())
        self.state["risk_exposure"] += step_risk_exposure
        self.state["safety_violation_count"] += step_safety_violations
        for idx, pos in enumerate(self.state["positions"]):
            self.trajectory_history[idx].append(pos.copy())

        transition_info = {
            "pair_collision_count": len(pairs),
            "obstacle_collision_count": int(obstacle_mask.sum()),
            "path_length_delta": float(movement.sum()),
            "step_risk_exposure": step_risk_exposure,
            "step_safety_violations": step_safety_violations,
            "num_agents": self.num_agents,
            "spatial_scale": self.runtime_params["spatial_scale"],
            "max_step_distance": self.runtime_params["max_speed"] * float(self.config["dt"]),
        }
        # Task reward owns goal/coverage/formation/risk objective terms; common
        # reward owns cross-task costs such as collisions, path length, time, and
        # no-fly-zone violations.
        task_result = self.current_task.compute_reward(self.current_task_state, prev_state, self.state, transition_info)
        if "formation_error" in task_result.metrics:
            self.state["formation_error_history"].append(task_result.metrics["formation_error"])
        common_reward_value, common_components = common_reward(self.config, transition_info)
        reward = float(task_result.reward + common_reward_value)
        reward_components = {**task_result.components, **common_components, "total_reward": reward}

        terminated = bool(task_result.success)
        truncated = bool(self.state["step_count"] >= int(self.state["max_steps"]))
        obs = self._build_observation()
        info = self._build_info(
            task_success=terminated,
            task_metrics=task_result.metrics,
            reward_components=reward_components,
            reward=reward,
        )
        self.last_observation = obs
        self.last_info = info
        return obs, reward, terminated, truncated, info

    def _sample_map_values(self, value_map: np.ndarray, positions: np.ndarray) -> np.ndarray:
        indices = world_to_grid(positions, self.grid_size, map_size=self.runtime_params["map_size"])
        return value_map[indices[:, 1], indices[:, 0]].astype(np.float32)

    def _task_targets(self) -> np.ndarray:
        if self.current_task.name in {"goal_nav", "risk_nav"}:
            return self.current_task_state["goals"]
        if self.current_task.name == "formation":
            return self.current_task_state["slots"]
        return np.zeros((0, 2), dtype=np.float32)

    def _build_info(self, task_success: bool, task_metrics: dict, reward_components: dict, reward: float) -> dict:
        """Assemble metrics for logging, evaluation CSVs, and visualizers."""

        collision_rate = self.state["collision_count"] / max(self.state["step_count"] * self.num_agents, 1)
        path_length_per_agent = self.state["path_length"] / max(self.num_agents, 1)
        risk_exposure_per_agent = self.state["risk_exposure"] / max(self.num_agents, 1)
        intrinsic_score = compute_intrinsic_score(
            self.current_task.name,
            {
                **task_metrics,
                "collision_rate": collision_rate,
                "path_length": path_length_per_agent,
                "cumulative_risk_exposure": risk_exposure_per_agent,
            },
            spatial_scale=self.runtime_params["spatial_scale"],
        )
        info = {
            "task_name": self.current_task.name,
            "success": bool(task_success),
            "collision_count": int(self.state["collision_count"]),
            "collision_rate": float(collision_rate),
            "coverage_ratio": float(task_metrics.get("coverage_ratio", 0.0)),
            "formation_error": float(task_metrics.get("formation_error", 0.0)),
            "risk_exposure": float(risk_exposure_per_agent),
            "risk_exposure_total": float(self.state["risk_exposure"]),
            "path_length": float(path_length_per_agent),
            "path_length_total": float(self.state["path_length"]),
            "reward": float(reward),
            "reward_components": reward_components,
            "per_agent_positions": self.state["positions"].copy(),
            "current_waypoints": self.state["waypoints"].copy(),
            "task_specific_metrics": task_metrics,
            "task_targets": self._task_targets().copy(),
            "trajectory_history": [np.asarray(traj, dtype=np.float32) for traj in self.trajectory_history],
            "full_task_field": self.full_task_field.copy(),
            "intrinsic_score": intrinsic_score,
            "normalized_score": intrinsic_score,
            "scaling_mode": self.scaling_mode,
            "map_size": self.runtime_params["map_size"],
            "num_agents": self.num_agents,
        }
        info.update(task_metrics)
        return info

    def get_metrics(self) -> dict:
        return dict(self.last_info)

    def _draw_render_scene(self, ax) -> None:
        map_size = self.runtime_params["map_size"]
        ax.imshow(self.state["risk_map"], origin="lower", cmap="Reds", alpha=0.3, extent=(0, map_size, 0, map_size))
        ax.imshow(self.state["obstacle_map"], origin="lower", cmap="gray_r", alpha=0.4, extent=(0, map_size, 0, map_size))
        for traj in self.trajectory_history:
            traj = np.asarray(traj)
            ax.plot(traj[:, 0], traj[:, 1], linewidth=2)
            ax.scatter(traj[-1, 0], traj[-1, 1], s=40)
        targets = self._task_targets()
        if len(targets) > 0:
            ax.scatter(targets[:, 0], targets[:, 1], marker="*", s=100, color="gold", edgecolors="black")
        ax.set_xlim(0, map_size)
        ax.set_ylim(0, map_size)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"{self.current_task.name} | N={self.num_agents}")

    def _paint_indices(self, image: np.ndarray, indices: np.ndarray, color: tuple[float, float, float], radius: int = 1) -> None:
        if indices.size == 0:
            return
        height, width = image.shape[:2]
        for x_idx, y_idx in indices.astype(int):
            x_min = max(x_idx - radius, 0)
            x_max = min(x_idx + radius + 1, width)
            y_min = max(y_idx - radius, 0)
            y_max = min(y_idx + radius + 1, height)
            image[y_min:y_max, x_min:x_max] = color

    def _render_rgb_array(self) -> np.ndarray:
        image = np.ones((self.grid_size, self.grid_size, 3), dtype=np.float32)
        risk = np.clip(self.state["risk_map"], 0.0, 1.0)
        obstacles = np.clip(self.state["obstacle_map"], 0.0, 1.0)
        image[..., 0] = np.clip(image[..., 0] + 0.55 * risk, 0.0, 1.0)
        image[..., 1] = np.clip(image[..., 1] - 0.35 * risk, 0.0, 1.0)
        image[..., 2] = np.clip(image[..., 2] - 0.35 * risk, 0.0, 1.0)
        obstacle_mask = obstacles > 0.5
        image[obstacle_mask] = np.array([0.12, 0.12, 0.12], dtype=np.float32)

        palette = [
            (0.12, 0.47, 0.71),
            (1.0, 0.5, 0.05),
            (0.17, 0.63, 0.17),
            (0.84, 0.15, 0.16),
            (0.58, 0.4, 0.74),
            (0.55, 0.34, 0.29),
        ]
        for agent_idx, traj in enumerate(self.trajectory_history):
            traj_array = np.asarray(traj, dtype=np.float32)
            if traj_array.size == 0:
                continue
            traj_indices = world_to_grid(traj_array, self.grid_size, map_size=self.runtime_params["map_size"])
            color = palette[agent_idx % len(palette)]
            self._paint_indices(image, traj_indices, color, radius=0)
            self._paint_indices(image, traj_indices[-1:, :], (0.0, 0.0, 0.0), radius=1)

        target_positions = self._task_targets()
        if len(target_positions) > 0:
            target_indices = world_to_grid(target_positions, self.grid_size, map_size=self.runtime_params["map_size"])
            self._paint_indices(image, target_indices, (1.0, 0.84, 0.0), radius=1)

        waypoint_indices = world_to_grid(self.state["waypoints"], self.grid_size, map_size=self.runtime_params["map_size"])
        self._paint_indices(image, waypoint_indices, (0.0, 0.75, 0.75), radius=1)
        return (np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)

    def render(self, mode: str = "human"):
        if mode == "human":
            fig, ax = plt.subplots(figsize=(5, 5))
            self._draw_render_scene(ax)
            plt.show(block=False)
            plt.pause(0.001)
            plt.close(fig)
            return None
        return self._render_rgb_array()
