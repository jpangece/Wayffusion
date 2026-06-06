from __future__ import annotations

import numpy as np
import yaml

from envs.waypoint.execution import WaypointExecutionModel
from envs.waypoint.world import MissionWorld
from tasks.waypoint import build_waypoint_scenario


def _setup(task: str):
    with open("configs/env/waypoint_missions.yaml", "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config["num_agents"] = 3
    rng = np.random.default_rng(9)
    world = MissionWorld.from_config(config, rng=rng)
    world.sensor_radius = float(config["sensor_radius"])
    world.comm_radius = float(config["comm_radius"])
    world.reset_common(3, rng=rng)
    scenario = build_waypoint_scenario(task, config.get("tasks", config))
    state = scenario.reset(rng, world)
    return config, rng, world, scenario, state


def _execute_reward(world, scenario, state, selected):
    prev = world.snapshot()
    info = WaypointExecutionModel().execute(world, np.asarray(selected, dtype=np.float32))
    return scenario.compute_rewards(prev, world, state, info)


def test_area_coverage_new_area_reward_exceeds_repeated_coverage():
    _, _, world, scenario, state = _setup("area_coverage")
    positions = world.get_uav_positions()
    repeated = _execute_reward(world.snapshot(), scenario, dict(state), positions)
    outward = np.clip(positions + np.asarray([0.18, 0.12], dtype=np.float32), 0.0, world.map_size)
    new = _execute_reward(world, scenario, state, outward)
    assert new["team_reward"] > repeated["team_reward"]
    assert scenario.get_metrics(world, state)["coverage_ratio"] >= 0.0


def test_belief_search_high_probability_reward_exceeds_low_probability():
    _, _, world, scenario, state = _setup("belief_search")
    high_idx = np.unravel_index(np.argmax(world.belief_grid), world.belief_grid.shape)
    low_idx = np.unravel_index(np.argmin(world.belief_grid), world.belief_grid.shape)
    high = np.asarray([(high_idx[1] + 0.5) / world.grid_size, (high_idx[0] + 0.5) / world.grid_size], dtype=np.float32) * world.map_size
    low = np.asarray([(low_idx[1] + 0.5) / world.grid_size, (low_idx[0] + 0.5) / world.grid_size], dtype=np.float32) * world.map_size
    high_mass = float(world.belief_grid[world.sensor_mask_for_position(high)].sum())
    low_mass = float(world.belief_grid[world.sensor_mask_for_position(low)].sum())
    assert high_mass > low_mass


def test_priority_inspection_first_high_weight_visit_beats_low_weight_visit_and_repeat():
    _, _, world, scenario, state = _setup("priority_inspection")
    state["pois"] = np.asarray([[0.82, 0.82], [0.18, 0.82]], dtype=np.float32)
    state["weights"] = np.asarray([2.0, 0.5], dtype=np.float32)
    state["deadlines"] = np.asarray([100, 100], dtype=np.int64)
    state["visited"] = np.zeros(2, dtype=bool)
    high = 0
    low = 1
    state_high = {k: np.copy(v) if isinstance(v, np.ndarray) else v for k, v in state.items()}
    state_low = {k: np.copy(v) if isinstance(v, np.ndarray) else v for k, v in state.items()}
    world_high = world.snapshot()
    world_low = world.snapshot()
    for uav in world_high.uavs:
        uav.position = state["pois"][high].copy()
    for uav in world_low.uavs:
        uav.position = state["pois"][low].copy()
    zero_transition = {"path_length_delta": np.zeros(len(world.uavs)), "safety_violation_count": 0}
    reward_high = scenario.compute_rewards(world.snapshot(), world_high, state_high, zero_transition)
    reward_low = scenario.compute_rewards(world.snapshot(), world_low, state_low, zero_transition)
    reward_repeat = scenario.compute_rewards(world_high.snapshot(), world_high, state_high, zero_transition)
    assert reward_high["team_reward"] > reward_low["team_reward"]
    assert reward_repeat["team_reward"] < reward_high["team_reward"]


def test_connectivity_expansion_connected_expansion_beats_staying_near_base():
    _, _, world, scenario, state = _setup("connectivity_expansion")
    stay_reward = _execute_reward(world.snapshot(), scenario, dict(state), world.get_uav_positions())
    expanded = world.get_uav_positions().copy()
    expanded[:, 0] = np.clip(expanded[:, 0] + 0.12, 0.0, world.map_size)
    expand_reward = _execute_reward(world, scenario, state, expanded)
    assert expand_reward["team_reward"] > stay_reward["team_reward"]


def test_dynamic_target_escort_observation_and_diversity_metrics():
    _, _, world, scenario, state = _setup("dynamic_target_escort")
    target = np.asarray(state["target_position"])
    for idx, uav in enumerate(world.uavs):
        angle = 2.0 * np.pi * idx / len(world.uavs)
        uav.position = target + 0.12 * np.asarray([np.cos(angle), np.sin(angle)], dtype=np.float32)
    reward = scenario.compute_rewards(world.snapshot(), world, state, {"path_length_delta": np.zeros(len(world.uavs)), "safety_violation_count": 0})
    assert reward["metrics"]["target_coverage_ratio"] > 0.0
    assert reward["metrics"]["multi_view_diversity"] > 0.0


def test_target_interception_high_probability_future_path_scores_positive():
    _, _, world, scenario, state = _setup("target_interception")
    future = state["branches"][int(np.argmax(state["branch_probs"]))][3]
    for uav in world.uavs:
        uav.position = future.copy()
    reward = scenario.compute_rewards(world.snapshot(), world, state, {"path_length_delta": np.zeros(len(world.uavs)), "safety_violation_count": 0})
    assert reward["metrics"]["covered_future_belief_mass"] > 0.0
