from __future__ import annotations

import numpy as np
import yaml

from envs.waypoint.safety import SafetyLayer
from envs.waypoint.world import MissionWorld
from tasks.waypoint import SCENARIO_REGISTRY, build_waypoint_scenario


def _world_and_config():
    with open("configs/env/waypoint_missions.yaml", "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config["num_agents"] = 3
    rng = np.random.default_rng(5)
    world = MissionWorld.from_config(config, rng=rng)
    world.sensor_radius = float(config["sensor_radius"])
    world.comm_radius = float(config["comm_radius"])
    world.reset_common(3, rng=rng)
    return world, config, rng


def test_each_waypoint_scenario_generates_valid_candidates_and_field():
    for name in SCENARIO_REGISTRY:
        world, config, rng = _world_and_config()
        scenario = build_waypoint_scenario(name, config.get("tasks", config))
        state = scenario.reset(rng, world)
        field = scenario.build_task_field(world, state)
        candidates, mask = scenario.generate_candidate_waypoints(world, state)
        candidates, mask = SafetyLayer().filter_candidates(world, candidates, mask, require_connectivity=name == "connectivity_expansion")
        candidate_count = int(config["tasks"].get("candidate_count", 12))
        assert field.shape == (5, world.grid_size, world.grid_size)
        assert candidates.shape == (3, candidate_count, 2)
        assert mask.shape == (3, candidate_count)
        assert mask.any(axis=1).all()
        assert world.check_geofence(candidates).all()


def test_no_fly_candidate_is_masked_and_fallback_exists():
    world, config, _ = _world_and_config()
    no_fly_center = np.asarray(config["no_fly_zones"][0]["center"], dtype=np.float32)
    candidate_count = int(config["tasks"].get("candidate_count", 12))
    candidates = np.repeat(no_fly_center[None, None, :], 3 * candidate_count, axis=0).reshape(3, candidate_count, 2)
    raw_mask = np.ones((3, candidate_count), dtype=bool)
    filtered, mask = SafetyLayer().filter_candidates(world, candidates, raw_mask)
    assert mask.any(axis=1).all()
    valid_points = filtered[mask]
    assert world.check_no_fly(valid_points).all()


def test_final_waypoint_filter_replaces_no_fly_waypoint_with_fallback():
    world, config, _ = _world_and_config()
    no_fly_center = np.asarray(config["no_fly_zones"][0]["center"], dtype=np.float32)
    selected = np.repeat(no_fly_center[None, :], 3, axis=0)
    filtered, valid, info = SafetyLayer().filter_waypoints(world, selected)
    assert not valid.all()
    assert info["no_fly_violation_count"] >= 1
    assert world.check_no_fly(filtered).all()
