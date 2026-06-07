from __future__ import annotations

from copy import deepcopy

import numpy as np
import yaml

from envs.waypoint_marl_env import WaypointMultiUAVEnv
from envs.waypoint.world import MissionWorld


def _config(task_name: str = "priority_inspection") -> dict:
    with open("configs/env/waypoint_missions.yaml", "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config["num_agents"] = 4
    config["task_name"] = task_name
    config["task_names"] = [task_name]
    return config


def test_domain_randomization_is_reproducible_per_seed_and_varies_across_seeds():
    env = WaypointMultiUAVEnv(_config())
    env.reset(seed=17)
    first_base = env.world.base_position.copy()
    first_zones = deepcopy(env.world.no_fly_zones)
    first_positions = env.world.get_uav_positions().copy()
    first_pois = np.asarray(env.task_state["pois"]).copy()
    first_weights = np.asarray(env.task_state["weights"]).copy()
    first_deadlines = np.asarray(env.task_state["deadlines"]).copy()

    env.reset(seed=17)
    np.testing.assert_allclose(env.world.base_position, first_base)
    assert env.world.no_fly_zones == first_zones
    np.testing.assert_allclose(env.world.get_uav_positions(), first_positions)
    np.testing.assert_allclose(env.task_state["pois"], first_pois)
    np.testing.assert_allclose(env.task_state["weights"], first_weights)
    np.testing.assert_array_equal(env.task_state["deadlines"], first_deadlines)

    env.reset(seed=18)
    assert not np.allclose(env.world.base_position, first_base)
    assert env.world.no_fly_zones != first_zones
    assert not np.allclose(env.world.get_uav_positions(), first_positions)


def test_randomized_spawn_and_task_targets_are_outside_no_fly_zones():
    for task_name in ["belief_search", "priority_inspection"]:
        env = WaypointMultiUAVEnv(_config(task_name))
        _, infos = env.reset(seed=29)
        assert 1 <= len(env.world.no_fly_zones) <= 2
        assert env.world.check_no_fly(env.world.base_position[None, :]).all()
        assert env.world.check_no_fly(env.world.get_uav_positions()).all()
        assert infos["uav_0"]["domain_randomization_enabled"] is True
        assert infos["uav_0"]["no_fly_zone_count"] == len(env.world.no_fly_zones)
        np.testing.assert_allclose(infos["uav_0"]["episode_base_position"], env.world.base_position)
        np.testing.assert_allclose(
            infos["uav_0"]["domain_randomization"]["spawn_positions"],
            env.world.get_uav_positions(),
        )
        if task_name == "belief_search":
            assert env.world.check_no_fly(np.asarray(env.task_state["peak_centers"])).all()
            assert env.world.check_no_fly(np.asarray(env.task_state["target_position"])[None, :]).all()
        else:
            assert env.world.check_no_fly(np.asarray(env.task_state["pois"])).all()


def test_belief_layout_and_target_are_reproducible_with_the_same_seed():
    env = WaypointMultiUAVEnv(_config("belief_search"))
    env.reset(seed=73)
    first_peaks = np.asarray(env.task_state["peak_centers"]).copy()
    first_target = np.asarray(env.task_state["target_position"]).copy()
    first_grid = env.world.belief_grid.copy()

    env.reset(seed=73)
    np.testing.assert_allclose(env.task_state["peak_centers"], first_peaks)
    np.testing.assert_allclose(env.task_state["target_position"], first_target)
    np.testing.assert_allclose(env.world.belief_grid, first_grid)


def test_coverage_ratio_uses_only_navigable_cells():
    config = _config("area_coverage")
    config["domain_randomization"]["enabled"] = False
    world = MissionWorld.from_config(config, rng=np.random.default_rng(3))
    navigable = world.navigable_grid_mask()
    world.coverage_grid.fill(0.0)
    world.coverage_grid[navigable] = 1.0
    assert np.isclose(world.coverage_ratio(), 1.0)


def test_disabling_domain_randomization_preserves_static_layout():
    config = _config("area_coverage")
    config["domain_randomization"]["enabled"] = False
    env = WaypointMultiUAVEnv(config)
    env.reset(seed=3)
    np.testing.assert_allclose(env.world.base_position, config["base_position"])
    assert env.world.no_fly_zones == config["no_fly_zones"]
    assert env.episode_randomization_info["enabled"] is False


def test_domain_randomization_keeps_sensing_and_communication_parameters_fixed():
    config = _config("area_coverage")
    expected = (
        float(config["sensor_radius"]),
        float(config["comm_radius"]),
        float(config["base_comm_radius"]),
    )
    env = WaypointMultiUAVEnv(config)
    for seed in [3, 7, 19]:
        env.reset(seed=seed)
        actual = (env.world.sensor_radius, env.world.comm_radius, env.world.base_comm_radius)
        np.testing.assert_allclose(actual, expected)
        for uav in env.world.uavs:
            assert np.isclose(uav.sensor_radius, expected[0])
            assert np.isclose(uav.comm_radius, expected[1])


def test_randomized_env_reset_and_step_are_valid():
    env = WaypointMultiUAVEnv(_config("area_coverage"))
    observations, infos = env.reset(seed=41)
    assert set(observations) == set(env.possible_agents)
    assert infos["uav_0"]["domain_randomization"]["base_position_randomized"] is True
    actions = {
        agent: env.world.uavs[idx].position.copy()
        for idx, agent in enumerate(env.possible_agents)
    }
    next_obs, rewards, terminations, truncations, step_infos = env.step(actions)
    assert set(next_obs) == set(env.possible_agents)
    assert all(np.isfinite(list(rewards.values())))
    assert set(terminations) == set(env.possible_agents)
    assert set(truncations) == set(env.possible_agents)
    assert step_infos["uav_0"]["domain_randomization"]["base_position"] == env.world.base_position.tolist()
