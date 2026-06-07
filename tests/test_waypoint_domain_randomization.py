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
    first_zones = deepcopy(env.world.no_fly_zones)
    first_positions = env.world.get_uav_positions().copy()
    first_pois = np.asarray(env.task_state["pois"]).copy()

    env.reset(seed=17)
    assert env.world.no_fly_zones == first_zones
    np.testing.assert_allclose(env.world.get_uav_positions(), first_positions)
    np.testing.assert_allclose(env.task_state["pois"], first_pois)

    env.reset(seed=18)
    assert env.world.no_fly_zones != first_zones
    assert not np.allclose(env.world.get_uav_positions(), first_positions)


def test_randomized_spawn_and_task_targets_are_outside_no_fly_zones():
    for task_name in ["belief_search", "priority_inspection"]:
        env = WaypointMultiUAVEnv(_config(task_name))
        _, infos = env.reset(seed=29)
        assert 1 <= len(env.world.no_fly_zones) <= 3
        assert env.world.check_no_fly(env.world.get_uav_positions()).all()
        assert infos["uav_0"]["domain_randomization_enabled"] is True
        assert infos["uav_0"]["no_fly_zone_count"] == len(env.world.no_fly_zones)
        if task_name == "belief_search":
            assert env.world.check_no_fly(np.asarray(env.task_state["peak_centers"])).all()
            assert env.world.check_no_fly(np.asarray(env.task_state["target_position"])[None, :]).all()
        else:
            assert env.world.check_no_fly(np.asarray(env.task_state["pois"])).all()


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
    assert env.world.no_fly_zones == config["no_fly_zones"]
    assert env.episode_randomization_info["enabled"] is False
