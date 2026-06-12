from __future__ import annotations

import copy

import numpy as np
import pytest
import yaml

from envs.waypoint.control import (
    MPECoreBackend,
    WaypointBehaviorFastBackend,
    WaypointBehaviorRealisticBackend,
    WaypointVelocityTracker,
    build_dynamics_backend,
)
from envs.waypoint.control.dynamics_base import DynamicsInfo
from envs.waypoint.world import MissionWorld
from envs.waypoint_marl_env import WaypointMultiUAVEnv


def _base_config() -> dict:
    with open("configs/env/waypoint_missions.yaml", "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config["num_agents"] = 3
    config["max_steps"] = 5
    config["task_name"] = "area_coverage"
    config["task_names"] = ["area_coverage"]
    config["domain_randomization"]["enabled"] = False
    config["no_fly_zones"] = []
    return config


def _world(config: dict, seed: int = 11) -> MissionWorld:
    rng = np.random.default_rng(seed)
    world = MissionWorld.from_config(config, rng=rng)
    world.sensor_radius = float(config["sensor_radius"])
    world.comm_radius = float(config["comm_radius"])
    world.base_comm_radius = float(config["base_comm_radius"])
    initial = np.asarray([[0.10, 0.10], [0.14, 0.10], [0.18, 0.10]], dtype=np.float32)
    world.reset_common(3, rng=rng, initial_positions=initial)
    return world


def _backend_config(name: str) -> dict:
    if name == "mpe_core":
        return {
            "name": "mpe_core",
            "source": "third_party_openai_mpe",
            "dt": 0.1,
            "substeps": 2,
            "damping": 0.25,
            "max_speed": 0.04,
            "agent_size": 0.02,
            "contact_force": 100.0,
            "contact_margin": 0.01,
        }
    if name == "waypoint_behavior_fast":
        return {
            "name": "waypoint_behavior_fast",
            "dt": 1.0,
            "max_speed": 0.04,
            "acceptance_radius": 0.025,
            "agent_size": 0.02,
            "enable_boundary_projection": True,
            "enable_no_fly_projection": True,
        }
    if name == "waypoint_behavior_realistic":
        return {
            "name": "waypoint_behavior_realistic",
            "dt": 1.0,
            "max_speed": 0.04,
            "acceptance_radius": 0.025,
            "agent_size": 0.02,
            "velocity_smoothing": 0.35,
            "speed_scale_range": [0.85, 1.15],
            "tracking_noise_std": 0.005,
            "command_latency_steps": 1,
            "enable_boundary_projection": True,
            "enable_no_fly_projection": True,
        }
    raise AssertionError(name)


def _adapter_output(world: MissionWorld, config: dict):
    adapter = WaypointVelocityTracker(config["action_adapter"])
    adapter.reset(world)
    targets = world.get_uav_positions() + np.asarray([0.12, 0.04], dtype=np.float32)
    return adapter.adapt(world, targets)


def _assert_complete_info(info: DynamicsInfo, backend_name: str, num_agents: int) -> None:
    assert isinstance(info, DynamicsInfo)
    assert info.path_length_delta.shape == (num_agents,)
    assert info.arrived_mask.shape == (num_agents,)
    assert info.rejected_mask.shape == (num_agents,)
    assert np.isfinite(info.path_length_delta).all()
    assert np.isfinite(info.min_pairwise_distance)
    assert np.isfinite(info.mean_speed)
    assert np.isfinite(info.max_speed_observed)
    assert np.isfinite(info.mean_control_norm)
    assert np.isfinite(info.disturbance_norm)
    transition = info.to_transition_info()
    for key in [
        "path_length_delta",
        "arrived_mask",
        "rejected_mask",
        "safety_violation_count",
        "min_pairwise_distance",
        "collision_count",
        "no_fly_violation_count",
        "geofence_violation_count",
        "mean_speed",
        "max_speed_observed",
        "mean_control_norm",
        "disturbance_norm",
        "substeps",
        "dynamics_backend",
        "dynamics_finite",
        "uses_real_mpe_core",
    ]:
        assert key in transition
    assert transition["dynamics_backend"] == backend_name


def test_build_dynamics_backend_supports_three_modes():
    assert isinstance(build_dynamics_backend({"name": "mpe_core"}), MPECoreBackend)
    assert isinstance(build_dynamics_backend({"name": "waypoint_behavior_fast"}), WaypointBehaviorFastBackend)
    assert isinstance(build_dynamics_backend({"name": "waypoint_behavior_realistic"}), WaypointBehaviorRealisticBackend)


@pytest.mark.parametrize("backend_name", ["mpe_core", "waypoint_behavior_fast", "waypoint_behavior_realistic"])
def test_backend_reset_step_returns_complete_dynamics_info(backend_name: str):
    config = _base_config()
    config["dynamics_backend"] = _backend_config(backend_name)
    world = _world(config)
    backend = build_dynamics_backend(config["dynamics_backend"])
    backend.reset(world)
    out = _adapter_output(world, config)
    info = backend.step(world, out.controls, out, safety_result=None)
    _assert_complete_info(info, backend_name, num_agents=3)
    assert world.step_count == 1
    assert world.communication_graph.shape == (4, 4)
    assert world.coverage_grid.sum() > 0.0
    assert len(world.uavs[0].trajectory) >= 1


def test_fast_backend_updates_position_path_step_comm_and_coverage_with_fixed_seed():
    config = _base_config()
    config["dynamics_backend"] = _backend_config("waypoint_behavior_fast")
    world = _world(config, seed=17)
    old_positions = world.get_uav_positions().copy()
    old_coverage = float(world.coverage_grid.sum())
    backend = build_dynamics_backend(config["dynamics_backend"])
    backend.reset(world)
    out = _adapter_output(world, config)
    info = backend.step(world, out.controls, out, safety_result=None)
    assert not np.allclose(world.get_uav_positions(), old_positions)
    assert float(sum(uav.path_length for uav in world.uavs)) == pytest.approx(float(info.path_length_delta.sum()))
    assert world.step_count == 1
    assert world.communication_graph.shape == (4, 4)
    assert float(world.coverage_grid.sum()) >= old_coverage
    assert info.info["uses_real_mpe_core"] is False
    assert info.substeps == 1


def test_realistic_backend_reproducible_with_fixed_seed():
    config = _base_config()
    config["dynamics_backend"] = _backend_config("waypoint_behavior_realistic")
    world_a = _world(config, seed=23)
    world_b = _world(config, seed=23)
    backend_a = build_dynamics_backend(config["dynamics_backend"])
    backend_b = build_dynamics_backend(config["dynamics_backend"])
    backend_a.reset(world_a)
    backend_b.reset(world_b)
    out_a = _adapter_output(world_a, config)
    out_b = _adapter_output(world_b, config)
    info_a = backend_a.step(world_a, out_a.controls, out_a, safety_result=None)
    info_b = backend_b.step(world_b, out_b.controls, out_b, safety_result=None)
    assert np.allclose(world_a.get_uav_positions(), world_b.get_uav_positions())
    assert np.allclose(info_a.path_length_delta, info_b.path_length_delta)
    assert info_a.info["uses_real_mpe_core"] is False
    assert info_a.info["tracking_noise_std"] == pytest.approx(config["dynamics_backend"]["tracking_noise_std"])


@pytest.mark.parametrize("backend_name", ["mpe_core", "waypoint_behavior_fast", "waypoint_behavior_realistic"])
def test_waypoint_env_switches_dynamics_backend_by_config(backend_name: str):
    config = _base_config()
    config["dynamics_backend"] = _backend_config(backend_name)
    env = WaypointMultiUAVEnv(copy.deepcopy(config))
    observations, _ = env.reset(seed=31)
    positions = env.world.get_uav_positions()
    actions = {agent: positions[idx] + np.asarray([0.05, 0.0], dtype=np.float32) for idx, agent in enumerate(env.possible_agents)}
    _, _, _, _, infos = env.step(actions)
    info = infos["uav_0"]
    assert set(observations) == set(env.possible_agents)
    assert info["dynamics_backend"] == backend_name
    assert info["dynamics_info"]["dynamics_backend"] == backend_name
    assert bool(info["dynamics_info"]["uses_real_mpe_core"]) is (backend_name == "mpe_core")
