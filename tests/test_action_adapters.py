from __future__ import annotations

import numpy as np
import yaml

from envs.waypoint.control import WaypointPDTracker, WaypointVelocityTracker
from envs.waypoint.world import MissionWorld


def _world():
    with open("configs/env/waypoint_missions.yaml", "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    rng = np.random.default_rng(3)
    world = MissionWorld.from_config(config, rng=rng)
    world.sensor_radius = float(config["sensor_radius"])
    world.comm_radius = float(config["comm_radius"])
    world.reset_common(3, rng=rng)
    return world, config


def test_waypoint_velocity_tracker_speed_scaling_and_accel_limit():
    world, config = _world()
    adapter = WaypointVelocityTracker(config["action_adapter"])
    adapter.reset(world)
    positions = world.get_uav_positions()
    far = positions + np.asarray([0.20, 0.0], dtype=np.float32)
    near = positions + np.asarray([0.02, 0.0], dtype=np.float32)
    far_out = adapter.adapt(world, far)
    adapter.reset(world)
    near_out = adapter.adapt(world, near)
    assert np.isfinite(far_out.controls).all()
    assert np.isfinite(far_out.desired_velocities).all()
    assert float(np.linalg.norm(far_out.desired_velocities, axis=-1).max()) <= config["action_adapter"]["max_speed"] + 1e-6
    assert float(np.linalg.norm(far_out.desired_velocities, axis=-1).mean()) > float(np.linalg.norm(near_out.desired_velocities, axis=-1).mean())
    assert float(far_out.control_norms.max()) <= config["action_adapter"]["max_accel"] + 1e-6
    assert far_out.info["meters_per_unit"] == config["action_adapter"]["meters_per_unit"]
    assert np.isfinite(far_out.info["mean_distance_to_waypoint_m"])
    assert np.isfinite(far_out.info["mean_desired_speed_mps"])
    assert 0.0 <= far_out.info["arrival_rate"] <= 1.0
    assert 0.0 <= far_out.info["command_update_rate"] <= 1.0


def test_waypoint_velocity_tracker_hover_inside_acceptance_radius():
    world, config = _world()
    adapter = WaypointVelocityTracker(config["action_adapter"])
    adapter.reset(world)
    out = adapter.adapt(world, world.get_uav_positions())
    assert np.allclose(out.desired_velocities, 0.0, atol=1e-6)
    assert np.isfinite(out.controls).all()
    assert out.info["arrival_rate"] == 1.0


def test_waypoint_adapter_default_scale_matches_env_and_policy_configs():
    _, env_config = _world()
    with open("configs/policy/mappo_waypoint.yaml", "r", encoding="utf-8") as handle:
        policy_config = yaml.safe_load(handle)
    with open("configs/policy/mappo_waypoint_debug.yaml", "r", encoding="utf-8") as handle:
        debug_policy_config = yaml.safe_load(handle)
    with open("configs/policy/direct_waypoint_debug.yaml", "r", encoding="utf-8") as handle:
        direct_policy_config = yaml.safe_load(handle)
    assert env_config["max_waypoint_distance"] == env_config["action_adapter"]["max_command_distance"]
    assert env_config["max_waypoint_distance"] == policy_config["max_waypoint_distance"]
    assert env_config["max_waypoint_distance"] == debug_policy_config["max_waypoint_distance"]
    assert env_config["max_waypoint_distance"] == direct_policy_config["max_delta"]
    assert env_config["max_speed"] == env_config["action_adapter"]["max_speed"]
    assert env_config["max_speed"] == env_config["dynamics_backend"]["max_speed"]
    assert env_config["physical_scale"]["meters_per_unit"] == env_config["action_adapter"]["meters_per_unit"]


def test_waypoint_pd_tracker_direction_damping_and_accel_limit():
    world, config = _world()
    cfg = dict(config["action_adapter"])
    cfg.update({"name": "waypoint_pd_tracker", "kp": 2.0, "kd": 2.8})
    adapter = WaypointPDTracker(cfg)
    adapter.reset(world)
    for uav in world.uavs:
        uav.velocity = np.asarray([0.10, 0.0], dtype=np.float32)
    targets = world.get_uav_positions() + np.asarray([0.10, 0.0], dtype=np.float32)
    out = adapter.adapt(world, targets)
    assert np.isfinite(out.controls).all()
    assert float(out.control_norms.max()) <= cfg["max_accel"] + 1e-6
    assert np.all(out.controls[:, 0] < cfg["kp"] * 0.10)
    for uav in world.uavs:
        uav.velocity = np.asarray([0.30, 0.0], dtype=np.float32)
    adapter.reset(world)
    brake = adapter.adapt(world, world.get_uav_positions())
    assert np.all(brake.controls[:, 0] < 0.0)
