from __future__ import annotations

import numpy as np
import yaml

from envs.waypoint.control import MPEParticleBackend, WaypointVelocityTracker
from envs.waypoint.world import MissionWorld


def _world(seed: int = 11):
    with open("configs/env/waypoint_missions.yaml", "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    rng = np.random.default_rng(seed)
    world = MissionWorld.from_config(config, rng=rng)
    world.sensor_radius = float(config["sensor_radius"])
    world.comm_radius = float(config["comm_radius"])
    world.reset_common(3, rng=rng)
    return world, config


def test_mpe_particle_backend_integrates_finite_continuous_motion():
    world, config = _world()
    backend = MPEParticleBackend(config["dynamics_backend"])
    adapter = WaypointVelocityTracker(config["action_adapter"])
    backend.reset(world)
    adapter.reset(world)
    old_positions = world.get_uav_positions().copy()
    targets = old_positions + np.asarray([0.18, 0.0], dtype=np.float32)
    out = adapter.adapt(world, targets)
    info = backend.step(world, out.controls, out)
    new_positions = world.get_uav_positions()
    assert np.isfinite(new_positions).all()
    assert np.isfinite([uav.velocity for uav in world.uavs]).all()
    assert not np.allclose(new_positions, targets)
    assert info.max_speed_observed <= config["dynamics_backend"]["max_speed"] + 1e-6
    assert np.isfinite(info.path_length_delta).all()
    assert info.substeps == config["dynamics_backend"]["substeps"]
    assert "dynamics_finite" in info.info


def test_mpe_particle_backend_is_deterministic_without_noise():
    world_a, config = _world(seed=19)
    world_b, _ = _world(seed=19)
    backend_a = MPEParticleBackend(config["dynamics_backend"])
    backend_b = MPEParticleBackend(config["dynamics_backend"])
    adapter_a = WaypointVelocityTracker(config["action_adapter"])
    adapter_b = WaypointVelocityTracker(config["action_adapter"])
    backend_a.reset(world_a)
    backend_b.reset(world_b)
    adapter_a.reset(world_a)
    adapter_b.reset(world_b)
    targets_a = world_a.get_uav_positions() + np.asarray([0.15, 0.03], dtype=np.float32)
    targets_b = world_b.get_uav_positions() + np.asarray([0.15, 0.03], dtype=np.float32)
    out_a = adapter_a.adapt(world_a, targets_a)
    out_b = adapter_b.adapt(world_b, targets_b)
    info_a = backend_a.step(world_a, out_a.controls, out_a)
    info_b = backend_b.step(world_b, out_b.controls, out_b)
    assert np.allclose(world_a.get_uav_positions(), world_b.get_uav_positions())
    assert np.allclose(info_a.path_length_delta, info_b.path_length_delta)


def test_mpe_particle_backend_collision_logic_stays_finite():
    world, config = _world()
    base = world.base_position.copy()
    for idx, uav in enumerate(world.uavs):
        uav.position = base + np.asarray([idx * 0.005, 0.0], dtype=np.float32)
        uav.velocity = np.zeros(2, dtype=np.float32)
    backend = MPEParticleBackend(config["dynamics_backend"])
    backend.reset(world)
    controls = np.zeros((len(world.uavs), 2), dtype=np.float32)
    info = backend.step(world, controls)
    assert np.isfinite(world.get_uav_positions()).all()
    assert np.isfinite(info.path_length_delta).all()
    assert info.collision_count >= 0
