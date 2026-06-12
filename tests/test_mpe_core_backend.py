from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

import envs.waypoint.control.dynamics_backends as dynamics_module
from envs.waypoint.control import MPECoreBackend, WaypointVelocityTracker, build_dynamics_backend
from envs.waypoint.world import MissionWorld


def _world_and_config():
    with open("configs/env/waypoint_missions.yaml", "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config["num_agents"] = 3
    rng = np.random.default_rng(13)
    world = MissionWorld.from_config(config, rng=rng)
    world.sensor_radius = float(config["sensor_radius"])
    world.comm_radius = float(config["comm_radius"])
    world.reset_common(3, rng=rng)
    return world, config


def _adapter_output(world: MissionWorld, config: dict):
    adapter = WaypointVelocityTracker(config["action_adapter"])
    adapter.reset(world)
    targets = world.get_uav_positions() + np.asarray([0.15, 0.03], dtype=np.float32)
    return adapter.adapt(world, targets)


def test_mpe_core_backend_creates_real_mpe_world_and_agents():
    world, config = _world_and_config()
    backend = MPECoreBackend(config["dynamics_backend"])
    backend.reset(world)
    assert backend.source == "third_party_openai_mpe"
    assert hasattr(backend, "mpe_world")
    assert backend.mpe_world is not None
    assert len(backend.mpe_agents) == 3
    assert backend.mpe_world.agents == backend.mpe_agents
    assert backend.mpe_world.dt == pytest.approx(config["dynamics_backend"]["dt"])
    assert backend.mpe_world.damping == pytest.approx(config["dynamics_backend"]["damping"])
    assert backend.mpe_world.contact_force == pytest.approx(config["dynamics_backend"]["contact_force"])
    assert backend.mpe_agents[0].max_speed == pytest.approx(config["dynamics_backend"]["max_speed"])
    assert backend.mpe_agents[0].size == pytest.approx(config["dynamics_backend"]["agent_size"])


def test_mpe_core_backend_step_calls_world_step_and_syncs_state(monkeypatch):
    world, config = _world_and_config()
    backend = MPECoreBackend(config["dynamics_backend"])
    backend.reset(world)
    out = _adapter_output(world, config)
    calls = {"count": 0}
    original_step = backend.mpe_world.step

    def spy_step():
        calls["count"] += 1
        return original_step()

    monkeypatch.setattr(backend.mpe_world, "step", spy_step)
    info = backend.step(world, out.controls, out, safety_result=None)
    assert calls["count"] == config["dynamics_backend"]["substeps"]
    assert info.info["uses_real_mpe_core"] is True
    assert info.info["mpe_world_step_calls"] == config["dynamics_backend"]["substeps"]
    for uav, agent in zip(world.uavs, backend.mpe_agents):
        assert np.allclose(uav.position, agent.state.p_pos)
        assert np.allclose(uav.velocity, agent.state.p_vel)
    assert np.isfinite(info.path_length_delta).all()
    assert np.isfinite(info.mean_speed)
    assert np.isfinite(info.max_speed_observed)
    assert info.max_speed_observed <= config["dynamics_backend"]["max_speed"] + 1e-6


def test_removed_self_written_backends_are_not_available():
    assert not hasattr(dynamics_module, "MPEParticleBackend")
    assert not hasattr(dynamics_module, "KinematicPointBackend")
    with pytest.raises(ValueError, match="waypoint_behavior_fast"):
        build_dynamics_backend({"name": "mpe_particle"})
    with pytest.raises(ValueError, match="waypoint_behavior_fast"):
        build_dynamics_backend({"name": "kinematic_point"})


def test_mpe_core_source_is_explicit_not_auto_fallback():
    local_backend = build_dynamics_backend({"name": "mpe_core"})
    assert local_backend.source == "third_party_openai_mpe"

    mpe2_backend = build_dynamics_backend({"name": "mpe_core", "source": "mpe2"})
    assert mpe2_backend.source == "mpe2"

    with pytest.raises(ValueError, match="Unsupported MPE core source"):
        build_dynamics_backend({"name": "mpe_core", "source": "missing_source"})


def test_backend_wrapper_does_not_contain_self_written_particle_integration():
    source = Path("envs/waypoint/control/dynamics_backends.py").read_text(encoding="utf-8")
    forbidden = [
        "(1.0 - self.damping) * velocities",
        "positions + velocities *",
        "_collision_forces",
        "contact_force * penetration",
    ]
    for pattern in forbidden:
        assert pattern not in source


def test_mpe_core_backend_finite_with_contact_force_enabled():
    world, config = _world_and_config()
    positions = world.get_uav_positions()
    positions[1] = positions[0] + np.asarray([0.025, 0.0], dtype=np.float32)
    for idx, uav in enumerate(world.uavs):
        uav.position = positions[idx].copy()
    backend = MPECoreBackend(config["dynamics_backend"])
    backend.reset(world)
    out = _adapter_output(world, config)
    info = backend.step(world, out.controls, out, safety_result=None)
    assert np.isfinite(world.get_uav_positions()).all()
    assert np.isfinite(np.asarray([uav.velocity for uav in world.uavs])).all()
    assert np.isfinite(info.path_length_delta).all()
    assert info.info["dynamics_finite"] is True
