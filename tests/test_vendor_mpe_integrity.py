from __future__ import annotations

from pathlib import Path

import numpy as np

from envs.waypoint.control import MPECoreBackend, build_dynamics_backend
from envs.waypoint.world import MissionWorld


def test_vendored_openai_mpe_core_files_exist():
    root = Path("third_party/openai_mpe")
    assert (root / "core.py").exists()
    assert (root / "README.md").exists()
    assert (root / "UPSTREAM_README.md").exists()
    assert (root / "LICENSE").exists()


def test_vendored_openai_mpe_core_exposes_world_and_agent():
    from third_party.openai_mpe import core

    assert hasattr(core, "World")
    assert hasattr(core, "Agent")
    world = core.World()
    agent = core.Agent()
    assert hasattr(world, "step")
    assert hasattr(agent, "state")
    assert hasattr(agent, "action")


def test_default_backend_uses_vendored_core_not_installed_mpe2():
    backend = build_dynamics_backend({"name": "mpe_core"})
    assert isinstance(backend, MPECoreBackend)
    assert backend.source == "third_party_openai_mpe"
    assert backend.mpe_core_module.__name__ == "third_party.openai_mpe.core"


def test_vendored_mpe_world_step_moves_agent():
    from third_party.openai_mpe import core

    world = core.World()
    agent = core.Agent()
    agent.name = "uav_0"
    agent.movable = True
    agent.collide = False
    agent.silent = True
    agent.max_speed = 0.08
    agent.accel = 1.0
    agent.state.p_pos = np.zeros(2, dtype=np.float32)
    agent.state.p_vel = np.zeros(2, dtype=np.float32)
    agent.action.u = np.asarray([1.0, 0.0], dtype=np.float32)
    agent.action.c = np.zeros(0, dtype=np.float32)
    world.agents = [agent]
    world.step()
    assert np.isfinite(agent.state.p_pos).all()
    assert np.isfinite(agent.state.p_vel).all()
    assert np.linalg.norm(agent.state.p_pos) > 0.0


def test_backend_reset_uses_vendored_world_class():
    config = {
        "map_size": 1.0,
        "grid_size": 8,
        "num_agents": 1,
        "dynamics_backend": {"name": "mpe_core", "source": "third_party_openai_mpe"},
    }
    rng = np.random.default_rng(7)
    world = MissionWorld.from_config(config, rng)
    world.reset_common(1, rng)
    backend = MPECoreBackend(config["dynamics_backend"])
    backend.reset(world)
    assert backend.source == "third_party_openai_mpe"
    assert backend.mpe_world.__class__.__module__ == "third_party.openai_mpe.core"
