from __future__ import annotations

import numpy as np
import yaml

from utils.waypoint_vector_env import ProcessWaypointEnvBatch, make_waypoint_env_batch


def test_process_waypoint_env_batch_reset_and_step():
    with open("configs/env/waypoint_missions.yaml", "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config["num_agents"] = 3
    config["max_steps"] = 2
    config["task_name"] = "area_coverage"
    config["task_names"] = ["area_coverage"]
    config["task_sampling_probs"] = {"area_coverage": 1.0}

    batch = make_waypoint_env_batch(config, 2, backend="process")
    try:
        assert isinstance(batch, ProcessWaypointEnvBatch)
        observations, infos = batch.reset(seeds=[7, 8])
        assert observations["all_uav_states"].shape[0] == 2
        assert len(infos) == 2

        positions = observations["all_uav_states"][..., :2]
        actions = np.clip(positions + 0.05, 0.0, 1.0).astype(np.float32)
        step = batch.step(actions)
        assert step.team_rewards.shape == (2,)
        assert step.per_agent_rewards.shape == (2, 3)
        assert step.done_for_reset.shape == (2,)
        assert np.isfinite(step.team_rewards).all()
        assert all(info["dynamics_info"]["uses_real_mpe_core"] for info in step.infos)
    finally:
        batch.close()
