from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from policies import build_policy
from utils.waypoint_evaluation import evaluate_waypoint_policy_episodes
from utils.waypoint_vector_env import make_waypoint_env_batch


def test_waypoint_rgb_render_and_gif_record(tmp_path: Path):
    with open("configs/env/waypoint_missions.yaml", "r", encoding="utf-8") as handle:
        env_config = yaml.safe_load(handle)
    with open("configs/policy/mappo_waypoint_debug.yaml", "r", encoding="utf-8") as handle:
        policy_config = yaml.safe_load(handle)
    env_config["num_agents"] = 3
    env_config["max_steps"] = 3
    env_config["task_name"] = "area_coverage"
    env_config["task_names"] = ["area_coverage"]
    batch = make_waypoint_env_batch(env_config, 1)
    batch.reset()
    start_len = len(batch.envs[0].world.uavs[0].trajectory)
    frame = batch.envs[0].render("rgb_array")
    assert frame.dtype == np.uint8
    assert frame.ndim == 3 and frame.shape[-1] == 3
    policy = build_policy(policy_config, batch.envs[0].global_observation_space, batch.envs[0].action_space_n)
    records = evaluate_waypoint_policy_episodes(
        env_config,
        policy,
        episodes=1,
        device=next(policy.parameters()).device,
        task_name="area_coverage",
        record_dir=tmp_path,
        record_episodes=1,
        record_format="gif",
        record_fps=4,
    )
    assert records and "recording_path" in records[0]
    assert Path(records[0]["recording_path"]).exists()
    assert len(batch.envs[0].world.uavs[0].trajectory) >= start_len
    batch.close()


def test_waypoint_low_detail_render_returns_rgb_array():
    with open("configs/env/waypoint_missions.yaml", "r", encoding="utf-8") as handle:
        env_config = yaml.safe_load(handle)
    env_config["num_agents"] = 8
    env_config["max_steps"] = 3
    env_config["task_name"] = "area_coverage"
    env_config["task_names"] = ["area_coverage"]
    env_config.setdefault("rendering", {})["render_detail"] = "low"
    batch = make_waypoint_env_batch(env_config, 1)
    batch.reset()
    frame = batch.envs[0].render("rgb_array")
    assert frame.dtype == np.uint8
    assert frame.ndim == 3 and frame.shape[-1] == 3
    assert frame.shape[0] < 780
    batch.close()
