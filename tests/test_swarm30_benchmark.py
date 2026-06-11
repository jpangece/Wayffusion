from __future__ import annotations

from copy import deepcopy
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch
import yaml

from envs.waypoint.randomization import build_episode_config
from envs.waypoint_marl_env import WaypointMultiUAVEnv
from policies import build_policy
from scripts.benchmarks.swarm30.run_benchmark import Swarm30Benchmark
from tasks.waypoint import build_waypoint_scenario
from utils.swarm30_benchmark import (
    SCALE_PROFILES,
    SWARM30_TASKS,
    build_scaled_env_config,
    normalized_task_progress,
    sync_policy_scale,
)


def _configs():
    with open("configs/env/waypoint_missions.yaml", "r", encoding="utf-8") as handle:
        env_config = yaml.safe_load(handle)
    with open("configs/policy/benchmarks/swarm30_standardized.yaml", "r", encoding="utf-8") as handle:
        policy_config = yaml.safe_load(handle)
    return env_config, policy_config


def test_scale_profiles_keep_physical_capabilities_constant():
    base, _ = _configs()
    configs = [build_scaled_env_config(base, count) for count in SCALE_PROFILES]
    assert [config["map_size"] for config in configs] == [1.0, 1.58, 2.24, 2.74]
    for config in configs:
        assert config["sensor_radius"] == base["sensor_radius"]
        assert config["comm_radius"] == base["comm_radius"]
        assert config["base_comm_radius"] == base["base_comm_radius"]
        assert config["max_speed"] == base["max_speed"]
        assert config["connectivity_action_filter"] is False
        assert config["connectivity_candidate_filter"] is False
        assert config["domain_randomization"]["spawn"]["spatial_units"] == "absolute"
        assert config["domain_randomization"]["no_fly_zones"]["spatial_units"] == "absolute"


def test_randomized_obstacle_size_is_absolute_across_scales():
    base, _ = _configs()
    radii = []
    for count in (4, 30):
        config = build_scaled_env_config(base, count)
        config["domain_randomization"]["no_fly_zones"].update(
            {"min_count": 1, "max_count": 1, "circle_probability": 1.0}
        )
        episode, _ = build_episode_config(config, np.random.default_rng(9))
        radii.append(float(episode["no_fly_zones"][0]["radius"]))
    assert np.isclose(radii[0], radii[1])


def test_dynamic_tasks_scale_to_multiple_targets_without_new_field_channels():
    base, _ = _configs()
    config = build_scaled_env_config(base, 30)
    for task_name in ("dynamic_target_escort", "target_interception"):
        env = WaypointMultiUAVEnv({**deepcopy(config), "task_name": task_name, "task_names": [task_name]})
        env.reset(seed=4)
        assert len(env.task_state["target_positions"]) == SCALE_PROFILES[30].dynamic_targets
        task_field = env.get_global_state()["task_field"]
        assert task_field.shape == (5, 88, 88)
        if task_name == "target_interception":
            assert float(task_field[2].std()) > 0.0
        positions = env.world.get_uav_positions()
        actions = {agent: positions[idx] for idx, agent in enumerate(env.possible_agents)}
        _, _, _, _, infos = env.step(actions)
        info = next(iter(infos.values()))
        assert np.isfinite(info["team_reward"])
        env.close()


def test_direct_policy_checkpoint_shape_is_agent_count_independent():
    base, policy_base = _configs()
    env4_config = build_scaled_env_config(base, 4, ["area_coverage"])
    env30_config = build_scaled_env_config(base, 30, ["area_coverage"])
    env4 = WaypointMultiUAVEnv(env4_config)
    env30 = WaypointMultiUAVEnv(env30_config)
    env4.reset(seed=1)
    env30.reset(seed=1)
    policy4 = build_policy(sync_policy_scale(policy_base, env4_config), env4.global_observation_space, env4.action_space_n)
    policy30 = build_policy(sync_policy_scale(policy_base, env30_config), env30.global_observation_space, env30.action_space_n)
    policy30.load_state_dict(policy4.state_dict(), strict=True)
    obs = {
        key: torch.as_tensor(value, dtype=torch.float32).unsqueeze(0)
        for key, value in env30.get_global_state().items()
    }
    with torch.no_grad():
        output = policy30.get_action_and_value(obs)
    assert output.env_action.shape == (1, 30, 2)
    assert output.logprob.shape == (1, 30)
    assert output.value.shape == (1,)
    env4.close()
    env30.close()


def test_normalized_progress_is_task_specific_and_bounded():
    assert normalized_task_progress("area_coverage", {"coverage_ratio": 0.85}) == 1.0
    assert normalized_task_progress("belief_search", {"searched_probability_mass": 0.425}) == 0.5
    assert normalized_task_progress("priority_inspection", {"weighted_poi_completion": 2.0}) == 1.0
    assert normalized_task_progress(
        "connectivity_expansion",
        {"coverage_ratio": 0.55, "effective_explored_radius": 0.45, "connectivity_violation_rate": 0.0},
    ) == 1.0
    assert set(SWARM30_TASKS) == {
        "area_coverage",
        "belief_search",
        "priority_inspection",
        "connectivity_expansion",
        "dynamic_target_escort",
        "target_interception",
    }


def test_benchmark_job_matrix_matches_protocol(tmp_path: Path):
    args = Namespace(
        runtime="test_runtime",
        benchmark_config="configs/benchmarks/swarm30_v1.yaml",
        env_config="configs/env/waypoint_missions.yaml",
        output_root=str(tmp_path),
        tracks=["standardized_specialists", "tuned_specialists", "generalist"],
        gpus=[0, 1, 2, 3],
        max_parallel=4,
        cpu_threads_per_job=1,
        poll_seconds=1,
        skip_evaluation=True,
        skip_email=True,
        smtp_env_file=".secrets/wayffusion_mail.env",
    )
    jobs = Swarm30Benchmark(args).prepare_training_jobs()
    assert len(jobs) == 39
    assert sum(job.track == "standardized_specialists" for job in jobs) == 18
    assert sum(job.track == "tuned_specialists" for job in jobs) == 18
    assert sum(job.track == "generalist" for job in jobs) == 3
    assert all(job.expected_updates == 2000 for job in jobs if job.track != "generalist")
    assert all(job.expected_updates == 12000 for job in jobs if job.track == "generalist")
    assert all("--tensorboard" in job.command for job in jobs)
    assert all("--train-randomization" in job.command for job in jobs)
