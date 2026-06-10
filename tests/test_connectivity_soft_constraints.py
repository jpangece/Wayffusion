from __future__ import annotations

import csv
from copy import deepcopy

import numpy as np
import torch
import yaml

from algorithms.mappo_waypoint import MAPPOWaypointTrainer, write_metrics_csv
from envs.waypoint.world import MissionWorld
from envs.waypoint_marl_env import WaypointMultiUAVEnv
from policies import build_policy
from tasks.waypoint.connectivity_expansion import ConnectivityExpansionScenario
from utils.waypoint_vector_env import make_waypoint_env_batch


def _load_env_config() -> dict:
    with open("configs/env/waypoint_missions.yaml", "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _connectivity_env_config() -> dict:
    config = _load_env_config()
    config["num_agents"] = 3
    config["max_steps"] = 3
    config["task_name"] = "connectivity_expansion"
    config["task_names"] = ["connectivity_expansion"]
    config["task_sampling_probs"] = {"connectivity_expansion": 1.0}
    config["domain_randomization"] = {"enabled": False}
    config["no_fly_zones"] = []
    config["base_position"] = [0.10, 0.10]
    config["comm_radius"] = 0.20
    config["base_comm_radius"] = 0.20
    config["max_waypoint_distance"] = 1.0
    config["min_uav_distance"] = 0.01
    config["connectivity_action_filter"] = False
    config["connectivity_candidate_filter"] = False
    config["strict_action_validation"] = False
    config.setdefault("action_adapter", {})["max_command_distance"] = 1.0
    config["action_adapter"]["clip_to_geofence"] = True
    config["action_adapter"]["reject_no_fly_waypoints"] = True
    return config


def _world_with_positions(positions: np.ndarray, comm_radius: float = 0.20, base_comm_radius: float = 0.20) -> MissionWorld:
    config = _connectivity_env_config()
    world = MissionWorld.from_config(config, rng=np.random.default_rng(7))
    world.comm_radius = float(comm_radius)
    world.base_comm_radius = float(base_comm_radius)
    world.sensor_radius = float(config["sensor_radius"])
    world.reset_common(len(positions), rng=np.random.default_rng(7), initial_positions=np.asarray(positions, dtype=np.float32))
    return world


def _reward_for_config(world: MissionWorld, task_cfg: dict, transition_info: dict | None = None) -> dict:
    scenario = ConnectivityExpansionScenario({"connectivity_expansion": task_cfg})
    state = scenario.reset(np.random.default_rng(3), world)
    info = {
        "path_length_delta": np.zeros(len(world.uavs), dtype=np.float32),
        "safety_violation_count": 0,
        "no_fly_violation_count": 0,
    }
    info.update(transition_info or {})
    return scenario.compute_rewards(world.snapshot(), world.snapshot(), state, info)


def test_connectivity_action_filter_false_does_not_replace_direct_waypoints():
    config = _connectivity_env_config()
    config["tasks"]["connectivity_expansion"]["action_risk_use_predicted_graph"] = True
    env = WaypointMultiUAVEnv(config)
    initial = np.asarray([[0.10, 0.10], [0.25, 0.10], [0.40, 0.10]], dtype=np.float32)
    env.reset(seed=0, options={"task_name": "connectivity_expansion", "initial_positions": initial})
    actions = {
        "uav_0": np.asarray([0.10, 0.10], dtype=np.float32),
        "uav_1": np.asarray([0.25, 0.10], dtype=np.float32),
        "uav_2": np.asarray([0.75, 0.10], dtype=np.float32),
    }

    _, _, _, _, infos = env.step(actions)
    info = infos["uav_0"]

    assert info["connectivity_action_filter_enabled"] is False
    assert info["connectivity_filter_replacement_count"] == 0
    assert np.allclose(info["raw_waypoints"], np.stack([actions[agent] for agent in env.possible_agents]))
    assert np.allclose(info["safe_waypoints"], info["raw_waypoints"])
    assert info["action_disconnect_risk"] > 0.0
    assert info["predicted_connected_to_base_ratio"] < 1.0
    env.close()


def test_connectivity_margin_penalty_positive_near_disconnect():
    positions = np.asarray([[0.10, 0.10], [0.25, 0.10], [0.44, 0.10]], dtype=np.float32)
    world = _world_with_positions(positions)

    stats = world.connectivity_margin_stats(threshold=0.05)

    assert stats["connectivity_margin_min"] < 0.05
    assert stats["connectivity_margin_penalty"] > 0.0


def test_action_disconnect_risk_positive_for_predicted_disconnect():
    config = _connectivity_env_config()
    config["tasks"]["connectivity_expansion"]["action_risk_use_predicted_graph"] = True
    env = WaypointMultiUAVEnv(config)
    initial = np.asarray([[0.10, 0.10], [0.25, 0.10], [0.40, 0.10]], dtype=np.float32)
    env.reset(seed=0, options={"task_name": "connectivity_expansion", "initial_positions": initial})

    diagnostics = env._action_connectivity_diagnostics(
        np.asarray([[0.10, 0.10], [0.25, 0.10], [0.75, 0.10]], dtype=np.float32)
    )

    assert diagnostics["action_disconnect_risk"] > 0.0
    assert diagnostics["predicted_connectivity_violation"] == diagnostics["action_disconnect_risk"]
    env.close()


def test_phase17_reward_includes_margin_penalty():
    positions = np.asarray([[0.10, 0.10], [0.25, 0.10], [0.44, 0.10]], dtype=np.float32)
    world = _world_with_positions(positions)
    base_cfg = {
        "w_expand": 0.0,
        "w_coverage": 0.0,
        "w_disconnect": 0.0,
        "w_uncovered_approach": 0.0,
        "w_repeat_footprint": 0.0,
        "w_connected_radius_hold": 0.0,
        "w_relay": 0.0,
        "w_distance": 0.0,
        "w_safety": 0.0,
        "connectivity_margin_threshold": 0.05,
    }
    without = _reward_for_config(world, {**base_cfg, "w_connectivity_margin": 0.0})
    with_margin = _reward_for_config(world, {**base_cfg, "w_connectivity_margin": 3.0})
    penalty = with_margin["metrics"]["connectivity_margin_penalty"]

    assert penalty > 0.0
    assert np.isclose(without["team_reward"] - with_margin["team_reward"], 3.0 * penalty)
    assert with_margin["components"]["connectivity_margin"] == -penalty


def test_phase18_reward_includes_action_risk_penalty():
    world = _world_with_positions(np.asarray([[0.10, 0.10], [0.25, 0.10], [0.40, 0.10]], dtype=np.float32))
    base_cfg = {
        "w_expand": 0.0,
        "w_coverage": 0.0,
        "w_disconnect": 0.0,
        "w_connectivity_margin": 0.0,
        "w_uncovered_approach": 0.0,
        "w_repeat_footprint": 0.0,
        "w_connected_radius_hold": 0.0,
        "w_relay": 0.0,
        "w_distance": 0.0,
        "w_safety": 0.0,
    }
    transition = {
        "action_disconnect_risk": 0.5,
        "predicted_connectivity_violation": 0.5,
        "predicted_connected_to_base_ratio": 0.5,
        "unsafe_action_ratio": 0.5,
        "predicted_disconnected_mask": np.asarray([False, False, True]),
    }
    without = _reward_for_config(world, {**base_cfg, "w_action_disconnect_risk": 0.0}, transition)
    with_risk = _reward_for_config(world, {**base_cfg, "w_action_disconnect_risk": 4.0}, transition)

    assert np.isclose(without["team_reward"] - with_risk["team_reward"], 2.0)
    assert with_risk["metrics"]["action_risk_penalty"] == 2.0
    assert with_risk["per_agent_rewards"][2] == without["per_agent_rewards"][2] - 4.0


def test_phase21_connected_coverage_reward_prefers_base_connected_new_coverage():
    prev_world = _world_with_positions(
        np.asarray([[0.10, 0.10], [0.25, 0.10], [0.40, 0.10]], dtype=np.float32),
        comm_radius=0.22,
        base_comm_radius=0.22,
    )
    world = prev_world.snapshot()
    world.uavs[0].position = np.asarray([0.18, 0.10], dtype=np.float32)
    world.uavs[1].position = np.asarray([0.34, 0.10], dtype=np.float32)
    world.uavs[2].position = np.asarray([0.88, 0.88], dtype=np.float32)
    world.update_communication_graph()
    world.accumulate_sensor_footprints()
    scenario = ConnectivityExpansionScenario(
        {
            "connectivity_expansion": {
                "w_expand": 0.0,
                "w_coverage": 0.0,
                "w_connected_new_coverage": 10.0,
                "w_disconnected_new_coverage": 10.0,
                "w_disconnect": 0.0,
                "w_connectivity_margin": 0.0,
                "w_action_disconnect_risk": 0.0,
                "w_uncovered_approach": 0.0,
                "w_repeat_footprint": 0.0,
                "w_connected_radius_hold": 0.0,
                "w_relay": 0.0,
                "w_distance": 0.0,
                "w_safety": 0.0,
            }
        }
    )
    state = scenario.reset(np.random.default_rng(3), prev_world)

    result = scenario.compute_rewards(
        prev_world,
        world,
        state,
        {
            "path_length_delta": np.zeros(len(world.uavs), dtype=np.float32),
            "safety_violation_count": 0,
            "no_fly_violation_count": 0,
        },
    )

    connected_gain = result["metrics"]["connected_new_coverage_gain"]
    disconnected_gain = result["metrics"]["disconnected_new_coverage_gain"]
    assert connected_gain > 0.0
    assert disconnected_gain > 0.0
    assert np.isclose(result["components"]["connected_new_coverage"], connected_gain)
    assert np.isclose(result["components"]["disconnected_new_coverage"], -disconnected_gain)
    assert np.isclose(result["team_reward"], 10.0 * (connected_gain - disconnected_gain))
    assert result["per_agent_rewards"][0] > 0.0
    assert result["per_agent_rewards"][2] < 0.0


def test_phase19_lagrangian_lambda_updates_from_cost_target_error():
    trainer = object.__new__(MAPPOWaypointTrainer)
    trainer.train_config = {
        "connectivity_cost_target": 0.05,
        "lagrangian_lr": 0.01,
        "lambda_cost_min": 0.0,
        "lambda_cost_max": 50.0,
    }
    trainer.use_lagrangian_connectivity_cost = True
    trainer.lambda_connectivity_cost = 1.0

    high = trainer._update_lagrangian_connectivity_cost(0.15)
    low = trainer._update_lagrangian_connectivity_cost(0.00)

    assert np.isclose(high["lambda_connectivity_cost"], 1.001)
    assert low["lambda_connectivity_cost"] < high["lambda_connectivity_cost"]
    assert low["lambda_connectivity_cost"] >= 0.0


def test_phase20_auxiliary_loss_forward_backward():
    config = _connectivity_env_config()
    with open("configs/policy/direct_waypoint_debug.yaml", "r", encoding="utf-8") as handle:
        policy_config = yaml.safe_load(handle)
    policy_config["map_size"] = config["map_size"]
    policy_config["max_delta"] = 0.14
    policy_config["use_connectivity_auxiliary_loss"] = True

    batch = make_waypoint_env_batch(config, 1)
    obs, _ = batch.reset()
    policy = build_policy(policy_config, batch.envs[0].global_observation_space, batch.envs[0].action_space_n)
    obs_t = {key: torch.as_tensor(value, dtype=torch.float32) for key, value in obs.items()}

    output = policy.get_action_and_value(obs_t)
    aux_loss = (
        output.aux["aux_pred_connectivity_violation"].mean()
        + output.aux["aux_pred_coverage_gain"].mean()
    )
    aux_loss.backward()

    aux_grads = [
        param.grad
        for name, param in policy.named_parameters()
        if name.startswith("aux_connectivity_head") or name.startswith("aux_coverage_head")
    ]
    assert aux_grads
    assert all(grad is not None and torch.isfinite(grad).all() for grad in aux_grads)
    batch.close()


def test_new_metrics_write_to_training_metrics_csv(tmp_path):
    path = tmp_path / "training_metrics.csv"
    write_metrics_csv(
        [
            {
                "update": 1,
                "connectivity_margin_penalty": 0.1,
                "action_disconnect_risk_mean": 0.2,
                "lambda_connectivity_cost": 1.2,
                "connectivity_action_filter_enabled": 0.0,
                "connectivity_filter_replacement_count": 0.0,
                "connected_new_coverage_gain": 0.1,
                "disconnected_new_coverage_gain": 0.02,
                "aux_total_loss": 0.3,
            }
        ],
        path,
    )

    with path.open("r", encoding="utf-8", newline="") as handle:
        header = next(csv.reader(handle))

    for key in [
        "connectivity_margin_penalty",
        "action_disconnect_risk_mean",
        "lambda_connectivity_cost",
        "connectivity_action_filter_enabled",
        "connectivity_filter_replacement_count",
        "connected_new_coverage_gain",
        "disconnected_new_coverage_gain",
        "aux_total_loss",
    ]:
        assert key in header


def test_non_connectivity_tasks_do_not_get_connectivity_action_risk_diagnostics():
    base = _connectivity_env_config()
    for task in ["area_coverage", "belief_search", "priority_inspection"]:
        config = deepcopy(base)
        config["task_name"] = task
        config["task_names"] = [task]
        config["task_sampling_probs"] = {task: 1.0}
        config["max_steps"] = 20
        env = WaypointMultiUAVEnv(config)
        obs, _ = env.reset(seed=0, options={"task_name": task})
        actions = {
            agent: env.world.get_uav_positions()[idx].copy()
            for idx, agent in enumerate(env.possible_agents)
        }
        _, _, _, _, infos = env.step(actions)
        info = next(iter(infos.values()))
        assert "action_disconnect_risk" not in info
        assert "predicted_connected_to_base_ratio" not in info
        assert "action_disconnect_risk" not in info.get("task_metrics", {})
        assert info["connectivity_action_filter_enabled"] is False
        assert info["connectivity_filter_replacement_count"] == 0
        assert obs
        env.close()
