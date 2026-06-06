from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algorithms.mappo_waypoint import MAPPOWaypointTrainer
from envs.waypoint_marl_env import WaypointMultiUAVEnv
from policies import build_policy
from utils.waypoint_evaluation import greedy_waypoint_action, random_waypoint_action, write_episode_media
from utils.waypoint_vector_env import make_waypoint_env_batch


def load_yaml(path: str | Path) -> dict:
    with open(ROOT / Path(path), "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def sync_policy_env_config(train_config: dict, env_config: dict) -> None:
    for key in ["map_size", "grid_size", "max_waypoint_distance", "min_uav_distance", "base_position", "no_fly_zones"]:
        if key in env_config:
            train_config[key] = env_config[key]
    if "max_delta" not in train_config and "max_waypoint_distance" in env_config:
        train_config["max_delta"] = env_config["max_waypoint_distance"]


def run_baseline(env_config: dict, tasks: list[str], method: str, episodes: int, record_dir: Path | None = None) -> tuple[float, dict, list[str]]:
    rng = np.random.default_rng(int(env_config.get("seed", 0)) + (17 if method == "greedy" else 11))
    returns = []
    metrics = []
    gif_paths = []
    for episode_idx in range(int(episodes)):
        cfg = deepcopy(env_config)
        cfg["task_name"] = tasks[episode_idx % len(tasks)]
        cfg["task_names"] = [cfg["task_name"]]
        env = WaypointMultiUAVEnv(cfg)
        env.reset(seed=int(cfg.get("seed", 0)) + episode_idx)
        done = False
        total = 0.0
        frames = [env.render("rgb_array")] if record_dir is not None and episode_idx == 0 else []
        info = {}
        while not done:
            action = greedy_waypoint_action(env) if method == "greedy" else random_waypoint_action(env, rng)
            _, _, terms, truncs, info_by_agent = env.step(action)
            info = next(iter(info_by_agent.values()))
            total += float(info.get("team_reward", 0.0))
            done = bool(any(terms.values()) or any(truncs.values()))
            if frames:
                frames.append(env.render("rgb_array"))
        if frames:
            gif_paths.append(str(write_episode_media(frames, record_dir, f"{method}_baseline_ep{episode_idx:03d}", "gif", 8)))
        returns.append(total)
        metrics.append(info)
        env.close()
    summary = {
        "success_rate": float(np.mean([item.get("success", False) for item in metrics])) if metrics else 0.0,
        "action_validity_rate": float(np.mean([1.0 - item.get("invalid_action_count", 0) / max(env_config.get("num_agents", 1), 1) for item in metrics])) if metrics else 1.0,
        "candidate_mask_empty_count": float(np.sum([item.get("candidate_mask_empty_count", 0) for item in metrics])) if metrics else 0.0,
        "mean_speed": float(np.mean([item.get("mean_speed", 0.0) for item in metrics])) if metrics else 0.0,
        "max_speed_observed": float(np.max([item.get("max_speed_observed", 0.0) for item in metrics])) if metrics else 0.0,
        "mean_control_norm": float(np.mean([item.get("mean_control_norm", 0.0) for item in metrics])) if metrics else 0.0,
        "collision_count": float(np.sum([item.get("collision_count", 0.0) for item in metrics])) if metrics else 0.0,
        "no_fly_violation_count": float(np.sum([item.get("no_fly_violation_count", 0.0) for item in metrics])) if metrics else 0.0,
        "geofence_violation_count": float(np.sum([item.get("geofence_violation_count", 0.0) for item in metrics])) if metrics else 0.0,
        "adapter_finite": bool(all(item.get("adapter_info", {}).get("adapter_finite", True) for item in metrics)) if metrics else True,
        "dynamics_finite": bool(all(item.get("dynamics_info", {}).get("dynamics_finite", True) for item in metrics)) if metrics else True,
    }
    return float(np.mean(returns)), summary, gif_paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/policy/mappo_waypoint_debug.yaml")
    parser.add_argument("--env-config", default="configs/env/waypoint_missions.yaml")
    parser.add_argument("--tasks", nargs="+", default=["area_coverage", "belief_search", "connectivity_expansion"])
    parser.add_argument("--policy-class", choices=["candidate_selection_waypoint", "direct_waypoint"], default="candidate_selection_waypoint")
    parser.add_argument("--dynamics-backend", choices=["mpe_particle", "kinematic_point"], default="mpe_particle")
    parser.add_argument("--action-adapter", choices=["waypoint_velocity_tracker", "waypoint_pd_tracker"], default="waypoint_velocity_tracker")
    parser.add_argument("--num_agents", type=int, default=3)
    parser.add_argument("--total_updates", type=int, default=5)
    parser.add_argument("--eval_episodes", type=int, default=2)
    parser.add_argument("--record_eval_episodes", type=int, default=1)
    parser.add_argument("--record_format", choices=["gif", "mp4"], default="gif")
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--run_timestamp", default=None)
    args = parser.parse_args()

    env_config = load_yaml(args.env_config)
    train_config = load_yaml(args.config)
    env_config["num_agents"] = int(args.num_agents)
    env_config["task_names"] = list(args.tasks)
    env_config["task_name"] = None
    env_config["action_mode"] = "waypoint"
    env_config["action_interface"] = "waypoint"
    env_config.setdefault("action_adapter", {})["name"] = str(args.action_adapter)
    env_config.setdefault("dynamics_backend", {})["name"] = str(args.dynamics_backend)
    env_config["execution_model"] = str(args.dynamics_backend)
    train_config["total_updates"] = int(args.total_updates)
    train_config["eval_interval"] = max(1, int(args.total_updates))
    train_config["policy_class"] = str(args.policy_class)
    sync_policy_env_config(train_config, env_config)
    output_root = ROOT / "outputs" / "debug" / "waypoint_mappo_validation" / str(args.run_timestamp or "latest")
    output_root.mkdir(parents=True, exist_ok=True)

    random_return, random_summary, random_gifs = run_baseline(env_config, args.tasks, "random", 2, output_root / "media")
    greedy_return, greedy_summary, greedy_gifs = run_baseline(env_config, args.tasks, "greedy", 2, output_root / "media")

    env_batch = make_waypoint_env_batch(env_config, int(train_config.get("num_envs", 2)), backend="sync")
    policy = build_policy(train_config, env_batch.envs[0].global_observation_space, env_batch.envs[0].action_space_n)
    trainer = MAPPOWaypointTrainer(env_batch, policy, train_config)
    with torch.no_grad():
        sample_output = policy.get_action_and_value({key: torch.as_tensor(value, dtype=torch.float32, device=trainer.device) for key, value in trainer.current_obs.items()})
    env_action_shape = list(sample_output.env_action.shape)
    history = trainer.train(
        output_root / "mappo_train",
        env_config=env_config,
        task_names=list(args.tasks),
        eval_episodes=int(args.eval_episodes),
        headless=bool(args.headless),
        record_eval_episodes=int(args.record_eval_episodes),
        record_format=args.record_format,
        record_fps=8,
        record_interval=1,
    )
    env_batch.close()
    last = history[-1] if history else {}
    gif_paths = random_gifs + greedy_gifs
    media_dir = last.get("eval_media_dir")
    if media_dir:
        gif_paths.extend(str(path) for path in Path(media_dir).rglob(f"*.{args.record_format}"))
    summary = {
        "random_return": random_return,
        "greedy_return": greedy_return,
        "mappo_return": float(last.get("eval_reward", 0.0)),
        "policy_class": str(args.policy_class),
        "action_interface": str(env_config.get("action_interface", "waypoint")),
        "action_mode": str(env_config.get("action_mode", "waypoint")),
        "action_adapter": str(args.action_adapter),
        "dynamics_backend": str(args.dynamics_backend),
        "env_action_shape": env_action_shape,
        "coverage_ratio": float(last.get("eval_area_coverage_coverage_ratio_mean", last.get("coverage_ratio", 0.0))),
        "searched_probability_mass": float(last.get("eval_belief_search_searched_probability_mass_mean", last.get("searched_probability_mass", 0.0))),
        "weighted_poi_completion": float(last.get("eval_priority_inspection_weighted_poi_completion_mean", 0.0)),
        "connectivity_violation_rate": float(last.get("eval_connectivity_expansion_connectivity_violation_rate_mean", last.get("connectivity_violation_rate", 0.0))),
        "action_validity_rate": float(min(random_summary["action_validity_rate"], greedy_summary["action_validity_rate"], last.get("eval_action_validity_rate", 1.0))),
        "candidate_mask_empty_count": float(random_summary["candidate_mask_empty_count"] + greedy_summary["candidate_mask_empty_count"] + last.get("candidate_mask_empty_count", 0.0)),
        "mean_speed": float(last.get("eval_mean_speed", last.get("mean_speed", max(random_summary["mean_speed"], greedy_summary["mean_speed"])))),
        "max_speed_observed": float(max(random_summary["max_speed_observed"], greedy_summary["max_speed_observed"], last.get("eval_max_speed_observed", last.get("max_speed_observed", 0.0)))),
        "mean_control_norm": float(last.get("eval_mean_control_norm", last.get("mean_control_norm", max(random_summary["mean_control_norm"], greedy_summary["mean_control_norm"])))),
        "collision_count": float(random_summary["collision_count"] + greedy_summary["collision_count"] + last.get("eval_collision_count", last.get("collision_count", 0.0))),
        "no_fly_violation_count": float(random_summary["no_fly_violation_count"] + greedy_summary["no_fly_violation_count"] + last.get("eval_no_fly_violation_count", last.get("no_fly_violation_count", 0.0))),
        "geofence_violation_count": float(random_summary["geofence_violation_count"] + greedy_summary["geofence_violation_count"] + last.get("eval_geofence_violation_count", last.get("geofence_violation_count", 0.0))),
        "adapter_finite": bool(random_summary["adapter_finite"] and greedy_summary["adapter_finite"] and np.isfinite(last.get("mean_control_norm", 0.0))),
        "dynamics_finite": bool(random_summary["dynamics_finite"] and greedy_summary["dynamics_finite"] and np.isfinite(last.get("mean_speed", 0.0))),
        "gif_paths": gif_paths,
    }
    max_speed_limit = float(env_config.get("dynamics_backend", {}).get("max_speed", env_config.get("max_speed", 0.08)))
    summary["passed"] = bool(
        np.isfinite(summary["random_return"])
        and np.isfinite(summary["greedy_return"])
        and np.isfinite(summary["mappo_return"])
        and np.isfinite(float(last.get("policy_loss", 0.0)))
        and np.isfinite(float(last.get("value_loss", 0.0)))
        and summary["action_validity_rate"] >= 0.99
        and summary["max_speed_observed"] <= max_speed_limit + 1e-5
        and summary["adapter_finite"]
        and summary["dynamics_finite"]
        and (summary["candidate_mask_empty_count"] == 0.0 if args.policy_class == "candidate_selection_waypoint" else True)
        and len(gif_paths) >= 1
    )
    (output_root / "validation_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not summary["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
