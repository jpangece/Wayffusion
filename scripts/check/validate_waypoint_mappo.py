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
    }
    return float(np.mean(returns)), summary, gif_paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/policy/mappo_waypoint_debug.yaml")
    parser.add_argument("--env-config", default="configs/env/waypoint_missions.yaml")
    parser.add_argument("--tasks", nargs="+", default=["area_coverage", "belief_search", "connectivity_expansion"])
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
    train_config["total_updates"] = int(args.total_updates)
    train_config["eval_interval"] = max(1, int(args.total_updates))
    output_root = ROOT / "outputs" / "debug" / "waypoint_mappo_validation" / str(args.run_timestamp or "latest")
    output_root.mkdir(parents=True, exist_ok=True)

    random_return, random_summary, random_gifs = run_baseline(env_config, args.tasks, "random", 2, output_root / "media")
    greedy_return, greedy_summary, greedy_gifs = run_baseline(env_config, args.tasks, "greedy", 2, output_root / "media")

    env_batch = make_waypoint_env_batch(env_config, int(train_config.get("num_envs", 2)), backend="sync")
    policy = build_policy(train_config, env_batch.envs[0].global_observation_space, env_batch.envs[0].action_space_n)
    trainer = MAPPOWaypointTrainer(env_batch, policy, train_config)
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
        "coverage_ratio": float(last.get("eval_area_coverage_coverage_ratio_mean", last.get("coverage_ratio", 0.0))),
        "searched_probability_mass": float(last.get("eval_belief_search_searched_probability_mass_mean", last.get("searched_probability_mass", 0.0))),
        "weighted_poi_completion": float(last.get("eval_priority_inspection_weighted_poi_completion_mean", 0.0)),
        "connectivity_violation_rate": float(last.get("eval_connectivity_expansion_connectivity_violation_rate_mean", last.get("connectivity_violation_rate", 0.0))),
        "action_validity_rate": float(min(random_summary["action_validity_rate"], greedy_summary["action_validity_rate"], last.get("eval_action_validity_rate", 1.0))),
        "candidate_mask_empty_count": float(random_summary["candidate_mask_empty_count"] + greedy_summary["candidate_mask_empty_count"] + last.get("candidate_mask_empty_count", 0.0)),
        "gif_paths": gif_paths,
    }
    summary["passed"] = bool(
        np.isfinite(summary["random_return"])
        and np.isfinite(summary["greedy_return"])
        and np.isfinite(summary["mappo_return"])
        and summary["action_validity_rate"] >= 0.99
        and summary["candidate_mask_empty_count"] == 0.0
        and len(gif_paths) >= 1
    )
    (output_root / "validation_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not summary["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
