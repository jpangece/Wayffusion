from __future__ import annotations

import argparse
import csv
import json
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs import CentralizedMultiUAVEnv
from fields.field_utils import grid_to_world
from policies import build_policy, observation_to_tensor
from scripts._common import ensure_dir, load_generic_config, observation_override_from_variant, prepare_env_config
from tasks.goal_nav import GoalNavigationTask
from baselines import make_baseline


def _pairwise_min_distance(points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.0
    distances = []
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            distances.append(float(np.linalg.norm(points[i] - points[j])))
    return float(min(distances)) if distances else 0.0


def _nearest_distances(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    if len(src) == 0 or len(dst) == 0:
        return np.zeros((len(src),), dtype=np.float32)
    return np.linalg.norm(src[:, None, :] - dst[None, :, :], axis=-1).min(axis=1).astype(np.float32)


def _obstacle_points(obstacle_map: np.ndarray, map_size: float) -> np.ndarray:
    obstacle_indices_yx = np.argwhere(np.asarray(obstacle_map) > 0.5)
    if len(obstacle_indices_yx) == 0:
        return np.zeros((0, 2), dtype=np.float32)
    obstacle_indices_xy = obstacle_indices_yx[:, [1, 0]]
    return grid_to_world(obstacle_indices_xy, obstacle_map.shape[0], map_size=map_size)


def _min_obstacle_distance(points: np.ndarray, obstacle_points: np.ndarray, map_size: float) -> float:
    if len(points) == 0:
        return 0.0
    if len(obstacle_points) == 0:
        return float(map_size)
    return float(_nearest_distances(points, obstacle_points).min())


def _alignment_to_goals(positions: np.ndarray, goals: np.ndarray, action: np.ndarray) -> tuple[float, float, float]:
    if len(goals) == 0 or len(positions) == 0:
        return 0.0, 0.0, float(np.linalg.norm(action, axis=-1).mean()) if len(action) else 0.0
    nearest = goals[np.argmin(np.linalg.norm(positions[:, None, :] - goals[None, :, :], axis=-1), axis=1)]
    target_vec = nearest - positions
    action_norm = np.linalg.norm(action, axis=-1)
    target_norm = np.linalg.norm(target_vec, axis=-1)
    valid = target_norm > 1e-6
    if not np.any(valid):
        return 0.0, 0.0, float(action_norm.mean())
    alignment = (action * target_vec).sum(axis=-1) / np.maximum(action_norm * target_norm, 1e-8)
    return float(alignment[valid].mean()), float((alignment[valid] > 0.25).mean()), float(action_norm.mean())


def _initial_episode_features(env: CentralizedMultiUAVEnv) -> dict[str, float]:
    goals = np.asarray(env.current_task_state["goals"], dtype=np.float32)
    positions = np.asarray(env.state["positions"], dtype=np.float32)
    map_size = float(env.runtime_params["map_size"])
    obstacle_map = np.asarray(env.state["obstacle_map"], dtype=np.float32)
    obstacle_points = _obstacle_points(obstacle_map, map_size)
    agent_to_goal = _nearest_distances(positions, goals)
    goal_to_agent = _nearest_distances(goals, positions)
    return {
        "num_goals": float(len(goals)),
        "initial_goal_cost": float(GoalNavigationTask._goal_cost(goals, positions)),
        "initial_agent_to_goal_mean": float(agent_to_goal.mean()) if len(agent_to_goal) else 0.0,
        "initial_agent_to_goal_max": float(agent_to_goal.max()) if len(agent_to_goal) else 0.0,
        "initial_goal_to_agent_mean": float(goal_to_agent.mean()) if len(goal_to_agent) else 0.0,
        "initial_goal_to_agent_max": float(goal_to_agent.max()) if len(goal_to_agent) else 0.0,
        "initial_goal_pair_min": _pairwise_min_distance(goals),
        "initial_agent_pair_min": _pairwise_min_distance(positions),
        "initial_goal_spread": float(np.linalg.norm(goals - goals.mean(axis=0, keepdims=True), axis=-1).mean()) if len(goals) else 0.0,
        "initial_agent_spread": float(np.linalg.norm(positions - positions.mean(axis=0, keepdims=True), axis=-1).mean()) if len(positions) else 0.0,
        "obstacle_fraction": float((obstacle_map > 0.5).mean()),
        "initial_goal_obstacle_min": _min_obstacle_distance(goals, obstacle_points, map_size),
        "initial_agent_obstacle_min": _min_obstacle_distance(positions, obstacle_points, map_size),
    }


def _summarize(rows: list[dict[str, float | int | str]], group_key: str) -> list[dict[str, float | int | str]]:
    groups: dict[str, list[dict[str, float | int | str]]] = {}
    for row in rows:
        groups.setdefault(str(row[group_key]), []).append(row)
    numeric_keys = [
        key
        for key, value in rows[0].items()
        if key != group_key and isinstance(value, (int, float, np.integer, np.floating))
    ]
    summaries: list[dict[str, float | int | str]] = []
    for group, group_rows in sorted(groups.items(), key=lambda item: item[0]):
        summary: dict[str, float | int | str] = {group_key: group, "episodes": len(group_rows)}
        for key in numeric_keys:
            values = np.asarray([float(row[key]) for row in group_rows], dtype=np.float64)
            summary[f"{key}_mean"] = float(values.mean())
            summary[f"{key}_std"] = float(values.std())
        summaries.append(summary)
    return summaries


def _write_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--policy-config", default=None)
    parser.add_argument("--baseline-policy", default=None)
    parser.add_argument("--env-config", default="configs/env/goal_nav.yaml")
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 23])
    parser.add_argument("--episodes", type=int, default=40)
    parser.add_argument("--obs_variant", default="multi_channel_field+task_id")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--label", default="policy")
    args = parser.parse_args()

    base_env_config = prepare_env_config(
        args.env_config,
        tasks=["goal_nav"],
        num_agents=4,
        scaling_mode="fixed_map",
        observation_override=observation_override_from_variant(args.obs_variant),
    )
    build_env = CentralizedMultiUAVEnv(base_env_config)
    device: torch.device | None = None
    if args.baseline_policy:
        policy = make_baseline(str(args.baseline_policy), base_env_config)
        checkpoint_path = None
        policy_config_path = None
    else:
        if not args.checkpoint or not args.policy_config:
            raise ValueError("--checkpoint and --policy-config are required unless --baseline-policy is set")
        policy_config = load_generic_config(args.policy_config)
        policy = build_policy(policy_config, build_env.observation_space, build_env.action_space)
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        policy.load_state_dict(checkpoint["model_state_dict"], strict=False)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        policy.to(device)
        policy.eval()
        checkpoint_path = str(args.checkpoint)
        policy_config_path = str(args.policy_config)

    output_dir = ensure_dir(args.output_dir)
    episode_rows: list[dict[str, float | int | str]] = []

    for eval_seed in [int(seed) for seed in args.seeds]:
        env_config = deepcopy(base_env_config)
        env_config["seed"] = eval_seed
        env = CentralizedMultiUAVEnv(env_config)
        for episode in range(int(args.episodes)):
            reset_seed = eval_seed + 1000 + episode
            observation, _ = env.reset(seed=reset_seed)
            initial_features = _initial_episode_features(env)
            first_action_alignment = 0.0
            first_moving_frac = 0.0
            first_action_norm = 0.0
            alignment_values = []
            moving_values = []
            action_norm_values = []
            total_reward = 0.0
            done = False
            last_info = {}
            step_count = 0
            while not done:
                if args.baseline_policy:
                    action = policy.act(observation)
                else:
                    assert device is not None
                    with torch.no_grad():
                        obs_tensor = observation_to_tensor(observation, device)
                        action = policy.act_deterministic(obs_tensor).squeeze(0).detach().cpu().numpy()
                remaining_goals = env.current_task_state["goals"][~env.current_task_state["goal_reached"]]
                alignment, moving_frac, action_norm = _alignment_to_goals(env.state["positions"], remaining_goals, action)
                if step_count == 0:
                    first_action_alignment = alignment
                    first_moving_frac = moving_frac
                    first_action_norm = action_norm
                alignment_values.append(alignment)
                moving_values.append(moving_frac)
                action_norm_values.append(action_norm)
                observation, reward, terminated, truncated, info = env.step(action)
                total_reward += float(reward)
                last_info = info
                done = bool(terminated or truncated)
                step_count += 1

            episode_rows.append(
                {
                    "label": str(args.label),
                    "eval_seed": eval_seed,
                    "episode": episode,
                    "reset_seed": reset_seed,
                    "success": float(last_info.get("success", False)),
                    "return": total_reward,
                    "goal_coverage_ratio": float(last_info.get("goal_coverage_ratio", 0.0)),
                    "remaining_goal_ratio": float(last_info.get("remaining_goal_ratio", 0.0)),
                    "collision_count": float(last_info.get("collision_count", 0.0)),
                    "collision_rate": float(last_info.get("collision_rate", 0.0)),
                    "obstacle_collision_count": float(env.state.get("obstacle_collision_count", 0.0)),
                    "obstacle_collision_rate": float(
                        env.state.get("obstacle_collision_count", 0.0) / max(step_count * env.num_agents, 1)
                    ),
                    "safety_violation_count": float(env.state.get("safety_violation_count", 0.0)),
                    "path_length": float(last_info.get("path_length", 0.0)),
                    "steps": float(step_count),
                    "first_action_alignment": first_action_alignment,
                    "first_moving_toward_goal_frac": first_moving_frac,
                    "first_action_norm": first_action_norm,
                    "mean_action_alignment": float(np.mean(alignment_values)) if alignment_values else 0.0,
                    "mean_moving_toward_goal_frac": float(np.mean(moving_values)) if moving_values else 0.0,
                    "mean_action_norm": float(np.mean(action_norm_values)) if action_norm_values else 0.0,
                    **initial_features,
                }
            )

    seed_summaries = _summarize(episode_rows, "eval_seed")
    success_rows = [row for row in episode_rows if float(row["success"]) >= 0.5]
    failure_rows = [row for row in episode_rows if float(row["success"]) < 0.5]
    success_failure_rows: list[dict[str, float | int | str]] = []
    for name, rows in (("success", success_rows), ("failure", failure_rows)):
        if not rows:
            continue
        tagged = [dict(row, outcome=name) for row in rows]
        success_failure_rows.extend(_summarize(tagged, "outcome"))

    episode_path = output_dir / f"{args.label}_episodes.csv"
    seed_summary_path = output_dir / f"{args.label}_seed_summary.csv"
    outcome_summary_path = output_dir / f"{args.label}_success_failure_summary.csv"
    _write_csv(episode_path, episode_rows)
    _write_csv(seed_summary_path, seed_summaries)
    _write_csv(outcome_summary_path, success_failure_rows)

    payload = {
        "label": args.label,
        "checkpoint": checkpoint_path,
        "policy_config": policy_config_path,
        "baseline_policy": str(args.baseline_policy) if args.baseline_policy else None,
        "env_config": str(args.env_config),
        "seeds": [int(seed) for seed in args.seeds],
        "episodes_per_seed": int(args.episodes),
        "episode_csv": str(episode_path),
        "seed_summary_csv": str(seed_summary_path),
        "success_failure_summary_csv": str(outcome_summary_path),
        "overall_success_rate": float(np.mean([float(row["success"]) for row in episode_rows])),
        "overall_collision_rate": float(np.mean([float(row["collision_rate"]) for row in episode_rows])),
        "overall_goal_coverage_ratio": float(np.mean([float(row["goal_coverage_ratio"]) for row in episode_rows])),
    }
    summary_path = output_dir / f"{args.label}_summary.json"
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"summary={summary_path}")
    print(f"episodes={episode_path}")
    print(f"seed_summary={seed_summary_path}")
    print(f"outcome_summary={outcome_summary_path}")


if __name__ == "__main__":
    main()
