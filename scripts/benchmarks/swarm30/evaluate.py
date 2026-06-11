from __future__ import annotations

import argparse
import csv
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import yaml


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.waypoint_marl_env import WaypointMultiUAVEnv
from policies import build_policy
from utils.swarm30_benchmark import (
    SWARM30_TASKS,
    build_scaled_env_config,
    normalized_task_progress,
    sync_policy_scale,
)
from utils.waypoint_evaluation import (
    env_config_for_randomization_mode,
    evaluate_waypoint_policy_episodes,
    greedy_waypoint_action,
    random_waypoint_action,
    write_episode_media,
)


def read_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows: list[dict]) -> dict:
    result: dict[str, float | str] = {}
    for key in sorted({key for row in rows for key in row}):
        values = [row[key] for row in rows if key in row]
        numeric = [float(value) for value in values if isinstance(value, (int, float, np.integer, np.floating, bool))]
        if numeric and len(numeric) == len(values):
            result[f"{key}_mean"] = float(np.mean(numeric))
            result[f"{key}_std"] = float(np.std(numeric))
        elif values:
            result[key] = str(values[0])
    return result


def load_policy(checkpoint_path: Path, policy_config_path: Path | None, env_config: dict, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if policy_config_path is not None:
        policy_config = read_yaml(policy_config_path)
    else:
        policy_config = deepcopy(checkpoint.get("train_config", {}))
    policy_config = sync_policy_scale(policy_config, env_config)
    probe = WaypointMultiUAVEnv(env_config)
    probe.reset(seed=int(env_config.get("seed", 0)))
    policy = build_policy(policy_config, probe.global_observation_space, probe.action_space_n)
    policy.load_state_dict(checkpoint["model_state_dict"], strict=True)
    policy.to(device)
    policy.eval()
    probe.close()
    return policy, policy_config


def evaluate_baseline(
    env_config: dict,
    task_name: str,
    episodes: int,
    baseline: str,
    output_dir: Path,
    record_episodes: int,
    record_format: str,
    record_fps: int,
) -> list[dict]:
    env = WaypointMultiUAVEnv(env_config)
    rng = np.random.default_rng(int(env_config.get("seed", 0)) + 9000)
    rows = []
    for episode in range(int(episodes)):
        env.reset(seed=int(env_config.get("seed", 0)) + 4000 + episode, options={"task_name": task_name})
        done = False
        total_return = 0.0
        frames = [env.render("rgb_array")] if episode < record_episodes else []
        info = {}
        steps = 0
        while not done:
            if baseline == "hold":
                positions = env.world.get_uav_positions()
                actions = {agent: positions[idx].copy() for idx, agent in enumerate(env.possible_agents)}
            elif baseline == "random":
                actions = random_waypoint_action(env, rng)
            elif baseline == "greedy":
                actions = greedy_waypoint_action(env)
            else:
                raise ValueError(f"Unsupported baseline: {baseline}")
            _, _, terms, truncs, infos = env.step(actions)
            info = next(iter(infos.values()))
            total_return += float(info.get("team_reward", 0.0))
            done = bool(any(terms.values()) or any(truncs.values()))
            steps += 1
            if episode < record_episodes:
                frames.append(env.render("rgb_array"))
        metrics = dict(info.get("task_metrics", {}))
        row = {
            "episode": episode,
            "return": total_return,
            "success_rate": float(info.get("success", False)),
            "action_validity_rate": float(
                1.0 - info.get("invalid_action_count", 0) / max(steps * env.num_agents, 1)
            ),
            "collision_rate": float(info.get("collision_rate", 0.0)),
            **{
                key: float(value)
                for key, value in metrics.items()
                if isinstance(value, (int, float, np.integer, np.floating, bool))
            },
        }
        row["normalized_progress"] = normalized_task_progress(task_name, row)
        if frames:
            media_path = write_episode_media(
                frames,
                output_dir / "media",
                f"{baseline}_{task_name}_ep{episode:03d}",
                record_format,
                record_fps,
            )
            row["recording_path"] = str(media_path)
        rows.append(row)
    env.close()
    return rows


def evaluate_checkpoint(args: argparse.Namespace) -> list[dict]:
    base_env = read_yaml(ROOT / args.env_config)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    all_summaries = []
    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_absolute():
        checkpoint = ROOT / checkpoint
    policy_config_path = None if args.policy_config is None else ROOT / args.policy_config
    for scale in args.scales:
        for task_name in args.tasks:
            scaled = build_scaled_env_config(base_env, scale, [task_name], randomization=True)
            scaled["seed"] = int(args.seed)
            policy, resolved_policy = load_policy(checkpoint, policy_config_path, scaled, device)
            for mode in args.modes:
                current_env = env_config_for_randomization_mode(scaled, mode)
                output_dir = Path(args.output_dir) / task_name / f"N{scale}" / mode
                output_dir.mkdir(parents=True, exist_ok=True)
                rows = evaluate_waypoint_policy_episodes(
                    current_env,
                    policy,
                    episodes=args.random_episodes if mode == "random" else args.fixed_episodes,
                    device=device,
                    task_name=task_name,
                    deterministic=True,
                    headless=True,
                    record_dir=output_dir / "media",
                    record_episodes=args.record_episodes,
                    record_format=args.record_format,
                    record_fps=args.record_fps,
                    record_prefix=f"{args.track}_{task_name}_N{scale}_{mode}",
                )
                for row in rows:
                    row["normalized_progress"] = normalized_task_progress(task_name, row)
                summary = {
                    "benchmark": "swarm30_v1",
                    "track": args.track,
                    "seed": int(args.seed),
                    "task_name": task_name,
                    "num_agents": int(scale),
                    "randomization_mode": mode,
                    "checkpoint": str(checkpoint),
                    **aggregate(rows),
                }
                summary["hard_filter_valid"] = bool(
                    float(summary.get("connectivity_action_filter_enabled_mean", 0.0)) == 0.0
                    and float(summary.get("connectivity_filter_replacement_count_mean", 0.0)) == 0.0
                )
                write_csv(output_dir / "episodes.csv", rows)
                (output_dir / "summary.json").write_text(
                    json.dumps(summary, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                (output_dir / "resolved_policy.yaml").write_text(
                    yaml.safe_dump(resolved_policy, sort_keys=False),
                    encoding="utf-8",
                )
                all_summaries.append(summary)
    return all_summaries


def evaluate_baselines(args: argparse.Namespace) -> list[dict]:
    base_env = read_yaml(ROOT / args.env_config)
    all_summaries = []
    for scale in args.scales:
        for task_name in args.tasks:
            for mode in args.modes:
                env_config = build_scaled_env_config(base_env, scale, [task_name], randomization=mode == "random")
                env_config["seed"] = int(args.seed)
                for baseline in args.baselines:
                    output_dir = Path(args.output_dir) / baseline / task_name / f"N{scale}" / mode
                    rows = evaluate_baseline(
                        env_config,
                        task_name,
                        args.random_episodes if mode == "random" else args.fixed_episodes,
                        baseline,
                        output_dir,
                        args.record_episodes,
                        args.record_format,
                        args.record_fps,
                    )
                    summary = {
                        "benchmark": "swarm30_v1",
                        "track": f"baseline_{baseline}",
                        "seed": int(args.seed),
                        "task_name": task_name,
                        "num_agents": int(scale),
                        "randomization_mode": mode,
                        **aggregate(rows),
                    }
                    write_csv(output_dir / "episodes.csv", rows)
                    (output_dir / "summary.json").write_text(
                        json.dumps(summary, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8",
                    )
                    all_summaries.append(summary)
    return all_summaries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate swarm30_v1 checkpoints and baselines.")
    parser.add_argument("--checkpoint")
    parser.add_argument("--policy-config")
    parser.add_argument("--track", default="checkpoint")
    parser.add_argument("--tasks", nargs="+", choices=SWARM30_TASKS, default=list(SWARM30_TASKS))
    parser.add_argument("--scales", nargs="+", type=int, default=[4, 10, 20, 30])
    parser.add_argument("--modes", nargs="+", choices=["fixed", "random"], default=["fixed", "random"])
    parser.add_argument("--random-episodes", type=int, default=100)
    parser.add_argument("--fixed-episodes", type=int, default=20)
    parser.add_argument("--record-episodes", type=int, default=2)
    parser.add_argument("--record-format", choices=["gif", "mp4"], default="gif")
    parser.add_argument("--record-fps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--env-config", default="configs/env/waypoint_missions.yaml")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--baselines", nargs="+", choices=["hold", "random", "greedy"])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.baselines:
        summaries = evaluate_baselines(args)
    else:
        if not args.checkpoint:
            raise ValueError("--checkpoint is required unless --baselines is used")
        summaries = evaluate_checkpoint(args)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "evaluation_index.json").write_text(
        json.dumps(summaries, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"[swarm30-eval] summaries={len(summaries)} output={output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
