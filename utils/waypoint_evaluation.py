from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch

from envs.waypoint_marl_env import WaypointMultiUAVEnv


def _to_tensor(obs: dict[str, np.ndarray], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: torch.as_tensor(value, dtype=torch.float32, device=device).unsqueeze(0) for key, value in obs.items()}


def write_episode_media(frames: list[np.ndarray], output_dir: str | Path, prefix: str, record_format: str, fps: int) -> Path:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    record_format = str(record_format).lower()
    path = output / f"{prefix}.{record_format}"
    frames = [np.asarray(frame, dtype=np.uint8) for frame in frames]
    if record_format == "mp4":
        try:
            with imageio.get_writer(path, fps=max(int(fps), 1)) as writer:
                for frame in frames:
                    writer.append_data(frame)
            return path
        except Exception:
            path = output / f"{prefix}.gif"
    imageio.mimsave(path, frames, duration=1.0 / float(max(int(fps), 1)), loop=0)
    return path


def _aggregate(records: list[dict]) -> dict:
    if not records:
        return {}
    keys = {key for record in records for key in record}
    summary = {}
    for key in keys:
        values = [record[key] for record in records if key in record]
        if values and isinstance(values[0], (int, float, np.integer, np.floating, bool)):
            summary[f"{key}_mean"] = float(np.mean(values))
        elif values:
            summary[key] = values[0]
    return summary


def evaluate_waypoint_policy_episodes(
    env_config: dict,
    policy,
    episodes: int,
    device: torch.device,
    task_name: str | None = None,
    deterministic: bool = True,
    headless: bool = True,
    record_dir: str | Path | None = None,
    record_episodes: int = 0,
    record_format: str = "gif",
    record_fps: int = 8,
    record_prefix: str = "episode",
) -> list[dict]:
    config = deepcopy(env_config)
    if task_name is not None:
        config["task_name"] = task_name
        config["task_names"] = [task_name]
    env = WaypointMultiUAVEnv(config)
    records = []
    record_root = Path(record_dir) if record_dir is not None else None
    for episode_idx in range(int(episodes)):
        env.reset(seed=int(config.get("seed", 0)) + 1000 + episode_idx)
        done = False
        total_reward = 0.0
        steps = 0
        info = {}
        frames = []
        should_record = record_root is not None and episode_idx < int(record_episodes)
        if should_record:
            frames.append(env.render("rgb_array"))
        while not done:
            obs = env.get_global_state()
            with torch.no_grad():
                if deterministic:
                    action_tensor = policy.act_deterministic(_to_tensor(obs, device))
                else:
                    action_tensor = policy.get_action_and_value(_to_tensor(obs, device))[0]
            action_row = action_tensor.squeeze(0).detach().cpu().numpy().astype(np.int64)
            action_dict = {agent: int(action_row[idx]) for idx, agent in enumerate(env.possible_agents)}
            _, _, terms, truncs, info_by_agent = env.step(action_dict)
            info = next(iter(info_by_agent.values()))
            total_reward += float(info.get("team_reward", 0.0))
            done = bool(any(terms.values()) or any(truncs.values()))
            steps += 1
            if should_record:
                frames.append(env.render("rgb_array"))
            if not headless:
                env.render("human")
        recording_path = None
        if should_record and frames:
            recording_path = write_episode_media(frames, record_root, f"{record_prefix}_ep{episode_idx:03d}_{info.get('task_name', task_name)}", record_format, record_fps)
        metrics = dict(info.get("task_metrics", {}))
        record = {
            "episode": episode_idx,
            "return": float(total_reward),
            "success_rate": float(info.get("success", False)),
            "path_length": float(info.get("path_length", 0.0)),
            "collision_rate": float(info.get("collision_rate", 0.0)),
            "steps": float(steps),
            "task_name": str(info.get("task_name", task_name or "unknown")),
            "num_agents": int(info.get("num_agents", config.get("num_agents", 0))),
            "action_validity_rate": float(1.0 - info.get("invalid_action_count", 0) / max(steps * int(config.get("num_agents", 1)), 1)),
            "candidate_mask_empty_count": float(info.get("candidate_mask_empty_count", 0)),
            **{key: float(value) for key, value in metrics.items() if isinstance(value, (int, float, np.integer, np.floating, bool))},
        }
        if recording_path is not None:
            record["recording_path"] = str(recording_path)
        records.append(record)
    env.close()
    return records


def evaluate_waypoint_policy_per_task(
    env_config: dict,
    policy,
    task_names: list[str],
    episodes: int,
    device: torch.device,
    headless: bool = True,
    record_dir: str | Path | None = None,
    record_episodes: int = 0,
    record_format: str = "gif",
    record_fps: int = 8,
    record_prefix: str = "eval",
) -> tuple[list[dict], dict[str, dict], dict]:
    all_records = []
    summaries = {}
    for task_name in task_names:
        task_record_dir = Path(record_dir) / task_name if record_dir is not None else None
        records = evaluate_waypoint_policy_episodes(
            env_config,
            policy,
            episodes,
            device,
            task_name=task_name,
            deterministic=True,
            headless=headless,
            record_dir=task_record_dir,
            record_episodes=record_episodes,
            record_format=record_format,
            record_fps=record_fps,
            record_prefix=f"{record_prefix}_{task_name}",
        )
        all_records.extend(records)
        summaries[task_name] = _aggregate(records)
    overall = _aggregate(all_records)
    return all_records, summaries, overall


def greedy_waypoint_action(env: WaypointMultiUAVEnv) -> dict[str, int]:
    features = env.get_global_state()["candidate_features"]
    mask = env.get_global_state()["candidate_mask"].astype(bool)
    task = env.current_scenario.name if env.current_scenario is not None else ""
    scores = features[..., 3] + 0.5 * features[..., 4] - 0.1 * features[..., 2]
    if task == "belief_search":
        scores = 2.0 * features[..., 4] + 0.5 * features[..., 3] - 0.1 * features[..., 2]
    elif task == "connectivity_expansion":
        positions = env.world.get_uav_positions()[:, None, :]
        outward = np.linalg.norm(env.last_candidate_waypoints - env.world.base_position[None, None, :], axis=-1)
        scores = outward / max(env.world.map_size, 1e-6) + 0.5 * features[..., 3]
    scores = np.where(mask, scores, -1e9)
    return {agent: int(np.argmax(scores[idx])) for idx, agent in enumerate(env.possible_agents)}


def random_waypoint_action(env: WaypointMultiUAVEnv, rng: np.random.Generator) -> dict[str, int]:
    actions = {}
    for idx, agent in enumerate(env.possible_agents):
        valid = np.where(env.last_candidate_mask[idx])[0]
        actions[agent] = int(rng.choice(valid)) if len(valid) else 0
    return actions
