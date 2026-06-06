from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from copy import deepcopy
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algorithms.mappo_waypoint import MAPPOWaypointTrainer
from envs.waypoint_marl_env import WaypointMultiUAVEnv
from policies import build_policy
from utils.waypoint_evaluation import greedy_waypoint_action, write_episode_media
from utils.waypoint_vector_env import make_waypoint_env_batch


PARAM_SPACE = {
    "max_command_distance": [0.15, 0.20, 0.25, 0.30],
    "max_speed": [0.03, 0.04, 0.05],
    "slowdown_radius": [0.05, 0.08, 0.10],
    "velocity_gain": [3.0, 4.0, 5.0],
    "max_accel": [0.04, 0.08, 0.10, 0.12],
    "control_smoothing": [0.30, 0.45, 0.60],
    "acceptance_radius": [0.015, 0.020, 0.030],
    "substeps": [10, 20],
}
FIXED_ADAPTER = {
    "name": "waypoint_velocity_tracker",
    "hover_when_arrived": True,
    "command_latency_steps": 0,
    "tracking_noise_std": 0.0,
    "clip_to_geofence": True,
    "reject_no_fly_waypoints": True,
}
ADAPTER_SETTING_KEYS = {
    "max_command_distance",
    "max_speed",
    "slowdown_radius",
    "velocity_gain",
    "max_accel",
    "control_smoothing",
    "acceptance_radius",
}
PROBES = [
    "single_step_far",
    "single_step_near",
    "square_path",
    "zigzag_path",
    "multi_agent_crossing",
    "connectivity_chain",
]


def load_yaml(path: str | Path) -> dict:
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = ROOT / resolved
    with open(resolved, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def write_yaml(path: str | Path, data: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def deep_update(base: dict, override: dict) -> dict:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value
    return result


def adapter_config(setting: dict[str, float]) -> dict:
    cfg = {**FIXED_ADAPTER, "meters_per_unit": 100.0}
    cfg.update({key: float(setting[key]) for key in ADAPTER_SETTING_KEYS if key in setting})
    return cfg


def env_config_for(base: dict, task: str, num_agents: int, seed: int, setting: dict[str, float]) -> dict:
    cfg = deepcopy(base)
    cfg["task_name"] = task
    cfg["task_names"] = [task]
    cfg["num_agents"] = int(num_agents)
    cfg["seed"] = int(seed)
    cfg["action_mode"] = "waypoint"
    cfg["action_interface"] = "waypoint"
    cfg["execution_model"] = "mpe_core"
    cfg["strict_source"] = True
    cfg["physical_scale"] = {"meters_per_unit": 100.0}
    cfg["max_waypoint_distance"] = float(setting.get("max_command_distance", cfg.get("max_waypoint_distance", 0.20)))
    cfg["max_speed"] = float(setting.get("max_speed", cfg.get("max_speed", 0.04)))
    cfg["dt_decision"] = float(setting.get("dt", 0.1)) * float(setting.get("substeps", cfg.get("dynamics_backend", {}).get("substeps", 20)))
    cfg["action_adapter"] = adapter_config(setting)
    cfg.setdefault("dynamics_backend", {})
    cfg["dynamics_backend"]["name"] = "mpe_core"
    cfg["dynamics_backend"]["source"] = "third_party_openai_mpe"
    cfg["dynamics_backend"]["dt"] = float(setting.get("dt", cfg["dynamics_backend"].get("dt", 0.1)))
    cfg["dynamics_backend"]["substeps"] = int(setting.get("substeps", cfg["dynamics_backend"].get("substeps", 20)))
    cfg["dynamics_backend"]["max_speed"] = float(setting.get("max_speed", cfg["dynamics_backend"].get("max_speed", cfg["max_speed"])))
    return cfg


def generate_settings(search_mode: str, max_settings: int) -> list[dict[str, float]]:
    base = {
        "max_command_distance": 0.20,
        "max_speed": 0.04,
        "slowdown_radius": 0.08,
        "velocity_gain": 4.0,
        "max_accel": 0.10,
        "control_smoothing": 0.45,
        "acceptance_radius": 0.020,
        "substeps": 20,
    }
    staged = [
        base,
        {**base, "max_command_distance": 0.20, "max_speed": 0.03, "velocity_gain": 3.5, "max_accel": 0.08, "control_smoothing": 0.40, "acceptance_radius": 0.025},
        {**base, "max_command_distance": 0.25, "max_speed": 0.04, "max_accel": 0.08},
        {**base, "max_command_distance": 0.30, "max_speed": 0.05, "slowdown_radius": 0.10, "velocity_gain": 5.0, "max_accel": 0.12, "control_smoothing": 0.60},
        {**base, "substeps": 10},
    ]
    keys = list(PARAM_SPACE)
    seen = {tuple(item[key] for key in keys) for item in staged}
    settings = list(staged)
    all_values = [PARAM_SPACE[key] for key in keys]
    combos = [dict(zip(keys, values)) for values in product(*all_values)]
    if search_mode == "random":
        rng = np.random.default_rng(20260606)
        rng.shuffle(combos)
    elif search_mode == "staged":
        combos.sort(
            key=lambda item: (
                abs(item["max_command_distance"] - base["max_command_distance"])
                + abs(item["slowdown_radius"] - base["slowdown_radius"])
                + 0.05 * abs(item["velocity_gain"] - base["velocity_gain"])
                + abs(item["max_accel"] - base["max_accel"])
                + abs(item["control_smoothing"] - base["control_smoothing"])
                + abs(item["acceptance_radius"] - base["acceptance_radius"])
                + abs(item["max_speed"] - base["max_speed"])
                + 0.01 * abs(item["substeps"] - base["substeps"])
            )
        )
    for item in combos:
        marker = tuple(item[key] for key in keys)
        if marker in seen:
            continue
        settings.append({key: float(item[key]) for key in keys})
        seen.add(marker)
        if len(settings) >= int(max_settings):
            break
    return settings[: int(max_settings)]


def waypoint_targets(probe: str, env: WaypointMultiUAVEnv, step_idx: int, total_steps: int) -> np.ndarray:
    positions = env.world.get_uav_positions()
    n = len(positions)
    map_size = float(env.world.map_size)
    if probe == "single_step_near":
        return env.world.project_to_geofence(positions + np.asarray([0.035, 0.0], dtype=np.float32))
    if probe == "single_step_far":
        return env.world.project_to_geofence(positions + np.asarray([0.18, 0.04], dtype=np.float32))
    if probe == "multi_agent_crossing":
        base = np.linspace(0.18, 0.82, n, dtype=np.float32)
        targets = np.stack([base[::-1], np.full(n, 0.78, dtype=np.float32)], axis=-1)
        return env.world.project_to_geofence(targets)
    if probe == "connectivity_chain":
        x = np.linspace(0.15, 0.75, n, dtype=np.float32)
        y = np.linspace(0.15, 0.55, n, dtype=np.float32)
        return env.world.project_to_geofence(np.stack([x, y], axis=-1))
    phase = (step_idx // max(total_steps // 4, 1)) % 4
    if probe == "square_path":
        corners = np.asarray([[0.22, 0.22], [0.78, 0.22], [0.78, 0.78], [0.22, 0.78]], dtype=np.float32)
        offsets = np.linspace(-0.04, 0.04, n, dtype=np.float32)
        targets = corners[phase][None, :] + np.stack([offsets, -offsets], axis=-1)
        return env.world.project_to_geofence(targets)
    # zigzag_path
    x = 0.2 + 0.6 * ((step_idx % total_steps) / max(total_steps - 1, 1))
    y = 0.25 if phase % 2 == 0 else 0.75
    offsets = np.linspace(-0.05, 0.05, n, dtype=np.float32)
    targets = np.stack([np.full(n, x, dtype=np.float32), np.full(n, y, dtype=np.float32) + offsets], axis=-1)
    return env.world.project_to_geofence(targets)


def summarize_values(values: list[float], default: float = 0.0, reducer=np.mean) -> float:
    clean = [float(value) for value in values if np.isfinite(value)]
    if not clean:
        return float(default)
    return float(reducer(clean))


def run_probe(base_config: dict, setting: dict[str, float], probe: str, num_agents: int, seed: int, record_path: Path | None = None) -> dict:
    cfg = env_config_for(base_config, "area_coverage", num_agents, seed, setting)
    cfg["max_steps"] = min(int(cfg.get("max_steps", 80)), 40)
    env = WaypointMultiUAVEnv(cfg)
    env.reset(seed=seed)
    agents = list(env.possible_agents)
    frames = [env.render("rgb_array")] if record_path is not None else []
    total_steps = 28
    errors = []
    final_errors = []
    arrival_steps: list[int] = []
    arrived = np.zeros(num_agents, dtype=bool)
    path_lengths = np.zeros(num_agents, dtype=np.float32)
    straight_dists = np.zeros(num_agents, dtype=np.float32)
    prev_positions = env.world.get_uav_positions().copy()
    prev_errors = None
    overshoots = 0
    oscillations = 0
    speed_violations = 0
    max_speed_observed = []
    mean_speed = []
    control_norm = []
    control_saturation = []
    control_changes = []
    last_control = None
    collisions = []
    no_fly = []
    geofence = []
    connectivity = []
    action_validity = []
    mpe_calls = []
    real_mpe = []
    finite = True
    last_targets = None
    for step_idx in range(total_steps):
        targets = waypoint_targets(probe, env, step_idx, total_steps).astype(np.float32)
        if last_targets is None:
            straight_dists = np.linalg.norm(targets - prev_positions, axis=-1).astype(np.float32)
        last_targets = targets.copy()
        actions = {agent: targets[idx] for idx, agent in enumerate(agents)}
        _, _, terms, truncs, info_by_agent = env.step(actions)
        info = next(iter(info_by_agent.values()))
        positions = env.world.get_uav_positions()
        step_errors = np.linalg.norm(positions - targets, axis=-1)
        errors.extend(float(item) for item in step_errors)
        just_arrived = (step_errors <= float(setting["acceptance_radius"])) & ~arrived
        for idx in np.where(just_arrived)[0]:
            arrival_steps.append(step_idx + 1)
        arrived |= step_errors <= float(setting["acceptance_radius"])
        path_lengths += np.linalg.norm(positions - prev_positions, axis=-1).astype(np.float32)
        if prev_errors is not None:
            overshoots += int(np.count_nonzero(step_errors > prev_errors + 0.02))
            prev_delta = positions - prev_positions
            target_delta = targets - positions
            oscillations += int(np.count_nonzero(np.sum(prev_delta * target_delta, axis=-1) < -1e-4))
        prev_errors = step_errors
        prev_positions = positions.copy()
        max_speed_observed.append(float(info.get("max_speed_observed", 0.0)))
        mean_speed.append(float(info.get("mean_speed", 0.0)))
        control_norm.append(float(info.get("mean_control_norm", 0.0)))
        max_control = float(info.get("adapter_info", {}).get("max_control_norm", 0.0))
        control_saturation.append(float(max_control >= float(setting["max_accel"]) * 0.98))
        if last_control is not None:
            control_changes.append(abs(float(info.get("mean_control_norm", 0.0)) - last_control))
        last_control = float(info.get("mean_control_norm", 0.0))
        collisions.append(float(info.get("collision_count", 0.0)))
        no_fly.append(float(info.get("no_fly_violation_count", 0.0)))
        geofence.append(float(info.get("geofence_violation_count", 0.0)))
        connectivity.append(float(info.get("connectivity_violation_rate", 0.0)))
        action_validity.append(1.0)
        dyn = info.get("dynamics_info", {})
        mpe_calls.append(float(dyn.get("mpe_world_step_calls", 0.0)))
        real_mpe.append(float(bool(dyn.get("uses_real_mpe_core", False))))
        finite = finite and bool(info.get("adapter_info", {}).get("adapter_finite", True)) and bool(dyn.get("dynamics_finite", True))
        if frames is not None and record_path is not None:
            frames.append(env.render("rgb_array"))
        if any(terms.values()) or any(truncs.values()):
            break
    positions = env.world.get_uav_positions()
    if last_targets is not None:
        final_errors = [float(item) for item in np.linalg.norm(positions - last_targets, axis=-1)]
    path_efficiency = np.divide(straight_dists, np.maximum(path_lengths, 1e-6))
    if record_path is not None and frames:
        write_episode_media(frames, record_path.parent, record_path.stem, record_path.suffix.lstrip(".") or "gif", 8)
    env.close()
    max_speed = float(cfg.get("dynamics_backend", {}).get("max_speed", cfg.get("max_speed", 0.08)))
    normalizer = max(total_steps * num_agents * int(cfg.get("dynamics_backend", {}).get("substeps", 1)), 1)
    return {
        "tracking_error_mean": summarize_values(errors),
        "tracking_error_final": summarize_values(final_errors),
        "arrival_rate": float(np.mean(arrived)) if len(arrived) else 0.0,
        "time_to_arrival_mean": summarize_values(arrival_steps, default=float(total_steps)),
        "overshoot_count": float(overshoots),
        "oscillation_score": float(oscillations / max(total_steps * num_agents, 1)),
        "path_efficiency": float(np.clip(np.mean(path_efficiency), 0.0, 1.5)),
        "mean_speed": summarize_values(mean_speed),
        "max_speed_observed": summarize_values(max_speed_observed, reducer=np.max),
        "speed_violation_rate": float(np.mean([item > max_speed + 1e-6 for item in max_speed_observed])) if max_speed_observed else 0.0,
        "mean_control_norm": summarize_values(control_norm),
        "control_saturation_rate": summarize_values(control_saturation),
        "control_smoothness": summarize_values(control_changes),
        "collision_rate": float(np.sum(collisions) / normalizer),
        "min_pairwise_distance_mean": 0.0,
        "no_fly_violation_rate": float(np.sum(no_fly) / normalizer),
        "geofence_violation_rate": float(np.sum(geofence) / normalizer),
        "connectivity_violation_rate": summarize_values(connectivity),
        "action_validity_rate": summarize_values(action_validity, default=1.0),
        "finite": bool(finite and np.isfinite(errors).all()),
        "uses_real_mpe_core": bool(real_mpe and min(real_mpe) >= 1.0),
        "mpe_source": "third_party_openai_mpe",
        "mpe_world_step_calls": float(np.sum(mpe_calls)),
        "mpe_world_step_calls_total": float(np.sum(mpe_calls)),
    }


def run_task_rollout(base_config: dict, setting: dict[str, float], task: str, num_agents: int, seed: int, episodes: int) -> dict:
    rewards = []
    successes = []
    progress = []
    coverage = []
    searched = []
    poi = []
    connected = []
    radius = []
    action_validity = []
    collisions = []
    no_fly = []
    geofence = []
    connectivity = []
    mpe_calls = []
    real_mpe = []
    finite = True
    for episode_idx in range(int(episodes)):
        cfg = env_config_for(base_config, task, num_agents, seed + episode_idx * 17, setting)
        env = WaypointMultiUAVEnv(cfg)
        env.reset(seed=seed + episode_idx * 17)
        done = False
        total = 0.0
        info = {}
        while not done:
            action = greedy_waypoint_action(env)
            _, _, terms, truncs, info_by_agent = env.step(action)
            info = next(iter(info_by_agent.values()))
            total += float(info.get("team_reward", 0.0))
            done = bool(any(terms.values()) or any(truncs.values()))
        metrics = dict(info.get("task_metrics", {}))
        rewards.append(total)
        successes.append(float(info.get("success", False)))
        coverage.append(float(metrics.get("coverage_ratio", info.get("coverage_ratio", 0.0))))
        searched.append(float(metrics.get("searched_probability_mass", info.get("searched_probability_mass", 0.0))))
        poi.append(float(metrics.get("weighted_poi_completion", info.get("weighted_poi_completion", 0.0))))
        connected.append(float(metrics.get("connected_to_base_ratio", info.get("connected_to_base_ratio", 1.0))))
        radius.append(float(metrics.get("effective_explored_radius", info.get("effective_explored_radius", 0.0))))
        task_progress = max(coverage[-1], searched[-1], poi[-1], min(radius[-1] / max(float(cfg.get("map_size", 1.0)), 1e-6), 1.0))
        progress.append(float(task_progress))
        action_validity.append(1.0)
        collisions.append(float(info.get("collision_count", 0.0)))
        no_fly.append(float(info.get("no_fly_violation_count", 0.0)))
        geofence.append(float(info.get("geofence_violation_count", 0.0)))
        connectivity.append(float(info.get("connectivity_violation_rate", 0.0)))
        dyn = info.get("dynamics_info", {})
        mpe_calls.append(float(dyn.get("mpe_world_step_calls", 0.0)))
        real_mpe.append(float(bool(dyn.get("uses_real_mpe_core", False))))
        finite = finite and bool(info.get("adapter_info", {}).get("adapter_finite", True)) and bool(dyn.get("dynamics_finite", True))
        env.close()
    max_steps = int(base_config.get("max_steps", 80))
    substeps = int(base_config.get("dynamics_backend", {}).get("substeps", 1))
    normalizer = max(int(episodes) * max_steps * num_agents * substeps, 1)
    return {
        "team_reward": summarize_values(rewards),
        "task_success_rate": summarize_values(successes),
        "normalized_task_progress": summarize_values(progress),
        "coverage_ratio": summarize_values(coverage),
        "searched_probability_mass": summarize_values(searched),
        "weighted_poi_completion": summarize_values(poi),
        "connected_to_base_ratio": summarize_values(connected, default=1.0),
        "effective_explored_radius": summarize_values(radius),
        "action_validity_rate": summarize_values(action_validity, default=1.0),
        "collision_rate": float(np.sum(collisions) / normalizer),
        "no_fly_violation_rate": float(np.sum(no_fly) / normalizer),
        "geofence_violation_rate": float(np.sum(geofence) / normalizer),
        "connectivity_violation_rate": summarize_values(connectivity),
        "finite": bool(finite),
        "uses_real_mpe_core": bool(real_mpe and min(real_mpe) >= 1.0),
        "mpe_source": "third_party_openai_mpe",
        "mpe_world_step_calls": float(np.sum(mpe_calls)),
    }


def merge_metrics(parts: list[dict]) -> dict:
    keys = {key for part in parts for key in part}
    merged: dict[str, Any] = {}
    for key in keys:
        values = [part[key] for part in parts if key in part]
        if not values:
            continue
        if isinstance(values[0], bool):
            merged[key] = bool(all(values))
        elif isinstance(values[0], str):
            merged[key] = values[0]
        elif isinstance(values[0], (int, float, np.integer, np.floating)):
            if key in {"mpe_world_step_calls", "mpe_world_step_calls_total", "overshoot_count"}:
                merged[key] = float(np.sum(values))
            elif key.startswith("max_"):
                merged[key] = float(np.max(values))
            else:
                merged[key] = float(np.mean(values))
    return merged


def score_row(row: dict) -> tuple[float, dict[str, float], bool, list[str]]:
    components = {
        "arrival_rate": 2.0 * float(row.get("arrival_rate", 0.0)),
        "path_efficiency": 1.5 * float(row.get("path_efficiency", 0.0)),
        "task_success_rate": 1.0 * float(row.get("task_success_rate", 0.0)),
        "normalized_task_progress": 1.0 * float(row.get("normalized_task_progress", 0.0)),
        "collision_rate": -2.0 * float(row.get("collision_rate", 0.0)),
        "connectivity_violation_rate": -2.0 * float(row.get("connectivity_violation_rate", 0.0)),
        "no_fly_violation_rate": -1.5 * float(row.get("no_fly_violation_rate", 0.0)),
        "speed_violation_rate": -1.0 * float(row.get("speed_violation_rate", 0.0)),
        "control_saturation_rate": -1.0 * float(row.get("control_saturation_rate", 0.0)),
        "oscillation_score": -0.8 * float(row.get("oscillation_score", 0.0)),
        "tracking_error_mean": -0.5 * float(row.get("tracking_error_mean", 0.0)),
        "control_smoothness": -0.3 * float(row.get("control_smoothness", 0.0)),
    }
    failures = []
    checks = {
        "finite": bool(row.get("finite", False)),
        "mpe_source": row.get("mpe_source", "") == "third_party_openai_mpe",
        "uses_real_mpe_core": bool(row.get("uses_real_mpe_core", False)),
        "mpe_world_step_calls": float(row.get("mpe_world_step_calls", 0.0)) > 0.0,
        "speed_violation_rate": float(row.get("speed_violation_rate", 1.0)) <= 0.01,
        "collision_rate": float(row.get("collision_rate", 1.0)) <= 0.03,
        "no_fly_violation_rate": float(row.get("no_fly_violation_rate", 1.0)) <= 0.01,
        "geofence_violation_rate": float(row.get("geofence_violation_rate", 1.0)) <= 0.01,
        "control_saturation_rate": float(row.get("control_saturation_rate", 1.0)) <= 0.50,
        "arrival_rate": float(row.get("arrival_rate", 0.0)) >= 0.65,
        "action_validity_rate": float(row.get("action_validity_rate", 0.0)) >= 0.99,
        "connectivity_violation_rate": float(row.get("connectivity_violation_rate", 1.0)) <= 0.15,
    }
    for key, ok in checks.items():
        if not ok:
            failures.append(key)
    return float(sum(components.values())), components, not failures, failures


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row})
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def record_top_media(base_config: dict, setting: dict[str, float], output_dir: Path, rank: int, tasks: list[str], record_format: str) -> list[str]:
    media_paths = []
    root = output_dir / "media" / f"top_{rank:03d}"
    root.mkdir(parents=True, exist_ok=True)
    for probe in ["square_path", "multi_agent_crossing"]:
        path = root / f"probe_{probe}.{record_format}"
        run_probe(base_config, setting, probe, 4, 100 + rank, path)
        media_paths.append(str(path))
    for task in [task for task in ["area_coverage", "connectivity_expansion"] if task in tasks]:
        cfg = env_config_for(base_config, task, 4, 200 + rank, setting)
        env = WaypointMultiUAVEnv(cfg)
        env.reset(seed=200 + rank)
        frames = [env.render("rgb_array")]
        done = False
        while not done:
            _, _, terms, truncs, _ = env.step(greedy_waypoint_action(env))
            frames.append(env.render("rgb_array"))
            done = bool(any(terms.values()) or any(truncs.values()))
        path = write_episode_media(frames, root, f"task_{task}", record_format, 8)
        media_paths.append(str(path))
        env.close()
    return media_paths


def run_mappo_smoke(base_config: dict, setting: dict[str, float], output_dir: Path, rank: int, tasks: list[str]) -> list[dict]:
    smoke_rows = []
    for policy_class, config_path in [
        ("candidate_selection_waypoint", "configs/policy/mappo_waypoint_debug.yaml"),
        ("direct_waypoint", "configs/policy/direct_waypoint_debug.yaml"),
    ]:
        env_config = env_config_for(base_config, tasks[0], 3, 300 + rank, setting)
        env_config["task_names"] = list(tasks[: min(len(tasks), 2)])
        env_config["task_name"] = None
        train_config = load_yaml(config_path)
        train_config["policy_class"] = policy_class
        train_config["num_envs"] = 2
        train_config["rollout_steps"] = 16
        train_config["total_updates"] = 1
        train_config["eval_interval"] = 1
        train_config["minibatch_size"] = 32
        for key in ["map_size", "grid_size", "max_waypoint_distance", "min_uav_distance", "base_position", "no_fly_zones"]:
            if key in env_config:
                train_config[key] = env_config[key]
        if "max_delta" not in train_config:
            train_config["max_delta"] = float(env_config.get("max_waypoint_distance", 0.22))
        env_batch = make_waypoint_env_batch(env_config, int(train_config["num_envs"]), backend="sync")
        policy = build_policy(train_config, env_batch.envs[0].global_observation_space, env_batch.envs[0].action_space_n)
        trainer = MAPPOWaypointTrainer(env_batch, policy, train_config)
        run_dir = output_dir / "mappo_smoke" / f"top_{rank:03d}_{policy_class}"
        history = trainer.train(
            run_dir,
            env_config=env_config,
            task_names=env_config["task_names"],
            eval_episodes=1,
            headless=True,
            record_eval_episodes=0,
            record_format="gif",
            record_fps=8,
            record_interval=1,
        )
        env_batch.close()
        last = history[-1] if history else {}
        smoke_rows.append(
            {
                "rank": rank,
                "policy_class": policy_class,
                "mappo_smoke_passed": bool(
                    np.isfinite(float(last.get("policy_loss", 0.0)))
                    and np.isfinite(float(last.get("value_loss", 0.0)))
                    and np.isfinite(float(last.get("eval_reward", 0.0)))
                    and float(last.get("eval_uses_real_mpe_core", 0.0)) >= 1.0
                    and str(last.get("eval_mpe_source", "")) == "third_party_openai_mpe"
                ),
                "mappo_smoke_eval_reward": float(last.get("eval_reward", 0.0)),
                "mappo_smoke_policy_loss": float(last.get("policy_loss", 0.0)),
                "mappo_smoke_value_loss": float(last.get("value_loss", 0.0)),
                "mappo_smoke_mpe_source": str(last.get("eval_mpe_source", "")),
                "mappo_smoke_output_dir": str(run_dir),
            }
        )
    return smoke_rows


def plot_results(output_dir: Path, rows: list[dict]) -> list[str]:
    warnings = []
    try:
        import matplotlib.pyplot as plt

        plots = output_dir / "plots"
        plots.mkdir(parents=True, exist_ok=True)
        xs = np.arange(len(rows))
        scores = [float(row.get("score", 0.0)) for row in rows]
        plt.figure(figsize=(8, 4))
        plt.scatter(xs, scores, s=14)
        plt.xlabel("setting index")
        plt.ylabel("score")
        plt.tight_layout()
        plt.savefig(plots / "score_vs_params.png")
        plt.close()
        for metric, filename in [
            ("tracking_error_mean", "tracking_error_boxplot.png"),
            ("control_saturation_rate", "control_saturation_boxplot.png"),
            ("max_speed_observed", "speed_distribution.png"),
            ("collision_rate", "collision_rate_by_setting.png"),
        ]:
            values = [float(row.get(metric, 0.0)) for row in rows]
            plt.figure(figsize=(5, 4))
            plt.boxplot(values)
            plt.ylabel(metric)
            plt.tight_layout()
            plt.savefig(plots / filename)
            plt.close()
    except Exception as exc:  # pragma: no cover - plotting is optional.
        warnings.append(f"matplotlib plotting failed: {type(exc).__name__}: {exc}")
    return warnings


def select_profiles(ranked: list[dict]) -> dict[str, dict] | None:
    passed = [row for row in ranked if row.get("passed")]
    if not passed:
        return None
    default = passed[0]
    conservative = min(
        passed,
        key=lambda row: (
            float(row.get("collision_rate", 0.0))
            + float(row.get("no_fly_violation_rate", 0.0))
            + float(row.get("geofence_violation_rate", 0.0))
            + float(row.get("control_saturation_rate", 0.0))
            + 0.2 * float(row.get("mean_control_norm", 0.0))
        ),
    )
    aggressive = max(
        passed,
        key=lambda row: (
            float(row.get("normalized_task_progress", 0.0))
            + float(row.get("arrival_rate", 0.0))
            + 0.2 * float(row.get("mean_speed", 0.0))
            - 0.5 * float(row.get("collision_rate", 0.0))
        ),
    )
    return {"default": default, "conservative": conservative, "aggressive": aggressive}


def write_recommended_configs(base_config: dict, profiles: dict[str, dict], output_dir: Path) -> dict[str, str]:
    config_dir = ROOT / "configs" / "env" / "adapters"
    config_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, row in profiles.items():
        setting = {key: float(row[key]) for key in PARAM_SPACE}
        cfg = adapter_config(setting)
        cfg["profile"] = name
        cfg["calibration_rank"] = int(row["rank"])
        cfg["calibration_score"] = float(row["score"])
        local_path = config_dir / f"waypoint_velocity_tracker_{name}.yaml"
        write_yaml(local_path, cfg)
        paths[name] = str(local_path)
        write_yaml(output_dir / "top_configs" / f"{name}.yaml", cfg)
    tuned = deepcopy(base_config)
    tuned["action_adapter"] = adapter_config({key: float(profiles["default"][key]) for key in PARAM_SPACE})
    tuned["physical_scale"] = {"meters_per_unit": 100.0}
    tuned["max_waypoint_distance"] = float(profiles["default"]["max_command_distance"])
    tuned["max_speed"] = float(profiles["default"]["max_speed"])
    tuned["dt_decision"] = 0.1 * float(profiles["default"]["substeps"])
    tuned.setdefault("dynamics_backend", {})["name"] = "mpe_core"
    tuned["dynamics_backend"]["source"] = "third_party_openai_mpe"
    tuned["dynamics_backend"]["dt"] = 0.1
    tuned["dynamics_backend"]["substeps"] = int(profiles["default"]["substeps"])
    tuned["dynamics_backend"]["max_speed"] = float(profiles["default"]["max_speed"])
    tuned["strict_source"] = True
    tuned["execution_model"] = "mpe_core"
    tuned_path = ROOT / "configs" / "env" / "waypoint_missions_tuned.yaml"
    write_yaml(tuned_path, tuned)
    paths["tuned_env"] = str(tuned_path)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-config", default="configs/env/waypoint_missions.yaml")
    parser.add_argument("--tasks", nargs="+", default=["area_coverage", "belief_search", "priority_inspection", "connectivity_expansion"])
    parser.add_argument("--num-agents-list", nargs="+", type=int, default=[3, 4])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--episodes-per-setting", type=int, default=2)
    parser.add_argument("--search-mode", choices=["grid", "random", "staged"], default="staged")
    parser.add_argument("--max-settings", type=int, default=80)
    parser.add_argument("--record-top-k", type=int, default=5)
    parser.add_argument("--record-format", choices=["gif", "mp4"], default="gif")
    parser.add_argument("--output-dir", default=str(ROOT / "outputs" / "debug" / "adapter_calibration" / "local_core_velocity_tracker_v1"))
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    base_config = load_yaml(args.env_config)
    base_config.setdefault("dynamics_backend", {})["name"] = "mpe_core"
    base_config["dynamics_backend"]["source"] = "third_party_openai_mpe"
    base_config["execution_model"] = "mpe_core"
    calibration_config = {
        "env_config": str(args.env_config),
        "tasks": list(args.tasks),
        "num_agents_list": list(args.num_agents_list),
        "seeds": list(args.seeds),
        "episodes_per_setting": int(args.episodes_per_setting),
        "search_mode": args.search_mode,
        "max_settings": int(args.max_settings),
        "record_top_k": int(args.record_top_k),
        "record_format": args.record_format,
        "output_dir": str(output_dir),
        "dynamics_backend": {"name": "mpe_core", "source": "third_party_openai_mpe", "strict_source": True},
        "param_space": PARAM_SPACE,
    }
    write_yaml(output_dir / "calibration_config.yaml", calibration_config)

    settings = generate_settings(args.search_mode, int(args.max_settings))
    rows = []
    for setting_idx, setting in enumerate(settings):
        stage_parts = []
        for num_agents in args.num_agents_list:
            for seed in args.seeds:
                for probe in PROBES:
                    stage_parts.append(run_probe(base_config, setting, probe, int(num_agents), int(seed)))
                for task in args.tasks:
                    stage_parts.append(run_task_rollout(base_config, setting, str(task), int(num_agents), int(seed), int(args.episodes_per_setting)))
        metrics = merge_metrics(stage_parts)
        row = {
            "setting_id": int(setting_idx),
            **{key: float(value) for key, value in setting.items()},
            **metrics,
        }
        score, components, passed, failures = score_row(row)
        row["score"] = float(score)
        row["score_components"] = json.dumps(components, sort_keys=True)
        row["passed"] = bool(passed)
        row["failed_constraints"] = ",".join(failures)
        rows.append(row)
        print(f"[adapter-calibration] setting={setting_idx + 1}/{len(settings)} score={score:.3f} passed={passed}", flush=True)

    ranked = sorted(rows, key=lambda item: float(item.get("score", 0.0)), reverse=True)
    for rank, row in enumerate(ranked, start=1):
        row["rank"] = int(rank)
    write_csv(output_dir / "adapter_calibration_results.csv", rows)
    write_csv(output_dir / "adapter_calibration_ranked.csv", ranked)

    media_paths = []
    for rank, row in enumerate(ranked[: max(int(args.record_top_k), 0)], start=1):
        setting = {key: float(row[key]) for key in PARAM_SPACE}
        media_paths.extend(record_top_media(base_config, setting, output_dir, rank, list(args.tasks), args.record_format))

    smoke_rows = []
    for rank, row in enumerate(ranked[: max(min(int(args.record_top_k), 5), 0)], start=1):
        setting = {key: float(row[key]) for key in PARAM_SPACE}
        smoke_rows.extend(run_mappo_smoke(base_config, setting, output_dir, rank, list(args.tasks)))
    write_csv(output_dir / "mappo_smoke_results.csv", smoke_rows)

    plot_warnings = plot_results(output_dir, rows)
    profiles = select_profiles(ranked)
    calibration_passed = profiles is not None
    config_paths = {}
    if profiles is not None:
        config_paths = write_recommended_configs(base_config, profiles, output_dir)
    closest = ranked[: min(10, len(ranked))]
    summary = {
        "passed": bool(calibration_passed),
        "calibration_failed": not bool(calibration_passed),
        "num_settings": len(settings),
        "num_passed": int(sum(1 for row in ranked if row.get("passed"))),
        "mpe_source_required": "third_party_openai_mpe",
        "uses_real_mpe_core_required": True,
        "top_rows": closest,
        "selected_profiles": {
            name: {key: value for key, value in row.items() if key in PARAM_SPACE or key in {"rank", "score", "passed", "failed_constraints"}}
            for name, row in (profiles or {}).items()
        },
        "config_paths": config_paths,
        "media_paths": media_paths,
        "mappo_smoke_results": smoke_rows,
        "plot_warnings": plot_warnings,
    }
    (output_dir / "adapter_calibration_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not calibration_passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
