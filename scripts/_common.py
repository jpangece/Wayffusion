from __future__ import annotations

import csv
from copy import deepcopy
from datetime import datetime
from numbers import Number
from pathlib import Path
import sys
from typing import Callable, Iterable
import warnings

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baselines import make_baseline
from envs import CentralizedMultiUAVEnv
from fields.visualization import save_rollout_plot, save_task_field_plot
from tasks import TASK_ORDER
from utils.evaluation import compute_reference_table, flatten_reward_components


def deep_update(base: dict, override: dict) -> dict:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value
    return result


def load_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load_env_config(config_path: str | Path, override: dict | None = None) -> dict:
    config_path = ROOT / Path(config_path)
    base_path = ROOT / "configs" / "env" / "base.yaml"
    config = load_yaml(base_path)
    if config_path.resolve() != base_path.resolve():
        config = deep_update(config, load_yaml(config_path))
    if override:
        config = deep_update(config, override)
    return config


def load_generic_config(config_path: str | Path) -> dict:
    return load_yaml(ROOT / Path(config_path))


def ensure_dir(path: str | Path) -> Path:
    directory = Path(path)
    if not directory.is_absolute():
        directory = ROOT / directory
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def timestamped_training_dir(algorithm: str, run_name: str, timestamp: str | None = None) -> Path:
    run_timestamp = str(timestamp or datetime.now().astimezone().strftime("%Y%m%d_%H%M%S"))
    return ensure_dir(Path("outputs/training") / str(algorithm) / run_timestamp / str(run_name))


def checkpoints_dir(run_dir: str | Path) -> Path:
    return ensure_dir(Path(run_dir) / "checkpoints")


def snapshot_dir(run_dir: str | Path) -> Path:
    return ensure_dir(Path(run_dir) / "snapshot")


def tensorboard_dir(run_dir: str | Path) -> Path:
    return ensure_dir(Path(run_dir) / "tensorboard")


def _write_yaml(path: str | Path, payload: dict) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=True)
    return output_path


def save_run_snapshot(
    run_dir: str | Path,
    train_config: dict,
    env_config: dict,
    cli_args: dict | None = None,
    model_state_dict: dict | None = None,
    extra_metadata: dict | None = None,
) -> Path:
    snapshot_root = snapshot_dir(run_dir)
    _write_yaml(snapshot_root / "train_config.yaml", deepcopy(train_config))
    _write_yaml(snapshot_root / "env_config.yaml", deepcopy(env_config))
    if cli_args is not None:
        _write_yaml(snapshot_root / "cli_args.yaml", deepcopy(cli_args))
    metadata = {"created_at": datetime.now().astimezone().isoformat()}
    if extra_metadata:
        metadata.update(deepcopy(extra_metadata))
    _write_yaml(snapshot_root / "metadata.yaml", metadata)
    if model_state_dict is not None:
        import torch

        torch.save({"model_state_dict": model_state_dict, "train_config": deepcopy(train_config)}, snapshot_root / "initial_model.pt")
    return snapshot_root


def run_dir_from_checkpoint(checkpoint_path: str | Path) -> Path:
    checkpoint = Path(checkpoint_path)
    if checkpoint.parent.name == "checkpoints":
        return checkpoint.parent.parent
    return checkpoint.parent


def build_env(config: dict, task_name: str | None = None) -> CentralizedMultiUAVEnv:
    env_config = deepcopy(config)
    if task_name is not None:
        env_config["task_name"] = task_name
    return CentralizedMultiUAVEnv(env_config)


def prepare_env_config(
    config_path: str | Path,
    tasks: list[str] | None = None,
    num_agents: int | None = None,
    scaling_mode: str | None = None,
    observation_override: dict | None = None,
    extra_override: dict | None = None,
) -> dict:
    override = {}
    if tasks:
        override["task_names"] = list(tasks)
        override["task_name"] = tasks[0] if len(tasks) == 1 else None
        override["task_sampling_probs"] = {
            name: (1.0 / len(tasks) if name in tasks else 0.0) for name in TASK_ORDER
        }
    if num_agents is not None:
        override["num_agents"] = int(num_agents)
    if scaling_mode is not None:
        override["scaling_mode"] = scaling_mode
    if observation_override:
        override.update(observation_override)
    if extra_override:
        override = deep_update(override, extra_override)
    return load_env_config(config_path, override=override if override else None)


def run_baseline_episode(env: CentralizedMultiUAVEnv, policy, seed: int | None = None) -> tuple[float, dict]:
    observation, info = env.reset(seed=seed)
    done = False
    total_reward = 0.0
    while not done:
        action = policy.act(observation)
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated
    info = dict(info)
    info["episode_reward"] = float(total_reward)
    return total_reward, info


def baseline_reference_episodes_for_agent_count(num_agents: int, requested_episodes: int) -> int:
    requested = max(int(requested_episodes), 2)
    if num_agents >= 80:
        return min(requested, 2)
    if num_agents >= 40:
        return max(requested, 3)
    return max(requested, 4)


def collect_baseline_episode_records(env_config: dict, method: str, episodes: int = 10) -> list[dict]:
    env = CentralizedMultiUAVEnv(deepcopy(env_config))
    policy = build_baseline(method, env_config)
    records = []
    for episode_idx in range(episodes):
        total_reward, info = run_baseline_episode(env, policy, seed=int(env_config.get("seed", 0)) + episode_idx)
        record = {
            "method": method,
            "algorithm": method,
            "architecture": "baseline",
            "observation_mode": env_config.get("observation_mode", "multi_channel_field"),
            "task_name": str(info.get("task_name", "unknown")),
            "num_agents": int(info.get("num_agents", env_config["num_agents"])),
            "scaling_mode": str(info.get("scaling_mode", env_config.get("scaling_mode", "fixed_map"))),
            "seed": int(env_config.get("seed", 0)) + episode_idx,
            "return": float(total_reward),
            "raw_return": float(total_reward),
            "success_rate": float(info.get("success", False)),
            "collision_rate": float(info.get("collision_rate", 0.0)),
            "path_length": float(info.get("path_length", 0.0)),
            "inference_latency_ms": 0.0,
            "rollout_steps_per_sec": 0.0,
            "wall_clock_time": 0.0,
            "memory_usage_mb": 0.0,
            "intrinsic_score": float(info.get("intrinsic_score", 0.0)),
            **flatten_reward_components(info),
        }
        for key, value in info.get("task_specific_metrics", {}).items():
            if isinstance(value, (int, float, np.floating)):
                record[key] = float(value)
        records.append(record)
    return records


def baseline_reference_table(env_config: dict, episodes: int = 10, methods: list[str] | None = None) -> dict[tuple[str, int, str], dict[str, float]]:
    methods = methods or ["random", "heuristic"]
    compact_records = []
    task_names = resolve_tasks(env_config)
    for task_name in task_names:
        single_task_config = deepcopy(env_config)
        single_task_config["task_name"] = task_name
        single_task_config["task_names"] = [task_name]
        single_task_config["task_sampling_probs"] = {
            name: (1.0 if name == task_name else 0.0) for name in TASK_ORDER
        }
        for method in methods:
            compact_records.extend(collect_baseline_episode_records(single_task_config, method, episodes=episodes))
    return compute_reference_table(compact_records)


def baseline_return_summary(env_config: dict, episodes: int = 10, methods: list[str] | None = None) -> dict[str, float]:
    reference_table = baseline_reference_table(env_config, episodes=episodes, methods=methods)
    task_names = resolve_tasks(env_config)
    if len(task_names) == 1:
        key = (task_names[0], int(env_config["num_agents"]), str(env_config.get("scaling_mode", "fixed_map")))
        refs = reference_table.get(key, {"random_return": 0.0, "heuristic_return": 1.0})
        return {"random": float(refs["random_return"]), "heuristic": float(refs["heuristic_return"])}
    random_values = [float(refs["random_return"]) for refs in reference_table.values()]
    heuristic_values = [float(refs["heuristic_return"]) for refs in reference_table.values()]
    return {
        "random": float(np.mean(random_values)) if random_values else 0.0,
        "heuristic": float(np.mean(heuristic_values)) if heuristic_values else 1.0,
    }


def save_rollout_artifacts(info: dict, output_dir: str | Path, prefix: str) -> None:
    output_dir = ensure_dir(output_dir)
    field_dir = ensure_dir(output_dir / "task_fields")
    traj_dir = ensure_dir(output_dir / "trajectories")
    save_task_field_plot(info["full_task_field"], field_dir / f"{prefix}_field.png", title=prefix)
    save_rollout_plot(
        obstacle_map=info["full_task_field"][0],
        risk_map=info["full_task_field"][4],
        trajectories=info["trajectory_history"],
        waypoints=info["current_waypoints"],
        goals=info.get("task_targets"),
        output_path=traj_dir / f"{prefix}_rollout.png",
        title=f"{prefix} | score={info.get('normalized_score', 0.0):.3f}",
        map_size=float(info.get("map_size", 1.0)),
    )


def write_metrics_csv(records: Iterable[dict], output_path: str | Path) -> None:
    records = list(records)
    if not records:
        return
    output_path = Path(output_path)
    if not output_path.is_absolute():
        output_path = ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted(
        {
            key
            for record in records
            for key, value in record.items()
            if not isinstance(value, (dict, list, tuple, np.ndarray))
        }
    )
    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            row = {key: value for key, value in record.items() if key in fieldnames}
            writer.writerow(row)


def build_baseline(policy_name: str, config: dict):
    return make_baseline(policy_name, config)


def resolve_tasks(config: dict) -> list[str]:
    if config.get("task_name"):
        return [config["task_name"]]
    return list(config.get("task_names", TASK_ORDER))


def build_reference_table_from_csv(records: Iterable[dict]) -> dict[tuple[str, int, str], dict[str, float]]:
    compact_records = []
    for record in records:
        compact_records.append(
            {
                "task_name": record["task_name"],
                "num_agents": int(record["num_agents"]),
                "scaling_mode": record.get("scaling_mode", "fixed_map"),
                "method": record["method"],
                "return_mean": float(record["return_mean"]),
            }
        )
    return compute_reference_table(compact_records)


def normalize_task_names(task_args: list[str] | None) -> list[str]:
    if not task_args or task_args == ["all"] or task_args == ["ALL"]:
        return list(TASK_ORDER)
    return [str(task_name) for task_name in task_args]


def format_task_set_name(task_names: list[str]) -> str:
    return "_".join(task_names)


def format_agent_set_name(agent_counts: list[int]) -> str:
    return "_".join(str(n) for n in agent_counts)


def format_obs_variant_name(obs_variant: str | None) -> str:
    variant = str(obs_variant or "multi_channel_field+task_id")
    return variant.replace("+", "_plus_").replace("-", "_").replace(" ", "_")


def observation_override_from_variant(obs_variant: str | None) -> dict:
    variant = str(obs_variant or "multi_channel_field+task_id")
    override = {
        "observation_mode": "multi_channel_field",
        "include_task_id": True,
        "include_agent_density": True,
        "drop_channels": [],
    }
    if variant == "task_id_only":
        warnings.warn(
            "'task_id_only' is a deprecated alias; use 'no_spatial_field' instead.",
            FutureWarning,
            stacklevel=2,
        )
        override["observation_mode"] = "no_spatial_field"
    elif variant == "no_spatial_field":
        override["observation_mode"] = "no_spatial_field"
    elif variant == "single_channel_field":
        override["observation_mode"] = "single_channel_field"
    elif variant in {"multi_channel_field", "multi_channel_field+task_id"}:
        pass
    elif variant == "multi_channel_field+agent_density_map":
        override["include_agent_density"] = True
    elif variant == "multi_channel_field_without_risk":
        override["drop_channels"] = ["risk"]
    elif variant == "multi_channel_field_without_desired_occupancy":
        override["drop_channels"] = ["desired_occupancy"]
    else:
        raise ValueError(f"Unsupported observation variant: {variant}")
    return override


def latest_checkpoint(directory: str | Path) -> Path | None:
    directory = Path(directory)
    if not directory.exists():
        return None
    checkpoints = sorted(directory.rglob("checkpoint*.pt"), key=lambda path: path.stat().st_mtime)
    return checkpoints[-1] if checkpoints else None


def create_summary_writer(run_dir: str | Path, enabled: bool = True):
    if not enabled:
        return None
    try:
        from torch.utils.tensorboard import SummaryWriter
    except Exception as exc:  # pragma: no cover - only hit in broken environments
        print(f"[tensorboard] disabled: {exc}", flush=True)
        return None
    return SummaryWriter(log_dir=str(tensorboard_dir(run_dir)))


def _is_numeric_scalar(value) -> bool:
    return isinstance(value, (Number, np.integer, np.floating, np.bool_))


TENSORBOARD_TRAIN_CORE_KEYS = frozenset(
    {
        "wall_clock_time",
        "mean_rollout_reward",
        "rollout_episode_success_rate",
        "rollout_terminal_goal_coverage",
        "rollout_terminal_collision_rate",
        "rollout_terminal_path_length",
        "policy_loss",
        "value_loss",
        "actor_loss",
        "critic_loss",
        "bc_loss",
        "entropy",
        "approx_kl",
        "kl_early_stop",
        "clip_frac",
        "ratio_mean",
        "reference_action_mse",
        "grad_norm",
        "explained_variance",
        "action_abs_mean",
        "action_saturation_frac",
        "log_std_mean",
        "policy_std_mean",
        "rollout_steps_per_sec",
        "episodes_completed",
        "cumulative_episodes",
        "memory_usage_mb",
        "inference_latency_ms",
        "eval_reward",
        "eval_success_rate",
        "eval_collision_rate",
        "eval_path_length",
        "eval_inference_latency_ms",
    }
)

TENSORBOARD_EVAL_CORE_METRIC_NAMES = frozenset(
    {
        "return",
        "success_rate",
        "collision_rate",
        "path_length",
        "normalized_score",
        "goal_coverage_ratio",
        "coverage_ratio",
        "repeated_coverage_ratio",
        "demand_revisit_excess",
        "formation_error",
        "cumulative_risk_exposure",
        "inference_latency_ms",
    }
)

TENSORBOARD_EVAL_CORE_SUFFIXES = tuple(
    f"_{metric_name}" for metric_name in TENSORBOARD_EVAL_CORE_METRIC_NAMES if metric_name != "return"
)

TENSORBOARD_FINAL_CORE_KEYS = frozenset(
    {
        f"{metric_name}_mean" for metric_name in TENSORBOARD_EVAL_CORE_METRIC_NAMES if metric_name != "reward"
    }
)


def normalize_tensorboard_metric_mode(mode: str | None) -> str:
    normalized = str(mode or "core").strip().lower()
    if normalized in {"core", "all"}:
        return normalized
    warnings.warn(
        f"Unknown tensorboard_metric_mode={mode!r}; falling back to 'core'. Valid modes are 'core' and 'all'.",
        RuntimeWarning,
        stacklevel=2,
    )
    return "core"


def _is_core_eval_metric_key(key: str, prefix: str = "eval_") -> bool:
    if not key.startswith(prefix):
        return False
    if key in TENSORBOARD_TRAIN_CORE_KEYS:
        return True
    metric_part = key[len(prefix) :]
    if metric_part == "return" or (metric_part.endswith("_return") and not metric_part.endswith("_raw_return")):
        return True
    return any(metric_part == suffix[1:] or metric_part.endswith(suffix) for suffix in TENSORBOARD_EVAL_CORE_SUFFIXES)


def _should_log_tensorboard_metric(
    key: str,
    *,
    metric_mode: str,
    metric_phase: str,
    include_keys: set[str] | None,
) -> bool:
    if metric_mode == "all":
        return True
    if include_keys is not None:
        return key in include_keys
    if metric_phase == "final_eval":
        return key in TENSORBOARD_FINAL_CORE_KEYS
    return key in TENSORBOARD_TRAIN_CORE_KEYS or _is_core_eval_metric_key(key)


def log_scalar_metrics(
    writer,
    namespace: str,
    step: int,
    metrics: dict,
    exclude_keys: set[str] | None = None,
    tensorboard_metric_mode: str = "core",
    include_keys: Iterable[str] | None = None,
    metric_phase: str = "train",
) -> None:
    if writer is None:
        return
    excluded = set(exclude_keys or set())
    metric_mode = normalize_tensorboard_metric_mode(tensorboard_metric_mode)
    included = set(include_keys) if include_keys is not None else None
    for key, value in metrics.items():
        if key in excluded or not _is_numeric_scalar(value):
            continue
        if not _should_log_tensorboard_metric(
            str(key),
            metric_mode=metric_mode,
            metric_phase=metric_phase,
            include_keys=included,
        ):
            continue
        scalar = float(value)
        if not np.isfinite(scalar):
            continue
        writer.add_scalar(f"{namespace}/{key}", scalar, global_step=int(step))


def _format_progress_value(value) -> str:
    if isinstance(value, (bool, np.bool_)):
        return str(int(value))
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.3f}"
    return str(value)


def print_progress_line(
    prefix: str,
    step_key: str,
    step_value,
    metrics: dict,
    key_order: list[str] | None = None,
) -> None:
    parts = [str(prefix), f"{step_key}={_format_progress_value(step_value)}"]
    keys = list(key_order or [])
    if keys:
        for key in keys:
            if key in metrics and _is_numeric_scalar(metrics[key]):
                parts.append(f"{key}={_format_progress_value(metrics[key])}")
        print(" | ".join(parts), flush=True)
        return
    seen = {step_key}
    for key in keys:
        if key in metrics and _is_numeric_scalar(metrics[key]):
            parts.append(f"{key}={_format_progress_value(metrics[key])}")
            seen.add(key)
    for key, value in metrics.items():
        if key in seen or key.startswith("_") or key.endswith("_path"):
            continue
        if _is_numeric_scalar(value):
            parts.append(f"{key}={_format_progress_value(value)}")
    print(" | ".join(parts), flush=True)


def build_metric_logger(
    run_dir: str | Path,
    namespace: str,
    step_key: str,
    tensorboard_enabled: bool = True,
    console_interval: int = 1,
    key_order: list[str] | None = None,
    tensorboard_metric_mode: str = "core",
    tensorboard_include_keys: Iterable[str] | None = None,
) -> tuple[object | None, Callable[[dict], None]]:
    writer = create_summary_writer(run_dir, enabled=tensorboard_enabled)
    interval = max(int(console_interval), 1)
    state = {"last_console_step": None}
    excluded = {step_key, "checkpoint_path", "eval_media_dir"}
    metric_mode = normalize_tensorboard_metric_mode(tensorboard_metric_mode)
    include_keys = set(tensorboard_include_keys) if tensorboard_include_keys is not None else None

    def log_record(record: dict) -> None:
        if step_key not in record:
            return
        step_value = int(record[step_key])
        log_scalar_metrics(
            writer,
            namespace,
            step_value,
            record,
            exclude_keys=excluded,
            tensorboard_metric_mode=metric_mode,
            include_keys=include_keys,
            metric_phase="train",
        )
        should_print = state["last_console_step"] is None
        if not should_print and abs(step_value - int(state["last_console_step"])) >= interval:
            should_print = True
        if not should_print and any(key.startswith("eval_") for key in record):
            should_print = True
        if should_print:
            print_progress_line(namespace, step_key, step_value, record, key_order=key_order)
            state["last_console_step"] = step_value

    return writer, log_record
