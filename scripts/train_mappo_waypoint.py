from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algorithms.mappo_waypoint import MAPPOWaypointTrainer, write_metrics_csv
from envs.waypoint_marl_env import WaypointMultiUAVEnv
from policies import build_policy
from utils.waypoint_vector_env import SyncWaypointEnvBatch, ThreadWaypointEnvBatch, make_waypoint_env_batch


def load_yaml(path: str | Path) -> dict:
    with open(ROOT / Path(path), "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def deep_update(base: dict, override: dict) -> dict:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value
    return result


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(value)).strip("_") or "run"


def sync_policy_env_config(train_config: dict, env_config: dict) -> None:
    for key in ["map_size", "grid_size", "max_waypoint_distance", "min_uav_distance", "base_position", "no_fly_zones"]:
        if key in env_config:
            train_config[key] = env_config[key]
    if "max_delta" not in train_config and "max_waypoint_distance" in env_config:
        train_config["max_delta"] = env_config["max_waypoint_distance"]


def make_output_dir(timestamp: str | None, run_name: str, output_dir: str | None = None) -> Path:
    if output_dir is not None:
        output = Path(output_dir)
        if not output.is_absolute():
            output = ROOT / output
        output.mkdir(parents=True, exist_ok=True)
        return output
    run_timestamp = timestamp or datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    output_dir = ROOT / "outputs" / "training" / "mappo_waypoint" / run_timestamp / safe_name(run_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def build_env_batch(env_config: dict, train_config: dict, tasks: list[str], envs_per_task: int | None, backend: str, workers: int | None):
    if envs_per_task is None:
        return make_waypoint_env_batch(env_config, int(train_config.get("num_envs", 1)), backend=backend, max_workers=workers)
    envs = []
    for task_idx, task_name in enumerate(tasks):
        for local_idx in range(int(envs_per_task)):
            cfg = deepcopy(env_config)
            cfg["task_name"] = task_name
            cfg["task_names"] = [task_name]
            cfg["seed"] = int(env_config.get("seed", 0)) + task_idx * int(envs_per_task) + local_idx
            envs.append(WaypointMultiUAVEnv(cfg))
    if backend == "sync":
        return SyncWaypointEnvBatch(envs)
    if backend == "thread":
        return ThreadWaypointEnvBatch(envs, max_workers=workers)
    raise ValueError(f"Unsupported env backend: {backend}")


def build_writer(output_dir: Path, enabled: bool):
    if not enabled:
        return None
    try:
        from torch.utils.tensorboard import SummaryWriter

        return SummaryWriter(str(output_dir / "tensorboard"))
    except Exception as exc:
        print(f"[warn] TensorBoard disabled: {exc}")
        return None


def log_tensorboard(writer, record: dict) -> None:
    if writer is None:
        return
    step = int(record.get("update", 0))
    for key, value in record.items():
        if isinstance(value, (int, float)) and key != "update":
            writer.add_scalar(key, float(value), step)
    writer.flush()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/policy/mappo_waypoint.yaml")
    parser.add_argument("--env-config", default="configs/env/waypoint_missions.yaml")
    parser.add_argument("--tasks", nargs="+", default=["area_coverage", "belief_search", "priority_inspection", "connectivity_expansion"])
    parser.add_argument("--num_agents", type=int, default=None)
    parser.add_argument("--num_envs", type=int, default=None)
    parser.add_argument("--envs_per_task", type=int, default=None)
    parser.add_argument("--total_updates", type=int, default=None)
    parser.add_argument("--rollout_steps", type=int, default=None)
    parser.add_argument("--eval_interval", type=int, default=None)
    parser.add_argument("--eval_episodes", type=int, default=5)
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tensorboard", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--record_eval_episodes", type=int, default=0)
    parser.add_argument("--record_format", choices=["gif", "mp4"], default="gif")
    parser.add_argument("--record_fps", type=int, default=8)
    parser.add_argument("--record_interval", type=int, default=1)
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--run_timestamp", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--env_backend", choices=["sync", "thread"], default="sync")
    parser.add_argument("--env_workers", type=int, default=None)
    args = parser.parse_args()

    train_config = load_yaml(args.config)
    env_config = load_yaml(args.env_config)
    task_names = [str(task) for task in args.tasks]
    env_config["task_names"] = task_names
    env_config["task_name"] = task_names[0] if len(task_names) == 1 else None
    env_config["task_sampling_probs"] = {task: 1.0 for task in task_names}
    if args.num_agents is not None:
        env_config["num_agents"] = int(args.num_agents)
    if args.seed is not None:
        env_config["seed"] = int(args.seed)
    if args.num_envs is not None:
        train_config["num_envs"] = int(args.num_envs)
    if args.total_updates is not None:
        train_config["total_updates"] = int(args.total_updates)
    if args.rollout_steps is not None:
        train_config["rollout_steps"] = int(args.rollout_steps)
    if args.eval_interval is not None:
        train_config["eval_interval"] = int(args.eval_interval)
    sync_policy_env_config(train_config, env_config)

    env_batch = build_env_batch(env_config, train_config, task_names, args.envs_per_task, args.env_backend, args.env_workers)
    policy = build_policy(train_config, env_batch.envs[0].global_observation_space, env_batch.envs[0].action_space_n)
    trainer = MAPPOWaypointTrainer(env_batch, policy, train_config)
    run_name = args.run_name or f"{train_config.get('name', 'mappo_waypoint')}_{'_'.join(task_names)}_N{env_config['num_agents']}"
    output_dir = make_output_dir(args.run_timestamp, run_name, args.output_dir)
    snapshot = output_dir / "snapshot"
    snapshot.mkdir(parents=True, exist_ok=True)
    (snapshot / "train_config.yaml").write_text(yaml.safe_dump(train_config, sort_keys=False), encoding="utf-8")
    (snapshot / "env_config.yaml").write_text(yaml.safe_dump(env_config, sort_keys=False), encoding="utf-8")
    (snapshot / "cli_args.yaml").write_text(yaml.safe_dump(vars(args), sort_keys=False), encoding="utf-8")
    (snapshot / "metadata.json").write_text(json.dumps({"created_at": datetime.now().astimezone().isoformat(), "main_env": "WaypointMultiUAVEnv"}, indent=2) + "\n", encoding="utf-8")
    writer = build_writer(output_dir, args.tensorboard)
    history = []

    def on_record(record: dict) -> None:
        history.append(dict(record))
        write_metrics_csv(history, output_dir / "training_metrics.csv")
        log_tensorboard(writer, record)
        update = int(record.get("update", 0))
        if update == 1 or update % max(int(train_config.get("eval_interval", 1)), 1) == 0:
            print(
                f"[waypoint-mappo] update={update} reward={record.get('mean_team_reward', 0.0):.3f} "
                f"eval_success={record.get('eval_success_rate', 0.0):.3f} eval_reward={record.get('eval_reward', 0.0):.3f}",
                flush=True,
            )

    final_history = trainer.train(
        output_dir,
        env_config=env_config,
        task_names=task_names,
        eval_episodes=int(args.eval_episodes),
        headless=bool(args.headless),
        record_eval_episodes=int(args.record_eval_episodes),
        record_format=args.record_format,
        record_fps=int(args.record_fps),
        record_interval=int(args.record_interval),
        log_callback=on_record,
    )
    if not history:
        history = final_history
    write_metrics_csv(history, output_dir / "training_metrics.csv")
    if writer is not None:
        writer.close()
    env_batch.close()
    print(f"[waypoint-mappo] output_dir={output_dir}", flush=True)


if __name__ == "__main__":
    main()
