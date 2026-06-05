from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algorithms import MAPPOTrainer
from policies import build_policy
from scripts.train_ppo import compact_task_set_name, safe_run_name
from scripts._common import (
    baseline_reference_episodes_for_agent_count,
    build_metric_logger,
    format_agent_set_name,
    format_task_set_name,
    log_scalar_metrics,
    load_generic_config,
    normalize_task_names,
    observation_override_from_variant,
    print_progress_line,
    prepare_env_config,
    save_run_snapshot,
    timestamped_training_dir,
    write_metrics_csv,
)
from utils import evaluate_policy_per_task, flatten_task_eval_summaries, make_env_batch, make_task_balanced_env_batch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/policy/mappo_shared.yaml")
    parser.add_argument("--env-config", default="configs/env/multitask.yaml")
    parser.add_argument("--tasks", nargs="+", default=["goal_nav", "coverage"])
    parser.add_argument("--agent_counts", nargs="+", type=int, default=[4])
    parser.add_argument("--scaling_mode", default="fixed_map")
    parser.add_argument("--obs_variant", default="multi_channel_field+task_id")
    parser.add_argument("--eval_episodes", type=int, default=5)
    parser.add_argument("--total_updates", type=int, default=None)
    parser.add_argument("--target_episodes", type=int, default=None)
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tensorboard", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--console_log_interval", type=int, default=5)
    parser.add_argument("--record_eval_episodes", type=int, default=0)
    parser.add_argument("--record_format", choices=["gif", "mp4"], default="gif")
    parser.add_argument("--record_fps", type=int, default=8)
    parser.add_argument("--record_interval", type=int, default=1)
    parser.add_argument("--env_backend", choices=["sync", "thread"], default="sync")
    parser.add_argument("--envs_per_task", type=int, default=None)
    parser.add_argument("--env_workers", type=int, default=None)
    parser.add_argument("--final_eval_source", choices=["best", "last"], default="best")
    parser.add_argument("--run_timestamp", default=None)
    parser.add_argument("--run_name", default=None)
    args = parser.parse_args()

    task_names = normalize_task_names(args.tasks)
    agent_counts = [int(n) for n in args.agent_counts]
    if len(set(agent_counts)) != 1:
        raise ValueError("Minimal MAPPO baseline currently supports one agent count per run.")

    train_config = load_generic_config(args.config)
    if args.total_updates is not None:
        train_config["total_updates"] = int(args.total_updates)
    if args.target_episodes is not None:
        train_config["target_episodes"] = int(args.target_episodes)
    train_config["algorithm"] = "mappo"

    num_agents = agent_counts[0]
    base_env_config = prepare_env_config(
        args.env_config,
        tasks=task_names,
        num_agents=num_agents,
        scaling_mode=args.scaling_mode,
        observation_override=observation_override_from_variant(args.obs_variant),
    )
    if args.envs_per_task is not None:
        env_batch = make_task_balanced_env_batch(
            base_env_config,
            task_names=task_names,
            envs_per_task=args.envs_per_task,
            backend=args.env_backend,
            max_workers=args.env_workers,
        )
    else:
        env_batch = make_env_batch(
            base_env_config,
            int(train_config.get("num_envs", 1)),
            backend=args.env_backend,
            max_workers=args.env_workers,
        )

    policy = build_policy(train_config, env_batch.envs[0].observation_space, env_batch.envs[0].action_space)
    trainer = MAPPOTrainer(env_batch, policy, train_config)

    run_name = safe_run_name(
        args.run_name or f"{train_config['name']}_{compact_task_set_name(task_names)}_N{format_agent_set_name(agent_counts)}"
    )
    tensorboard_metric_mode = str(train_config.get("tensorboard_metric_mode", "core"))
    tensorboard_task_namespace = safe_run_name(format_task_set_name(task_names))
    output_dir = timestamped_training_dir("mappo", run_name, timestamp=args.run_timestamp)
    save_run_snapshot(
        output_dir,
        train_config=train_config,
        env_config=base_env_config,
        cli_args=vars(args),
        model_state_dict=trainer.policy.state_dict(),
        extra_metadata={"task_names": task_names, "agent_counts": agent_counts, "output_root": "mappo"},
    )
    writer, log_record = build_metric_logger(
        output_dir,
        namespace=f"mappo/{tensorboard_task_namespace}/train",
        step_key="update",
        tensorboard_enabled=args.tensorboard,
        console_interval=args.console_log_interval,
        tensorboard_metric_mode=tensorboard_metric_mode,
        key_order=[
            "mean_rollout_reward",
            "policy_loss",
            "value_loss",
            "entropy",
            "approx_kl",
            "clip_frac",
            "ratio_mean",
            "grad_norm",
            "eval_reward",
            "eval_success_rate",
            "eval_collision_rate",
            "eval_path_length",
        ],
    )
    metrics_csv_path = output_dir / "training_metrics.csv"
    metrics_flush_interval = max(int(args.console_log_interval), 1)
    persisted_history: list[dict] = []

    def log_and_persist(record: dict) -> None:
        persisted_history.append(dict(record))
        log_record(record)
        update_idx = int(record.get("update", 0))
        if update_idx % metrics_flush_interval == 0 or "checkpoint_path" in record:
            write_metrics_csv(persisted_history, metrics_csv_path)

    try:
        history = trainer.train(
            output_dir,
            eval_env=env_batch.envs[0],
            eval_task_names=task_names,
            eval_base_env_config=base_env_config,
            eval_episodes=args.eval_episodes,
            headless=args.headless,
            record_eval_episodes=args.record_eval_episodes,
            record_format=args.record_format,
            record_fps=args.record_fps,
            record_interval=args.record_interval,
            log_callback=log_and_persist,
        )
        if persisted_history:
            history = persisted_history
        write_metrics_csv(history, metrics_csv_path)

        final_eval_checkpoint = output_dir / "checkpoints" / "checkpoint_best_eval.pt"
        final_eval_source = "last"
        if args.final_eval_source == "best" and final_eval_checkpoint.exists():
            checkpoint = torch.load(final_eval_checkpoint, map_location=trainer.device)
            trainer.policy.load_state_dict(checkpoint["model_state_dict"], strict=False)
            final_eval_source = "best"

        eval_records = []
        final_update = int(history[-1].get("update", 0)) if history else 0
        final_media_dir = output_dir / "final_eval_media" if args.record_eval_episodes > 0 else None
        _, task_summaries, overall_summary = evaluate_policy_per_task(
            base_env_config,
            trainer.policy,
            task_names,
            args.eval_episodes,
            trainer.device,
            headless=args.headless,
            record_dir=final_media_dir,
            record_episodes=min(int(args.record_eval_episodes), int(args.eval_episodes)),
            record_format=args.record_format,
            record_fps=args.record_fps,
            record_prefix="final_eval",
            normalize_with_reference=True,
            reference_episodes=baseline_reference_episodes_for_agent_count(num_agents, max(4, args.eval_episodes // 2)),
        )
        for task_name, task_summary in task_summaries.items():
            log_scalar_metrics(
                writer,
                f"mappo/{tensorboard_task_namespace}/final_eval/N{num_agents}/{task_name}",
                final_update,
                task_summary,
                tensorboard_metric_mode=tensorboard_metric_mode,
                metric_phase="final_eval",
            )
            print_progress_line(
                f"mappo-final/{task_name}",
                "num_agents",
                num_agents,
                task_summary,
                key_order=["return_mean", "normalized_score_mean", "success_rate_mean", "collision_rate_mean"],
            )
            eval_records.append(
                {
                    **task_summary,
                    "method": "mappo",
                    "algorithm": "mappo",
                    "architecture": train_config["policy_class"],
                    "observation_mode": base_env_config.get("observation_mode", "multi_channel_field"),
                    "obs_variant": args.obs_variant,
                    "task_set": format_task_set_name(task_names),
                    "task_name": task_name,
                    "eval_group": "per_task",
                    "final_eval_source": final_eval_source,
                    "num_agents": num_agents,
                    "scaling_mode": args.scaling_mode,
                    "seed": int(base_env_config.get("seed", 0)),
                    "normalized_score": float(task_summary.get("normalized_score_mean", 0.0)),
                }
            )

        log_scalar_metrics(
            writer,
            f"mappo/{tensorboard_task_namespace}/final_eval/N{num_agents}/overall",
            final_update,
            overall_summary,
            tensorboard_metric_mode=tensorboard_metric_mode,
            metric_phase="final_eval",
        )
        print_progress_line(
            "mappo-final/overall",
            "num_agents",
            num_agents,
            overall_summary,
            key_order=["return_mean", "normalized_score_mean", "success_rate_mean", "collision_rate_mean"],
        )
        eval_records.append(
            {
                **overall_summary,
                "method": "mappo",
                "algorithm": "mappo",
                "architecture": train_config["policy_class"],
                "observation_mode": base_env_config.get("observation_mode", "multi_channel_field"),
                "obs_variant": args.obs_variant,
                "task_set": format_task_set_name(task_names),
                "task_name": "overall",
                "eval_group": "overall",
                "final_eval_source": final_eval_source,
                "num_agents": num_agents,
                "scaling_mode": args.scaling_mode,
                "seed": int(base_env_config.get("seed", 0)),
                "normalized_score": float(overall_summary.get("normalized_score_mean", 0.0)),
            }
        )
        write_metrics_csv(eval_records, output_dir / "eval_metrics.csv")
        (output_dir / "run_summary.json").write_text(
            json.dumps(
                {
                    "output_dir": str(output_dir),
                    "final_eval_source": final_eval_source,
                    "final_update": final_update,
                    "task_names": task_names,
                    "agent_counts": agent_counts,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    finally:
        if writer is not None:
            writer.close()
    print(f"mappo_output={output_dir}")


if __name__ == "__main__":
    main()
