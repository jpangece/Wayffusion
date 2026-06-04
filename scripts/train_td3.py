from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algorithms import TD3Trainer
from policies import build_policy
from scripts._common import (
    baseline_reference_episodes_for_agent_count,
    build_metric_logger,
    format_agent_set_name,
    format_obs_variant_name,
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
from utils import evaluate_policy_per_task, make_env_batch, make_task_balanced_env_batch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/policy/td3_cnn_deepsets.yaml")
    parser.add_argument("--env-config", default="configs/env/multitask.yaml")
    parser.add_argument("--tasks", nargs="+", default=["goal_nav", "coverage"])
    parser.add_argument("--agent_counts", nargs="+", type=int, default=[4])
    parser.add_argument("--scaling_mode", default="fixed_map")
    parser.add_argument("--obs_variant", default="multi_channel_field+task_id")
    parser.add_argument("--init_checkpoint", default=None)
    parser.add_argument("--eval_episodes", type=int, default=5)
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tensorboard", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--console_log_interval", type=int, default=1024)
    parser.add_argument("--record_eval_episodes", type=int, default=0)
    parser.add_argument("--record_format", choices=["gif", "mp4"], default="gif")
    parser.add_argument("--record_fps", type=int, default=8)
    parser.add_argument("--record_interval", type=int, default=1)
    parser.add_argument("--env_backend", choices=["sync", "thread"], default="sync")
    parser.add_argument("--envs_per_task", type=int, default=None)
    parser.add_argument("--env_workers", type=int, default=None)
    args = parser.parse_args()

    task_names = normalize_task_names(args.tasks)
    agent_count = int(args.agent_counts[0])
    config = load_generic_config(args.config)
    env_config = prepare_env_config(
        args.env_config,
        tasks=task_names,
        num_agents=agent_count,
        scaling_mode=args.scaling_mode,
        observation_override=observation_override_from_variant(args.obs_variant),
    )
    if args.envs_per_task is not None:
        env_batch = make_task_balanced_env_batch(
            env_config,
            task_names=task_names,
            envs_per_task=args.envs_per_task,
            backend=args.env_backend,
            max_workers=args.env_workers,
        )
    else:
        env_batch = make_env_batch(
            env_config,
            int(config.get("num_envs", 1)),
            backend=args.env_backend,
            max_workers=args.env_workers,
        )
    policy = build_policy(config, env_batch.envs[0].observation_space, env_batch.envs[0].action_space)
    if args.init_checkpoint:
        checkpoint = torch.load(args.init_checkpoint, map_location="cpu")
        policy.load_state_dict(checkpoint["model_state_dict"], strict=False)
    trainer = TD3Trainer(env_batch, policy, config)
    output_root = "bc_td3" if args.init_checkpoint else "td3"
    run_name = f"{config['name']}_{format_task_set_name(task_names)}_N{format_agent_set_name([agent_count])}_{format_obs_variant_name(args.obs_variant)}"
    output_dir = timestamped_training_dir(output_root, run_name)
    save_run_snapshot(
        output_dir,
        train_config=config,
        env_config=env_config,
        cli_args=vars(args),
        model_state_dict=trainer.actor.state_dict(),
        extra_metadata={"task_names": task_names, "agent_counts": [agent_count], "output_root": output_root},
    )
    writer, log_record = build_metric_logger(
        output_dir,
        namespace=f"{output_root}/train",
        step_key="step",
        tensorboard_enabled=args.tensorboard,
        console_interval=args.console_log_interval,
        tensorboard_metric_mode=str(config.get("tensorboard_metric_mode", "core")),
        key_order=[
            "rollout_reward",
            "actor_loss",
            "critic_loss",
            "memory_usage_mb",
            "inference_latency_ms",
            "eval_reward",
            "eval_success_rate",
        ],
    )
    try:
        metrics = trainer.train(
            output_dir,
            eval_env=env_batch.envs[0],
            eval_task_names=task_names,
            eval_base_env_config=env_config,
            eval_episodes=args.eval_episodes,
            headless=args.headless,
            record_eval_episodes=args.record_eval_episodes,
            record_format=args.record_format,
            record_fps=args.record_fps,
            record_interval=args.record_interval,
            log_callback=log_record,
        )
        write_metrics_csv(metrics, output_dir / "training_metrics.csv")

        final_media_dir = output_dir / "final_eval_media" if args.record_eval_episodes > 0 else None
        _, task_summaries, overall_summary = evaluate_policy_per_task(
            env_config,
            trainer.actor,
            task_names,
            args.eval_episodes,
            trainer.device,
            headless=args.headless,
            record_dir=final_media_dir,
            record_episodes=min(int(args.record_eval_episodes), int(args.eval_episodes)),
            record_format=args.record_format,
            record_fps=args.record_fps,
            record_prefix=f"final_eval_N{agent_count}",
            normalize_with_reference=True,
            reference_episodes=baseline_reference_episodes_for_agent_count(agent_count, max(4, args.eval_episodes // 2)),
        )
        final_step = int(metrics[-1].get("step", trainer.total_steps)) if metrics else int(trainer.total_steps)
        eval_rows = []
        for task_name, task_summary in task_summaries.items():
            log_scalar_metrics(
                writer,
                f"{output_root}/final_eval/N{agent_count}/{task_name}",
                final_step,
                task_summary,
                tensorboard_metric_mode=str(config.get("tensorboard_metric_mode", "core")),
                metric_phase="final_eval",
            )
            print_progress_line(
                f"{output_root}-final/{task_name}",
                "num_agents",
                agent_count,
                task_summary,
                key_order=["return_mean", "normalized_score_mean", "success_rate_mean", "collision_rate_mean"],
            )
            eval_rows.append(
                {
                    **task_summary,
                    "method": "bc_td3" if args.init_checkpoint else "td3",
                    "algorithm": "td3",
                    "architecture": config["policy_class"],
                    "observation_mode": env_config.get("observation_mode", "multi_channel_field"),
                    "obs_variant": args.obs_variant,
                    "task_set": format_task_set_name(task_names),
                    "task_name": task_name,
                    "eval_group": "per_task",
                    "num_agents": agent_count,
                    "scaling_mode": args.scaling_mode,
                    "seed": int(env_config.get("seed", 0)),
                    "normalized_score": float(task_summary.get("normalized_score_mean", 0.0)),
                }
            )
        log_scalar_metrics(
            writer,
            f"{output_root}/final_eval/N{agent_count}/overall",
            final_step,
            overall_summary,
            tensorboard_metric_mode=str(config.get("tensorboard_metric_mode", "core")),
            metric_phase="final_eval",
        )
        print_progress_line(
            f"{output_root}-final/overall",
            "num_agents",
            agent_count,
            overall_summary,
            key_order=["return_mean", "normalized_score_mean", "success_rate_mean", "collision_rate_mean"],
        )
        eval_rows.append(
            {
                **overall_summary,
                "method": "bc_td3" if args.init_checkpoint else "td3",
                "algorithm": "td3",
                "architecture": config["policy_class"],
                "observation_mode": env_config.get("observation_mode", "multi_channel_field"),
                "obs_variant": args.obs_variant,
                "task_set": format_task_set_name(task_names),
                "task_name": "overall",
                "eval_group": "overall",
                "num_agents": agent_count,
                "scaling_mode": args.scaling_mode,
                "seed": int(env_config.get("seed", 0)),
                "normalized_score": float(overall_summary.get("normalized_score_mean", 0.0)),
            }
        )
        write_metrics_csv(eval_rows, output_dir / "eval_metrics.csv")
    finally:
        if writer is not None:
            writer.close()
    print(f"td3_output={output_dir}")


if __name__ == "__main__":
    main()
