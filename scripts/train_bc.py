from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algorithms import BCTrainer
from policies import build_policy
from scripts._common import (
    baseline_reference_episodes_for_agent_count,
    baseline_reference_table,
    build_metric_logger,
    ensure_dir,
    format_agent_set_name,
    format_obs_variant_name,
    format_task_set_name,
    latest_checkpoint,
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
from utils import ExpertDataset, aggregate_episode_records, apply_reference_normalization, evaluate_policy_episodes


def infer_dataset_path(task_names: list[str], agent_counts: list[int], obs_variant: str) -> Path:
    base = ensure_dir("outputs/datasets")
    obs_tag = format_obs_variant_name(obs_variant)
    if len(task_names) == 1 and len(agent_counts) == 1:
        return base / f"expert_{task_names[0]}_N{agent_counts[0]}_{obs_tag}.npz"
    return base / f"expert_{format_task_set_name(task_names)}_N{format_agent_set_name(agent_counts)}_{obs_tag}.npz"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/policy/bc_cnn_deepsets.yaml")
    parser.add_argument("--tasks", nargs="+", default=["goal_nav", "coverage"])
    parser.add_argument("--agent_counts", nargs="+", type=int, default=[4])
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--init_checkpoint", default=None)
    parser.add_argument("--env-config", default="configs/env/multitask.yaml")
    parser.add_argument("--scaling_mode", default="fixed_map")
    parser.add_argument("--obs_variant", default="multi_channel_field+task_id")
    parser.add_argument("--eval_episodes", type=int, default=5)
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tensorboard", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--console_log_interval", type=int, default=1)
    parser.add_argument("--record_eval_episodes", type=int, default=0)
    parser.add_argument("--record_format", choices=["gif", "mp4"], default="gif")
    parser.add_argument("--record_fps", type=int, default=8)
    args = parser.parse_args()

    task_names = normalize_task_names(args.tasks)
    agent_counts = [int(n) for n in args.agent_counts]
    policy_config = load_generic_config(args.config)
    if policy_config["policy_class"] == "mlp" and len(set(agent_counts)) > 1:
        raise ValueError("Flatten MLP BC is restricted to fixed-N training. Use a single agent count.")

    dataset_path = Path(args.dataset) if args.dataset else infer_dataset_path(task_names, agent_counts, args.obs_variant)
    dataset = ExpertDataset(dataset_path)
    build_num_agents = max(agent_counts) if policy_config["policy_class"] != "mlp" else agent_counts[0]
    env_config = prepare_env_config(
        args.env_config,
        tasks=task_names,
        num_agents=build_num_agents,
        scaling_mode=args.scaling_mode,
        observation_override=observation_override_from_variant(args.obs_variant),
    )
    policy_config.setdefault("bc_waypoint_step", float(env_config.get("max_waypoint_step", 1.0)))
    from envs import CentralizedMultiUAVEnv

    build_env = CentralizedMultiUAVEnv(env_config)
    policy = build_policy(policy_config, build_env.observation_space, build_env.action_space)
    trainer = BCTrainer(policy, policy_config)
    if args.init_checkpoint:
        checkpoint = torch.load(args.init_checkpoint, map_location=trainer.device)
        current_state = trainer.policy.state_dict()
        compatible_state = {
            key: value
            for key, value in checkpoint["model_state_dict"].items()
            if key in current_state and current_state[key].shape == value.shape
        }
        skipped_keys = sorted(set(checkpoint["model_state_dict"]) - set(compatible_state))
        trainer.policy.load_state_dict(compatible_state, strict=False)
        if skipped_keys:
            preview = ", ".join(skipped_keys[:8])
            suffix = "..." if len(skipped_keys) > 8 else ""
            print(f"bc_init_skipped_mismatched_or_missing={len(skipped_keys)} [{preview}{suffix}]")
    run_name = f"{policy_config['name']}_{format_task_set_name(task_names)}_N{format_agent_set_name(agent_counts)}_{format_obs_variant_name(args.obs_variant)}"
    output_dir = timestamped_training_dir("bc", run_name)
    save_run_snapshot(
        output_dir,
        train_config=policy_config,
        env_config=env_config,
        cli_args=vars(args),
        model_state_dict=trainer.policy.state_dict(),
        extra_metadata={
            "task_names": task_names,
            "agent_counts": agent_counts,
            "output_root": "bc",
            "init_checkpoint": args.init_checkpoint,
        },
    )
    writer, log_record = build_metric_logger(
        output_dir,
        namespace="bc/train",
        step_key="epoch",
        tensorboard_enabled=args.tensorboard,
        console_interval=args.console_log_interval,
        tensorboard_metric_mode=str(policy_config.get("tensorboard_metric_mode", "core")),
        key_order=["bc_loss", "epoch_time_sec", "wall_clock_time", "memory_usage_mb"],
    )
    try:
        history = trainer.train(dataset, output_dir, log_callback=log_record)
        write_metrics_csv(history, output_dir / "training_metrics.csv")

        final_checkpoint = latest_checkpoint(output_dir)
        if final_checkpoint is None:
            raise RuntimeError("BC training did not produce a checkpoint.")
        eval_records = []
        trainer.policy.load_state_dict(torch.load(final_checkpoint, map_location=trainer.device)["model_state_dict"], strict=False)
        final_epoch = int(history[-1].get("epoch", 0)) if history else 0
        for agent_count in agent_counts:
            eval_env_config = prepare_env_config(
                args.env_config,
                tasks=task_names,
                num_agents=agent_count,
                scaling_mode=args.scaling_mode,
                observation_override=observation_override_from_variant(args.obs_variant),
            )
            eval_env = CentralizedMultiUAVEnv(eval_env_config)
            final_media_dir = output_dir / "final_eval_media" / f"N{agent_count}" if args.record_eval_episodes > 0 else None
            records = evaluate_policy_episodes(
                eval_env,
                trainer.policy,
                args.eval_episodes,
                trainer.device,
                headless=args.headless,
                record_dir=final_media_dir,
                record_episodes=min(int(args.record_eval_episodes), int(args.eval_episodes)),
                record_format=args.record_format,
                record_fps=args.record_fps,
                record_prefix=f"final_eval_N{agent_count}",
            )
            reference_table = baseline_reference_table(
                eval_env_config,
                episodes=baseline_reference_episodes_for_agent_count(agent_count, max(4, args.eval_episodes // 2)),
            )
            normalized_records = [apply_reference_normalization(record, reference_table) for record in records]
            summary = aggregate_episode_records(normalized_records)
            log_scalar_metrics(
                writer,
                f"bc/final_eval/N{agent_count}",
                final_epoch,
                summary,
                tensorboard_metric_mode=str(policy_config.get("tensorboard_metric_mode", "core")),
                metric_phase="final_eval",
            )
            print_progress_line(
                "bc-final",
                "num_agents",
                agent_count,
                summary,
                key_order=["return_mean", "normalized_score_mean", "success_rate_mean", "collision_rate_mean"],
            )
            eval_records.append(
                {
                    "method": "bc",
                    "algorithm": "bc",
                    "architecture": policy_config["policy_class"],
                    "observation_mode": eval_env_config.get("observation_mode", "multi_channel_field"),
                    "task_set": format_task_set_name(task_names),
                    "task_name": "multitask" if len(task_names) > 1 else task_names[0],
                    "num_agents": agent_count,
                    "scaling_mode": args.scaling_mode,
                    "seed": int(eval_env_config.get("seed", 0)),
                    "normalized_score": float(summary.get("normalized_score_mean", 0.0)),
                    **summary,
                }
            )
        write_metrics_csv(eval_records, output_dir / "eval_metrics.csv")
    finally:
        if writer is not None:
            writer.close()
    print(f"bc_output={output_dir}")


if __name__ == "__main__":
    main()
