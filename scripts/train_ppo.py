from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algorithms import PPOTrainer
from policies import build_policy
from scripts._common import (
    baseline_reference_episodes_for_agent_count,
    build_metric_logger,
    checkpoints_dir,
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


def compact_task_set_name(task_names: list[str]) -> str:
    task_aliases = {
        "goal_nav": "goal",
        "coverage": "cov",
        "formation": "form",
        "risk_nav": "risk",
    }
    canonical_all_tasks = {"goal_nav", "coverage", "formation", "risk_nav"}
    if set(task_names) == canonical_all_tasks and len(task_names) == len(canonical_all_tasks):
        return "all4"
    return "_".join(task_aliases.get(task_name, str(task_name)) for task_name in task_names)


def safe_run_name(value: str) -> str:
    keep = []
    for char in str(value):
        if char.isalnum() or char in {"-", "_"}:
            keep.append(char)
        else:
            keep.append("_")
    return "".join(keep).strip("_") or "ppo_run"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/policy/ppo_cnn_deepsets.yaml")
    parser.add_argument("--env-config", default="configs/env/multitask.yaml")
    parser.add_argument("--tasks", nargs="+", default=["goal_nav", "coverage"])
    parser.add_argument("--agent_counts", nargs="+", type=int, default=[4])
    parser.add_argument("--scaling_mode", default="fixed_map")
    parser.add_argument("--init_checkpoint", default=None)
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
    train_config = load_generic_config(args.config)
    agent_counts = [int(n) for n in args.agent_counts]
    if args.total_updates is not None:
        train_config["total_updates"] = int(args.total_updates)
    if args.target_episodes is not None:
        train_config["target_episodes"] = int(args.target_episodes)
    variable_n = len(set(agent_counts)) > 1
    if variable_n and train_config["policy_class"] == "mlp":
        raise ValueError("Flatten MLP PPO is restricted to fixed-N training. Use one agent count or switch architecture.")

    build_num_agents = agent_counts[0] if train_config["policy_class"] == "mlp" else max(agent_counts)
    base_env_config = prepare_env_config(
        args.env_config,
        tasks=task_names,
        num_agents=build_num_agents,
        scaling_mode=args.scaling_mode,
        observation_override=observation_override_from_variant(args.obs_variant),
    )
    if args.envs_per_task is not None:
        env_build = make_task_balanced_env_batch(
            base_env_config,
            task_names=task_names,
            envs_per_task=args.envs_per_task,
            backend=args.env_backend,
            max_workers=args.env_workers,
        )
    else:
        env_build = make_env_batch(
            base_env_config,
            int(train_config.get("num_envs", 1)),
            backend=args.env_backend,
            max_workers=args.env_workers,
        )
    policy = build_policy(train_config, env_build.envs[0].observation_space, env_build.envs[0].action_space)
    trainer = PPOTrainer(env_build, policy, train_config)
    if args.init_checkpoint:
        trainer.load_checkpoint(args.init_checkpoint)
        if float(train_config.get("reference_policy_coef", 0.0) or 0.0) > 0.0:
            trainer.set_reference_policy_from_current()

    output_root = "bc_ppo" if args.init_checkpoint else "ppo"
    run_name = safe_run_name(
        args.run_name or f"{train_config['name']}_{compact_task_set_name(task_names)}_N{format_agent_set_name(agent_counts)}"
    )
    tensorboard_metric_mode = str(train_config.get("tensorboard_metric_mode", "core"))
    tensorboard_task_namespace = safe_run_name(format_task_set_name(task_names))
    output_dir = timestamped_training_dir(
        output_root,
        run_name,
        timestamp=args.run_timestamp,
    )
    save_run_snapshot(
        output_dir,
        train_config=train_config,
        env_config=base_env_config,
        cli_args=vars(args),
        model_state_dict=trainer.policy.state_dict(),
        extra_metadata={"task_names": task_names, "agent_counts": agent_counts, "output_root": output_root},
    )
    writer, log_record = build_metric_logger(
        output_dir,
        namespace=f"{output_root}/{tensorboard_task_namespace}/train",
        step_key="update",
        tensorboard_enabled=args.tensorboard,
        console_interval=args.console_log_interval,
        tensorboard_metric_mode=tensorboard_metric_mode,
        key_order=[
            "mean_rollout_reward",
            "policy_loss",
            "value_loss",
            "entropy",
            "episodes_completed",
            "cumulative_episodes",
            "rollout_steps_per_sec",
            "eval_reward",
            "eval_success_rate",
        ],
    )
    metrics_csv_path = output_dir / "training_metrics.csv"
    metrics_flush_interval = max(int(args.console_log_interval), 1)

    try:
        if not variable_n:
            persisted_history: list[dict] = []

            def log_and_persist(record: dict) -> None:
                persisted_history.append(dict(record))
                log_record(record)
                update_idx = int(record.get("update", 0))
                if update_idx % metrics_flush_interval == 0 or "checkpoint_path" in record:
                    write_metrics_csv(persisted_history, metrics_csv_path)

            history = trainer.train(
                output_dir,
                eval_env=env_build.envs[0],
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
        else:
            env_batches = {
                n: (
                    make_task_balanced_env_batch(
                        prepare_env_config(
                            args.env_config,
                            tasks=task_names,
                            num_agents=n,
                            scaling_mode=args.scaling_mode,
                            observation_override=observation_override_from_variant(args.obs_variant),
                        ),
                        task_names=task_names,
                        envs_per_task=args.envs_per_task,
                        backend=args.env_backend,
                        max_workers=args.env_workers,
                    )
                    if args.envs_per_task is not None
                    else make_env_batch(
                        prepare_env_config(
                            args.env_config,
                            tasks=task_names,
                            num_agents=n,
                            scaling_mode=args.scaling_mode,
                            observation_override=observation_override_from_variant(args.obs_variant),
                        ),
                        int(train_config.get("num_envs", 1)),
                        backend=args.env_backend,
                        max_workers=args.env_workers,
                    )
                )
                for n in agent_counts
            }
            history = []
            total_updates = int(train_config["total_updates"])
            target_episodes = int(train_config.get("target_episodes", 0) or 0)
            eval_interval = int(train_config.get("eval_interval", total_updates))
            rng = np.random.default_rng(int(base_env_config.get("seed", 0)))
            eval_count = 0
            best_eval_success = float("-inf")
            best_eval_reward = float("-inf")
            for update_idx in range(1, total_updates + 1):
                trainer._set_lr(update_idx, total_updates)
                chosen_n = int(rng.choice(agent_counts))
                trainer.set_env_batch(env_batches[chosen_n])
                batch, rollout_stats = trainer.collect_rollout()
                train_stats = trainer.update(batch)
                record = {"update": update_idx, "sampled_num_agents": chosen_n, **rollout_stats, **train_stats}
                if update_idx % eval_interval == 0 or update_idx == total_updates:
                    eval_count += 1
                    media_dir = None
                    media_episodes = 0
                    if args.record_eval_episodes > 0 and eval_count % max(args.record_interval, 1) == 0:
                        media_dir = output_dir / "media" / f"eval_{update_idx:04d}"
                        media_episodes = int(args.record_eval_episodes)
                    eval_overall_returns = []
                    eval_overall_success_rates = []
                    eval_overall_collision_rates = []
                    eval_overall_path_lengths = []
                    eval_overall_inference_latencies = []
                    for eval_agent_count in agent_counts:
                        eval_base_env_config = prepare_env_config(
                            args.env_config,
                            tasks=task_names,
                            num_agents=eval_agent_count,
                            scaling_mode=args.scaling_mode,
                            observation_override=observation_override_from_variant(args.obs_variant),
                        )
                        task_record_dir = media_dir / f"N{eval_agent_count}" if media_dir is not None else None
                        _, task_summaries, overall_summary = evaluate_policy_per_task(
                            eval_base_env_config,
                            trainer.policy,
                            task_names,
                            args.eval_episodes,
                            trainer.device,
                            headless=args.headless,
                            record_dir=task_record_dir,
                            record_episodes=media_episodes,
                            record_format=args.record_format,
                            record_fps=args.record_fps,
                            record_prefix=f"update_{update_idx:04d}_N{eval_agent_count}",
                            normalize_with_reference=False,
                        )
                        record.update(
                            flatten_task_eval_summaries(
                                task_summaries,
                                overall_summary,
                                prefix=f"eval_N{eval_agent_count}",
                            )
                        )
                        if "return_mean" in overall_summary:
                            eval_overall_returns.append(float(overall_summary["return_mean"]))
                        if "success_rate_mean" in overall_summary:
                            eval_overall_success_rates.append(float(overall_summary["success_rate_mean"]))
                        if "collision_rate_mean" in overall_summary:
                            eval_overall_collision_rates.append(float(overall_summary["collision_rate_mean"]))
                        if "path_length_mean" in overall_summary:
                            eval_overall_path_lengths.append(float(overall_summary["path_length_mean"]))
                        if "inference_latency_ms_mean" in overall_summary:
                            eval_overall_inference_latencies.append(float(overall_summary["inference_latency_ms_mean"]))
                    if eval_overall_returns:
                        record["eval_overall_return"] = float(np.mean(eval_overall_returns))
                        record["eval_reward"] = record["eval_overall_return"]
                    if eval_overall_success_rates:
                        record["eval_overall_success_rate"] = float(np.mean(eval_overall_success_rates))
                        record["eval_success_rate"] = record["eval_overall_success_rate"]
                    if eval_overall_collision_rates:
                        record["eval_overall_collision_rate"] = float(np.mean(eval_overall_collision_rates))
                        record["eval_collision_rate"] = record["eval_overall_collision_rate"]
                    if eval_overall_path_lengths:
                        record["eval_overall_path_length"] = float(np.mean(eval_overall_path_lengths))
                        record["eval_path_length"] = record["eval_overall_path_length"]
                    if eval_overall_inference_latencies:
                        record["eval_overall_inference_latency_ms"] = float(np.mean(eval_overall_inference_latencies))
                        record["eval_inference_latency_ms"] = record["eval_overall_inference_latency_ms"]
                    if media_dir is not None and media_episodes > 0:
                        record["eval_media_dir"] = str(media_dir)
                    checkpoint_path = checkpoints_dir(output_dir) / f"checkpoint_{update_idx:04d}.pt"
                    torch.save({"model_state_dict": trainer.policy.state_dict(), "train_config": train_config}, checkpoint_path)
                    record["checkpoint_path"] = str(checkpoint_path)
                    eval_success = float(record.get("eval_success_rate", float("-inf")))
                    eval_reward = float(record.get("eval_reward", float("-inf")))
                    is_best = False
                    if eval_success > best_eval_success:
                        is_best = True
                    elif eval_success == best_eval_success and eval_reward > best_eval_reward:
                        is_best = True
                    if is_best:
                        best_eval_success = eval_success
                        best_eval_reward = eval_reward
                        best_path = checkpoints_dir(output_dir) / "checkpoint_best_eval.pt"
                        torch.save({"model_state_dict": trainer.policy.state_dict(), "train_config": train_config}, best_path)
                        record["best_checkpoint_path"] = str(best_path)
                        (output_dir / "best_eval_summary.json").write_text(
                            json.dumps(
                                {
                                    "update": int(update_idx),
                                    "eval_success_rate": eval_success,
                                    "eval_reward": eval_reward,
                                    "checkpoint_path": str(best_path),
                                },
                                indent=2,
                                sort_keys=True,
                            )
                            + "\n",
                            encoding="utf-8",
                        )
                history.append(record)
                log_record(record)
                should_stop = target_episodes > 0 and trainer.completed_episodes >= target_episodes
                if update_idx % metrics_flush_interval == 0 or "checkpoint_path" in record or should_stop:
                    write_metrics_csv(history, metrics_csv_path)
                if should_stop:
                    break
        write_metrics_csv(history, metrics_csv_path)

        final_eval_checkpoint = checkpoints_dir(output_dir) / "checkpoint_best_eval.pt"
        final_eval_source = "last"
        if args.final_eval_source == "best" and final_eval_checkpoint.exists():
            checkpoint = torch.load(final_eval_checkpoint, map_location=trainer.device)
            trainer.policy.load_state_dict(checkpoint["model_state_dict"], strict=False)
            final_eval_source = "best"

        eval_records = []
        final_update = int(history[-1].get("update", 0)) if history else 0
        for agent_count in agent_counts:
            eval_env_config = prepare_env_config(
                args.env_config,
                tasks=task_names,
                num_agents=agent_count,
                scaling_mode=args.scaling_mode,
                observation_override=observation_override_from_variant(args.obs_variant),
            )
            final_media_dir = output_dir / "final_eval_media" / f"N{agent_count}" if args.record_eval_episodes > 0 else None
            _, task_summaries, overall_summary = evaluate_policy_per_task(
                eval_env_config,
                trainer.policy,
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
            for task_name, task_summary in task_summaries.items():
                log_scalar_metrics(
                    writer,
                    f"{output_root}/{tensorboard_task_namespace}/final_eval/N{agent_count}/{task_name}",
                    final_update,
                    task_summary,
                    tensorboard_metric_mode=tensorboard_metric_mode,
                    metric_phase="final_eval",
                )
                print_progress_line(
                    f"{output_root}-final/{task_name}",
                    "num_agents",
                    agent_count,
                    task_summary,
                    key_order=["return_mean", "normalized_score_mean", "success_rate_mean", "collision_rate_mean"],
                )
                eval_records.append(
                    {
                        **task_summary,
                        "method": "bc_ppo" if args.init_checkpoint else "ppo",
                        "algorithm": "ppo",
                        "architecture": train_config["policy_class"],
                        "observation_mode": eval_env_config.get("observation_mode", "multi_channel_field"),
                        "obs_variant": args.obs_variant,
                        "task_set": format_task_set_name(task_names),
                        "task_name": task_name,
                        "eval_group": "per_task",
                        "final_eval_source": final_eval_source,
                        "num_agents": agent_count,
                        "scaling_mode": args.scaling_mode,
                        "seed": int(eval_env_config.get("seed", 0)),
                        "normalized_score": float(task_summary.get("normalized_score_mean", 0.0)),
                    }
                )
            log_scalar_metrics(
                writer,
                f"{output_root}/{tensorboard_task_namespace}/final_eval/N{agent_count}/overall",
                final_update,
                overall_summary,
                tensorboard_metric_mode=tensorboard_metric_mode,
                metric_phase="final_eval",
            )
            print_progress_line(
                f"{output_root}-final/overall",
                "num_agents",
                agent_count,
                overall_summary,
                key_order=["return_mean", "normalized_score_mean", "success_rate_mean", "collision_rate_mean"],
            )
            eval_records.append(
                {
                    **overall_summary,
                    "method": "bc_ppo" if args.init_checkpoint else "ppo",
                    "algorithm": "ppo",
                    "architecture": train_config["policy_class"],
                    "observation_mode": eval_env_config.get("observation_mode", "multi_channel_field"),
                    "obs_variant": args.obs_variant,
                    "task_set": format_task_set_name(task_names),
                    "task_name": "overall",
                    "eval_group": "overall",
                    "final_eval_source": final_eval_source,
                    "num_agents": agent_count,
                    "scaling_mode": args.scaling_mode,
                    "seed": int(eval_env_config.get("seed", 0)),
                    "normalized_score": float(overall_summary.get("normalized_score_mean", 0.0)),
                }
            )
        write_metrics_csv(eval_records, output_dir / "eval_metrics.csv")
    finally:
        if writer is not None:
            writer.close()
    print(f"ppo_output={output_dir}")


if __name__ == "__main__":
    main()
