from __future__ import annotations

import argparse
import csv
import json
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.waypoint_marl_env import WaypointMultiUAVEnv
from policies import build_policy
from scripts.benchmarks.swarm30.evaluate import read_yaml
from utils.swarm30_benchmark import build_scaled_env_config, sync_policy_scale
from utils.waypoint_evaluation import write_episode_media
from utils.waypoint_vector_env import make_waypoint_env_batch


def now_runtime() -> str:
    return time.strftime("%Y%m%d_%H%M", time.gmtime())


def sync_device(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def elapsed(fn):
    start = time.perf_counter()
    value = fn()
    return value, time.perf_counter() - start


def to_tensor_batch(obs: dict[str, np.ndarray], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: torch.as_tensor(value, dtype=torch.float32, device=device).unsqueeze(0) for key, value in obs.items()}


def to_tensor_existing_batch(obs: dict[str, np.ndarray], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: torch.as_tensor(value, dtype=torch.float32, device=device) for key, value in obs.items()}


def build_env_and_policy(env_config: dict[str, Any], policy_config: dict[str, Any], device: torch.device):
    env = WaypointMultiUAVEnv(env_config)
    env.reset(seed=int(env_config.get("seed", 0)))
    resolved_policy = sync_policy_scale(policy_config, env_config)
    policy = build_policy(resolved_policy, env.global_observation_space, env.action_space_n)
    policy.to(device)
    policy.eval()
    return env, policy, resolved_policy


@torch.no_grad()
def policy_action(policy, obs: dict[str, np.ndarray], device: torch.device, deterministic: bool = True) -> np.ndarray:
    tensor_obs = to_tensor_batch(obs, device)
    if deterministic:
        action = policy.act_deterministic(tensor_obs)
    else:
        action = policy.get_action_and_value(tensor_obs).env_action
    sync_device(device)
    return action.squeeze(0).detach().cpu().numpy().astype(np.float32)


def action_dict(env: WaypointMultiUAVEnv, action: np.ndarray) -> dict[str, np.ndarray]:
    return {agent: action[idx].astype(np.float32) for idx, agent in enumerate(env.possible_agents)}


def safe_mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def safe_std(values: list[float]) -> float:
    return float(np.std(values)) if values else 0.0


def profile_single_env(
    env_config: dict[str, Any],
    policy_config: dict[str, Any],
    task_name: str,
    scale: int,
    device: torch.device,
    steps: int,
    reset_repeats: int,
) -> list[dict[str, Any]]:
    cfg = deepcopy(env_config)
    cfg["task_name"] = task_name
    cfg["task_names"] = [task_name]
    env, policy, _ = build_env_and_policy(cfg, policy_config, device)
    rows: list[dict[str, Any]] = []

    reset_times = []
    for idx in range(reset_repeats):
        _, reset_time = elapsed(lambda idx=idx: env.reset(seed=int(cfg.get("seed", 0)) + idx))
        reset_times.append(reset_time)
    rows.append(make_row("single_env_reset", task_name, scale, len(reset_times), reset_times, units="reset"))

    get_obs_times: list[float] = []
    policy_times: list[float] = []
    step_times: list[float] = []
    total_start = time.perf_counter()
    done_count = 0
    for step_idx in range(steps):
        obs, obs_time = elapsed(env.get_global_state)
        get_obs_times.append(obs_time)

        def run_policy():
            return policy_action(policy, obs, device, deterministic=True)

        action, policy_time = elapsed(run_policy)
        policy_times.append(policy_time)

        def run_step():
            return env.step(action_dict(env, action))

        (_, _, terms, truncs, _), step_time = elapsed(run_step)
        step_times.append(step_time)
        if bool(any(terms.values()) or any(truncs.values())):
            done_count += 1
            env.reset(seed=int(cfg.get("seed", 0)) + 1000 + done_count)
    total_time = time.perf_counter() - total_start
    rows.append(make_row("single_env_get_global_state", task_name, scale, steps, get_obs_times, units="step"))
    rows.append(make_row("single_env_policy_action", task_name, scale, steps, policy_times, units="step"))
    rows.append(make_row("single_env_env_step", task_name, scale, steps, step_times, units="step"))
    rows.append(
        {
            "component": "single_env_loop_total",
            "task": task_name,
            "scale": scale,
            "count": steps,
            "total_sec": total_time,
            "mean_sec": total_time / max(steps, 1),
            "std_sec": 0.0,
            "units": "step",
            "steps_per_sec": steps / max(total_time, 1e-9),
            "env_steps_per_sec": steps / max(total_time, 1e-9),
        }
    )
    env.close()
    return rows


def profile_render_and_media(
    env_config: dict[str, Any],
    policy_config: dict[str, Any],
    task_name: str,
    scale: int,
    device: torch.device,
    frames: int,
    output_dir: Path,
) -> list[dict[str, Any]]:
    cfg = deepcopy(env_config)
    cfg["task_name"] = task_name
    cfg["task_names"] = [task_name]
    env, policy, _ = build_env_and_policy(cfg, policy_config, device)
    env.reset(seed=int(cfg.get("seed", 0)) + 77)
    render_times: list[float] = []
    step_times: list[float] = []
    frames_out = []
    for idx in range(frames):
        frame, render_time = elapsed(lambda: env.render("rgb_array"))
        render_times.append(render_time)
        frames_out.append(frame)
        obs = env.get_global_state()
        action = policy_action(policy, obs, device, deterministic=True)
        (_, _, terms, truncs, _), step_time = elapsed(lambda: env.step(action_dict(env, action)))
        step_times.append(step_time)
        if bool(any(terms.values()) or any(truncs.values())):
            env.reset(seed=int(cfg.get("seed", 0)) + 2000 + idx)
    media_path, write_time = elapsed(
        lambda: write_episode_media(
            frames_out,
            output_dir / "media_probe",
            f"{task_name}_N{scale}_{frames}frames",
            "gif",
            8,
        )
    )
    rows = [
        make_row("render_rgb_array", task_name, scale, len(render_times), render_times, units="frame"),
        make_row("render_probe_env_step", task_name, scale, len(step_times), step_times, units="step"),
        {
            "component": "gif_write",
            "task": task_name,
            "scale": scale,
            "count": len(frames_out),
            "total_sec": write_time,
            "mean_sec": write_time / max(len(frames_out), 1),
            "std_sec": 0.0,
            "units": "frame",
            "steps_per_sec": len(frames_out) / max(write_time, 1e-9),
            "env_steps_per_sec": 0.0,
            "artifact": str(media_path),
        },
    ]
    env.close()
    return rows


@torch.no_grad()
def profile_process_batch(
    env_config: dict[str, Any],
    policy_config: dict[str, Any],
    task_name: str,
    scale: int,
    device: torch.device,
    num_envs: int,
    batch_steps: int,
) -> list[dict[str, Any]]:
    cfg = deepcopy(env_config)
    cfg["task_name"] = task_name
    cfg["task_names"] = [task_name]
    batch = make_waypoint_env_batch(cfg, num_envs=num_envs, task_name=task_name, backend="process")
    try:
        obs, _ = batch.reset(seeds=[int(cfg.get("seed", 0)) + idx for idx in range(num_envs)])
        probe_env = WaypointMultiUAVEnv(cfg)
        probe_env.reset(seed=int(cfg.get("seed", 0)))
        resolved_policy = sync_policy_scale(policy_config, cfg)
        policy = build_policy(resolved_policy, probe_env.global_observation_space, probe_env.action_space_n)
        policy.to(device)
        policy.eval()
        probe_env.close()

        policy_times: list[float] = []
        batch_step_times: list[float] = []
        total_start = time.perf_counter()
        for _ in range(batch_steps):
            def run_policy():
                tensor_obs = to_tensor_existing_batch(obs, device)
                action = policy.get_action_and_value(tensor_obs).env_action
                sync_device(device)
                return action.detach().cpu().numpy().astype(np.float32)

            actions, policy_time = elapsed(run_policy)
            policy_times.append(policy_time)
            step_result, step_time = elapsed(lambda: batch.step(actions))
            batch_step_times.append(step_time)
            obs = step_result.observations
        total_time = time.perf_counter() - total_start
        rows = [
            make_row("process_batch_policy_action_B8", task_name, scale, batch_steps, policy_times, units="batch_step"),
            make_row("process_batch_env_step_B8", task_name, scale, batch_steps, batch_step_times, units="batch_step"),
            {
                "component": "process_batch_collect_total_B8",
                "task": task_name,
                "scale": scale,
                "count": batch_steps,
                "num_envs": num_envs,
                "total_sec": total_time,
                "mean_sec": total_time / max(batch_steps, 1),
                "std_sec": 0.0,
                "units": "batch_step",
                "steps_per_sec": batch_steps / max(total_time, 1e-9),
                "env_steps_per_sec": (batch_steps * num_envs) / max(total_time, 1e-9),
            },
        ]
        return rows
    finally:
        batch.close()


def profile_backward(policy_config: dict[str, Any], env_config: dict[str, Any], device: torch.device, repeats: int) -> list[dict[str, Any]]:
    env = WaypointMultiUAVEnv(env_config)
    env.reset(seed=int(env_config.get("seed", 0)))
    policy = build_policy(sync_policy_scale(policy_config, env_config), env.global_observation_space, env.action_space_n)
    policy.to(device)
    policy.train()
    optimizer = torch.optim.Adam(policy.parameters(), lr=2e-4)
    obs_np = env.get_global_state()
    batch_obs = {
        key: torch.as_tensor(np.repeat(value[None, ...], 64, axis=0), dtype=torch.float32, device=device)
        for key, value in obs_np.items()
    }
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        output = policy.get_action_and_value(batch_obs)
        loss = output.value.pow(2).mean() - 0.001 * output.entropy.mean() + output.logprob.mean().pow(2)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
        optimizer.step()
        sync_device(device)
        times.append(time.perf_counter() - start)
    env.close()
    return [make_row("policy_backward_fake_minibatch64", "all_tasks", int(env_config["num_agents"]), repeats, times, units="minibatch")]


def make_row(
    component: str,
    task_name: str,
    scale: int,
    count: int,
    values: list[float],
    units: str,
) -> dict[str, Any]:
    total = float(np.sum(values))
    return {
        "component": component,
        "task": task_name,
        "scale": int(scale),
        "count": int(count),
        "total_sec": total,
        "mean_sec": safe_mean(values),
        "std_sec": safe_std(values),
        "units": units,
        "steps_per_sec": count / max(total, 1e-9),
        "env_steps_per_sec": count / max(total, 1e-9) if units == "step" else 0.0,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    keys = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, Any]], output_dir: Path, benchmark_cfg: dict[str, Any]) -> dict[str, Any]:
    by_component = {row["component"]: row for row in rows if row.get("task") in {"area_coverage", "all_tasks"} and int(row.get("scale", -1)) == 30}
    single_step = by_component.get("single_env_loop_total", {})
    process_collect = by_component.get("process_batch_collect_total_B8", {})
    process_policy = by_component.get("process_batch_policy_action_B8", {})
    process_env = by_component.get("process_batch_env_step_B8", {})
    render = by_component.get("render_rgb_array", {})
    gif = by_component.get("gif_write", {})
    backward = by_component.get("policy_backward_fake_minibatch64", {})

    train_cfg = benchmark_cfg["train"]
    eval_cfg = benchmark_cfg["evaluation"]
    rollout_steps = int(train_cfg["rollout_steps"])
    num_envs = int(train_cfg["num_envs"])
    eval_episodes = int(train_cfg["eval_episodes"])
    record_eval_episodes = int(train_cfg["record_eval_episodes"])
    max_steps = 220
    collect_step_sec = float(process_collect.get("mean_sec", 0.0))
    eval_step_sec = float(single_step.get("mean_sec", 0.0))
    render_sec = float(render.get("mean_sec", 0.0))
    gif_sec = float(gif.get("mean_sec", 0.0))
    backward_sec = float(backward.get("mean_sec", 0.0))

    estimated_update = {
        "collect_sec_per_update": collect_step_sec * rollout_steps,
        "fake_update_sec_per_4_epochs_approx": backward_sec * 4 * max((rollout_steps * num_envs) // 512, 1),
        "eval_no_media_sec_per_eval": eval_step_sec * eval_episodes * max_steps,
        "eval_render_sec_per_eval": render_sec * record_eval_episodes * max_steps,
        "eval_gif_write_sec_per_eval": gif_sec * record_eval_episodes * max_steps,
    }
    total_update_with_eval = (
        estimated_update["collect_sec_per_update"] * 100
        + estimated_update["fake_update_sec_per_4_epochs_approx"] * 100
        + estimated_update["eval_no_media_sec_per_eval"]
        + estimated_update["eval_render_sec_per_eval"]
        + estimated_update["eval_gif_write_sec_per_eval"]
    )
    percentages = {key: 100.0 * value / max(total_update_with_eval, 1e-9) for key, value in {
        "collect": estimated_update["collect_sec_per_update"] * 100,
        "ppo_update_approx": estimated_update["fake_update_sec_per_4_epochs_approx"] * 100,
        "eval_step": estimated_update["eval_no_media_sec_per_eval"],
        "eval_render": estimated_update["eval_render_sec_per_eval"],
        "eval_gif_write": estimated_update["eval_gif_write_sec_per_eval"],
    }.items()}
    summary = {
        "note": "Estimates use measured N=30 timing and benchmark swarm30_v1 defaults. PPO update is an approximate fake-minibatch backward timing.",
        "measured_reference": {
            "single_env_step_mean_sec": eval_step_sec,
            "process_batch_collect_mean_sec_per_8env_step": collect_step_sec,
            "process_batch_policy_mean_sec": float(process_policy.get("mean_sec", 0.0)),
            "process_batch_env_mean_sec": float(process_env.get("mean_sec", 0.0)),
            "render_frame_mean_sec": render_sec,
            "gif_write_frame_mean_sec": gif_sec,
            "fake_backward_minibatch64_mean_sec": backward_sec,
        },
        "estimated_benchmark_period_100_updates_sec": {
            **estimated_update,
            "total_sec": total_update_with_eval,
            "percentages": percentages,
        },
        "estimated_evaluate_one_task_20_fixed_100_random_sec": {
            "episode_step_sec": eval_step_sec * 120 * max_steps,
            "render_sec": render_sec * 4 * max_steps,
            "gif_write_sec": gif_sec * 4 * max_steps,
            "total_sec": eval_step_sec * 120 * max_steps + render_sec * 4 * max_steps + gif_sec * 4 * max_steps,
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def write_report(output_dir: Path, rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    lines = [
        "# Swarm30 Runtime Profiling Report",
        "",
        "This profiling run does not modify environment, dynamics, reward, or policy code.",
        "",
        "## Main Finding",
        "",
    ]
    measured = summary["measured_reference"]
    pct = summary["estimated_benchmark_period_100_updates_sec"]["percentages"]
    lines.extend(
        [
            f"- N=30 single-env step mean: `{measured['single_env_step_mean_sec']:.4f}s`.",
            f"- N=30 process batch collect mean for 8 envs: `{measured['process_batch_collect_mean_sec_per_8env_step']:.4f}s` per batch step.",
            f"- Policy forward within process batch: `{measured['process_batch_policy_mean_sec']:.4f}s`; env batch step: `{measured['process_batch_env_mean_sec']:.4f}s`.",
            f"- Approximate 100-update benchmark period: collect `{pct['collect']:.1f}%`, PPO update `{pct['ppo_update_approx']:.1f}%`, eval stepping `{pct['eval_step']:.1f}%`, render `{pct['eval_render']:.1f}%`, GIF write `{pct['eval_gif_write']:.1f}%`.",
            "",
            "## Component Table",
            "",
            "| Component | Task | N | Count | Total sec | Mean sec | Unit |",
            "|---|---|---:|---:|---:|---:|---|",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row.get('component')} | {row.get('task')} | {row.get('scale')} | {row.get('count')} | "
            f"{float(row.get('total_sec', 0.0)):.3f} | {float(row.get('mean_sec', 0.0)):.5f} | {row.get('units', '')} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- If process batch env step dominates, the bottleneck is CPU-side environment simulation: MPE substeps, N=30 pairwise communication/collision checks, task grid updates, and IPC.",
            "- If render/GIF dominates evaluate, reduce media frequency for intermediate runs and keep full media only for final evaluation.",
            "- If PPO backward is small, adding GPU or larger network optimization will not solve wall-clock time; env throughput must be improved first.",
        ]
    )
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Profile swarm30 runtime component costs without changing environment code.")
    parser.add_argument("--benchmark-config", default="configs/benchmarks/swarm30_v1.yaml")
    parser.add_argument("--env-config", default="configs/env/waypoint_missions.yaml")
    parser.add_argument("--policy-config", default="configs/policy/benchmarks/swarm30_standardized.yaml")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--tasks", nargs="+", default=["area_coverage", "belief_search", "connectivity_expansion"])
    parser.add_argument("--scales", nargs="+", type=int, default=[4, 30])
    parser.add_argument("--single-steps", type=int, default=80)
    parser.add_argument("--reset-repeats", type=int, default=5)
    parser.add_argument("--render-frames", type=int, default=60)
    parser.add_argument("--process-batch-steps", type=int, default=40)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--backward-repeats", type=int, default=8)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir or f"outputs/debug/perf_profile/swarm30_runtime_breakdown/{now_runtime()}")
    output_dir.mkdir(parents=True, exist_ok=True)
    benchmark_cfg = read_yaml(ROOT / args.benchmark_config)
    base_env = read_yaml(ROOT / args.env_config)
    policy_config = read_yaml(ROOT / args.policy_config)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    rows: list[dict[str, Any]] = []
    for scale in args.scales:
        for task in args.tasks:
            scaled = build_scaled_env_config(base_env, int(scale), [task], randomization=True)
            scaled["seed"] = 123
            rows.extend(
                profile_single_env(
                    scaled,
                    policy_config,
                    task,
                    int(scale),
                    device,
                    steps=int(args.single_steps),
                    reset_repeats=int(args.reset_repeats),
                )
            )
    scaled30 = build_scaled_env_config(base_env, 30, ["area_coverage"], randomization=True)
    scaled30["seed"] = 321
    rows.extend(
        profile_render_and_media(
            scaled30,
            policy_config,
            "area_coverage",
            30,
            device,
            frames=int(args.render_frames),
            output_dir=output_dir,
        )
    )
    rows.extend(
        profile_process_batch(
            scaled30,
            policy_config,
            "area_coverage",
            30,
            device,
            num_envs=int(args.num_envs),
            batch_steps=int(args.process_batch_steps),
        )
    )
    rows.extend(profile_backward(policy_config, scaled30, device, repeats=int(args.backward_repeats)))

    write_csv(output_dir / "component_timings.csv", rows)
    summary = summarize(rows, output_dir, benchmark_cfg)
    write_report(output_dir, rows, summary)
    print(json.dumps({"output_dir": str(output_dir), "summary": summary}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
