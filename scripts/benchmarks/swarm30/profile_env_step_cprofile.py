from __future__ import annotations

import argparse
import cProfile
import io
import pstats
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.waypoint_marl_env import WaypointMultiUAVEnv
from scripts.benchmarks.swarm30.evaluate import read_yaml
from utils.swarm30_benchmark import build_scaled_env_config


def random_local_action(env: WaypointMultiUAVEnv, rng: np.random.Generator) -> dict[str, np.ndarray]:
    positions = env.world.get_uav_positions()
    deltas = rng.normal(0.0, 0.08, size=positions.shape).astype(np.float32)
    targets = np.clip(positions + deltas, 0.0, float(env.world.map_size)).astype(np.float32)
    return {agent: targets[idx] for idx, agent in enumerate(env.possible_agents)}


def main() -> int:
    parser = argparse.ArgumentParser(description="cProfile N=30 env.step internals without modifying env code.")
    parser.add_argument("--env-config", default="configs/env/waypoint_missions.yaml")
    parser.add_argument("--task", default="connectivity_expansion")
    parser.add_argument("--scale", type=int, default=30)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir or f"outputs/debug/perf_profile/env_step_cprofile/{time.strftime('%Y%m%d_%H%M', time.gmtime())}")
    output_dir.mkdir(parents=True, exist_ok=True)
    base_env = read_yaml(ROOT / args.env_config)
    config: dict[str, Any] = build_scaled_env_config(base_env, int(args.scale), [args.task], randomization=True)
    config["seed"] = 2026
    env = WaypointMultiUAVEnv(config)
    env.reset(seed=2026, options={"task_name": args.task})
    rng = np.random.default_rng(2026)
    profile = cProfile.Profile()
    step_times = []
    for idx in range(int(args.steps)):
        action = random_local_action(env, rng)
        start = time.perf_counter()
        profile.enable()
        _, _, terms, truncs, _ = env.step(action)
        profile.disable()
        step_times.append(time.perf_counter() - start)
        if bool(any(terms.values()) or any(truncs.values())):
            env.reset(seed=3000 + idx, options={"task_name": args.task})
    env.close()

    stats_path = output_dir / f"{args.task}_N{args.scale}_step_profile.prof"
    txt_path = output_dir / f"{args.task}_N{args.scale}_step_profile.txt"
    profile.dump_stats(str(stats_path))
    stream = io.StringIO()
    stats = pstats.Stats(profile, stream=stream).strip_dirs().sort_stats("cumtime")
    stats.print_stats(80)
    txt_path.write_text(
        f"task={args.task}\nscale={args.scale}\nsteps={args.steps}\n"
        f"mean_step_sec={float(np.mean(step_times)):.6f}\n"
        f"total_step_sec={float(np.sum(step_times)):.6f}\n\n"
        + stream.getvalue(),
        encoding="utf-8",
    )
    print(txt_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
