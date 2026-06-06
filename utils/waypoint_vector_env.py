from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass

import numpy as np

from envs.waypoint_marl_env import WaypointMultiUAVEnv


def stack_global_observations(observations: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    return {key: np.stack([obs[key] for obs in observations], axis=0) for key in observations[0]}


@dataclass
class WaypointBatchStep:
    observations: dict[str, np.ndarray]
    team_rewards: np.ndarray
    per_agent_rewards: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    done_for_reset: np.ndarray
    bootstrap_mask: np.ndarray
    bootstrap_observations: dict[str, np.ndarray]
    infos: list[dict]


class SyncWaypointEnvBatch:
    def __init__(self, envs: list[WaypointMultiUAVEnv]):
        self.envs = envs
        self.num_envs = len(envs)

    def reset(self, seeds: list[int] | None = None, options: dict | None = None):
        observations = []
        infos = []
        for idx, env in enumerate(self.envs):
            seed = None if seeds is None else seeds[idx]
            _, info_by_agent = env.reset(seed=seed, options=options)
            observations.append(env.get_global_state())
            infos.append(next(iter(info_by_agent.values())))
        return stack_global_observations(observations), infos

    def step(self, actions: np.ndarray) -> WaypointBatchStep:
        next_obs = []
        bootstrap_obs = []
        team_rewards = []
        per_agent_rewards = []
        terminated = []
        truncated = []
        done_for_reset = []
        infos = []
        for env, action_row in zip(self.envs, np.asarray(actions, dtype=np.int64)):
            action_dict = {agent: int(action_row[idx]) for idx, agent in enumerate(env.possible_agents)}
            _, _, terms, truncs, info_by_agent = env.step(action_dict)
            info = next(iter(info_by_agent.values()))
            is_terminated = bool(any(terms.values()))
            is_truncated = bool(any(truncs.values()))
            done = is_terminated or is_truncated
            terminal_global = env.get_global_state()
            if done:
                info = dict(info)
                info["terminal_info"] = dict(info)
                info["terminal_global_state"] = {key: value.copy() for key, value in terminal_global.items()}
                env.reset()
            current_global = env.get_global_state()
            next_obs.append(current_global)
            bootstrap_obs.append(terminal_global if done else current_global)
            team_rewards.append(float(info.get("team_reward", 0.0)))
            per_agent_rewards.append(np.asarray(info.get("per_agent_rewards", np.zeros(env.num_agents)), dtype=np.float32))
            terminated.append(is_terminated)
            truncated.append(is_truncated)
            done_for_reset.append(done)
            infos.append(info)
        return WaypointBatchStep(
            observations=stack_global_observations(next_obs),
            team_rewards=np.asarray(team_rewards, dtype=np.float32),
            per_agent_rewards=np.stack(per_agent_rewards, axis=0).astype(np.float32),
            terminated=np.asarray(terminated, dtype=bool),
            truncated=np.asarray(truncated, dtype=bool),
            done_for_reset=np.asarray(done_for_reset, dtype=bool),
            bootstrap_mask=(1.0 - np.asarray(terminated, dtype=np.float32)),
            bootstrap_observations=stack_global_observations(bootstrap_obs),
            infos=infos,
        )

    def close(self) -> None:
        for env in self.envs:
            env.close()


class ThreadWaypointEnvBatch(SyncWaypointEnvBatch):
    def __init__(self, envs: list[WaypointMultiUAVEnv], max_workers: int | None = None):
        super().__init__(envs)
        self.executor = ThreadPoolExecutor(max_workers=max_workers or len(envs))

    def reset(self, seeds: list[int] | None = None, options: dict | None = None):
        futures = []
        for idx, env in enumerate(self.envs):
            seed = None if seeds is None else seeds[idx]
            futures.append(self.executor.submit(env.reset, seed=seed, options=options))
        infos = []
        observations = []
        for env, future in zip(self.envs, futures):
            _, info_by_agent = future.result()
            observations.append(env.get_global_state())
            infos.append(next(iter(info_by_agent.values())))
        return stack_global_observations(observations), infos

    def step(self, actions: np.ndarray) -> WaypointBatchStep:
        futures = []
        for env, action_row in zip(self.envs, np.asarray(actions, dtype=np.int64)):
            action_dict = {agent: int(action_row[idx]) for idx, agent in enumerate(env.possible_agents)}
            futures.append(self.executor.submit(self._step_one, env, action_dict))
        results = [future.result() for future in futures]
        return WaypointBatchStep(
            observations=stack_global_observations([result[0] for result in results]),
            team_rewards=np.asarray([result[2] for result in results], dtype=np.float32),
            per_agent_rewards=np.stack([result[3] for result in results], axis=0).astype(np.float32),
            terminated=np.asarray([result[4] for result in results], dtype=bool),
            truncated=np.asarray([result[5] for result in results], dtype=bool),
            done_for_reset=np.asarray([result[6] for result in results], dtype=bool),
            bootstrap_mask=np.asarray([0.0 if result[4] else 1.0 for result in results], dtype=np.float32),
            bootstrap_observations=stack_global_observations([result[1] for result in results]),
            infos=[result[7] for result in results],
        )

    def _step_one(self, env: WaypointMultiUAVEnv, action_dict: dict[str, int]):
        _, _, terms, truncs, info_by_agent = env.step(action_dict)
        info = next(iter(info_by_agent.values()))
        is_terminated = bool(any(terms.values()))
        is_truncated = bool(any(truncs.values()))
        done = is_terminated or is_truncated
        terminal_global = env.get_global_state()
        if done:
            info = dict(info)
            info["terminal_info"] = dict(info)
            info["terminal_global_state"] = {key: value.copy() for key, value in terminal_global.items()}
            env.reset()
        return (
            env.get_global_state(),
            terminal_global if done else env.get_global_state(),
            float(info.get("team_reward", 0.0)),
            np.asarray(info.get("per_agent_rewards", np.zeros(env.num_agents)), dtype=np.float32),
            is_terminated,
            is_truncated,
            done,
            info,
        )

    def close(self) -> None:
        self.executor.shutdown(wait=True)
        super().close()


def make_waypoint_env_batch(
    config: dict,
    num_envs: int,
    task_name: str | None = None,
    backend: str = "sync",
    max_workers: int | None = None,
):
    envs = []
    for idx in range(int(num_envs)):
        env_config = deepcopy(config)
        env_config["seed"] = int(config.get("seed", 0)) + idx
        if task_name is not None:
            env_config["task_name"] = task_name
            env_config["task_names"] = [task_name]
        envs.append(WaypointMultiUAVEnv(env_config))
    if backend == "sync":
        return SyncWaypointEnvBatch(envs)
    if backend == "thread":
        return ThreadWaypointEnvBatch(envs, max_workers=max_workers)
    raise ValueError(f"Unsupported waypoint env backend: {backend}")
