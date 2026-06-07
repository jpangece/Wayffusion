from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass
import multiprocessing as mp
import traceback

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
        action_array = np.asarray(actions)
        for env, action_row in zip(self.envs, action_array):
            action_dict = self._action_dict(env, action_row)
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

    def _action_dict(self, env: WaypointMultiUAVEnv, action_row: np.ndarray) -> dict:
        if env.action_mode == "candidate_index_legacy":
            row = np.asarray(action_row, dtype=np.int64)
            return {agent: int(row[idx]) for idx, agent in enumerate(env.possible_agents)}
        row = np.asarray(action_row, dtype=np.float32)
        return {agent: row[idx].astype(np.float32) for idx, agent in enumerate(env.possible_agents)}

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
        action_array = np.asarray(actions)
        for env, action_row in zip(self.envs, action_array):
            action_dict = self._action_dict(env, action_row)
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

    def _step_one(self, env: WaypointMultiUAVEnv, action_dict: dict):
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


def _process_worker(connection, config: dict) -> None:
    env = None
    try:
        env = WaypointMultiUAVEnv(config)
        while True:
            command, payload = connection.recv()
            if command == "reset":
                observations, info_by_agent = env.reset(
                    seed=payload.get("seed"),
                    options=payload.get("options"),
                )
                del observations
                connection.send(("ok", (env.get_global_state(), next(iter(info_by_agent.values())))))
                continue
            if command == "step":
                _, _, terms, truncs, info_by_agent = env.step(payload)
                info = next(iter(info_by_agent.values()))
                is_terminated = bool(any(terms.values()))
                is_truncated = bool(any(truncs.values()))
                done = is_terminated or is_truncated
                terminal_global = env.get_global_state()
                if done:
                    info = dict(info)
                    info["terminal_info"] = dict(info)
                    info["terminal_global_state"] = {
                        key: value.copy() for key, value in terminal_global.items()
                    }
                    env.reset()
                current_global = env.get_global_state()
                connection.send(
                    (
                        "ok",
                        (
                            current_global,
                            terminal_global if done else current_global,
                            float(info.get("team_reward", 0.0)),
                            np.asarray(
                                info.get("per_agent_rewards", np.zeros(env.num_agents)),
                                dtype=np.float32,
                            ),
                            is_terminated,
                            is_truncated,
                            done,
                            info,
                        ),
                    )
                )
                continue
            if command == "close":
                connection.send(("ok", None))
                break
            raise ValueError(f"Unsupported process env command: {command}")
    except (EOFError, KeyboardInterrupt):
        pass
    except Exception:
        try:
            connection.send(("error", traceback.format_exc()))
        except (BrokenPipeError, EOFError):
            pass
    finally:
        if env is not None:
            env.close()
        connection.close()


class ProcessWaypointEnvBatch:
    """Run each environment in its own process to bypass the Python GIL."""

    def __init__(self, configs: list[dict], start_method: str = "fork"):
        if not configs:
            raise ValueError("ProcessWaypointEnvBatch requires at least one environment config")
        available_methods = mp.get_all_start_methods()
        method = start_method if start_method in available_methods else "spawn"
        self.context = mp.get_context(method)
        self.num_envs = len(configs)
        self.envs = [WaypointMultiUAVEnv(deepcopy(configs[0]))]
        self.connections = []
        self.processes = []
        self.closed = False
        for config in configs:
            parent, child = self.context.Pipe()
            process = self.context.Process(
                target=_process_worker,
                args=(child, deepcopy(config)),
                daemon=True,
            )
            process.start()
            child.close()
            self.connections.append(parent)
            self.processes.append(process)

    @staticmethod
    def _receive(connection):
        status, payload = connection.recv()
        if status == "error":
            raise RuntimeError(f"Waypoint environment worker failed:\n{payload}")
        return payload

    def reset(self, seeds: list[int] | None = None, options: dict | None = None):
        for idx, connection in enumerate(self.connections):
            seed = None if seeds is None else seeds[idx]
            connection.send(("reset", {"seed": seed, "options": options}))
        results = [self._receive(connection) for connection in self.connections]
        return (
            stack_global_observations([result[0] for result in results]),
            [result[1] for result in results],
        )

    def step(self, actions: np.ndarray) -> WaypointBatchStep:
        action_array = np.asarray(actions)
        prototype = self.envs[0]
        for connection, action_row in zip(self.connections, action_array):
            if prototype.action_mode == "candidate_index_legacy":
                row = np.asarray(action_row, dtype=np.int64)
                action_dict = {
                    agent: int(row[idx])
                    for idx, agent in enumerate(prototype.possible_agents)
                }
            else:
                row = np.asarray(action_row, dtype=np.float32)
                action_dict = {
                    agent: row[idx].astype(np.float32)
                    for idx, agent in enumerate(prototype.possible_agents)
                }
            connection.send(("step", action_dict))
        results = [self._receive(connection) for connection in self.connections]
        return WaypointBatchStep(
            observations=stack_global_observations([result[0] for result in results]),
            team_rewards=np.asarray([result[2] for result in results], dtype=np.float32),
            per_agent_rewards=np.stack([result[3] for result in results], axis=0).astype(np.float32),
            terminated=np.asarray([result[4] for result in results], dtype=bool),
            truncated=np.asarray([result[5] for result in results], dtype=bool),
            done_for_reset=np.asarray([result[6] for result in results], dtype=bool),
            bootstrap_mask=np.asarray(
                [0.0 if result[4] else 1.0 for result in results],
                dtype=np.float32,
            ),
            bootstrap_observations=stack_global_observations([result[1] for result in results]),
            infos=[result[7] for result in results],
        )

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        for connection, process in zip(self.connections, self.processes):
            if process.is_alive():
                try:
                    connection.send(("close", None))
                except (BrokenPipeError, EOFError):
                    pass
        for connection, process in zip(self.connections, self.processes):
            if process.is_alive():
                try:
                    self._receive(connection)
                except (BrokenPipeError, EOFError, RuntimeError):
                    pass
            connection.close()
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
        for env in self.envs:
            env.close()


def make_waypoint_env_batch(
    config: dict,
    num_envs: int,
    task_name: str | None = None,
    backend: str = "sync",
    max_workers: int | None = None,
):
    configs = []
    for idx in range(int(num_envs)):
        env_config = deepcopy(config)
        env_config["seed"] = int(config.get("seed", 0)) + idx
        if task_name is not None:
            env_config["task_name"] = task_name
            env_config["task_names"] = [task_name]
        configs.append(env_config)
    if backend == "process":
        return ProcessWaypointEnvBatch(configs)
    envs = [WaypointMultiUAVEnv(env_config) for env_config in configs]
    if backend == "sync":
        return SyncWaypointEnvBatch(envs)
    if backend == "thread":
        return ThreadWaypointEnvBatch(envs, max_workers=max_workers)
    raise ValueError(f"Unsupported waypoint env backend: {backend}")
