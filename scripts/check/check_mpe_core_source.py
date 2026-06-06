from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.waypoint_marl_env import WaypointMultiUAVEnv


def load_yaml(path: str | Path) -> dict:
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = ROOT / resolved
    with open(resolved, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def module_origin(name: str) -> str | None:
    try:
        spec = importlib.util.find_spec(name)
    except ModuleNotFoundError:
        return None
    return None if spec is None else str(spec.origin)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-config", default="configs/env/waypoint_missions.yaml")
    parser.add_argument("--output-dir", default=str(ROOT / "outputs" / "debug" / "mpe_core_source_check"))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_yaml(args.env_config)
    config["num_agents"] = int(config.get("num_agents", 4))
    config["task_names"] = ["area_coverage"]
    config["task_name"] = "area_coverage"
    config.setdefault("dynamics_backend", {})["name"] = "mpe_core"
    config.setdefault("dynamics_backend", {}).setdefault("source", "third_party_openai_mpe")

    env = WaypointMultiUAVEnv(config)
    observations, infos = env.reset(seed=int(config.get("seed", 0)))
    agents = list(env.agents)
    actions = {agent: env.world.uavs[idx].position.copy() for idx, agent in enumerate(agents)}
    _, _, _, _, step_infos = env.step(actions)
    first_info = step_infos[agents[0]]
    dynamics = dict(first_info.get("dynamics_info", {}))
    backend = env.dynamics_backend

    summary = {
        "python_executable": sys.executable,
        "python_version": sys.version,
        "sys_path": list(sys.path),
        "mpe2_installed": module_origin("mpe2") is not None,
        "mpe2_origin": module_origin("mpe2"),
        "mpe2_core_origin": module_origin("mpe2._mpe_utils.core"),
        "pettingzoo_installed": module_origin("pettingzoo") is not None,
        "pettingzoo_origin": module_origin("pettingzoo"),
        "pettingzoo_mpe_core_origin": module_origin("pettingzoo.mpe._mpe_utils.core"),
        "third_party_core_origin": module_origin("third_party.openai_mpe.core"),
        "dynamics_backend_name": getattr(backend, "name", ""),
        "env_dynamics_backend_source": getattr(backend, "source", ""),
        "env_requested_mpe_source": getattr(backend, "requested_source", ""),
        "mpe_core_file": str(getattr(getattr(backend, "mpe_core_module", None), "__file__", "")),
        "mpe_world_class": dynamics.get("mpe_world_class", ""),
        "mpe_source": dynamics.get("mpe_source", ""),
        "uses_real_mpe_core": bool(dynamics.get("uses_real_mpe_core", False)),
        "mpe_world_step_calls": int(dynamics.get("mpe_world_step_calls", 0)),
        "dynamics_finite": bool(dynamics.get("dynamics_finite", False)),
    }
    summary["passed"] = bool(
        summary["dynamics_backend_name"] == "mpe_core"
        and summary["env_dynamics_backend_source"] == "third_party_openai_mpe"
        and summary["mpe_source"] == "third_party_openai_mpe"
        and summary["uses_real_mpe_core"]
        and summary["mpe_world_step_calls"] > 0
        and summary["dynamics_finite"]
    )

    path = output_dir / "source_check_summary.json"
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    env.close()
    if not summary["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
