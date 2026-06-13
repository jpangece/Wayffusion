from __future__ import annotations

from copy import deepcopy

import numpy as np
import yaml

from envs.waypoint.world import MissionWorld
from tasks.waypoint.area_coverage import AreaCoverageScenario
from tasks.waypoint.base import uncovered_approach_gain
from tasks.waypoint.belief_search import BeliefSearchScenario
from tasks.waypoint.connectivity_expansion import ConnectivityExpansionScenario
from tasks.waypoint.priority_inspection import PriorityInspectionScenario
from tasks.waypoint.target_interception import TargetInterceptionScenario
from utils.swarm30_benchmark import build_scaled_env_config


def _base_config() -> dict:
    with open("configs/env/waypoint_missions.yaml", "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config["domain_randomization"] = {"enabled": False}
    config.setdefault("runtime_acceleration", {})["sensor_footprint_mode"] = "exact"
    config["runtime_acceleration"]["geometry_backend"] = "numpy"
    return config


def _make_world(
    *,
    num_agents: int = 4,
    grid_size: int = 32,
    map_size: float = 1.0,
    no_fly_zones: list[dict] | None = None,
) -> MissionWorld:
    config = _base_config()
    config.update(
        {
            "num_agents": num_agents,
            "grid_size": grid_size,
            "map_size": map_size,
            "geofence": [0.0, map_size, 0.0, map_size],
            "no_fly_zones": no_fly_zones or [],
        }
    )
    world = MissionWorld.from_config(config, rng=np.random.default_rng(17))
    world.sensor_radius = float(config["sensor_radius"])
    world.comm_radius = float(config["comm_radius"])
    side = int(np.ceil(np.sqrt(num_agents)))
    coords = np.linspace(0.12 * map_size, 0.42 * map_size, side, dtype=np.float32)
    positions = np.asarray([(x, y) for y in coords for x in coords], dtype=np.float32)[:num_agents]
    world.reset_common(num_agents, rng=np.random.default_rng(17), initial_positions=positions)
    return world


def _move_and_accumulate(world: MissionWorld, offset: np.ndarray) -> MissionWorld:
    moved = world.snapshot()
    for uav, delta in zip(moved.uavs, np.broadcast_to(offset, (len(moved.uavs), 2))):
        uav.position = np.clip(uav.position + delta, 0.0, moved.map_size).astype(np.float32)
    moved._invalidate_communication_cache()
    moved._invalidate_sensor_mask_cache()
    moved.update_communication_graph()
    moved.accumulate_sensor_footprints()
    moved.step_count += 1
    return moved


def _legacy_coverage_stats(prev_world: MissionWorld, world: MissionWorld) -> dict[str, np.ndarray]:
    navigable = world.navigable_grid_mask()
    positions = world.get_uav_positions()
    radii = np.asarray([uav.sensor_radius for uav in world.uavs], dtype=np.float32)
    masks = world.sensor_masks_for_positions(positions, radii) & navigable[None, :, :]
    before_covered = prev_world.coverage_grid > 0.0
    before_visited = prev_world.visit_count_grid > 0.0
    denominator = max(float(navigable.sum()), 1.0)
    gains = (masks & ~before_covered[None, :, :]).sum(axis=(1, 2)) / denominator
    footprint_sizes = np.maximum(masks.sum(axis=(1, 2)).astype(np.float32), 1.0)
    repeats = (masks & before_visited[None, :, :]).sum(axis=(1, 2)) / footprint_sizes
    return {
        "masks": masks,
        "new_masks": masks & ~before_covered[None, :, :],
        "gains": np.asarray(gains, dtype=np.float32),
        "repeats": repeats,
        "newly_covered": masks.any(axis=0) & ~before_covered,
    }


def _legacy_uncovered_approach_gain(
    prev_world: MissionWorld,
    world: MissionWorld,
    scale: float,
) -> float:
    navigable = world.navigable_grid_mask()
    uncovered = (prev_world.coverage_grid <= 0.0) & navigable
    if not bool(uncovered.any()) or not world.uavs:
        return 0.0
    points = world.grid_cell_centers()[uncovered]

    def score(positions: np.ndarray) -> float:
        distances = np.linalg.norm(positions[:, None, :] - points[None, :, :], axis=-1)
        proximity = np.exp(-distances / scale).max(axis=0)
        return float(proximity.mean())

    return max(score(world.get_uav_positions()) - score(prev_world.get_uav_positions()), 0.0)


def _legacy_relay_credit(world: MissionWorld) -> float:
    graph = world.communication_graph.copy()
    n = len(world.uavs)
    original_connected = world.base_connected_nodes_for_graph(graph)[1:]
    original_count = int(np.count_nonzero(original_connected))
    if original_count <= 1:
        return 0.0
    credit = 0.0
    for remove_idx in range(1, n + 1):
        if not bool(original_connected[remove_idx - 1]):
            continue
        reduced = graph.copy()
        reduced[remove_idx, :] = False
        reduced[:, remove_idx] = False
        visited = np.zeros(n + 1, dtype=bool)
        visited[0] = True
        stack = [0]
        while stack:
            node = stack.pop()
            for nxt in np.where(reduced[node])[0]:
                if not visited[nxt]:
                    visited[nxt] = True
                    stack.append(int(nxt))
        if original_count > np.count_nonzero(visited[1:]):
            credit += 1.0
    return credit


def _legacy_interception_belief(world: MissionWorld, task_state: dict) -> float:
    masses = []
    for branches, probs, current_idx in zip(
        task_state["target_branches"],
        task_state["target_branch_probs"],
        task_state["target_indices"],
    ):
        covered = 0.0
        total = 0.0
        for branch, prob in zip(branches, probs):
            future = branch[int(current_idx) :]
            point_mass = float(prob) / max(len(future), 1)
            total += point_mass * len(future)
            for point in future:
                if any(np.linalg.norm(uav.position - point) <= uav.sensor_radius for uav in world.uavs):
                    covered += point_mass
        masses.append(covered / max(total, 1e-8))
    return float(np.mean(masses)) if masses else 0.0


def _legacy_interception_field(
    scenario: TargetInterceptionScenario,
    world: MissionWorld,
    task_state: dict,
) -> np.ndarray:
    field = super(TargetInterceptionScenario, scenario).build_task_field(world, task_state)
    future_belief = np.zeros((world.grid_size, world.grid_size), dtype=np.float32)
    for branches, probs, current_idx in zip(
        task_state["target_branches"],
        task_state["target_branch_probs"],
        task_state["target_indices"],
    ):
        for branch, probability in zip(branches, probs):
            future = branch[int(current_idx) :]
            if not len(future):
                continue
            indices = world.world_to_grid(future)
            point_mass = float(probability) / len(future)
            for x_idx, y_idx in indices:
                future_belief[y_idx, x_idx] += point_mass
    future_belief[~world.navigable_grid_mask()] = 0.0
    if future_belief.max() > 1e-8:
        future_belief /= float(future_belief.max())
    field[2] = future_belief
    return field


def test_coverage_transition_cache_matches_legacy_formulas():
    prev_world = _make_world()
    world = _move_and_accumulate(prev_world, np.asarray([0.11, 0.07], dtype=np.float32))
    legacy = _legacy_coverage_stats(prev_world, world)
    stats = world.coverage_transition_stats(prev_world)

    assert np.array_equal(stats.sensor_masks, legacy["masks"])
    assert np.array_equal(stats.per_agent_new_masks, legacy["new_masks"])
    assert np.array_equal(stats.newly_covered_mask, legacy["newly_covered"])
    assert np.array_equal(stats.per_agent_new_coverage, legacy["gains"])
    assert np.array_equal(stats.repeat_footprint_ratios, legacy["repeats"])

    first = stats
    assert world.coverage_transition_stats(prev_world) is first
    world.uavs[0].position += np.asarray([0.02, 0.0], dtype=np.float32)
    assert world.coverage_transition_stats(prev_world) is not first


def test_uncovered_approach_kdtree_matches_dense_n4_and_n30():
    cases = [
        (_make_world(), np.asarray([0.08, 0.05], dtype=np.float32), 0.20),
    ]
    config = build_scaled_env_config(_base_config(), 30, ["area_coverage"], randomization=False)
    config.setdefault("runtime_acceleration", {})["sensor_footprint_mode"] = "exact"
    world30 = MissionWorld.from_config(config, rng=np.random.default_rng(23))
    world30.sensor_radius = float(config["sensor_radius"])
    world30.comm_radius = float(config["comm_radius"])
    xs = np.linspace(0.18, 1.18, 6, dtype=np.float32)
    ys = np.linspace(0.18, 1.02, 5, dtype=np.float32)
    positions = np.asarray([(x, y) for y in ys for x in xs], dtype=np.float32)
    world30.reset_common(30, rng=np.random.default_rng(23), initial_positions=positions)
    cases.append((world30, np.asarray([0.035, 0.025], dtype=np.float32), 0.20))

    for prev_world, offset, scale in cases:
        world = _move_and_accumulate(prev_world, offset)
        expected = _legacy_uncovered_approach_gain(prev_world, world, scale)
        actual = uncovered_approach_gain(prev_world, world, scale)
        assert np.isclose(actual, expected, rtol=0.0, atol=1e-7)


def test_belief_search_reward_matches_legacy_overlap_semantics():
    config = _base_config()
    world = _make_world()
    scenario = BeliefSearchScenario(config["tasks"])
    state = scenario.reset(np.random.default_rng(31), world)
    prev_world = world.snapshot()
    moved = _move_and_accumulate(world, np.asarray([0.04, 0.03], dtype=np.float32))
    optimized_world = moved.snapshot()
    optimized_state = deepcopy(state)
    transition = {
        "path_length_delta": np.full(len(world.uavs), 0.05, dtype=np.float32),
        "safety_violation_count": 0,
        "no_fly_violation_count": 0,
    }

    legacy_world = moved.snapshot()
    legacy_state = deepcopy(state)
    prev_belief = prev_world.belief_grid.copy()
    newly_mass = 0.0
    repeated_mass = 0.0
    per_agent = np.zeros(len(legacy_world.uavs), dtype=np.float32)
    for idx, uav in enumerate(legacy_world.uavs):
        mask = legacy_world.sensor_mask_for_position(uav.position, uav.sensor_radius)
        mass = float(prev_belief[mask].sum())
        newly_mass += mass
        repeated_mass += float((prev_world.visit_count_grid[mask] > 0.0).mean()) * 0.01
        legacy_world.belief_grid[mask] = 0.0
        per_agent[idx] = float(scenario.cfg("w_prob", 120.0)) * mass
    target = np.asarray(legacy_state["target_position"], dtype=np.float32)
    detected_now = any(
        np.linalg.norm(uav.position - target) <= uav.sensor_radius for uav in legacy_world.uavs
    )
    detect_bonus = 0.0
    if detected_now and not legacy_state.get("detected", False):
        legacy_state["detected"] = True
        legacy_state["time_to_detection"] = legacy_world.step_count
        detect_bonus = float(scenario.cfg("detection_bonus", 8.0))
    distance = float(np.asarray(transition["path_length_delta"], dtype=np.float32).sum())
    expected_reward = (
        float(scenario.cfg("w_prob", 120.0)) * newly_mass
        + detect_bonus
        - float(scenario.cfg("w_repeat", 1.0)) * repeated_mass
        - float(scenario.cfg("w_distance", 0.15)) * distance
    )

    actual = scenario.compute_rewards(
        prev_world,
        optimized_world,
        optimized_state,
        transition,
    )
    assert np.isclose(actual["team_reward"], expected_reward, rtol=0.0, atol=1e-7)
    assert np.array_equal(actual["per_agent_rewards"], per_agent)
    assert np.array_equal(optimized_world.belief_grid, legacy_world.belief_grid)
    assert actual["metrics"]["detection_rate"] == float(legacy_state["detected"])
    assert actual["metrics"]["time_to_detection"] == float(legacy_state["time_to_detection"])


def test_connectivity_breakdown_and_relay_match_legacy():
    prev_world = _make_world()
    world = _move_and_accumulate(prev_world, np.asarray([0.10, 0.02], dtype=np.float32))
    scenario = ConnectivityExpansionScenario(_base_config()["tasks"])
    connected = world.connected_to_base_mask()
    result = scenario._connected_coverage_breakdown(prev_world, world, connected)
    legacy = _legacy_coverage_stats(prev_world, world)
    denominator = max(float(world.navigable_grid_mask().sum()), 1.0)
    connected_mask = legacy["new_masks"][connected].any(axis=0)
    disconnected_mask = legacy["new_masks"][~connected].any(axis=0)
    disconnected_only = disconnected_mask & ~connected_mask

    assert result["connected_new_coverage_cells"] == float(connected_mask.sum())
    assert result["disconnected_new_coverage_cells"] == float(disconnected_only.sum())
    assert result["connected_new_coverage_gain"] == float(connected_mask.sum()) / denominator
    assert result["disconnected_new_coverage_gain"] == float(disconnected_only.sum()) / denominator
    assert scenario._relay_credit(world) == _legacy_relay_credit(world)


def test_target_interception_vectorization_matches_legacy():
    config = _base_config()
    world = _make_world()
    scenario = TargetInterceptionScenario(config["tasks"])
    state = scenario.reset(np.random.default_rng(41), world)
    state["target_indices"][:] = 2
    scenario._sync_legacy_aliases(state)

    expected_mass = _legacy_interception_belief(world, state)
    expected_field = _legacy_interception_field(scenario, world, state)
    actual_mass = scenario._covered_future_belief(world, state)
    actual_field = scenario.build_task_field(world, state)

    assert np.isclose(actual_mass, expected_mass, rtol=0.0, atol=1e-12)
    assert np.allclose(actual_field, expected_field, rtol=0.0, atol=1e-7)


def test_priority_gaussian_field_cache_preserves_legacy_field():
    config = _base_config()
    config["tasks"]["priority_inspection"]["poi_target_field"] = "gaussian"
    world = _make_world()
    scenario = PriorityInspectionScenario(config["tasks"])
    state = scenario.reset(np.random.default_rng(53), world)
    actual = scenario.build_task_field(world, state)
    cached = scenario.build_task_field(world, state)

    expected = super(PriorityInspectionScenario, scenario).build_task_field(world, state)
    target = np.zeros_like(expected[4], dtype=np.float32)
    centers = world.grid_cell_centers()
    sigma = max(scenario.distance_cfg("poi_target_sigma", 0.08, world), 1e-6)
    for point, weight, is_visited in zip(state["pois"], state["weights"], state["visited"]):
        if is_visited:
            continue
        distance = np.linalg.norm(centers - point[None, None, :], axis=-1)
        target += float(weight) * np.exp(-0.5 * (distance / sigma) ** 2).astype(np.float32)
    target[~world.navigable_grid_mask()] = 0.0
    if target.max() > 1e-8:
        target /= float(target.max())
    expected[4] = target

    assert np.array_equal(actual, expected)
    assert np.array_equal(cached, expected)
    assert "_poi_kernels" in state


def test_risk_grid_pairwise_cache_and_reward_snapshot_are_safe():
    zones = [
        {"center": [0.55, 0.52], "radius": 0.10},
        {"x_min": 0.72, "x_max": 0.82, "y_min": 0.20, "y_max": 0.34},
    ]
    world = _make_world(no_fly_zones=zones)
    centers = world.grid_cell_centers()
    expected_risk = np.zeros((world.grid_size, world.grid_size), dtype=np.float32)
    for zone in zones:
        if "center" in zone:
            distance = np.linalg.norm(
                centers - np.asarray(zone["center"], dtype=np.float32),
                axis=-1,
            )
            expected_risk = np.maximum(
                expected_risk,
                (distance <= float(zone["radius"])).astype(np.float32),
            )
        else:
            inside = (
                (centers[..., 0] >= float(zone["x_min"]))
                & (centers[..., 0] <= float(zone["x_max"]))
                & (centers[..., 1] >= float(zone["y_min"]))
                & (centers[..., 1] <= float(zone["y_max"]))
            )
            expected_risk = np.maximum(expected_risk, inside.astype(np.float32))
    assert np.array_equal(world.risk_grid(), expected_risk)
    assert world.risk_grid() is world.risk_grid()

    positions = world.get_uav_positions()
    expected_pairwise = np.linalg.norm(
        positions[:, None, :] - positions[None, :, :],
        axis=-1,
    ).astype(np.float32)
    first_pairwise = world.compute_pairwise_distances()
    assert np.array_equal(first_pairwise, expected_pairwise)
    assert world.compute_pairwise_distances() is first_pairwise

    snapshot = world.snapshot_for_reward()
    assert len(snapshot.uavs[0].trajectory) == 1
    snapshot.coverage_grid.fill(1.0)
    snapshot.uavs[0].position[:] = 0.9
    assert not np.array_equal(snapshot.coverage_grid, world.coverage_grid)
    assert not np.array_equal(snapshot.uavs[0].position, world.uavs[0].position)
    assert snapshot.grid_cell_centers() is world.grid_cell_centers()
