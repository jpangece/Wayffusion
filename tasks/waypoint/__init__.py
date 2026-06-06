from __future__ import annotations

from tasks.waypoint.area_coverage import AreaCoverageScenario
from tasks.waypoint.base import WAYPOINT_TASKS, WaypointScenario
from tasks.waypoint.belief_search import BeliefSearchScenario
from tasks.waypoint.connectivity_expansion import ConnectivityExpansionScenario
from tasks.waypoint.dynamic_target_escort import DynamicTargetEscortScenario
from tasks.waypoint.priority_inspection import PriorityInspectionScenario
from tasks.waypoint.target_interception import TargetInterceptionScenario


SCENARIO_REGISTRY = {
    "area_coverage": AreaCoverageScenario,
    "belief_search": BeliefSearchScenario,
    "priority_inspection": PriorityInspectionScenario,
    "connectivity_expansion": ConnectivityExpansionScenario,
    "dynamic_target_escort": DynamicTargetEscortScenario,
    "target_interception": TargetInterceptionScenario,
}


def build_waypoint_scenario(name: str, config: dict) -> WaypointScenario:
    if name not in SCENARIO_REGISTRY:
        raise ValueError(f"Unsupported waypoint scenario: {name}")
    return SCENARIO_REGISTRY[name](config)


__all__ = [
    "AreaCoverageScenario",
    "BeliefSearchScenario",
    "ConnectivityExpansionScenario",
    "DynamicTargetEscortScenario",
    "PriorityInspectionScenario",
    "TargetInterceptionScenario",
    "WAYPOINT_TASKS",
    "WaypointScenario",
    "build_waypoint_scenario",
]
