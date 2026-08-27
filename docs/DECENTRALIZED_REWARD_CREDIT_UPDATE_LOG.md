# Decentralized Reward and Credit Update Log

This document tracks research and development work on `feature/decentralized-reward-credit`.

The branch starts from the completed heatmap-conditioned waypoint-policy baseline at commit `0bb63b0`. The existing heatmap implementation is preserved as the reference baseline.

## Current motivation

- UAVs receive weak or sparse feedback while approaching useful target regions.
- Shared team rewards provide weak individual credit assignment, making productive and inactive UAV behavior difficult to distinguish.
- The actor consumes centralized observations and therefore does not represent decentralized execution.

## Development order

1. Add a dense distance/progress reward.
2. Define the local observation contract.
3. Introduce a decentralized shared actor.
4. Establish clearer per-agent credit assignment.
5. Evaluate stronger credit-assignment or CTDE methods only if justified.
6. Re-evaluate the heatmap contribution after the learning architecture is stable.

VDN and QMIX are not selected architectures. They are possible later comparisons whose suitability must first be evaluated against the current continuous waypoint-policy setup.

## Validation rule

Each stage should progress through focused implementation, focused tests, a small smoke run, a seed-0 diagnostic, a limited comparison, and multi-seed evaluation only when justified.

## 2026-08-27 — Branch initialization

### Objective

- Separate the new reward, decentralization, and credit-assignment work from the completed heatmap baseline.

### Baseline

- Source branch: `feature/heatmap-task-adapters`
- Baseline commit: `0bb63b0`
- Development branch: `feature/decentralized-reward-credit`
- No reward or policy implementation has been changed yet.

### Next step

- Perform a read-only audit of the current priority-inspection reward path before defining the dense reward formulation.

## 2026-08-27 — Priority-inspection reward audit and V1 design

### Audit findings

- The completed heatmap OFF, REAL, and ZERO experiments use the default `PriorityInspectionScenario` reward coefficients.
- First visits, repeats, lateness, flight distance, and safety contribute to the team reward.
- `_approach_gain` exists, but `w_approach` defaults to `0.0` in the completed heatmap experiments and therefore did not contribute to training reward.
- The environment produces both a team reward and per-agent rewards. Per-agent rewards are stored as `[T,E,N]`, while MAPPO GAE and PPO optimization use only the scalar team reward `[T,E]`.
- `PriorityInspectionScenario.compute_rewards()` receives pre-transition and post-transition UAV positions before mutating the current-step visited mask, making it the appropriate location for V1 progress shaping.
- Reward can use task and world state directly and remain identical across future OFF, REAL, and ZERO heatmap conditions.

### V1 dense-progress decision

- Add `w_progress` with a default value of `0.0` for backward compatibility.
- For each UAV, select the nearest POI from the pre-step unvisited set using the pre-transition UAV position, then freeze that POI identity for both distance calculations within the step.
- Measure distance to the target region as `max(center_distance - sensor_radius, 0)`.
- Define signed progress as `d_previous - d_current`: approaching is positive, moving away is negative, and no movement is zero.
- Mean the per-agent progress values before adding the term to the existing team reward.
- Expose per-agent progress for diagnostics without changing GAE, PPO, the critic, or policy architecture in V1.
- Do not priority-weight or clip progress in V1.
- Keep expired but unvisited POIs eligible, matching current task semantics.
- Include a newly completed POI in progress for its completion step by using the same pre-step target set for both distances.
- Keep the existing `_approach_gain` implementation unchanged.
- Use `w_progress: 1.0` only in a new experimental configuration; do not change any frozen baseline configuration.

### Required diagnostics before seed-0

- Mean per-agent progress.
- Positive, negative, and zero progress fractions.
- Mean target-region distance before and after the transition.
- Weighted progress reward contribution.

V1 addresses dense reward feedback only. It does not solve multi-agent credit assignment because MAPPO still trains from a shared team advantage.

### Next step

- Implement V1 test-first and add a dedicated debug configuration. The frozen heatmap configurations must remain unchanged.
