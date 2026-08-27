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
