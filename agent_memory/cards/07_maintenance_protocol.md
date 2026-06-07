# Maintenance Protocol

## Purpose

This card defines the operational rule for keeping repo-local agent memory aligned with the actual codebase.

## Hard rule

Any change that modifies repository behavior or the engineering contract must update `agent_memory/` in the same task before the work is considered complete.

This includes:

- code changes
- config changes
- CLI changes
- output directory changes
- training and evaluation workflow changes
- documentation changes that redefine the current source of truth
- long-horizon research debug changes that add `outputs/debug_long/<timestamp>/` records, diagnostics, or dataset builders
- robustness changes to debug artifact writers, such as atomic save behavior for `.npz` / JSON sidecars

## Minimum required updates

For every qualifying change:

1. update at least one relevant memory card
2. append the change to `cards/03_recent_modifications.md`
3. update `cards/05_reuse_rules.md` if reuse guidance changed
4. update this card if the maintenance process itself changed
5. update `manifest.yaml` if cards were added, removed, or renamed

## Debug-long rule

Long-horizon debug work is only considered complete when both of the following are true:

- the code/config change is applied and verified
- the corresponding `outputs/debug_long/<timestamp>/` notes capture the command, result, and conclusion

This keeps research-oriented repairs auditable even when the final answer is still partial or negative.

## Review checklist

Before closing a task, confirm:

- the code matches the claimed behavior
- docs and CLI examples reflect the new behavior
- tests cover the new contract where practical
- agent memory reflects the new source of truth

## Non-compliance examples

The following are considered incomplete work:

- adding a new training flag without updating memory
- moving outputs to a new directory without updating reuse rules
- changing checkpoint behavior without updating the training pipeline card
- closing an audit or implementation task while memory still describes the old contract

## Trusted conclusion

`agent_memory/` is part of the repository contract, not optional commentary. Future agents should treat memory synchronization as required maintenance, not as an afterthought.

## Output registry maintenance rule (2026-06-07)

For every newly launched run, maintenance is incomplete unless:

1. the output path follows the current debug/formal layout;
2. the appropriate `run_registry.yaml` contains the run before execution;
3. the same record is updated after execution;
4. purpose and status are human-readable;
5. config snapshots and metrics remain inside the run directory.

Historical layouts may be read but must not be used as templates for new work.

## Direct MAPPO specialist layout override

Direct specialist maintenance must verify:

- debug path is `<debug-root>/phaseXX_<purpose>/<runtime>/<task>/trialXX/sN/`;
- formal path is `<training-root>/<runtime>/<task>/sN/`, unless multiple formal trials require `trialXX/sN/`;
- `phase_name` validation occurs before creating outputs;
- run names are only `sN` or `trialXX_sN`;
- summary/report records complete parameters and best checkpoint.
