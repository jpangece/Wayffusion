# Heatmap-Guided Multi-UAV Adaptation — Development Update Log

This document records incremental design, implementation, testing, and evaluation progress for the Wayffusion heatmap-guided multi-UAV adaptation upgrade.

## Reusable entry template

### Date

`YYYY-MM-DD`

### Objective

- State the focused objective for this increment.

### Changes

- List the files or behavior changed.

### Design decisions

- Record decisions and their compatibility implications.

### Validation

- Record tests, inspections, and results.

### Known limitations

- Record work intentionally deferred or not yet verified.

### Next step

- State the next small, reviewable increment.

### Related commit

- Add the commit subject and hash when available; do not invent a hash.

## 2026-07-21 — Repository architecture audit and implementation planning

### Objective

- Review the existing Wayffusion platform before changing code.
- Identify safe extension points for heatmap generation, heatmap-conditioned policy training, three task-element adapters, and evaluation.

### Changes

- Added the repository architecture audit document.
- No production code, test, configuration, or dependency changes were made.

### Design decisions

- Preserve the existing five-channel `task_field`.
- Introduce a separate optional task-element heatmap observation.
- Preserve existing baseline policy classes.
- Add a new heatmap-conditioned policy class later.
- Keep the initial implementation two-dimensional.
- Start with baseline contract tests before implementing new functionality.
- Develop in small reviewed increments rather than one large change.
- Keep implementation commits and update-log commits small and focused.

### Validation

- Confirmed that the audit is based on verified repository paths and symbols.
- Confirmed that no production files were modified.
- Confirmed that the working branch is `feature/heatmap-task-adapters`.

### Known limitations

- No baseline training or test run has been performed in this documentation task.
- No heatmap generation or adapter implementation exists yet.
- Server/Docker execution has not yet been validated for this branch.

### Next step

- Add baseline contract tests for observations, actions, rollout tensors, and default backward-compatible behavior.

### Related commit

- Pending commit: `docs: document Wayffusion architecture for heatmap adaptation`

## 2026-07-21 — Baseline observation, action, and rollout contract tests

### Objective

- Lock down current Wayffusion interfaces before adding heatmap and adapter functionality.
- Protect baseline behavior from accidental regression.

### Changes

- Created `tests/test_heatmap_baseline_contracts.py` with focused global observation, per-agent observation, direct-policy, candidate-policy, default compatibility, and synthetic rollout-buffer contract tests.
- Updated `docs/HEATMAP_ADAPTATION_UPDATE_LOG.md` with this validation record.
- No production code or configuration was changed.

### Design decisions

- Test current contracts before adding new observation keys.
- Preserve the existing five-channel task field.
- Keep tests CPU-only, deterministic, and small.
- Focus on keys, shapes, types, and baseline compatibility.
- Avoid changing production code for test convenience.

### Validation

- `pytest -q tests/test_heatmap_baseline_contracts.py` — collection failed with one error because `/opt/anaconda3/bin/python3` could not import `torch` (`ModuleNotFoundError: No module named 'torch'`).
- `/Users/a...../.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -m pytest -q tests/test_heatmap_baseline_contracts.py` — command failed because the bundled Python does not contain the `pytest` module.
- `/opt/anaconda3/envs/ml-env/bin/python -m pytest -q tests/test_heatmap_baseline_contracts.py` — command failed because the ML environment does not contain the `pytest` module.
- `PYTHONPATH=/opt/anaconda3/lib/python3.12/site-packages /opt/anaconda3/envs/ml-env/bin/python -m pytest -q tests/test_heatmap_baseline_contracts.py` — collection failed with one error because Python 3.10 attempted to load the Python 3.12 NumPy binary extension and could not import `numpy.core._multiarray_umath`.
- `/opt/anaconda3/envs/ml-env/bin/python -c "import sys; sys.path.append('/opt/anaconda3/lib/python3.12/site-packages'); import pytest; raise SystemExit(pytest.main(['-q','tests/test_heatmap_baseline_contracts.py']))"` — collection failed with one error because the ML environment could not import `gymnasium`.

### Known limitations

- No requested contract required a production change, so no implementation limitations were identified in this step.
- The tests could not be executed successfully in the available local Python environments: the pytest-enabled environment lacks PyTorch, while the PyTorch-enabled environment lacks pytest and Gymnasium. Dependencies were not installed because this task prohibits dependency changes.

### Next step

- Define the optional task-element heatmap observation contract while keeping it disabled by default.

### Related commit

- Pending commit: `test: lock down baseline waypoint contracts`
- Pending commit: `docs: record baseline contract validation`

## 2026-07-21 — Optional heatmap observation contract

### Objective

- Add only the optional task-element heatmap observation contract.
- Preserve all default observation, reward, policy, and training behavior.

### Changes

- Added the optional `heatmap_observation` environment configuration with `enabled: false` by default and a configurable positive `channels` count (default `1`).
- When enabled, global and per-agent observations expose `task_element_heatmap [C,G,G]` as a `float32` tensor bounded to `[0,1]`.
- Added focused observation-contract tests for default absence, enabled spaces and values, step persistence, and channel validation.

### Design decisions

- The existing five-channel `task_field` is unchanged.
- The new observation is absent unless explicitly enabled, preserving exact default keys and spaces.
- This increment initializes the heatmap to zeros. Scenario-specific generation and semantic channel definitions are intentionally deferred.
- No reward, policy, rollout, optimization, or training code was changed.

### Validation

- Server: `admin1-Super-Server`.
- Container: `pjs_wayffusion`.
- Base image: `yejun-cu128-snapshot:with-tools`.
- Python: `3.10.12`.
- PyTorch: `2.7.1+cu128`.
- CUDA runtime used by PyTorch: `12.8`.
- GPU: `NVIDIA GeForce RTX 5090`.
- `git diff --check` — passed with no output.
- `CUDA_VISIBLE_DEVICES=0 python3 -m pytest -q tests/test_heatmap_observation_contract.py tests/test_heatmap_baseline_contracts.py` — 8 passed in 2.12s.
- `CUDA_VISIBLE_DEVICES=0 python3 -m pytest -q` — 114 passed in 8.49s.

### Known limitations

- The heatmap carries no task-element semantics yet; this increment defines only its optional fixed-shape observation boundary.

### Next step

- Define task-element channel semantics and generation in a separate reviewed increment.

### Related commit

- None; changes intentionally left uncommitted.

## 2026-07-22 — Deterministic task-element priority heatmap generator

### Objective

- Specify and implement the first deterministic task-element heatmap generator.
- Keep generation standalone and leave the existing optional observation integration unchanged.

### Changes

- Added the production utility `envs/waypoint/task_element_heatmap.py`.
- Added the focused test specification `tests/test_task_element_heatmap_generator.py`.
- Added `generate_task_element_heatmap(grid_size, task_elements, channels=1)` for a single normalized spatial-priority channel.

### Design decisions

- The output contract is `[1,G,G]`, with grid coordinates written as `output[0,y,x]`.
- Each valid task element contributes its non-negative priority to its grid cell; priorities in the same cell accumulate.
- The completed channel is normalized by its maximum accumulated value.
- Empty input and all-zero priorities produce an all-zero `float32` heatmap.
- Output is deterministic, input-order independent, `float32`, and bounded to `[0,1]`.
- Invalid coordinates, negative priorities, invalid grid sizes, and unsupported channel counts are rejected instead of clipped or ignored.
- This increment supports exactly one normalized priority channel and does not integrate the generator into `WaypointMultiUAVEnv`.

### Validation

- Local `git diff --check` passed.
- Local generator-focused validation passed: 11 tests in 0.26s.
- The existing compatibility tests could not collect locally because the local interpreter lacked `gymnasium` and `torch`; this was a local dependency limitation, not a product test failure.
- No dependencies were installed locally, and `PYTHONPATH` was not changed.
- The IRMV server could not reach GitHub directly over HTTPS. The two commits were transferred as a git format-patch file and applied with `git am`; no successful server-side GitHub pull is claimed.
- The Mac/GitHub commits were `dfead2c test: specify task element heatmap generator` and `6ad5ad0 feat: add deterministic task element heatmap generator`.
- Applying the same patch content on the IRMV server created equivalent local commits `570e201 test: specify task element heatmap generator` and `2409602 feat: add deterministic task element heatmap generator`.
- IRMV Docker validation used container `pjs_wayffusion`, constrained GPU visibility with `CUDA_VISIBLE_DEVICES=0`, and executed tests as the numeric `pjs` UID/GID to avoid root-owned files in the bind-mounted repository.
- IRMV focused validation passed: 19 tests in 2.81s.
- IRMV full regression validation passed: 125 tests in 8.27s.

### Known limitations

- The generator exists as a standalone utility only.
- No task-element source or environment observation update currently invokes it.
- The current semantic model supports exactly one normalized priority channel.

### Next step

- Define and test the environment integration contract that supplies task elements to the generator and writes the generated result into the optional `task_element_heatmap` observation without changing disabled-mode behavior.

### Related commit

- Test-first specification: `dfead2c test: specify task element heatmap generator`.
- Production implementation: `6ad5ad0 feat: add deterministic task element heatmap generator`.
- Equivalent IRMV-applied test commit: `570e201 test: specify task element heatmap generator`.
- Equivalent IRMV-applied production commit: `2409602 feat: add deterministic task element heatmap generator`.

## 2026-07-22 — Task-element heatmap environment integration

### Objective

- Integrate the existing deterministic task-element heatmap generator into `WaypointMultiUAVEnv`.
- Preserve disabled-mode observations and all existing compatibility behavior.

### Changes

- Added the focused integration specification `tests/test_heatmap_environment_integration.py`.
- Updated `envs/waypoint_marl_env.py` to parse `heatmap_observation.task_elements`, defaulting to an empty list while retaining the existing `channels` default of `1`.
- In enabled mode, reset calls `generate_task_element_heatmap` with the environment grid size, configured task elements, and channel count, then stores the result in `self.task_element_heatmap`.
- Global and every per-agent observation expose the same stored values under `task_element_heatmap`.
- The configured heatmap remains static throughout the episode; it is not regenerated, decayed, or mutated during a step.

### Design decisions

- Invalid configured task elements propagate `ValueError` from the standalone generator without environment-side clipping or repair.
- Disabled mode preserves the exact legacy global and per-agent observation keys and does not invoke the generator.
- Empty enabled task-element input preserves the existing all-zero `float32` behavior.
- Existing enabled multi-channel zero-placeholder behavior remains unchanged when no task elements are configured.
- Non-empty task elements use the generator's current single-channel semantic contract.
- No reward, policy, rollout-buffer, training, task-logic, or configuration file was changed.

### Validation

- Before production integration, IRMV Docker execution of the new integration file produced 4 passed and 2 failed tests. The expected failures showed that configured heatmap values remained zero and that an invalid coordinate did not raise `ValueError`, confirming that environment integration was still missing.
- After implementation, local `git diff --check` passed.
- Local test collection was blocked because the local interpreter lacked `gymnasium` and `torch`; this was a local dependency limitation, not a product failure.
- No dependencies were installed locally, and `PYTHONPATH` was not changed.
- The IRMV server could not pull from GitHub directly. The two commits were transferred as the same patch content and applied with `git am`; no successful server-side GitHub pull is claimed.
- The Mac/GitHub commits were `cadd1d7 test: specify heatmap environment integration` and `3b0108f feat: integrate task element heatmap observation`.
- Applying that same patch content on the IRMV server created equivalent commits `c7a8ebf test: specify heatmap environment integration` and `995bbf6 feat: integrate task element heatmap observation`.
- IRMV Docker validation used container `pjs_wayffusion`, set `CUDA_VISIBLE_DEVICES=0`, and executed tests as the numeric `pjs` UID/GID to avoid root-owned files in the bind-mounted repository.
- IRMV focused validation passed: 25 tests in 2.35s.
- IRMV full regression validation passed: 131 tests in 8.03s.

### Known limitations

- Task elements are currently static configuration data.
- The heatmap is generated only on reset and does not mutate during an episode.
- No task-specific adapter currently derives task elements from live mission state.
- The generator supports one semantic priority channel for non-empty task elements.
- Existing empty multi-channel placeholder behavior is compatibility-only and is not yet a semantic multi-channel implementation.

### Next step

- Define a task-element provider or adapter interface that derives normalized task elements from live task state without coupling task-specific semantics directly into `WaypointMultiUAVEnv`.
- Keep the next increment test-first and backward-compatible.

### Related commit

- Test-first specification: `cadd1d7 test: specify heatmap environment integration`.
- Production implementation: `3b0108f feat: integrate task element heatmap observation`.
- Equivalent IRMV-applied test commit: `c7a8ebf test: specify heatmap environment integration`.
- Equivalent IRMV-applied production commit: `995bbf6 feat: integrate task element heatmap observation`.

## 2026-07-22 — Task-element provider selection and static fallback

### Objective

- Define a test-first task-element provider selection and static fallback contract.
- Add a small provider registry and resolver without integrating it into the environment yet.

### Changes

- Added `envs/waypoint/task_element_provider.py` with the provider protocol, registry operations, and source resolver.
- Added the focused specification `tests/test_task_element_provider_selection.py`.
- Defined `TaskElement` as a mapping containing raw task-element fields.
- Defined `TaskElementProvider.build_task_elements` with `scenario`, `world`, and `task_state` inputs.
- Added `register_task_element_provider`, `clear_task_element_provider_registry`, and `resolve_task_elements`.

### Design decisions

- If heatmap observation is disabled, resolution returns the configured static elements unchanged and neither constructs nor invokes a provider.
- If provider use is disabled, resolution returns the static elements unchanged.
- If both flags are enabled and the task is registered, the resolver constructs the provider once, invokes it once, and returns its result unchanged.
- An empty registered-provider result means there are no live elements and does not trigger static fallback.
- An unregistered task falls back to configured static elements without error.
- The resolver does not validate, copy, sort, normalize, clip, mutate, or grid-convert task elements, and it does not call the heatmap generator.
- Coordinate and priority validation, same-cell accumulation, and normalization remain owned by the standalone generator.
- The registry rejects empty task names and duplicate registrations.
- No environment, scenario, configuration, reward, policy, rollout-buffer, or training file was changed.

### Validation

- Local `git diff --check` passed.
- Local provider-selection validation passed: 10 tests in 0.01s.
- Local standalone-generator compatibility validation passed: 11 tests in 0.19s.
- These focused tests required neither `gymnasium` nor `torch`; no dependencies were installed, and `PYTHONPATH` was not changed.
- The IRMV server could not pull from GitHub directly. The two commits were transferred as the same patch content and applied with `git am`; no successful server-side GitHub pull is claimed.
- The Mac/GitHub commits were `2d2582f test: specify task element provider selection` and `4142f4d feat: add task element provider registry`.
- Applying that same patch content on the IRMV server created equivalent commits `bdf9aa5 test: specify task element provider selection` and `afd090b feat: add task element provider registry`.
- IRMV Docker validation used container `pjs_wayffusion`, set `CUDA_VISIBLE_DEVICES=0`, and executed tests as the numeric `pjs` UID/GID to avoid root-owned files in the bind-mounted repository.
- IRMV focused validation passed: 35 tests in 2.85s.
- IRMV full regression validation passed: 141 tests in 8.62s.

### Known limitations

- The provider registry is not yet integrated into `WaypointMultiUAVEnv`.
- No real task-specific provider is registered.
- Static configured task elements remain the only source used by the environment.
- Provider construction occurs per resolver call by contract; the environment lifecycle has not yet fixed whether resolution occurs once per reset.
- Dynamic step-time task-element refresh is not implemented.

### Next step

- Add a test-first environment integration contract for the provider resolver.
- The environment should parse an optional provider-enabled flag, call the resolver once during reset after scenario and task-state initialization, preserve static fallback, and remain unchanged when provider support is disabled.
- Do not implement the `priority_inspection` adapter in that same increment.

### Related commit

- Test-first specification: `2d2582f test: specify task element provider selection`.
- Production implementation: `4142f4d feat: add task element provider registry`.
- Equivalent IRMV-applied test commit: `bdf9aa5 test: specify task element provider selection`.
- Equivalent IRMV-applied production commit: `afd090b feat: add task element provider registry`.

## 2026-07-22 — Task-element provider environment integration

### Objective

- Integrate the existing task-element provider resolver into the `WaypointMultiUAVEnv` reset lifecycle.
- Preserve static source behavior and disabled-mode compatibility by default.

### Changes

- Added the focused environment integration specification `tests/test_task_element_provider_environment_integration.py`.
- Updated `envs/waypoint_marl_env.py` to parse the optional `heatmap_observation.provider.enabled` setting.
- Provider use defaults to `false` when either the provider section or its `enabled` field is absent, so existing configurations remain backward-compatible.
- Reset now resolves the task-element source and passes the resulting raw elements to the existing heatmap generator.

### Design decisions

- Resolution occurs exactly once per reset after active scenario construction, `scenario.reset`, task-state initialization, and mission-goal metadata injection.
- The active task name comes from `self.current_scenario.name`, reflecting the scenario selected for the current episode.
- The resolver receives `self.current_scenario`, `self.world`, and `self.task_state` by identity.
- A second reset performs one new resolution; neither `step` nor `step_global` resolves or invokes a provider.
- When heatmap observation is disabled, providers are not constructed or invoked.
- When provider use is disabled, static configured task elements remain the source.
- When an enabled registered provider exists, its result replaces static elements.
- An empty registered-provider result produces a zero heatmap without static fallback, while an unregistered task uses static configured elements.
- `resolve_task_elements` owns source selection only. `generate_task_element_heatmap` retains ownership of validation, accumulation, normalization, shape, and dtype.
- The environment does not modify, normalize, sort, clip, or grid-convert provider output.
- No scenario, reward, policy, rollout-buffer, training, or configuration file was changed.

### Validation

- Before production integration, IRMV Docker execution of the new integration specification produced 4 passed and 3 failed tests. The expected failures showed provider construction counts remaining zero, proving that the environment was not yet calling the resolver.
- After implementation, local `git diff --check` passed.
- Local test collection was blocked because the local interpreter lacked `gymnasium` and `torch`; this was a local dependency limitation, not a product failure.
- No dependencies were installed locally, and `PYTHONPATH` was not changed.
- The IRMV server could not pull from GitHub directly. The two commits were transferred as the same patch content and applied with `git am`; no successful server-side GitHub pull is claimed.
- The Mac/GitHub commits were `a2b9c86 test: specify provider environment integration` and `eb262d6 feat: integrate task element provider resolution`.
- Applying that same patch content on the IRMV server created equivalent commits `4698025 test: specify provider environment integration` and `937de3a feat: integrate task element provider resolution`.
- IRMV Docker validation used container `pjs_wayffusion`, set `CUDA_VISIBLE_DEVICES=0`, and executed tests as the numeric `pjs` UID/GID to avoid root-owned files in the bind-mounted repository.
- IRMV focused validation passed: 42 tests in 3.30s.
- IRMV full regression validation passed: 148 tests in 8.29s.

### Known limitations

- No real task-specific provider is registered yet.
- Provider-derived task elements are generated only during reset.
- Dynamic step-time refresh is not implemented.
- Static configuration remains the fallback for unregistered tasks.
- The semantic generator supports one channel for non-empty task elements.

### Next step

- Implement the first real task-specific provider for `priority_inspection` using a test-first increment.
- Map unvisited POIs and their weights from `task_state` into raw grid-space task elements.
- Do not add dynamic step-time refresh in the same increment.

### Related commit

- Test-first specification: `a2b9c86 test: specify provider environment integration`.
- Production implementation: `eb262d6 feat: integrate task element provider resolution`.
- Equivalent IRMV-applied test commit: `4698025 test: specify provider environment integration`.
- Equivalent IRMV-applied production commit: `937de3a feat: integrate task element provider resolution`.

## 2026-07-22 — Priority-inspection task-element provider

### Objective

- Add the first real task-specific task-element provider for `priority_inspection`.
- Keep provider registration and environment selection outside this increment.

### Changes

- Added `envs/waypoint/priority_inspection_task_element_provider.py` with `PriorityInspectionTaskElementProvider`.
- Added the focused test specification `tests/test_priority_inspection_task_element_provider.py`.
- The provider reads `task_state["pois"]`, `task_state["weights"]`, and `task_state["visited"]`; deadlines and repeat-visit counts are intentionally ignored.
- Each unvisited POI produces one raw element of the form `{"x": grid_x, "y": grid_y, "priority": weight}`.

### Design decisions

- Visited POIs produce no elements, all-visited input returns an empty sequence, and source order is preserved.
- Multiple POIs mapping to one grid cell remain separate raw elements in source order; the provider does not accumulate or normalize priorities.
- Missing required keys raise `ValueError`.
- POIs must have shape `[N,2]`, while weights and visited flags must have shape `[N]`; inconsistent lengths raise `ValueError`.
- POI coordinates must be finite, and weights must be finite and non-negative.
- World-space bounds are validated before calling `world.world_to_grid`, preventing that method's clipping behavior from silently accepting invalid physical positions.
- Valid POIs are converted through `world.world_to_grid`, preserving grid XY orientation.
- The provider does not generate a heatmap or mutate `task_state`, the scenario, or the world.
- Task-specific filtering and world-to-grid conversion belong to the provider. Same-cell accumulation, normalization, output shape, dtype, and grid-coordinate bounds validation remain owned by the standalone generator.
- The provider is not registered yet, and `WaypointMultiUAVEnv` was not modified.
- No scenario, configuration, reward, policy, rollout-buffer, or training file was changed.

### Validation

- Before implementation, the new test failed during collection because `envs.waypoint.priority_inspection_task_element_provider` did not exist; this was the expected test-first failure.
- After implementation, local `git diff --check` passed.
- Local provider-focused validation passed: 23 tests in 0.17s.
- Local provider-selection and heatmap-generator compatibility validation passed: 21 tests in 0.12s.
- No dependencies were installed locally, and `PYTHONPATH` was not changed.
- The IRMV server could not pull from GitHub directly. Equivalent patch content was transferred and applied with `git am`; no successful server-side GitHub pull is claimed.
- The Mac/GitHub commits were `3918908 test: specify priority inspection task element provider` and `09ef582 feat: add priority inspection task element provider`.
- Applying equivalent patch content on the IRMV server created commits `2276a6b test: specify priority inspection task element provider` and `d5ff9ad feat: add priority inspection task element provider`.
- IRMV Docker validation used container `pjs_wayffusion`, set `CUDA_VISIBLE_DEVICES=0`, and executed tests as the numeric `pjs` UID/GID.
- IRMV focused validation passed: 65 tests in 3.24s.
- IRMV full regression validation passed: 171 tests in 9.00s.

### Known limitations

- The provider is not yet registered for `priority_inspection`, so the environment cannot automatically select it.
- Provider output will be generated only during reset once registration is added.
- Dynamic step-time refresh after POIs become visited is not implemented.
- Deadline and repeat-visit semantics are not represented.
- The semantic heatmap remains single-channel for non-empty elements.
- Current unit tests use a dependency-light `GridWorld` stand-in rather than a full `MissionWorld` instance.

### Next step

- Add a test-first registration and real-environment integration contract for `priority_inspection`.
- Verify that enabling provider resolution for `priority_inspection` automatically selects the new provider.
- Use a real `MissionWorld` or real `WaypointMultiUAVEnv` episode to verify coordinate orientation and generated heatmap values.
- Do not add dynamic step-time refresh in the same increment.

### Related commit

- Test-first specification: `3918908 test: specify priority inspection task element provider`.
- Production implementation: `09ef582 feat: add priority inspection task element provider`.
- Equivalent IRMV-applied test commit: `2276a6b test: specify priority inspection task element provider`.
- Equivalent IRMV-applied production commit: `d5ff9ad feat: add priority inspection task element provider`.

## 2026-07-22 — Priority-inspection provider registration and real-environment integration

### Objective

- Add the explicit production registration mechanism for the real `priority_inspection` task-element provider.
- Verify automatic provider selection through a real `WaypointMultiUAVEnv` episode and real `MissionWorld`.

### Changes

- Added `register_priority_inspection_task_element_provider()` to `envs/waypoint/priority_inspection_task_element_provider.py`.
- Added the real-environment integration specification `tests/test_priority_inspection_provider_registration_integration.py`.
- The registration function registers `PriorityInspectionTaskElementProvider` for the exact task name `priority_inspection` through `register_task_element_provider`.
- The provider class itself is the factory, creating a fresh provider instance for every resolver call.
- Registration is explicit and does not occur as an import side effect.

### Design decisions

- Repeated task-specific registration calls are idempotent, while direct duplicate calls to the generic provider registry still raise `ValueError`.
- No permanent module-level registration flag is used, so registration succeeds again after `clear_task_element_provider_registry()`.
- Only the exact duplicate-registration `ValueError` for `priority_inspection` is suppressed; unrelated `ValueError` exceptions propagate.
- Existing configurations are unchanged unless both heatmap observation and provider resolution are enabled.
- `PriorityInspectionTaskElementProvider` owns POI filtering, weight mapping, and world-to-grid conversion.
- `resolve_task_elements` owns provider-versus-static source selection.
- `generate_task_element_heatmap` owns accumulation, normalization, output shape, dtype, and generator validation.
- `WaypointMultiUAVEnv` owns reset lifecycle and observation exposure.
- No scenario, generic registry, environment, configuration, reward, policy, rollout-buffer, or training file was changed by the registration implementation.

### Validation

- Before implementation, the new integration test failed during collection because `register_priority_inspection_task_element_provider` did not exist; this was the expected test-first failure.
- The real-environment tests used `task_name` and `task_names` set to `priority_inspection`, a deterministic seed, disabled domain randomization, and `heatmap_observation.provider.enabled: true`.
- Expected heatmaps were independently constructed with NumPy from live `task_state["pois"]`, `task_state["weights"]`, `task_state["visited"]`, and `world.world_to_grid`, without calling `generate_task_element_heatmap`.
- Validation covered `[1,G,G]` shape, `float32` dtype, unvisited-weight placement at `[0,y,x]`, same-cell accumulation, maximum normalization, global/per-agent equality, provider precedence over static elements, default and explicit-disabled static behavior, re-registration after clearing, and non-transposed coordinate orientation.
- Initial IRMV execution after registration produced five failures before provider resolution because the test fixture used `max_steps=2`.
- `PriorityInspectionScenario` computes deadlines with `rng.integers(max(8, world.max_steps // 4), world.max_steps + 1, ...)`; with `max_steps=2`, this produced `low=8`, `high=3`, and `ValueError: low >= high`.
- This was a malformed test fixture, not a provider or environment product failure. The fixture was corrected from `max_steps=2` to `max_steps=16`, with no production-code change.
- Local `git diff --check` passed.
- Local dependency-light provider and registry validation passed: 33 tests in 0.17s.
- Real-environment tests could not collect locally because `gymnasium` was missing; this was a local dependency limitation, not a product failure.
- No dependencies were installed locally, and `PYTHONPATH` was not changed.
- The IRMV server could not pull from GitHub directly. Equivalent patch content was transferred and applied with `git am`; no successful server-side GitHub pull is claimed.
- The Mac/GitHub commits were `6601f34 test: specify priority inspection provider registration`, `c38215c feat: register priority inspection task element provider`, and `4fa0e6a test: fix priority inspection integration fixture`.
- Applying equivalent patch content on the IRMV server created commits `42257f8 test: specify priority inspection provider registration`, `6ee7c85 feat: register priority inspection task element provider`, and `e846921 test: fix priority inspection integration fixture`.
- IRMV Docker validation used container `pjs_wayffusion`, set `CUDA_VISIBLE_DEVICES=0`, and executed tests as the numeric `pjs` UID/GID.
- After the fixture correction, IRMV focused validation passed: 70 tests in 3.11s.
- After the fixture correction, IRMV full regression validation passed: 176 tests in 8.28s.

### Known limitations

- The `priority_inspection` heatmap is generated only during reset and does not refresh when POIs become visited during an episode.
- Deadline and repeat-visit semantics are not represented.
- Only `priority_inspection` has a real task-specific provider.
- Static configured elements remain the fallback for unregistered tasks.
- Non-empty semantic provider output remains single-channel.

### Next step

- Define a test-first dynamic refresh lifecycle contract for provider-derived heatmaps, focused only on `priority_inspection` visited-state changes.
- Decide and verify whether refresh occurs after `scenario.step_update` and before the next observation is built.
- Ensure provider resolution occurs once per step only when dynamic refresh is explicitly enabled.
- Preserve reset-only static behavior by default.
- Do not add another task adapter in the same increment.

### Related commit

- Test-first specification: `6601f34 test: specify priority inspection provider registration`.
- Production implementation: `c38215c feat: register priority inspection task element provider`.
- Test fixture correction: `4fa0e6a test: fix priority inspection integration fixture`.
- Equivalent IRMV-applied test commit: `42257f8 test: specify priority inspection provider registration`.
- Equivalent IRMV-applied production commit: `6ee7c85 feat: register priority inspection task element provider`.
- Equivalent IRMV-applied fixture correction: `e846921 test: fix priority inspection integration fixture`.

## 2026-07-22 — Optional dynamic provider heatmap refresh

### Objective

- Add optional step-time refresh for provider-derived task-element heatmaps.
- Preserve the existing reset-only behavior unless dynamic refresh is explicitly enabled.

### Changes

- Added the focused specification `tests/test_task_element_provider_dynamic_refresh.py`.
- Updated `envs/waypoint_marl_env.py` to parse `heatmap_observation.provider.refresh_on_step`, which defaults to `false`.
- Extracted the existing provider-resolution and heatmap-generation sequence into `_refresh_task_element_heatmap()` so reset and step-time refresh share the same construction path.
- Reset continues to use the shared helper without changing its existing resolution semantics.
- `step` and `step_global` each refresh exactly once after task-state mutation and reward computation, before constructing or returning same-transition observations and global states.

### Design decisions

- Step-time refresh occurs only when heatmap observation, provider resolution, and `refresh_on_step` are all enabled.
- With `refresh_on_step` absent or `false`, provider-derived heatmaps remain reset-only and existing configurations retain their prior behavior.
- `step` and `step_global` each contain one guarded refresh at the corresponding transition point and do not delegate refresh to observation accessors, preventing duplicate provider calls.
- `get_global_state` and per-agent observation construction do not invoke provider resolution.
- Provider instances are not cached; reset performs one provider construction and build, and each dynamically refreshed step performs one additional construction and build under the resolver contract.
- Terminal and truncated transitions refresh before final observations are assembled, so returned observations and `infos[agent]["terminal_observation"]` reflect final task state.
- Scenarios own task-state mutation, providers own task-state-to-element conversion, the resolver owns provider-versus-static source selection, the generator owns accumulation, normalization, shape, and dtype, and `WaypointMultiUAVEnv` owns refresh timing and observation exposure.
- No provider, registry, generator, scenario, configuration, reward, policy, rollout-buffer, or training file was changed.

### Validation

- Before implementation, IRMV Docker execution of the test-first specification produced 5 passed and 5 failed tests in 1.13s.
- The passing cases confirmed that an absent `refresh_on_step` defaults to `false`, explicit `false` preserves reset-only behavior, disabled heatmap observation and disabled provider resolution prevent provider invocation, and unregistered tasks preserve static fallback.
- The expected failures showed that `step` and `step_global` performed no additional provider resolution, terminal observations retained the reset-only heatmap, and `priority_inspection` visited-state changes were not reflected. These failures demonstrated that dynamic environment refresh was still missing.
- Before the production commit, local `git diff --check` passed.
- Local environment-test collection was blocked because the local interpreter lacked `gymnasium`; this was a local dependency limitation, not a production failure.
- No dependencies were installed locally, and `PYTHONPATH` was not changed.
- The IRMV server could not pull from GitHub directly. Equivalent patches were transferred and applied with `git am`; no successful server-side GitHub pull is claimed.
- The Mac/GitHub commits were `bd7defe test: specify provider heatmap dynamic refresh` and `fbe049a feat: refresh provider heatmap after steps`.
- Applying equivalent patch content on the IRMV server created commits `e524457 test: specify provider heatmap dynamic refresh` and `d58b577 feat: refresh provider heatmap after steps`.
- IRMV Docker validation used container `pjs_wayffusion`, set `CUDA_VISIBLE_DEVICES=0`, and executed tests with the numeric `pjs` UID/GID.
- IRMV focused validation passed: 80 tests in 3.25s.
- IRMV full regression validation passed: 186 tests in 8.50s.

### Verified environment behavior

- Reset performs one provider resolution.
- Each step performs exactly one additional resolution only when dynamic refresh is enabled; `step` and `step_global` do not double-refresh.
- Terminal and truncated observations contain the refreshed heatmap.
- Visited `priority_inspection` POIs disappear from the refreshed heatmap, and remaining POIs are accumulated and normalized again.
- Global and all per-agent observations contain equal refreshed values.
- Calling `get_global_state` does not cause additional provider calls.
- Default, disabled, unregistered-task, and static-fallback behavior remains compatible.

### Known limitations

- Dynamic refresh is opt-in and disabled by default.
- Only `priority_inspection` currently has a real semantic provider.
- Deadline and repeat-visit semantics are excluded.
- Semantic provider output remains single-channel.
- Policy, encoder, rollout-buffer, critic, and MAPPO training consumption of `task_element_heatmap` has not yet been verified; the validation above establishes environment behavior only.
- No performance improvement has been measured.

### Next step

- Perform a read-only audit of the complete observation-to-policy path.
- Trace `task_element_heatmap` through wrappers, preprocessing, rollout collection, buffers, actor input, critic/global-state input, encoders, and MAPPO training.
- Determine whether the key is consumed, ignored, rejected, or automatically flattened without modifying code during the audit.

### Related commit

- Test-first specification: `bd7defe test: specify provider heatmap dynamic refresh`.
- Production implementation: `fbe049a feat: refresh provider heatmap after steps`.
- Equivalent IRMV-applied test commit: `e524457 test: specify provider heatmap dynamic refresh`.
- Equivalent IRMV-applied production commit: `d58b577 feat: refresh provider heatmap after steps`.

## 2026-07-23 — Optional DirectWaypointPolicy heatmap consumption

### Objective

- Add optional `task_element_heatmap` consumption to `DirectWaypointPolicy` while preserving the existing default architecture.
- Encode the heatmap spatially and condition both the actor and centralized critic without changing the environment, trainer, rollout buffer, candidate policy, or checked-in configurations.

### Changes

- Added the focused specification `tests/test_direct_waypoint_policy_heatmap_consumption.py`.
- Added the optional policy flag `use_task_element_heatmap`, which defaults to `false`.
- Updated `policies.build_policy` to pass the flag only to `DirectWaypointPolicy`; `CandidateSelectionWaypointPolicy` was not changed.
- Updated `policies/direct_waypoint_policy.py` with observation-space and runtime validation, a separate `heatmap_encoder`, and actor/critic feature fusion.

### Design decisions

- Disabled mode creates no heatmap encoder module, parameter, or buffer; existing modules, `state_dict` keys, parameter shapes, tensor shapes, and strict checkpoint compatibility remain unchanged.
- Observations without `task_element_heatmap` remain valid in disabled mode, and a present heatmap key is ignored.
- Enabled mode requires `task_element_heatmap` in the observation space and infers its exact `[C,G,G]` shape rather than hard-coding channels or grid size.
- One or more channels are supported, and channel and spatial dimensions must be positive.
- Runtime heatmaps must be floating-point tensors with exact batched shape `[B,C,G,G]`. Missing keys, wrong rank, channels, spatial dimensions, batch size, or tensor type raise a clear error mentioning `task_element_heatmap`.
- The existing `task_field` encoder and its input-channel count remain unchanged.
- The separate `heatmap_encoder` uses `Conv2d`, `ReLU`, a second `Conv2d`, `ReLU`, `AdaptiveAvgPool2d((2,2))`, and `Flatten`; raw heatmaps are not flattened directly.
- The encoded feature is appended to every actor context and to the centralized critic context. Actor output remains `[B,N,2]`, critic value remains `[B]`, and agent-mask behavior is unchanged.
- The actor and critic share the encoded feature within a forward pass without detaching it, so either loss can backpropagate into the heatmap encoder.
- Enabled architectures add parameters and widen fusion layers; strict loading of an old disabled checkpoint into an enabled architecture is expected to fail, and no checkpoint migration was added.

### Validation

- Before implementation, IRMV execution of the test-first specification produced 1 passed and 13 failed tests in 1.75s.
- The passing test confirmed disabled compatibility. The expected failures showed that `DirectWaypointPolicy` did not accept `use_task_element_heatmap`, `build_policy` did not expose the option, and no `heatmap_encoder` existed.
- Local `git diff --check` passed, and dependency-free AST parsing of the modified policy files passed.
- PyTorch-based tests could not collect locally because the local interpreter lacked `torch`; this was a local dependency limitation, not a production failure.
- No dependencies were installed locally, and `PYTHONPATH` was not changed.
- An initial local command referenced two nonexistent test files and was corrected to the existing focused policy tests.
- The IRMV server could not pull from GitHub directly. Equivalent patches were transferred and applied with `git am`; no successful server-side GitHub pull is claimed.
- The Mac/GitHub commits were `9a697a7 test: specify direct policy heatmap consumption` and `9a72eeb feat: condition direct policy on task heatmap`.
- Applying equivalent patch content on the IRMV server created commits `4a3ea15 test: specify direct policy heatmap consumption` and `6bd178a feat: condition direct policy on task heatmap`.
- IRMV Docker validation used container `pjs_wayffusion`, set `CUDA_VISIBLE_DEVICES=0`, and executed tests with the numeric `pjs` UID/GID.
- IRMV focused policy validation passed: 26 tests in 5.13s.
- IRMV heatmap-pipeline regression validation passed: 32 tests in 1.28s.
- IRMV full regression validation passed: 200 tests in 9.95s.

### Verified policy behavior

- `build_policy` exposes explicit heatmap opt-in.
- The disabled architecture remains strict-checkpoint compatible with the prior architecture.
- The enabled architecture creates a separate spatial encoder while leaving the `task_field` encoder input channels unchanged.
- Batch sizes of one and greater than one preserve actor, critic, and agent-mask output shapes.
- Equal-mass heatmaps with different spatial arrangements can change both actor outputs and critic values.
- Separate actor and critic backward passes produce finite, non-`None` gradients for heatmap-encoder parameters.
- Malformed runtime heatmaps are rejected.
- Strict loading of a disabled checkpoint into an enabled architecture fails as expected because enabled mode adds parameters and wider fusion layers.
- Existing direct-policy, UVFA, candidate-policy, environment-heatmap-pipeline, and full-repository tests remain compatible.

### Ownership

- The environment owns semantic heatmap generation and refresh.
- Vector environments, the rollout buffer, and the trainer preserve the dictionary tensor.
- `DirectWaypointPolicy` owns heatmap encoding and actor/critic fusion.
- `CandidateSelectionWaypointPolicy` remains unchanged and still does not consume `task_element_heatmap`.

### Known limitations

- `use_task_element_heatmap` is not enabled by the default policy configuration.
- Heatmap observation, provider resolution, and dynamic refresh are not enabled by the default environment configuration.
- The main training launcher does not register the real `priority_inspection` provider.
- No training smoke test has yet confirmed an end-to-end optimizer update with heatmap support enabled.
- No performance improvement or ablation result has been measured; the validation establishes policy consumption, not training benefit.
- Enabled architecture is not strict-load compatible with old disabled checkpoints.
- `CandidateSelectionWaypointPolicy` still ignores `task_element_heatmap`.
- The active actor consumes centralized global state rather than a strictly decentralized local observation.

### Next step

- Define a minimal test-first training-activation contract.
- Add an explicit checked-in experimental configuration or launcher path that enables heatmap observation, provider resolution, step-time refresh, real `priority_inspection` provider registration, and `DirectWaypointPolicy` heatmap consumption.
- Verify environment construction, policy construction, one rollout, and one optimizer update without beginning full training or performance evaluation in the same increment.

### Related commit

- Test-first specification: `9a697a7 test: specify direct policy heatmap consumption`.
- Production implementation: `9a72eeb feat: condition direct policy on task heatmap`.
- Equivalent IRMV-applied test commit: `4a3ea15 test: specify direct policy heatmap consumption`.
- Equivalent IRMV-applied production commit: `6bd178a feat: condition direct policy on task heatmap`.

## 2026-07-23 — End-to-end heatmap training activation smoke

### Objective

- Verify that semantic task-element heatmaps can be activated through the real waypoint MAPPO training path.
- Verify one real rollout, one real optimizer update, and heatmap-encoder participation in optimization.
- Preserve disabled-policy behavior and unregistered-provider static fallback.

### Changes

- Added the focused smoke specification `tests/test_heatmap_training_activation_smoke.py`.
- Updated `policies/direct_waypoint_policy.py` to normalize observation-space lookup for ordinary mapping-like spaces and real `gymnasium.spaces.Dict` instances.
- Added the public launcher hook `register_heatmap_task_element_providers(env_config: dict) -> None` to `scripts/train_mappo_waypoint.py`.
- Integrated the registration hook after CLI task selection and configuration synchronization and before environment construction.
- Default YAML files, the trainer, environment, vector environment, rollout buffer, providers, candidate policy, evaluation, rewards, and scenarios were not changed.

### Test-first findings and production fixes

- Initial IRMV execution produced 2 passed and 2 failed tests in 3.04s.
- One expected failure raised `task_element_heatmap must be present in observation_space when enabled` even though the real `gymnasium.spaces.Dict` visibly contained `task_element_heatmap: Box(..., (1,8,8), float32)`.
- The old lookup treated the Gymnasium Dict space object like an ordinary mapping. The production fix uses the public `observation_space.spaces` mapping for Gymnasium Dict and the original object for ordinary mapping-like test spaces; no private Gymnasium attributes are used.
- The normalized lookup supplies task-field or global-task-field, heatmap, UAV-state, task-ID, global-info, and UVFA goal-space dimensions.
- The other expected failure showed that `scripts.train_mappo_waypoint` did not expose `register_heatmap_task_element_providers`; this was the intended missing launcher activation hook.

### Launcher registration contract

- `register_heatmap_task_element_providers` returns without registration unless both `heatmap_observation.enabled` and `heatmap_observation.provider.enabled` are true; step-time refresh is not required for registration.
- The hook examines both `task_name` and `task_names` and registers the real `priority_inspection` provider only when that task is active.
- Registration calls `register_priority_inspection_task_element_provider()` and remains explicit and idempotent through the existing task-specific registration contract.
- Importing the launcher or provider module has no registration side effect.
- Heatmap-disabled, provider-disabled, and unrelated-task launches remain unchanged.
- Sync and thread backends use the parent-process registry. Fork-based process workers inherit parent registration, while spawn-based workers may require a future worker-side registration hook; spawn support was not verified in this increment.

### Smoke configuration and production path

- The test-local environment configuration selects `priority_inspection`, enables a one-channel heatmap, provider resolution, and `refresh_on_step`, disables domain randomization, and uses deterministic seed 83.
- The smoke setup uses two UAVs, grid size 8, `max_steps: 16`, four POIs, the sync vector backend, and one environment.
- The compact direct policy enables `use_task_element_heatmap`, uses small CNN and hidden dimensions, and disables unrelated optional encoders where practical.
- Training uses rollout length 2, one PPO epoch, minibatch size 2, CPU-compatible execution, no checkpoint resume, and no evaluation loop.
- The exercised production path is `register_priority_inspection_task_element_provider` → `make_waypoint_env_batch` → real `WaypointMultiUAVEnv` → `build_policy` → real `MAPPOWaypointTrainer` → `collect_rollout` → real `WaypointRolloutBuffer` → `trainer.update` → real `optimizer.step`.

### Verified enabled behavior

- The initial vector observation contains finite `float32` `task_element_heatmap` values with batched shape `[1,1,8,8]`, and the policy observation space declares `[1,8,8]`.
- `DirectWaypointPolicy.use_task_element_heatmap` is true and `heatmap_encoder` exists.
- Actor output remains `[1,2,2]`, and critic value remains `[1]`.
- Real rollout collection succeeds and stores finite heatmaps with shape `[2,1,1,8,8]`.
- One real `trainer.update` succeeds with finite `policy_loss`, `value_loss`, `entropy`, and `grad_norm`.
- At least one `heatmap_encoder` parameter changes after `optimizer.step`; where gradients remain observable, they are finite and non-`None`.

### Verified disabled and fallback behavior

- With policy heatmap consumption disabled, the environment may still emit and store the heatmap, but no `heatmap_encoder` exists and the existing architecture remains valid.
- The disabled policy completes the same real rollout and one optimizer update successfully.
- Without provider registration, configured static task elements remain the heatmap source. Provider absence was not made an error, and resolver semantics remain unchanged.

### Validation

- Local `git diff --check` and dependency-free AST parsing passed.
- Local PyTorch tests were blocked during collection because `torch` was not installed; this was a local dependency limitation, not a production failure.
- No dependencies were installed locally, and `PYTHONPATH` was not changed. IRMV Docker validation is the authoritative runtime result.
- The IRMV server did not pull from GitHub directly. Equivalent patches were transferred from the Mac and applied with `git am`; no matching commit hashes or successful server-side GitHub pull is claimed.
- The Mac/GitHub commits were `3d083b7 test: specify heatmap training activation smoke` and `d524ae3 feat: activate heatmap provider training path`.
- Applying equivalent patch content on the IRMV server created `8ce57cc test: specify heatmap training activation smoke` and `378f516 feat: activate heatmap provider training path`.
- IRMV Docker validation used container `pjs_wayffusion`, set `CUDA_VISIBLE_DEVICES=0`, and ran tests with the numeric `pjs` UID/GID.
- Training-activation smoke validation passed: 4 tests in 3.81s.
- Focused policy/trainer regression passed: 31 tests in 5.03s. It covered direct-policy heatmap consumption, existing direct-policy tests, MAPPO waypoint trainer tests, UVFA goal conditioning, and candidate-selection policy tests.
- Full repository regression passed: 204 tests in 9.15s.

### Compatibility and ownership

- Heatmap-disabled and provider-disabled launches, unrelated tasks, static fallback, existing policy outputs, and trainer behavior remain compatible.
- Default YAML files were not enabled or modified, and `CandidateSelectionWaypointPolicy` remains unchanged.
- The environment owns semantic heatmap generation and refresh; the provider registry owns task-specific source resolution; and the launcher owns explicit provider activation before environment creation.
- The vector environment owns observation batching, the rollout buffer and trainer preserve the heatmap tensor, `DirectWaypointPolicy` owns encoding and actor/critic fusion, and the optimizer updates the encoder through the normal PPO objective.

### SSH workflow note

- A dedicated Ed25519 key was created for the IRMV server, and public-key authentication was verified with password authentication disabled.
- The Mac SSH alias is `irmv-wayffusion`; future transfers can use `ssh irmv-wayffusion` and `scp <file> irmv-wayffusion:/data0/pjs/`.
- No account password is recorded in this document.

### Known limitations

- Default YAML files still do not enable semantic heatmap training, and no checked-in experimental heatmap-enabled training configuration exists.
- No long training run, heatmap-on versus heatmap-off comparison, multi-seed evaluation, ablation, or statistical-significance analysis has been completed; this increment verifies integration, not performance benefit.
- Spawn-based process-worker provider registration is not verified.
- `CandidateSelectionWaypointPolicy` still ignores the heatmap.
- The actor consumes centralized global state rather than a strictly decentralized local observation.
- Enabled checkpoints are not strict-load compatible with older disabled architectures.

### Next step

- Define checked-in, explicit experimental configurations for a short deterministic heatmap-enabled training sanity run and an otherwise identical heatmap-disabled control.
- Keep default configurations unchanged and verify that both experimental configurations launch and complete a very short training run.
- Do not begin long training, multi-seed evaluation, or performance claims in the same increment.

### Related commit

- Test-first specification: `3d083b7 test: specify heatmap training activation smoke`.
- Production implementation: `d524ae3 feat: activate heatmap provider training path`.
- Equivalent IRMV-applied test commit: `8ce57cc test: specify heatmap training activation smoke`.
- Equivalent IRMV-applied production commit: `378f516 feat: activate heatmap provider training path`.

## 2026-07-23 — Paired heatmap debug configurations and sanity-run artifact audit

### Objective

- Add explicit paired short-debug experiment configurations for `priority_inspection`.
- Keep training, environment, and policy settings identical except for semantic heatmap activation.
- Verify that both configurations launch through the real combined-config training path and complete two PPO updates.
- Audit the generated snapshots, metrics, and checkpoints without claiming a performance benefit from the short runs.

### Test-first specification

- Added `tests/test_heatmap_experiment_config_contract.py`.
- The Mac/GitHub test commit was `c3f124b test: specify paired heatmap experiment configs`.
- Equivalent patch content was applied on IRMV as `bdc99b3 test: specify paired heatmap experiment configs`.
- Initial local validation produced 1 failed, 1 passed, and 4 skipped tests in 0.05s.
- Initial IRMV validation produced 1 failed, 1 passed, and 4 skipped tests in 0.07s.
- The expected failure identified the missing `configs/experiments/priority_inspection_heatmap_on_debug.yaml` and `configs/experiments/priority_inspection_heatmap_off_debug.yaml` files.
- The default-config protection test passed; the skipped tests depended on the missing experiment files.

### Changes

- Created `configs/experiments/priority_inspection_heatmap_on_debug.yaml`.
- Created `configs/experiments/priority_inspection_heatmap_off_debug.yaml`.
- Updated `scripts/train_mappo_waypoint.py` with combined experiment loading and optional `--experiment-config` support.
- The Mac/GitHub implementation commit was `709e60e feat: add paired heatmap debug experiments`.
- Equivalent patch content was applied on IRMV as `6f14cfe feat: add paired heatmap debug experiments`.
- The IRMV server could not pull directly from GitHub; equivalent patches were transferred and applied with `git am`, and no successful server-side GitHub pull is claimed.

### Combined experiment loader and CLI

- Added public `load_experiment_config(path: str | Path) -> tuple[dict, dict, dict]`.
- The loader requires top-level `experiment_name`, `seed`, `environment`, `policy`, and `training` fields and rejects malformed mappings or missing fields with clear errors.
- Parsed data is deep-copied. `environment` becomes `env_config`; `policy` and `training` are merged into `train_config`; and the top-level seed is injected into both.
- `env_backend`, `init_checkpoint`, and `evaluation_enabled` are extracted into `launch_config`.
- Combined mode cannot be explicitly mixed with `--config` or `--env-config` and reuses the existing training path.
- Configured task names, environment backend, and initialization checkpoint are used unless explicitly overridden.
- With `evaluation_enabled: false`, evaluation and recording episode counts are set to zero and `trainer.evaluate` is replaced with a no-op result.
- Existing separate-config behavior remains unchanged.

### Paired configuration contract

- Both configurations use deterministic seed 0, only `priority_inspection`, two agents, grid size 8, map size 1.0, `max_steps: 16`, `max_waypoint_distance: 0.20`, disabled domain randomization, four POIs, and `num_pois_range: null`.
- Both use `DirectWaypointPolicy` with CNN channels `[8,16,32]`, agent hidden dimension 32, joint hidden dimension 64, and attention and unrelated optional encoders disabled.
- Both use the sync vector backend, one environment, four rollout steps, two total updates, one epoch, minibatch size 4, learning rate 0.0003, disabled reward normalization, enabled advantage normalization, a constant learning-rate schedule, disabled evaluation, and no initialization checkpoint.
- Heatmap ON sets `environment.heatmap_observation.enabled`, `provider.enabled`, `provider.refresh_on_step`, and `policy.use_task_element_heatmap` to `true`, with one heatmap channel.
- Heatmap OFF sets the same four activation fields to `false`, while retaining one declared heatmap channel.
- Default environment and policy configurations were not modified and remain heatmap-disabled.

### Configuration and regression validation

- IRMV contract validation passed: 6 tests in 0.10s.
- Focused heatmap-training regression passed: 23 tests in 4.51s.
- Full repository regression passed: 210 tests in 9.32s.

### Sanity training and exact metrics

- Both runs used `scripts/train_mappo_waypoint.py --experiment-config ...` and completed exactly two updates.
- The output directories were `outputs/heatmap_off_debug_sanity` and `outputs/heatmap_on_debug_sanity`.
- Matplotlib could not create `/.config/matplotlib` and used a temporary `/tmp` cache directory. The warning did not interrupt training; future container commands can set `MPLCONFIGDIR=/tmp/matplotlib-pjs`, and no production-code change is required.
- Rounded OFF console rewards were approximately -0.009 at update 1 and -1.008 at update 2.
- Rounded ON console rewards were approximately -0.009 at update 1 and -1.008 at update 2; these rounded values are not asserted to be exactly equal.
- OFF exact `mean_rollout_reward` values were -0.009005821775645018 and -1.0083090085536242 for updates 1 and 2. OFF exact `value_loss` values were 0.0001763524196576327 and 5.3765764236450195. Weighted POI completion remained 0.0.
- ON exact `mean_rollout_reward` values were -0.008946843212470412 and -1.008230492239818 for updates 1 and 2. ON exact `value_loss` values were 0.0002475009532645345 and 5.401249885559082. Weighted POI completion remained 0.0.

### Artifact audit

- Each run produced `training_metrics.csv`, a TensorBoard event file, `snapshot/cli_args.yaml`, `snapshot/env_config.yaml`, `snapshot/eval_config.yaml`, `snapshot/metadata.json`, `snapshot/train_config.yaml`, `checkpoints/checkpoint_0002.pt`, `checkpoints/checkpoint_best_eval.pt`, and `best_eval_summary.json`.
- The OFF output directory was approximately 244 KB; `checkpoint_0002.pt` was 91,526 bytes and `checkpoint_best_eval.pt` was 91,671 bytes.
- The ON output directory was approximately 300 KB; `checkpoint_0002.pt` was 120,154 bytes and `checkpoint_best_eval.pt` was 120,319 bytes.
- The OFF checkpoint contained 23 state tensors and 20,453 total state elements, no heatmap-encoder tensors or elements, `use_task_element_heatmap: false`, and an effectively disabled heatmap.
- The ON checkpoint contained 27 state tensors and 27,229 total state elements, four heatmap-encoder tensors containing 632 elements, `use_task_element_heatmap: true`, and an effectively enabled heatmap.
- The final checkpoint size difference was 28,628 bytes and the state-element difference was 6,776. Of those additional state elements, 632 belong directly to the heatmap encoder; the remaining 6,144 are consistent with widened enabled-mode actor/critic fusion layers.
- Both effective environment snapshots identify only `priority_inspection`. OFF records heatmap disabled; ON records heatmap, provider, and refresh enabled. Both train snapshots correctly record the policy heatmap flag, and both metrics files contain exactly two update rows.
- The audit completed with `artifact_audit=PASS`.

### Performance interpretation

- The small numerical differences are not evidence of a heatmap benefit.
- The runs used only four rollout steps and two updates. They support no success, completion, convergence, generalization, or performance claim.
- This increment verifies launchability and artifact production only.

### Artifact provenance and known issues

- `snapshot/cli_args.yaml` records raw parser/default values rather than fully resolved launch values. It records `env_backend: process` even though the effective experiment backend was sync, and it records the parser's full default task list even though the effective environment snapshot contains only `priority_inspection`.
- Therefore, `snapshot/cli_args.yaml` is not authoritative for resolved execution configuration.
- Although evaluation was disabled and evaluation episodes were zero, both runs created `checkpoint_best_eval.pt` and `best_eval_summary.json`, recording zero evaluation reward and success rate. These artifacts are misleading when evaluation is disabled.
- Spawn-based process-worker provider registration remains unverified.
- The authoritative artifacts for these runs are `snapshot/env_config.yaml`, `snapshot/train_config.yaml`, `training_metrics.csv`, `checkpoint_0002.pt`, and `snapshot/metadata.json`.

### Ownership

- Combined experiment YAML owns explicit experiment intent.
- `load_experiment_config` owns experiment splitting and seed propagation.
- The launcher owns resolved task, backend, checkpoint, and evaluation selection.
- Effective environment and training snapshots own final environment and policy/trainer configurations.
- Checkpoints own model-state persistence, and `training_metrics.csv` owns per-update training metrics.

### Next step

- Add a test-first artifact-semantics contract so disabled evaluation neither runs evaluation logic nor creates best-evaluation checkpoint or summary artifacts.
- Add a resolved launch snapshot separate from raw CLI arguments, or clarify snapshot semantics so effective backend, task names, checkpoint source, and evaluation activation are recorded unambiguously while preserving raw CLI arguments separately if useful.
- Complete this before controlled pilot training; do not begin long training or make performance claims in the same increment.

### Related commit

- Test-first specification: `c3f124b test: specify paired heatmap experiment configs`.
- Production implementation: `709e60e feat: add paired heatmap debug experiments`.
- Equivalent IRMV-applied test commit: `bdc99b3 test: specify paired heatmap experiment configs`.
- Equivalent IRMV-applied production commit: `6f14cfe feat: add paired heatmap debug experiments`.

## 2026-07-23 — Resolved launch provenance and evaluation-disabled artifact semantics

### Objective

- Separate raw parsed CLI provenance from effective resolved launch provenance.
- Prevent evaluation-disabled training runs from invoking evaluation logic or producing misleading best-evaluation artifacts.
- Preserve ordinary checkpoint and metrics generation.
- Verify the corrected behavior through paired heatmap OFF and ON sanity runs.

### Test-first specification

- Added `tests/test_training_artifact_semantics.py`.
- The Mac/GitHub test commit was `3ca2faa test: specify training artifact semantics`.
- Equivalent patch content was applied on IRMV as `1d4165b test: specify training artifact semantics`.
- Initial local validation produced 3 failed and 5 skipped tests in 0.08s; initial IRMV validation produced 3 failed and 5 skipped tests in 0.15s.
- Existing experiment-config compatibility remained intact with 6 passing tests locally and 6 passing tests on IRMV.
- The expected failures showed that `scripts.train_mappo_waypoint` did not expose `build_resolved_launch_snapshot`, the launcher did not write `snapshot/resolved_launch.yaml`, and `MAPPOWaypointTrainer.train` did not receive `evaluation_enabled` independently from `eval_episodes`.
- The missing contracts also included module-level `should_run_evaluation` and `should_write_best_eval_artifacts` helpers in `algorithms.mappo_waypoint`.

### Changes

- Modified `scripts/train_mappo_waypoint.py` and `algorithms/mappo_waypoint.py`.
- The Mac/GitHub implementation commit was `0476718 fix: clarify training artifact semantics`.
- Equivalent patch content was applied on IRMV as `83f8d99 fix: clarify training artifact semantics`.

### Resolved launch provenance

- Added public `build_resolved_launch_snapshot(...)`.
- `snapshot/cli_args.yaml` remains the raw parsed CLI argument record.
- Added `snapshot/resolved_launch.yaml` as the record of effective launcher decisions after combined-config loading and CLI override resolution.
- The resolved snapshot records `experiment_config`, `config`, `env_config`, `eval_config`, `task_names`, `env_backend`, `init_checkpoint`, `evaluation_enabled`, `training_seed`, `output_dir`, `run_name`, `num_envs`, `rollout_steps`, and `total_updates`.
- Path-like values are serialized as strings. An absent experiment configuration or initialization checkpoint remains `null`.

### Evaluation and checkpoint semantics

- Added public `should_run_evaluation(...)`. It returns `false` when evaluation is disabled or `eval_episodes` is zero or negative; otherwise it preserves the existing interval and final-update schedule.
- Added public `should_write_best_eval_artifacts(...)`. It returns `false` when evaluation is disabled or no real evaluation result exists, while permitting the existing numerical best-score comparison when enabled evaluation returns a result.
- `MAPPOWaypointTrainer.train` now accepts `evaluation_enabled: bool = True`; the default preserves compatibility for existing callers.
- `self.evaluate` is not called when evaluation is disabled, and the previous launcher-side `trainer.evaluate` lambda replacement was removed.
- Ordinary scheduled and final checkpoint creation is independent of evaluation. Ordinary checkpoint contents and `training_metrics.csv` behavior remain unchanged.
- Evaluation-enabled runs retain existing best-checkpoint behavior.

### Evaluation-disabled artifact contract

- Verified creation of `training_metrics.csv`, `checkpoints/checkpoint_0002.pt`, `snapshot/cli_args.yaml`, `snapshot/resolved_launch.yaml`, `snapshot/env_config.yaml`, `snapshot/train_config.yaml`, `snapshot/eval_config.yaml`, `snapshot/metadata.json`, and a TensorBoard event file.
- Verified absence of `checkpoints/checkpoint_best_eval.pt` and `best_eval_summary.json`.

### IRMV validation

- Artifact-semantics contract passed: 8 tests in 0.12s.
- Experiment-config compatibility passed: 6 tests in 0.10s.
- Focused trainer and heatmap regression passed: 23 tests in 4.53s.
- Full repository regression passed: 218 tests in 9.73s.
- Validation used container `pjs_wayffusion`, `CUDA_VISIBLE_DEVICES=0`, `MPLCONFIGDIR=/tmp/matplotlib-pjs`, and the numeric `pjs` UID/GID.

### Paired runtime validation

- The OFF run wrote to `outputs/heatmap_off_debug_semantics_v2`; `checkpoint_0002.pt` was 91,526 bytes, `resolved_launch.yaml` was 501 bytes, and `training_metrics.csv` was 6,078 bytes.
- The ON run wrote to `outputs/heatmap_on_debug_semantics_v2`; `checkpoint_0002.pt` was 120,154 bytes, `resolved_launch.yaml` was 499 bytes, and `training_metrics.csv` was 6,081 bytes.
- Both resolved launch snapshots recorded only `priority_inspection`, sync backend, `init_checkpoint: null`, `evaluation_enabled: false`, training seed 0, one environment, four rollout steps, and two total updates.
- Both runs completed exactly two PPO updates, produced `checkpoint_0002.pt`, `training_metrics.csv`, and raw and resolved snapshots, and produced neither `checkpoint_best_eval.pt` nor `best_eval_summary.json`.
- OFF completed with `artifact_semantics=PASS`; ON completed with `artifact_semantics=PASS`; the final paired result was `paired_artifact_semantics=PASS`.

### Console logging note

- Console output still prints `eval_success=0.000` and `eval_reward=0.000` when evaluation is disabled.
- These are logging fallback values, not evidence that evaluation ran; no best-evaluation artifacts were generated.
- This cosmetic ambiguity does not invalidate the artifact-semantics contract and may be clarified later, but it is not required before pilot training.

### Compatibility

- Combined experiment-config mode and separate environment/policy config mode remain supported.
- Raw `cli_args.yaml` and existing environment, training, evaluation, and metadata snapshots remain available.
- Existing checkpoint contents, default YAML files, and heatmap ON/OFF experiment values remain unchanged.
- No performance metric meaning was changed.

### Interpretation

- This increment verifies provenance and artifact correctness; it does not demonstrate a heatmap performance benefit.
- The two-update runs remain launch and artifact sanity checks only.

### Known limitations

- No controlled medium-duration pilot training, multi-seed comparison, or statistical analysis has been completed.
- Spawn-based process-worker provider registration remains unverified.
- `CandidateSelectionWaypointPolicy` still ignores heatmaps, and the actor still consumes centralized global state.
- Console evaluation fields remain visually ambiguous when evaluation is disabled, although generated artifacts are now correct.

### Next step

- Define a controlled medium-duration paired pilot experiment with identical ON/OFF settings except for heatmap activation and with explicit evaluation settings.
- Validate one deterministic seed and runtime duration before expanding to multiple deterministic seeds.
- Define primary metrics before execution: weighted POI completion, mean rollout reward, goal achievement, arrival rate, evaluation success rate, and evaluation reward.
- Keep the first pilot limited enough to detect configuration, runtime, reward, or stability problems before multi-seed experiments, and do not make performance claims from a single pilot seed.

### Related commit

- Test-first specification: `3ca2faa test: specify training artifact semantics`.
- Production implementation: `0476718 fix: clarify training artifact semantics`.
- Equivalent IRMV-applied test commit: `1d4165b test: specify training artifact semantics`.
- Equivalent IRMV-applied production commit: `83f8d99 fix: clarify training artifact semantics`.
