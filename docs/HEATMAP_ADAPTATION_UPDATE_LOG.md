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
