# Deprecated Dynamics Note

The previous ActionAdapter + handwritten particle-dynamics refactor has been
removed from the runnable Wayffusion mainline.

Removed backend names:

- `mpe_particle`
- `mpe_like_particle`
- `kinematic_point`

Removed classes:

- `MPEParticleBackend`
- `KinematicPointBackend`

This note is historical. The handwritten MPE-like and kinematic backends remain
removed, but Wayffusion now supports three explicit backend modes:
`waypoint_behavior_fast`, `waypoint_behavior_realistic`, and `mpe_core`.
`mpe_core` calls an actual MPE `World.step()` implementation and keeps
Wayffusion-specific task/reward/observation logic in Wayffusion.

See `docs/dynamics_backend_modes_zh.md` for current backend choices, and
`docs/mpe_core_backend_zh.md` for the MPE core source details.
