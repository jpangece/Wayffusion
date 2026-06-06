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

The only supported dynamics backend is now `MPECoreBackend` with
`dynamics_backend.name: mpe_core`. It calls an actual MPE `World.step()`
implementation and keeps Wayffusion-specific task/reward/observation logic in
Wayffusion.

See `docs/mpe_core_backend_zh.md` for the current architecture.
