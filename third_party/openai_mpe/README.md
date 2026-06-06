# Vendored OpenAI MPE Core

This directory vendors the upstream OpenAI Multi-Agent Particle Environment
`multiagent/core.py` file so Wayffusion can call a real MPE `World.step()`
implementation even when `mpe2` or `pettingzoo` are not installed locally.

Upstream:

- Repository: `https://github.com/openai/multiagent-particle-envs`
- Vendored file: `multiagent/core.py`
- Vendored as: `third_party/openai_mpe/core.py`
- Upstream README snapshot: `third_party/openai_mpe/UPSTREAM_README.md`

Wayffusion-specific adaptation lives outside this directory in
`envs/waypoint/control/dynamics_backends.py`. Keep this vendored core as close
to upstream as possible.
