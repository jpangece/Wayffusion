"""Vendored OpenAI Multi-Agent Particle Environment core.

Wayffusion imports these classes through a thin wrapper in
``envs.waypoint.control.dynamics_backends``. Do not modify this package for
Wayffusion-specific behavior.
"""

from third_party.openai_mpe.core import Action, Agent, AgentState, Entity, EntityState, Landmark, World

__all__ = ["Action", "Agent", "AgentState", "Entity", "EntityState", "Landmark", "World"]
