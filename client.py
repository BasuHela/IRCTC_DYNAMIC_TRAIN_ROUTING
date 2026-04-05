"""
WebSocket client for IRCTC Dynamic Train Routing environment.
Extends openenv EnvClient for persistent session management.
"""

from openenv.core.env_client import EnvClient
from server.models import Action, Observation, State


class IRCTCEnv(EnvClient[Action, Observation, State]):
    """
    Client for the IRCTC Dynamic Train Routing environment.

    Usage:
        # Async
        async with IRCTCEnv(base_url="https://your-space.hf.space") as env:
            obs = await env.reset(task_id=1)
            obs = await env.step(Action(command="search_trains", source_stn="DEL", dest_stn="KOTA"))

        # Sync
        with IRCTCEnv(base_url="https://your-space.hf.space").sync() as env:
            obs = env.reset(task_id=1)
            obs = env.step(Action(command="search_trains", source_stn="DEL", dest_stn="KOTA"))
    """

    @property
    def action_cls(self):
        return Action

    @property
    def observation_cls(self):
        return Observation

    @property
    def state_cls(self):
        return State
