"""
WebSocket client for IRCTC Dynamic Train Routing environment.
Extends openenv EnvClient for persistent session management.
"""

from typing import Any, Dict

from openenv.core.env_client import EnvClient
from openenv.core.client_types import StepResult
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

    def _step_payload(self, action: Action) -> Dict[str, Any]:
        return action.model_dump(exclude_none=True)

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[Observation]:
        # Server wraps fields as {"observation": {...}, "reward": ..., "done": ...}
        obs_data = payload.get("observation", payload)
        obs = Observation(**{k: v for k, v in obs_data.items() if k in Observation.model_fields})
        return StepResult(
            observation=obs,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> State:
        return State(**{k: v for k, v in payload.items() if k in State.model_fields})
