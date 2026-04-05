"""
FastAPI server for the IRCTC Dynamic Train Routing environment.
Uses OpenEnv's create_fastapi_app for spec compliance (schema, health, /ws)
and adds stateful /session/* endpoints for direct multi-turn HTTP interaction.
"""

from typing import Optional
from pydantic import BaseModel

from openenv.core.env_server import create_fastapi_app
from server.models import Action, Observation, State
from server.environment import IRCTCEnvironment

# ── OpenEnv standard app ──
# Provides: /health, /schema, /metadata, /reset (stateless), /step (stateless),
#           /state, /ws (WebSocket for persistent sessions), /mcp
app = create_fastapi_app(
    IRCTCEnvironment,
    Action,
    Observation,
)

# ── Stateful session endpoints ──
# These maintain a single shared environment instance for sequential
# HTTP reset→step→step→...→finish interaction without WebSocket.

_session_env = IRCTCEnvironment()


class SessionResetRequest(BaseModel):
    task_id: int = 1
    seed: Optional[int] = None


class SessionStepResponse(BaseModel):
    observation: dict
    reward: Optional[float]
    done: bool


@app.post("/session/reset", response_model=dict, tags=["Session"])
def session_reset(req: SessionResetRequest):
    """Reset the shared session environment for a given task."""
    obs = _session_env.reset(seed=req.seed, task_id=req.task_id)
    return obs.model_dump()


@app.post("/session/step", response_model=SessionStepResponse, tags=["Session"])
def session_step(action: Action):
    """Step the shared session environment with an action."""
    obs = _session_env.step(action)
    return SessionStepResponse(
        observation=obs.model_dump(),
        reward=obs.reward,
        done=obs.done,
    )


@app.get("/session/state", response_model=dict, tags=["Session"])
def session_state():
    """Return the full internal state of the shared session environment."""
    return _session_env.state.model_dump()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
