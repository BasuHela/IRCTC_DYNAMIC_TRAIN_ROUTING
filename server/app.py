"""
FastAPI server for the IRCTC Dynamic Train Routing environment.
Uses OpenEnv's create_fastapi_app for spec compliance (schema, health, /ws, /mcp).
Tasks 1/2/3 are defined in openenv.yaml; the autograder selects them via reset(task_id=N).
"""

import uvicorn
from openenv.core.env_server import create_fastapi_app
from server.models import Action, Observation
from server.environment import IRCTCEnvironment

# ── OpenEnv standard app ──
app = create_fastapi_app(
    IRCTCEnvironment,
    Action,
    Observation,
)

def main():
    """Entry point for the OpenEnv multi-mode deployment."""
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
