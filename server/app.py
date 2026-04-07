"""
FastAPI server for the IRCTC Dynamic Train Routing environment.
Uses OpenEnv's create_fastapi_app for spec compliance (schema, health, /ws, /mcp).
"""

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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)
