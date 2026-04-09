"""
FastAPI server for the IRCTC Dynamic Train Routing environment.
Uses OpenEnv's create_fastapi_app for spec compliance (schema, health, /ws, /mcp).
"""

import uvicorn
from openenv.core.env_server import create_fastapi_app, Task
from server.models import Action, Observation, State
from server.environment import IRCTCEnvironment

# ── Independent Grader Logic ──
def irctc_grader(state: State) -> float:
    """Computes the final score based on the agent's final state for the OpenEnv harness."""
    # Exploration
    exploration = 0.05 * min(state.searches_made, 4)

    # Destination reached
    destination = 0.30 if state.current_location == state.target_dest else 0.0

    # Budget efficiency
    spent = state.budget - state.wallet_balance
    budget_eff = 0.25 * max(0, (state.budget - spent) / state.budget) if state.budget > 0 else 0.0

    # Confirmation quality
    if state.bookings_made:
        cnf_scores = []
        for b in state.bookings_made:
            if b["status"] == "CNF":
                cnf_scores.append(1.0)
            else:
                cnf_scores.append(b.get("wl_confirm_prob", 0.0))
        confirmation = 0.25 * (sum(cnf_scores) / len(cnf_scores))
    else:
        confirmation = 0.0

    # Penalties
    penalty = 0.0
    penalty += 0.10 * state.invalid_actions      
    penalty += 0.20 * state.time_violations       
    penalty += 0.05 * state.duplicate_searches    

    reward = exploration + destination + budget_eff + confirmation - penalty
    return round(max(0.0, min(1.0, reward)), 4)


# ── Register Tasks ──
# The hackathon requires at least 3 tasks to be registered with the grader.
tasks = [
    Task(id="1", grader=irctc_grader),
    Task(id="2", grader=irctc_grader),
    Task(id="3", grader=irctc_grader),
]

# ── OpenEnv standard app ──
app = create_fastapi_app(
    IRCTCEnvironment,
    Action,
    Observation,
    state_cls=State,
    tasks=tasks
)

def main():
    """Entry point for the OpenEnv multi-mode deployment."""
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
