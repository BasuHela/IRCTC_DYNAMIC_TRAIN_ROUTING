# IRCTC Dynamic Train Routing — OpenEnv Environment

A multi-turn reinforcement learning environment simulating Indian Railways ticket booking. The agent must navigate Waitlisted (WL) constraints, manage a budget, handle timing conflicts, and reason about multi-hop split journeys across 3 difficulty tiers.

## Motivation

Train booking with waitlist constraints is a problem 1.4 billion Indians understand. The environment captures real decision-making challenges: should you take a risky waitlisted direct train, or spend more time finding a confirmed multi-leg route? This creates a rich optimization problem spanning cost, availability, and temporal feasibility.

## Action Space

All agent actions are JSON objects with a `command` field and optional parameters:

| Command              | Required Fields          | Description                              |
|----------------------|--------------------------|------------------------------------------|
| `search_trains`      | `source_stn`, `dest_stn` | Search for trains between two stations   |
| `check_availability` | `train_no`               | Check detailed status of a specific train|
| `book_ticket`        | `train_no`               | Book a ticket (deducts from wallet)      |
| `finish`             | —                        | End the episode and compute final reward |

**Example action:**
```json
{"command": "search_trains", "source_stn": "DEL", "dest_stn": "BOM"}
```

## Observation Space

After each action, the agent receives an `Observation` (extends `openenv.core` base) with:

| Field                | Type           | Description                                        |
|----------------------|----------------|----------------------------------------------------|
| `done`               | `bool`         | Whether the episode has ended (inherited)          |
| `reward`             | `float\|None`  | Step/final reward (inherited)                      |
| `message`            | `str`          | Human-readable status message                      |
| `search_results`     | `List[Dict]`   | Train results (if search was performed)            |
| `availability_status`| `str \| null`  | "CNF" or "WL" (if availability was checked)        |
| `wl_probability`     | `float \| null`| WL confirmation probability (0.0–1.0)              |
| `wallet_balance`     | `float`        | Remaining budget                                   |
| `booked_itinerary`   | `List[Dict]`   | All booked legs so far                             |
| `current_location`   | `str`          | Agent's current station                            |

## Tasks

### Task 1: Direct Confirmed Booking (Easy)
- **Route:** DEL → KOTA
- **Budget:** ₹3,000
- **Challenge:** A direct confirmed train exists. Agent must search, find it, and book.
- **Expected score:** 0.85–1.0

### Task 2: WL Avoidance via Split Journey (Medium)
- **Route:** DEL → BOM
- **Budget:** ₹3,000
- **Challenge:** Direct train is Waitlisted (~15% confirm). Agent must discover a 2-leg confirmed split via KOTA.
- **Expected score:** 0.5–0.8

### Task 3: Multi-Constraint Optimization (Hard)
- **Route:** DEL → BOM
- **Budget:** ₹2,200 (tight)
- **Challenge:** All direct trains are WL. Multiple splits exist but some exceed budget, have timing conflicts, or include WL legs. Only 1–2 valid paths per episode.
- **Expected score:** 0.2–0.5

## Reward Function

Continuous reward with partial progress signals:

```
reward = exploration + destination + budget_eff + confirmation − penalties
```

| Component        | Formula                                    | Max   |
|------------------|--------------------------------------------|-------|
| Exploration      | `0.05 × min(unique_searches, 4)`           | +0.20 |
| Destination      | `0.30 if reached destination`              | +0.30 |
| Budget Eff.      | `0.25 × (budget − spent) / budget`         | +0.25 |
| Confirmation     | `0.25 × avg(cnf_score per leg)`            | +0.25 |
| Invalid action   | `−0.10 per invalid station/train`          | —     |
| Time violation   | `−0.20 per timing conflict`                | —     |
| Duplicate search | `−0.05 per redundant search`               | —     |

Final reward is clamped to `[0.0, 1.0]`.

## Setup & Usage

### Prerequisites
- Python 3.10+
- Docker (for containerized deployment)

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Start the environment server
python -m server.app
# OR
uvicorn server.app:app --host 0.0.0.0 --port 7860

# In another terminal, run the inference agent
HF_TOKEN=your_token python inference.py
```

### Using the Client

```python
from client import IRCTCEnv
from server.models import Action

# Sync usage
with IRCTCEnv(base_url="https://your-space.hf.space").sync() as env:
    obs = env.reset(task_id=1)
    obs = env.step(Action(command="search_trains", source_stn="DEL", dest_stn="KOTA"))
    print(obs.message, obs.reward, obs.done)
```

### Docker

```bash
docker build -t irctc-router .
docker run -p 7860:7860 -e HF_TOKEN=your_token irctc-router
```

### Environment Variables

| Variable       | Description                        | Default                                              |
|----------------|------------------------------------|------------------------------------------------------|
| `HF_TOKEN`     | Hugging Face / API key             | —                                                    |
| `API_BASE_URL` | LLM API endpoint                   | `https://api-inference.huggingface.co/v1/`           |
| `MODEL_NAME`   | Model identifier for inference     | `meta-llama/Meta-Llama-3-70B-Instruct`               |

## Baseline Scores

With `Meta-Llama-3-70B-Instruct`:

| Task | Difficulty | Expected Score |
|------|------------|----------------|
| 1    | Easy       | 0.85 – 1.0    |
| 2    | Medium     | 0.50 – 0.80   |
| 3    | Hard       | 0.20 – 0.50   |

## Repository Structure

```
irctc_router/
├── Dockerfile          # At root (NOT in server/)
├── openenv.yaml        # Environment metadata
├── inference.py        # Baseline agent script
├── client.py           # WebSocket client (EnvClient)
├── README.md           # This file
├── pyproject.toml      # Python project config
├── requirements.txt    # Dependencies
└── server/
    ├── __init__.py
    ├── models.py       # Pydantic schemas (Action, Observation, State)
    ├── environment.py  # reset/step/state logic (Environment ABC)
    ├── app.py          # OpenEnv create_app server
    └── requirements.txt
```
