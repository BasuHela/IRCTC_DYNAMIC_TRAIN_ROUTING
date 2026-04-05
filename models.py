"""
Pydantic models for the IRCTC Dynamic Train Routing environment.
Inherits from openenv.core base types for full framework compliance.
"""

from pydantic import Field
from typing import Literal, Optional, List, Dict
from openenv.core.env_server.types import (
    Action as BaseAction,
    Observation as BaseObservation,
    State as BaseState,
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ACTION SPACE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class Action(BaseAction):
    """Agent's action: search for trains, check availability, book a ticket, or finish."""
    command: Literal["search_trains", "check_availability", "book_ticket", "finish"] = Field(
        ..., description="The command to execute"
    )
    source_stn: Optional[str] = Field(
        default=None, description="Source station code (for search_trains)"
    )
    dest_stn: Optional[str] = Field(
        default=None, description="Destination station code (for search_trains)"
    )
    train_no: Optional[str] = Field(
        default=None, description="Train number (for check_availability / book_ticket)"
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# OBSERVATION SPACE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class Observation(BaseObservation):
    """
    What the agent sees after each step.
    Inherits `done: bool` and `reward: Optional[float]` from BaseObservation.
    """
    message: str = Field(
        default="", description="Human-readable status message"
    )
    search_results: Optional[List[Dict]] = Field(
        default=None, description="Train search results"
    )
    availability_status: Optional[str] = Field(
        default=None, description="CNF or WL status"
    )
    wl_probability: Optional[float] = Field(
        default=None, description="WL confirmation probability 0.0-1.0"
    )
    wallet_balance: float = Field(
        default=0.0, description="Remaining budget"
    )
    booked_itinerary: List[Dict] = Field(
        default_factory=list, description="All booked legs"
    )
    current_location: str = Field(
        default="", description="Agent's current station"
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# INTERNAL STATE (hidden from agent, used by grader)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class State(BaseState):
    """
    Full internal state — only visible to grading harness via state property.
    Inherits `episode_id: Optional[str]` and `step_count: int` from BaseState.
    """
    task_id: int = Field(default=1, description="Task 1, 2, or 3")
    target_source: str = Field(default="", description="Origin station")
    target_dest: str = Field(default="", description="Destination station")
    budget: float = Field(default=0.0, description="Total budget")
    wallet_balance: float = Field(default=0.0, description="Remaining balance")
    train_database: List[Dict] = Field(
        default_factory=list, description="Full synthetic timetable"
    )
    current_location: str = Field(default="", description="Current position")
    searches_made: int = Field(default=0)
    duplicate_searches: int = Field(default=0)
    invalid_actions: int = Field(default=0)
    time_violations: int = Field(default=0)
    bookings_made: List[Dict] = Field(default_factory=list)
    optimal_cost: float = Field(default=0.0, description="For reward normalization")
    seed: int = Field(default=42, description="Episode seed for reproducibility")
    search_history: List[str] = Field(default_factory=list)
    valid_stations: List[str] = Field(default_factory=list)
