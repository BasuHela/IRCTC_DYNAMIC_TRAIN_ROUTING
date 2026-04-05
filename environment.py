"""
IRCTC Dynamic Train Routing Environment.
Extends openenv.core Environment ABC for full framework compliance.
Supports 3 tasks: Easy (direct booking), Medium (WL avoidance), Hard (multi-constraint).

Reflection fixes applied:
  - Exploration reward uses searches_made (not unique), per blueprint formula
  - Intermediate steps return reward=0.0; all penalties tracked in state,
    applied only in _compute_final_reward to avoid double-counting
  - WebSocket task selection via episode_id prefix: "task_<N>_..." or "task_<N>"
"""

import random
import copy
from uuid import uuid4
from typing import Optional, Any, Dict, List

from openenv.core.env_server.interfaces import Environment
from server.models import Action, Observation, State


class IRCTCEnvironment(Environment):
    """Multi-turn RL environment simulating Indian Railways ticket booking."""

    MAX_STEPS = 30

    def __init__(self):
        self._state: Optional[State] = None

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  TASK CONFIGURATIONS
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _build_task1(self, rng: random.Random) -> dict:
        """Task 1 (Easy): DEL → KOTA, direct confirmed train, generous budget."""
        trains = [
            {
                "train_no": "12955",
                "name": "Kota Jan Shatabdi",
                "source": "DEL",
                "dest": "KOTA",
                "price": round(800 * rng.uniform(0.9, 1.1)),
                "status": "CNF",
                "wl_confirm_prob": None,
                "depart_time": "06:00",
                "arrive_time": "11:30",
                "duration_hrs": 5.5,
            },
            {
                "train_no": "12956",
                "name": "Delhi Jaipur Express",
                "source": "DEL",
                "dest": "JP",
                "price": round(600 * rng.uniform(0.9, 1.1)),
                "status": "CNF",
                "wl_confirm_prob": None,
                "depart_time": "07:00",
                "arrive_time": "11:00",
                "duration_hrs": 4,
            },
            {
                "train_no": "12957",
                "name": "Jaipur Kota Express",
                "source": "JP",
                "dest": "KOTA",
                "price": round(800 * rng.uniform(0.9, 1.1)),
                "status": "CNF",
                "wl_confirm_prob": None,
                "depart_time": "12:00",
                "arrive_time": "15:30",
                "duration_hrs": 3.5,
            },
        ]
        return {
            "target_source": "DEL",
            "target_dest": "KOTA",
            "budget": 3000.0,
            "trains": trains,
            "optimal_cost": 800.0,
            "valid_stations": ["DEL", "JP", "KOTA"],
        }

    def _build_task2(self, rng: random.Random) -> dict:
        """Task 2 (Medium): DEL → BOM, direct is WL, split via KOTA is CNF."""
        wl_prob_direct = round(rng.uniform(0.10, 0.25), 2)
        trains = [
            {
                "train_no": "12951",
                "name": "Mumbai Rajdhani",
                "source": "DEL",
                "dest": "BOM",
                "price": round(1500 * rng.uniform(0.9, 1.1)),
                "status": "WL",
                "wl_confirm_prob": wl_prob_direct,
                "depart_time": "16:00",
                "arrive_time": "08:00+1",
                "duration_hrs": 16,
            },
            {
                "train_no": "12952",
                "name": "Delhi Kota Express",
                "source": "DEL",
                "dest": "KOTA",
                "price": round(800 * rng.uniform(0.9, 1.1)),
                "status": "CNF",
                "wl_confirm_prob": None,
                "depart_time": "06:00",
                "arrive_time": "11:30",
                "duration_hrs": 5.5,
            },
            {
                "train_no": "12953",
                "name": "Kota Mumbai SF",
                "source": "KOTA",
                "dest": "BOM",
                "price": round(1100 * rng.uniform(0.9, 1.1)),
                "status": "CNF",
                "wl_confirm_prob": None,
                "depart_time": "13:00",
                "arrive_time": "23:00",
                "duration_hrs": 10,
            },
            {
                "train_no": "12954",
                "name": "Delhi Surat Express",
                "source": "DEL",
                "dest": "ST",
                "price": round(1800 * rng.uniform(0.9, 1.1)),
                "status": "CNF",
                "wl_confirm_prob": None,
                "depart_time": "08:00",
                "arrive_time": "22:00",
                "duration_hrs": 14,
            },
            {
                "train_no": "12958",
                "name": "Surat Mumbai Express",
                "source": "ST",
                "dest": "BOM",
                "price": round(1400 * rng.uniform(0.9, 1.1)),
                "status": "CNF",
                "wl_confirm_prob": None,
                "depart_time": "23:30",
                "arrive_time": "05:00+1",
                "duration_hrs": 5.5,
            },
            {
                "train_no": "12959",
                "name": "Delhi Vadodara Exp",
                "source": "DEL",
                "dest": "BRC",
                "price": round(1200 * rng.uniform(0.9, 1.1)),
                "status": "CNF",
                "wl_confirm_prob": None,
                "depart_time": "09:00",
                "arrive_time": "21:00",
                "duration_hrs": 12,
            },
            {
                "train_no": "12960",
                "name": "Vadodara Mumbai Exp",
                "source": "BRC",
                "dest": "BOM",
                "price": round(900 * rng.uniform(0.9, 1.1)),
                "status": "CNF",
                "wl_confirm_prob": None,
                "depart_time": "22:30",
                "arrive_time": "04:30+1",
                "duration_hrs": 6,
            },
        ]
        return {
            "target_source": "DEL",
            "target_dest": "BOM",
            "budget": 3000.0,
            "trains": trains,
            "optimal_cost": 1900.0,
            "valid_stations": ["DEL", "KOTA", "ST", "BRC", "BOM"],
        }

    def _build_task3(self, rng: random.Random) -> dict:
        """Task 3 (Hard): DEL → BOM, tight budget, WL + timing constraints."""
        trains = [
            {
                "train_no": "13001",
                "name": "Rajdhani Express",
                "source": "DEL",
                "dest": "BOM",
                "price": round(1800 * rng.uniform(0.9, 1.1)),
                "status": "WL",
                "wl_confirm_prob": round(rng.uniform(0.10, 0.20), 2),
                "depart_time": "16:00",
                "arrive_time": "08:00+1",
                "duration_hrs": 16,
            },
            {
                "train_no": "13002",
                "name": "Delhi Jaipur SF",
                "source": "DEL",
                "dest": "JP",
                "price": round(400 * rng.uniform(0.9, 1.1)),
                "status": "CNF",
                "wl_confirm_prob": None,
                "depart_time": "06:00",
                "arrive_time": "10:00",
                "duration_hrs": 4,
            },
            {
                "train_no": "13003",
                "name": "Jaipur Kota Exp",
                "source": "JP",
                "dest": "KOTA",
                "price": round(350 * rng.uniform(0.9, 1.1)),
                "status": "CNF",
                "wl_confirm_prob": None,
                "depart_time": "11:30",
                "arrive_time": "15:00",
                "duration_hrs": 3.5,
            },
            {
                "train_no": "13004",
                "name": "Kota Mumbai SF",
                "source": "KOTA",
                "dest": "BOM",
                "price": round(1100 * rng.uniform(0.9, 1.1)),
                "status": "CNF",
                "wl_confirm_prob": None,
                "depart_time": "16:30",
                "arrive_time": "04:00+1",
                "duration_hrs": 11.5,
            },
            {
                "train_no": "13005",
                "name": "Delhi Ahmedabad Mail",
                "source": "DEL",
                "dest": "ADI",
                "price": round(900 * rng.uniform(0.9, 1.1)),
                "status": "WL",
                "wl_confirm_prob": round(rng.uniform(0.15, 0.30), 2),
                "depart_time": "20:00",
                "arrive_time": "08:00+1",
                "duration_hrs": 12,
            },
            {
                "train_no": "13006",
                "name": "Ahmedabad Mumbai Shatabdi",
                "source": "ADI",
                "dest": "BOM",
                "price": round(700 * rng.uniform(0.9, 1.1)),
                "status": "CNF",
                "wl_confirm_prob": None,
                "depart_time": "09:30",
                "arrive_time": "16:00",
                "duration_hrs": 6.5,
            },
            {
                "train_no": "13007",
                "name": "Delhi Kota Janshatabdi",
                "source": "DEL",
                "dest": "KOTA",
                "price": round(700 * rng.uniform(0.9, 1.1)),
                "status": "CNF",
                "wl_confirm_prob": None,
                "depart_time": "07:00",
                "arrive_time": "12:30",
                "duration_hrs": 5.5,
            },
            {
                "train_no": "13008",
                "name": "Kota Surat Exp",
                "source": "KOTA",
                "dest": "ST",
                "price": round(600 * rng.uniform(0.9, 1.1)),
                "status": "CNF",
                "wl_confirm_prob": None,
                "depart_time": "14:00",
                "arrive_time": "22:00",
                "duration_hrs": 8,
            },
            {
                "train_no": "13009",
                "name": "Surat Mumbai WL",
                "source": "ST",
                "dest": "BOM",
                "price": round(500 * rng.uniform(0.9, 1.1)),
                "status": "WL",
                "wl_confirm_prob": round(rng.uniform(0.10, 0.20), 2),
                "depart_time": "23:30",
                "arrive_time": "04:30+1",
                "duration_hrs": 5,
            },
            {
                "train_no": "13010",
                "name": "Jaipur Mumbai Superfast",
                "source": "JP",
                "dest": "BOM",
                "price": round(1600 * rng.uniform(0.9, 1.1)),
                "status": "CNF",
                "wl_confirm_prob": None,
                "depart_time": "12:00",
                "arrive_time": "06:00+1",
                "duration_hrs": 18,
            },
            {
                "train_no": "13011",
                "name": "Vadodara Mumbai Exp",
                "source": "BRC",
                "dest": "BOM",
                "price": round(500 * rng.uniform(0.9, 1.1)),
                "status": "CNF",
                "wl_confirm_prob": None,
                "depart_time": "10:00",
                "arrive_time": "16:00",
                "duration_hrs": 6,
            },
            {
                "train_no": "13012",
                "name": "Panvel Mumbai Local",
                "source": "PNVL",
                "dest": "BOM",
                "price": round(100 * rng.uniform(0.9, 1.1)),
                "status": "CNF",
                "wl_confirm_prob": None,
                "depart_time": "07:00",
                "arrive_time": "08:30",
                "duration_hrs": 1.5,
            },
        ]
        return {
            "target_source": "DEL",
            "target_dest": "BOM",
            "budget": 2200.0,
            "trains": trains,
            "optimal_cost": 1850.0,
            "valid_stations": ["DEL", "JP", "KOTA", "ADI", "ST", "BRC", "PNVL", "BOM"],
        }

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  TASK ID RESOLUTION
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    @staticmethod
    def _resolve_task_id(
        episode_id: Optional[str] = None,
        task_id_kwarg: Optional[int] = None,
    ) -> int:
        """
        Determine task_id from kwargs or episode_id prefix.
        Priority: explicit kwarg > episode_id prefix > default (1).
        Episode ID format: "task_<N>" or "task_<N>_<uuid>" where N = 1, 2, or 3.
        """
        if task_id_kwarg is not None:
            return task_id_kwarg
        if episode_id and episode_id.startswith("task_"):
            try:
                tid = int(episode_id.split("_")[1])
                if tid in (1, 2, 3):
                    return tid
            except (IndexError, ValueError):
                pass
        return 1  # default to easy

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  OPENENV API: reset()
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> Observation:
        """Initialize environment for the given task. Returns opening observation."""
        task_id = self._resolve_task_id(episode_id, kwargs.get("task_id"))
        if task_id not in (1, 2, 3):
            raise ValueError(f"Invalid task_id: {task_id}. Must be 1, 2, or 3.")

        ep_seed = seed if seed is not None else random.randint(0, 2**31)
        ep_id = episode_id or str(uuid4())
        rng = random.Random(ep_seed)

        builders = {1: self._build_task1, 2: self._build_task2, 3: self._build_task3}
        config = builders[task_id](rng)

        self._state = State(
            episode_id=ep_id,
            step_count=0,
            task_id=task_id,
            target_source=config["target_source"],
            target_dest=config["target_dest"],
            budget=config["budget"],
            wallet_balance=config["budget"],
            train_database=config["trains"],
            current_location=config["target_source"],
            searches_made=0,
            duplicate_searches=0,
            invalid_actions=0,
            time_violations=0,
            bookings_made=[],
            optimal_cost=config["optimal_cost"],
            seed=ep_seed,
            search_history=[],
            valid_stations=config["valid_stations"],
        )

        stations_str = ", ".join(config["valid_stations"])
        return Observation(
            done=False,
            reward=0.0,
            message=(
                f"You are at {config['target_source']}. "
                f"Destination: {config['target_dest']}. "
                f"Budget: ₹{config['budget']:.0f}. "
                f"Available stations: {stations_str}. "
                f"Use search_trains to find routes."
            ),
            wallet_balance=config["budget"],
            booked_itinerary=[],
            current_location=config["target_source"],
        )

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  OPENENV API: step()
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def step(self, action: Action, timeout_s: Optional[float] = None, **kwargs: Any) -> Observation:
        """
        Process one agent action. Returns Observation with done/reward.

        Design: intermediate steps always return reward=0.0.
        All penalties are tracked in internal state and applied only
        in _compute_final_reward() when the agent calls 'finish' or
        the episode auto-terminates. This prevents double-counting
        when the caller accumulates rewards.
        """
        if self._state is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")

        self._state.step_count += 1
        obs_message = ""
        search_results = None
        availability_status = None
        wl_probability = None

        # ── Auto-terminate after MAX_STEPS ──
        if self._state.step_count >= self.MAX_STEPS:
            final_reward = self._compute_final_reward()
            return Observation(
                done=True,
                reward=final_reward,
                message="Maximum steps reached. Episode auto-terminated.",
                wallet_balance=self._state.wallet_balance,
                booked_itinerary=[self._booking_summary(b) for b in self._state.bookings_made],
                current_location=self._state.current_location,
            )

        # ── SEARCH TRAINS ──
        if action.command == "search_trains":
            src = (action.source_stn or "").upper()
            dst = (action.dest_stn or "").upper()

            if src not in self._state.valid_stations or dst not in self._state.valid_stations:
                self._state.invalid_actions += 1
                stations_str = ", ".join(self._state.valid_stations)
                obs_message = (
                    f"Invalid station code(s): {src} → {dst}. "
                    f"Valid stations: {stations_str}."
                )
            else:
                search_key = f"{src}-{dst}"
                if search_key in self._state.search_history:
                    self._state.duplicate_searches += 1
                else:
                    self._state.search_history.append(search_key)

                self._state.searches_made += 1
                results = [
                    t for t in self._state.train_database
                    if t["source"] == src and t["dest"] == dst
                ]
                search_results = [self._train_display(t) for t in results]
                obs_message = (
                    f"Found {len(results)} train(s) from {src} to {dst}."
                    if results
                    else f"No trains found from {src} to {dst}."
                )

        # ── CHECK AVAILABILITY ──
        elif action.command == "check_availability":
            train = self._find_train(action.train_no)
            if not train:
                self._state.invalid_actions += 1
                obs_message = f"Train {action.train_no} not found in database."
            else:
                availability_status = train["status"]
                wl_probability = train.get("wl_confirm_prob")
                obs_message = (
                    f"Train {train['train_no']} ({train['name']}): "
                    f"{train['source']} → {train['dest']}, "
                    f"Status: {train['status']}"
                )
                if train["status"] == "WL":
                    obs_message += f", WL confirmation probability: {wl_probability}"
                obs_message += (
                    f", Price: ₹{train['price']}, "
                    f"Departs: {train['depart_time']}, Arrives: {train['arrive_time']}"
                )

        # ── BOOK TICKET ──
        elif action.command == "book_ticket":
            train = self._find_train(action.train_no)
            if not train:
                self._state.invalid_actions += 1
                obs_message = f"Train {action.train_no} not found in database."
            elif any(b["train_no"] == train["train_no"] for b in self._state.bookings_made):
                obs_message = f"Train {train['train_no']} already booked."
            elif train["price"] > self._state.wallet_balance:
                obs_message = (
                    f"Insufficient funds. Balance: ₹{self._state.wallet_balance:.0f}, "
                    f"Cost: ₹{train['price']}."
                )
            elif train["source"] != self._state.current_location:
                self._state.invalid_actions += 1
                obs_message = (
                    f"Cannot book train from {train['source']}. "
                    f"You are currently at {self._state.current_location}."
                )
            else:
                if self._has_time_conflict(train):
                    self._state.time_violations += 1
                    obs_message = (
                        f"Timing conflict! Train {train['train_no']} departs at "
                        f"{train['depart_time']} but your previous leg hasn't arrived "
                        f"with enough buffer (≥60 min required)."
                    )
                else:
                    self._state.wallet_balance -= train["price"]
                    self._state.current_location = train["dest"]
                    self._state.bookings_made.append(copy.deepcopy(train))
                    obs_message = (
                        f"Booked! Train {train['train_no']} ({train['name']}): "
                        f"{train['source']} → {train['dest']}, ₹{train['price']}. "
                        f"Status: {train['status']}. "
                        f"You are now at {train['dest']}. "
                        f"Remaining balance: ₹{self._state.wallet_balance:.0f}."
                    )

        # ── FINISH ──
        elif action.command == "finish":
            final_reward = self._compute_final_reward()
            return Observation(
                done=True,
                reward=final_reward,
                message="Journey finished. Computing final score.",
                wallet_balance=self._state.wallet_balance,
                booked_itinerary=[self._booking_summary(b) for b in self._state.bookings_made],
                current_location=self._state.current_location,
            )

        else:
            obs_message = f"Unknown command: {action.command}. Valid: search_trains, check_availability, book_ticket, finish."

        # Intermediate steps always return reward=0.0
        return Observation(
            done=False,
            reward=0.0,
            message=obs_message,
            search_results=search_results,
            availability_status=availability_status,
            wl_probability=wl_probability,
            wallet_balance=self._state.wallet_balance,
            booked_itinerary=[self._booking_summary(b) for b in self._state.bookings_made],
            current_location=self._state.current_location,
        )

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  OPENENV API: state (property)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    @property
    def state(self) -> State:
        """Return the full internal state for grading/debugging."""
        if self._state is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")
        return self._state.model_copy(deep=True)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  REWARD COMPUTATION
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _compute_final_reward(self) -> float:
        """
        Continuous reward function per blueprint §5.1:
          reward = exploration + destination + budget_eff + confirmation - penalties
        Clamped to [0.0, 1.0].
        """
        s = self._state

        # Exploration: 0.05 * min(searches_made, 4) → max 0.20
        # Uses searches_made (total count), per blueprint formula.
        exploration = 0.05 * min(s.searches_made, 4)

        # Destination reached: +0.30
        destination = 0.30 if s.current_location == s.target_dest else 0.0

        # Budget efficiency: 0.25 * (budget - spent) / budget → 0 to 0.25
        spent = s.budget - s.wallet_balance
        budget_eff = 0.25 * max(0, (s.budget - spent) / s.budget) if s.budget > 0 else 0.0

        # Confirmation quality: 0.25 * avg(cnf_score per leg) → 0 to 0.25
        if s.bookings_made:
            cnf_scores = []
            for b in s.bookings_made:
                if b["status"] == "CNF":
                    cnf_scores.append(1.0)
                else:
                    cnf_scores.append(b.get("wl_confirm_prob", 0.0))
            confirmation = 0.25 * (sum(cnf_scores) / len(cnf_scores))
        else:
            confirmation = 0.0

        # Penalties (per blueprint §5.1)
        penalty = 0.0
        penalty += 0.10 * s.invalid_actions      # invalid station/train
        penalty += 0.20 * s.time_violations       # time-travel violation
        penalty += 0.05 * s.duplicate_searches    # redundant duplicate search

        reward = exploration + destination + budget_eff + confirmation - penalty
        return round(max(0.0, min(1.0, reward)), 4)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  HELPERS
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _find_train(self, train_no: Optional[str]) -> Optional[Dict]:
        if not train_no or not self._state:
            return None
        for t in self._state.train_database:
            if t["train_no"] == train_no:
                return t
        return None

    def _parse_time_minutes(self, time_str: str) -> int:
        """Convert 'HH:MM' or 'HH:MM+1' to minutes from midnight."""
        day_offset = 0
        clean = time_str
        if "+1" in time_str:
            day_offset = 1440  # 24 * 60
            clean = time_str.replace("+1", "")
        parts = clean.strip().split(":")
        return int(parts[0]) * 60 + int(parts[1]) + day_offset

    def _has_time_conflict(self, new_train: Dict) -> bool:
        """Check if new train departs >= 60 min after last booked train arrives."""
        if not self._state.bookings_made:
            return False
        last_booking = self._state.bookings_made[-1]
        last_arrive = self._parse_time_minutes(last_booking["arrive_time"])
        new_depart = self._parse_time_minutes(new_train["depart_time"])
        return new_depart < last_arrive + 60

    def _train_display(self, train: Dict) -> Dict:
        """Agent-visible train info (hides internal fields)."""
        display = {
            "train_no": train["train_no"],
            "name": train["name"],
            "source": train["source"],
            "dest": train["dest"],
            "price": train["price"],
            "status": train["status"],
            "depart_time": train["depart_time"],
            "arrive_time": train["arrive_time"],
        }
        if train["status"] == "WL":
            display["wl_confirm_prob"] = train["wl_confirm_prob"]
        return display

    def _booking_summary(self, booking: Dict) -> Dict:
        return {
            "train_no": booking["train_no"],
            "name": booking["name"],
            "source": booking["source"],
            "dest": booking["dest"],
            "price": booking["price"],
            "status": booking["status"],
        }
