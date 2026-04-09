"""
IRCTC Dynamic Train Routing Environment.
Extends openenv.core Environment ABC for full framework compliance.
Supports 3 tasks: Easy (direct booking), Medium (WL avoidance), Hard (multi-constraint).
"""

import random
import copy
from uuid import uuid4
from typing import Optional, Any, Dict, List

from openenv.core.env_server.interfaces import Environment
from server.models import Action, Observation, State

class IRCTCEnvironment(Environment):
    MAX_STEPS = 30

    def __init__(self):
        self._state: Optional[State] = None

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  TASK CONFIGURATIONS
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _build_task1(self, rng: random.Random) -> dict:
        trains = [
            {"train_no": "12955", "name": "Kota Jan Shatabdi", "source": "DEL", "dest": "KOTA", "price": round(800 * rng.uniform(0.9, 1.1)), "status": "CNF", "wl_confirm_prob": None, "depart_time": "06:00", "arrive_time": "11:30", "duration_hrs": 5.5},
            {"train_no": "12956", "name": "Delhi Jaipur Express", "source": "DEL", "dest": "JP", "price": round(600 * rng.uniform(0.9, 1.1)), "status": "CNF", "wl_confirm_prob": None, "depart_time": "07:00", "arrive_time": "11:00", "duration_hrs": 4},
            {"train_no": "12957", "name": "Jaipur Kota Express", "source": "JP", "dest": "KOTA", "price": round(800 * rng.uniform(0.9, 1.1)), "status": "CNF", "wl_confirm_prob": None, "depart_time": "12:00", "arrive_time": "15:30", "duration_hrs": 3.5},
        ]
        return {"target_source": "DEL", "target_dest": "KOTA", "budget": 3000.0, "trains": trains, "optimal_cost": 800.0, "valid_stations": ["DEL", "JP", "KOTA"]}

    def _build_task2(self, rng: random.Random) -> dict:
        wl_prob_direct = round(rng.uniform(0.10, 0.25), 2)
        trains = [
            {"train_no": "12951", "name": "Mumbai Rajdhani", "source": "DEL", "dest": "BOM", "price": round(1500 * rng.uniform(0.9, 1.1)), "status": "WL", "wl_confirm_prob": wl_prob_direct, "depart_time": "16:00", "arrive_time": "08:00+1", "duration_hrs": 16},
            {"train_no": "12952", "name": "Delhi Kota Express", "source": "DEL", "dest": "KOTA", "price": round(800 * rng.uniform(0.9, 1.1)), "status": "CNF", "wl_confirm_prob": None, "depart_time": "06:00", "arrive_time": "11:30", "duration_hrs": 5.5},
            {"train_no": "12953", "name": "Kota Mumbai SF", "source": "KOTA", "dest": "BOM", "price": round(1100 * rng.uniform(0.9, 1.1)), "status": "CNF", "wl_confirm_prob": None, "depart_time": "13:00", "arrive_time": "23:00", "duration_hrs": 10},
            {"train_no": "12954", "name": "Delhi Surat Express", "source": "DEL", "dest": "ST", "price": round(1800 * rng.uniform(0.9, 1.1)), "status": "CNF", "wl_confirm_prob": None, "depart_time": "08:00", "arrive_time": "22:00", "duration_hrs": 14},
            {"train_no": "12958", "name": "Surat Mumbai Express", "source": "ST", "dest": "BOM", "price": round(1400 * rng.uniform(0.9, 1.1)), "status": "CNF", "wl_confirm_prob": None, "depart_time": "23:30", "arrive_time": "05:00+1", "duration_hrs": 5.5},
            {"train_no": "12959", "name": "Delhi Vadodara Exp", "source": "DEL", "dest": "BRC", "price": round(1200 * rng.uniform(0.9, 1.1)), "status": "CNF", "wl_confirm_prob": None, "depart_time": "09:00", "arrive_time": "21:00", "duration_hrs": 12},
            {"train_no": "12960", "name": "Vadodara Mumbai Exp", "source": "BRC", "dest": "BOM", "price": round(900 * rng.uniform(0.9, 1.1)), "status": "CNF", "wl_confirm_prob": None, "depart_time": "22:30", "arrive_time": "04:30+1", "duration_hrs": 6},
        ]
        return {"target_source": "DEL", "target_dest": "BOM", "budget": 3000.0, "trains": trains, "optimal_cost": 1900.0, "valid_stations": ["DEL", "KOTA", "ST", "BRC", "BOM"]}

    def _build_task3(self, rng: random.Random) -> dict:
        trains = [
            {"train_no": "13001", "name": "Rajdhani Express", "source": "DEL", "dest": "BOM", "price": round(1800 * rng.uniform(0.9, 1.1)), "status": "WL", "wl_confirm_prob": round(rng.uniform(0.10, 0.20), 2), "depart_time": "16:00", "arrive_time": "08:00+1", "duration_hrs": 16},
            {"train_no": "13002", "name": "Delhi Jaipur SF", "source": "DEL", "dest": "JP", "price": round(400 * rng.uniform(0.9, 1.1)), "status": "CNF", "wl_confirm_prob": None, "depart_time": "06:00", "arrive_time": "10:00", "duration_hrs": 4},
            {"train_no": "13003", "name": "Jaipur Kota Exp", "source": "JP", "dest": "KOTA", "price": round(350 * rng.uniform(0.9, 1.1)), "status": "CNF", "wl_confirm_prob": None, "depart_time": "11:30", "arrive_time": "15:00", "duration_hrs": 3.5},
            {"train_no": "13004", "name": "Kota Mumbai SF", "source": "KOTA", "dest": "BOM", "price": round(1100 * rng.uniform(0.9, 1.1)), "status": "CNF", "wl_confirm_prob": None, "depart_time": "16:30", "arrive_time": "04:00+1", "duration_hrs": 11.5},
            {"train_no": "13005", "name": "Delhi Ahmedabad Mail", "source": "DEL", "dest": "ADI", "price": round(900 * rng.uniform(0.9, 1.1)), "status": "WL", "wl_confirm_prob": round(rng.uniform(0.15, 0.30), 2), "depart_time": "20:00", "arrive_time": "08:00+1", "duration_hrs": 12},
            {"train_no": "13006", "name": "Ahmedabad Mumbai Shatabdi", "source": "ADI", "dest": "BOM", "price": round(700 * rng.uniform(0.9, 1.1)), "status": "CNF", "wl_confirm_prob": None, "depart_time": "09:30", "arrive_time": "16:00", "duration_hrs": 6.5},
            {"train_no": "13007", "name": "Delhi Kota Janshatabdi", "source": "DEL", "dest": "KOTA", "price": round(700 * rng.uniform(0.9, 1.1)), "status": "CNF", "wl_confirm_prob": None, "depart_time": "07:00", "arrive_time": "12:30", "duration_hrs": 5.5},
            {"train_no": "13008", "name": "Kota Surat Exp", "source": "KOTA", "dest": "ST", "price": round(600 * rng.uniform(0.9, 1.1)), "status": "CNF", "wl_confirm_prob": None, "depart_time": "14:00", "arrive_time": "22:00", "duration_hrs": 8},
            {"train_no": "13009", "name": "Surat Mumbai WL", "source": "ST", "dest": "BOM", "price": round(500 * rng.uniform(0.9, 1.1)), "status": "WL", "wl_confirm_prob": round(rng.uniform(0.10, 0.20), 2), "depart_time": "23:30", "arrive_time": "04:30+1", "duration_hrs": 5},
            {"train_no": "13010", "name": "Jaipur Mumbai Superfast", "source": "JP", "dest": "BOM", "price": round(1600 * rng.uniform(0.9, 1.1)), "status": "CNF", "wl_confirm_prob": None, "depart_time": "12:00", "arrive_time": "06:00+1", "duration_hrs": 18},
            {"train_no": "13011", "name": "Vadodara Mumbai Exp", "source": "BRC", "dest": "BOM", "price": round(500 * rng.uniform(0.9, 1.1)), "status": "CNF", "wl_confirm_prob": None, "depart_time": "10:00", "arrive_time": "16:00", "duration_hrs": 6},
            {"train_no": "13012", "name": "Panvel Mumbai Local", "source": "PNVL", "dest": "BOM", "price": round(100 * rng.uniform(0.9, 1.1)), "status": "CNF", "wl_confirm_prob": None, "depart_time": "07:00", "arrive_time": "08:30", "duration_hrs": 1.5},
        ]
        return {"target_source": "DEL", "target_dest": "BOM", "budget": 2200.0, "trains": trains, "optimal_cost": 1850.0, "valid_stations": ["DEL", "JP", "KOTA", "ADI", "ST", "BRC", "PNVL", "BOM"]}

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  TASK ID RESOLUTION (UPDATED FOR SAFETY)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    @staticmethod
    def _resolve_task_id(episode_id: Optional[str] = None, task_id_kwarg: Optional[Any] = None) -> int:
        if task_id_kwarg is not None:
            try:
                return int(task_id_kwarg)
            except ValueError:
                pass
                
        if episode_id and episode_id.startswith("task_"):
            try:
                tid = int(episode_id.split("_")[1])
                if tid in (1, 2, 3):
                    return tid
            except (IndexError, ValueError):
                pass
        return 1  # Default fallback

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  OPENENV API: reset() (UPDATED FOR SAFETY)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None, **kwargs: Any) -> Observation:
        episode_id = kwargs.get("episode_id")
        
        # Safely extract task_id from kwargs or OpenEnv's standard 'options' dict
        raw_task_id = kwargs.get("task_id")
        if options and "task_id" in options:
            raw_task_id = options["task_id"]

        task_id = self._resolve_task_id(episode_id, raw_task_id)
        if task_id not in (1, 2, 3):
            task_id = 1

        ep_seed = seed if seed is not None else random.randint(0, 2**31)
        ep_id = episode_id or str(uuid4())
        rng = random.Random(ep_seed)

        builders = {1: self._build_task1, 2: self._build_task2, 3: self._build_task3}
        config = builders[task_id](rng)

        self._state = State(
            episode_id=ep_id, step_count=0, task_id=task_id, target_source=config["target_source"],
            target_dest=config["target_dest"], budget=config["budget"], wallet_balance=config["budget"],
            train_database=config["trains"], current_location=config["target_source"],
            searches_made=0, duplicate_searches=0, invalid_actions=0, time_violations=0,
            bookings_made=[], optimal_cost=config["optimal_cost"], seed=ep_seed, search_history=[], valid_stations=config["valid_stations"]
        )

        stations_str = ", ".join(config["valid_stations"])
        return Observation(
            done=False,
            reward=0.0,
            message=f"You are at {config['target_source']}. Destination: {config['target_dest']}. Budget: ₹{config['budget']:.0f}. Available stations: {stations_str}. Use search_trains to find routes.",
            wallet_balance=config["budget"],
            booked_itinerary=[],
            current_location=config["target_source"],
        )

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  OPENENV API: step()
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    def step(self, action: Action, timeout_s: Optional[float] = None, **kwargs: Any) -> Observation:
        if self._state is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")

        self._state.step_count += 1
        obs_message, search_results, availability_status, wl_probability = "", None, None, None

        if self._state.step_count >= self.MAX_STEPS:
            final_reward = self._compute_final_reward()
            return Observation(done=True, reward=final_reward, message="Maximum steps reached. Episode auto-terminated.", wallet_balance=self._state.wallet_balance, booked_itinerary=[self._booking_summary(b) for b in self._state.bookings_made], current_location=self._state.current_location)

        if action.command == "search_trains":
            src, dst = (action.source_stn or "").upper(), (action.dest_stn or "").upper()
            if src not in self._state.valid_stations or dst not in self._state.valid_stations:
                self._state.invalid_actions += 1
                obs_message = f"Invalid station code(s): {src} → {dst}. Valid stations: {', '.join(self._state.valid_stations)}."
            else:
                search_key = f"{src}-{dst}"
                if search_key in self._state.search_history:
                    self._state.duplicate_searches += 1
                else:
                    self._state.search_history.append(search_key)
                self._state.searches_made += 1
                results = [t for t in self._state.train_database if t["source"] == src and t["dest"] == dst]
                search_results = [self._train_display(t) for t in results]
                obs_message = f"Found {len(results)} train(s) from {src} to {dst}." if results else f"No trains found from {src} to {dst}."

        elif action.command == "check_availability":
            train = self._find_train(action.train_no)
            if not train:
                self._state.invalid_actions += 1
                obs_message = f"Train {action.train_no} not found in database."
            else:
                availability_status, wl_probability = train["status"], train.get("wl_confirm_prob")
                obs_message = f"Train {train['train_no']} ({train['name']}): {train['source']} → {train['dest']}, Status: {train['status']}"
                if train["status"] == "WL": obs_message += f", WL confirmation probability: {wl_probability}"
                obs_message += f", Price: ₹{train['price']}, Departs: {train['depart_time']}, Arrives: {train['arrive_time']}"

        elif action.command == "book_ticket":
            train = self._find_train(action.train_no)
            if not train:
                self._state.invalid_actions += 1
                obs_message = f"Train {action.train_no} not found in database."
            elif any(b["train_no"] == train["train_no"] for b in self._state.bookings_made):
                obs_message = f"Train {train['train_no']} already booked."
            elif train["price"] > self._state.wallet_balance:
                obs_message = f"Insufficient funds. Balance: ₹{self._state.wallet_balance:.0f}, Cost: ₹{train['price']}."
            elif train["source"] != self._state.current_location:
                self._state.invalid_actions += 1
                obs_message = f"Cannot book train from {train['source']}. You are currently at {self._state.current_location}."
            else:
                if self._has_time_conflict(train):
                    self._state.time_violations += 1
                    obs_message = f"Timing conflict! Train {train['train_no']} departs at {train['depart_time']} but your previous leg hasn't arrived with enough buffer (≥60 min required)."
                else:
                    self._state.wallet_balance -= train["price"]
                    self._state.current_location = train["dest"]
                    self._state.bookings_made.append(copy.deepcopy(train))
                    obs_message = f"Booked! Train {train['train_no']} ({train['name']}): {train['source']} → {train['dest']}, ₹{train['price']}. Status: {train['status']}. You are now at {train['dest']}. Remaining balance: ₹{self._state.wallet_balance:.0f}."

        elif action.command == "finish":
            final_reward = self._compute_final_reward()
            return Observation(done=True, reward=final_reward, message="Journey finished. Computing final score.", wallet_balance=self._state.wallet_balance, booked_itinerary=[self._booking_summary(b) for b in self._state.bookings_made], current_location=self._state.current_location)

        else:
            obs_message = f"Unknown command: {action.command}. Valid: search_trains, check_availability, book_ticket, finish."

        return Observation(done=False, reward=0.0, message=obs_message, search_results=search_results, availability_status=availability_status, wl_probability=wl_probability, wallet_balance=self._state.wallet_balance, booked_itinerary=[self._booking_summary(b) for b in self._state.bookings_made], current_location=self._state.current_location)

    @property
    def state(self) -> State:
        if self._state is None: raise RuntimeError("Environment not initialized. Call reset() first.")
        return self._state.model_copy(deep=True)

    def _compute_final_reward(self) -> float:
        s = self._state
        exploration = 0.05 * min(s.searches_made, 4)
        destination = 0.30 if s.current_location == s.target_dest else 0.0
        spent = s.budget - s.wallet_balance
        budget_eff = 0.25 * max(0, (s.budget - spent) / s.budget) if s.budget > 0 else 0.0
        confirmation = 0.25 * (sum([1.0 if b["status"] == "CNF" else b.get("wl_confirm_prob", 0.0) for b in s.bookings_made]) / len(s.bookings_made)) if s.bookings_made else 0.0
        penalty = (0.10 * s.invalid_actions) + (0.20 * s.time_violations) + (0.05 * s.duplicate_searches)
        return round(max(0.0, min(1.0, exploration + destination + budget_eff + confirmation - penalty)), 4)

    def _find_train(self, train_no: Optional[str]) -> Optional[Dict]:
        return next((t for t in self._state.train_database if t["train_no"] == train_no), None) if train_no and self._state else None

    def _parse_time_minutes(self, time_str: str) -> int:
        clean, day_offset = (time_str.replace("+1", ""), 1440) if "+1" in time_str else (time_str, 0)
        parts = clean.strip().split(":")
        return int(parts[0]) * 60 + int(parts[1]) + day_offset

    def _has_time_conflict(self, new_train: Dict) -> bool:
        if not self._state.bookings_made: return False
        return self._parse_time_minutes(new_train["depart_time"]) < self._parse_time_minutes(self._state.bookings_made[-1]["arrive_time"]) + 60

    def _train_display(self, train: Dict) -> Dict:
        display = {k: train[k] for k in ["train_no", "name", "source", "dest", "price", "status", "depart_time", "arrive_time"]}
        if train["status"] == "WL": display["wl_confirm_prob"] = train["wl_confirm_prob"]
        return display

    def _booking_summary(self, booking: Dict) -> Dict:
        return {k: booking[k] for k in ["train_no", "name", "source", "dest", "price", "status"]}
