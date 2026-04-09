"""
Baseline inference script for IRCTC Dynamic Train Routing.
Strictly conforms to the OpenEnv stdout formatting requirements for the Meta Hackathon.
"""

import asyncio
import os
import json
import re
import sys
import textwrap
from typing import List, Optional

from openai import OpenAI
from server.models import Action
from client import IRCTCEnv

# ── Mandatory Environment Variables ──
HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("API_KEY", "")
API_BASE_URL = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

# Phase 2 Validation injects the target port into ENV_URL
ENV_URL = os.environ.get("ENV_URL", "http://localhost:7860")

# ── Dynamic Autograder Variables ──
BENCHMARK = os.environ.get("IRCTC_DYNAMIC_ROUTER_BENCHMARK", "irctc-dynamic-router")

try:
    task_env = os.environ.get("IRCTC_DYNAMIC_ROUTER_TASK", "1")
    TASK_ID = int(task_env)
except ValueError:
    TASK_ID = 1

TASK_NAME = str(TASK_ID)

MAX_STEPS = 30
TEMPERATURE = 0.1
MAX_TOKENS = 256
SUCCESS_SCORE_THRESHOLD = 0.5  # Agent needs >= 0.5 to be considered "successful"

# ── System Prompt ──
SYSTEM_PROMPT = textwrap.dedent("""
    You are an IRCTC Train Booking Agent. Book tickets from source to destination within budget using CONFIRMED (CNF) tickets only.

    STRICT WORKFLOW:
    1. search_trains — use the exact station codes shown in "GOAL" and "Available stations". Never invent station codes.
    2. check_availability(train_no) — pick a CNF train from the search results using its exact train_no
    3. book_ticket(train_no) — book it using the same exact train_no
    4. If ALL trains on a route are WL, split the journey via an intermediate station listed in "Available stations"
    5. Call finish once you have reached the destination

    RULES:
    - Use ONLY station codes and train numbers that appear in the observation — never invent them
    - NEVER repeat a search for the same route
    - check_availability and book_ticket require only train_no
    - Multi-leg trips: next departure must be ≥60 min after previous arrival
    - Only book CNF tickets
    - Stay within wallet balance

    Output ONLY valid JSON, one command per step, no markdown, no explanation. Examples:
    {"command": "search_trains", "source_stn": "<SOURCE_FROM_OBSERVATION>", "dest_stn": "<DEST_FROM_OBSERVATION>"}
    {"command": "check_availability", "train_no": "<TRAIN_NO_FROM_RESULTS>"}
    {"command": "book_ticket", "train_no": "<TRAIN_NO_FROM_RESULTS>"}
    {"command": "finish"}
""").strip()

# ── Formatting Helpers (Strictly matches hackathon spec) ──
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error.replace('\n', ' ') if error else "null"
    action_val = action.replace('\n', ' ')
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action_val} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

def build_user_prompt(step: int, obs_dict: dict, history: List[str], goal: Optional[str] = None) -> str:
    """Build context string from observation."""
    parts = [f"Step: {step}"]
    if goal:
        parts.append(f"GOAL: {goal}")
    parts.append(obs_dict.get("message", ""))

    if obs_dict.get("search_results"):
        parts.append("Search Results:")
        for t in obs_dict["search_results"]:
            status_info = t.get("status", "")
            if status_info == "WL" and "wl_confirm_prob" in t:
                status_info += f" (confirm prob: {t['wl_confirm_prob']})"
            parts.append(
                f"  Train {t['train_no']}: {t['source']}→{t['dest']}, ₹{t['price']}, "
                f"Status: {status_info}, Depart: {t.get('depart_time', 'N/A')}, Arrive: {t.get('arrive_time', 'N/A')}"
            )
        cnf_trains = [t for t in obs_dict["search_results"] if t.get("status") == "CNF"]
        if cnf_trains:
            parts.append(f"ACTION REQUIRED: Call check_availability for train_no={cnf_trains[0]['train_no']} (CNF train found)")
        else:
            parts.append("WARNING: All trains are WL — search intermediate stations to split the journey")

    if obs_dict.get("availability_status"):
        avail = obs_dict["availability_status"]
        parts.append(f"Availability Status: {avail}")
        if avail == "CNF":
            parts.append("ACTION REQUIRED: Call book_ticket with the same train_no")
        else:
            parts.append("WARNING: Train is WL — try a different train or intermediate route")

    if obs_dict.get("booked_itinerary"):
        parts.append("Booked Itinerary:")
        for b in obs_dict["booked_itinerary"]:
            parts.append(f"  {b['train_no']}: {b['source']}→{b['dest']}, ₹{b['price']} ({b['status']})")

    parts.append(f"Current Location: {obs_dict.get('current_location', 'N/A')}")
    parts.append(f"Wallet Balance: ₹{obs_dict.get('wallet_balance', 0):.0f}")

    if history:
        history_block = "\n".join(history[-6:])
        parts.append(f"\nPrevious steps:\n{history_block}")

    parts.append("\nSend your next action (JSON only).")
    return "\n".join(parts)

def parse_action(response_text: str) -> Action:
    text = response_text.strip()
    
    if text.startswith("```"):
        text = "\n".join([l for l in text.split("\n") if not l.startswith("```")]).strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        json_match = re.search(r'\{.*\}', text, flags=re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group())
            except:
                raise ValueError("LLM output is not valid JSON.")
        else:
            raise ValueError("No JSON block found in LLM output.")

    return Action(
        command=data.get("command", "finish"),
        source_stn=data.get("source_stn"),
        dest_stn=data.get("dest_stn"),
        train_no=data.get("train_no"),
    )

async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)
    
    env = IRCTCEnv(base_url=ENV_URL)
    
    try:
        await env.connect()
        obs = await env.reset(seed=42, task_id=TASK_ID)

        # Extract goal string from initial observation message
        init_obs = obs.observation.model_dump() if hasattr(obs, "observation") else obs.model_dump()
        goal = init_obs.get("message", "")

        for step in range(1, MAX_STEPS + 1):
            if obs.done:
                break

            # Handle both StepResult wrappers and direct Observations depending on the OpenEnv version
            obs_data = obs.observation.model_dump() if hasattr(obs, "observation") else obs.model_dump()
            user_prompt = build_user_prompt(step, obs_data, history, goal=goal)
            action_obj = None
            action_str = ""
            error_msg = None

            # 1. Model Request
            try:
                completion = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=TEMPERATURE,
                    max_tokens=MAX_TOKENS,
                )
                text = (completion.choices[0].message.content or "").strip()
                action_obj = parse_action(text)
                action_str = action_obj.model_dump_json(exclude_none=True)
            except Exception as exc:
                error_msg = f"LLM Error: {str(exc)}"
                action_str = '{"command":"finish"}'
                action_obj = Action(command="finish")

            # 2. Environment Step
            try:
                obs = await env.step(action_obj)
            except Exception as exc:
                error_msg = f"Env Error: {str(exc)}"
                obs.done = True

            reward = obs.reward or 0.0
            done = obs.done

            rewards.append(reward)
            steps_taken = step

            # 3. Log Step
            log_step(step=step, action=action_str, reward=reward, done=done, error=error_msg)

            # Build observation summary for history so model knows what results it received
            obs_summary_parts = []
            obs_data_next = obs.observation.model_dump() if hasattr(obs, "observation") else obs.model_dump()
            if obs_data_next.get("search_results"):
                trains = obs_data_next["search_results"]
                cnf = [t["train_no"] for t in trains if t.get("status") == "CNF"]
                wl = [t["train_no"] for t in trains if t.get("status") != "CNF"]
                if cnf:
                    obs_summary_parts.append(f"CNF trains: {cnf}")
                if wl:
                    obs_summary_parts.append(f"WL trains: {wl}")
            if obs_data_next.get("availability_status"):
                obs_summary_parts.append(f"availability={obs_data_next['availability_status']}")
            obs_summary = f" -> {'; '.join(obs_summary_parts)}" if obs_summary_parts else f" -> reward {reward:+.2f}"
            history.append(f"Step {step}: {action_str}{obs_summary}")

            if done:
                break

        score = rewards[-1] if rewards else 0.0
        score = min(max(score, 0.0), 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD

    except asyncio.CancelledError:
        # Crucial catch: Prevents the grader timeout from crashing the script
        print("[DEBUG] Episode timed out (CancelledError). Terminating loop.", flush=True)
    except BaseException as e:
        # Crucial catch: Catches terminal interrupts and low-level loop errors
        print(f"[DEBUG] Fatal system error: {e}", flush=True)
    finally:
        try:
            await env.close()
        except BaseException as e:
            pass
        
        # 4. Log End (This will now ALWAYS execute, satisfying the autograder)
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit(0)
