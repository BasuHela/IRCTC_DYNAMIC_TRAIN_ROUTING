"""
Baseline inference script for IRCTC Dynamic Train Routing.
Strictly conforms to the OpenEnv stdout formatting requirements for the Meta Hackathon.
"""

import asyncio
import os
import json
import re
import textwrap
from typing import List, Optional

from openai import OpenAI
from server.models import Action
from client import IRCTCEnv

# ── Mandatory Environment Variables ──
HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("API_KEY", "")
API_BASE_URL = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

# The autograder runs the Docker container locally and injects its own ENV_URL
ENV_URL = os.environ.get("ENV_URL", "http://localhost:7860")

# ── Dynamic Autograder Variables ──
# The Meta grader injects variables based on your openenv.yaml name (irctc-dynamic-router)
BENCHMARK = os.environ.get("IRCTC_DYNAMIC_ROUTER_BENCHMARK", "irctc-dynamic-router")

# Read task ID dynamically from the grader
try:
    task_env = os.environ.get("IRCTC_DYNAMIC_ROUTER_TASK", "1")
    TASK_ID = int(task_env)
except ValueError:
    TASK_ID = 1

# The grader expects the exact string ID in the STDOUT log
TASK_NAME = str(TASK_ID)

MAX_STEPS = 30
TEMPERATURE = 0.1
MAX_TOKENS = 256
SUCCESS_SCORE_THRESHOLD = 0.5  # Agent needs >= 0.5 to be considered "successful"

# ── System Prompt ──
SYSTEM_PROMPT = textwrap.dedent("""
    You are an IRCTC Train Booking Agent.
    Your goal: get from the source station to the destination station, under budget, with CONFIRMED (CNF) tickets.

    Rules:
    1. Use search_trains to discover available trains between stations.
    2. If a direct train is Waitlisted (WL), search for intermediate stations to split the journey into multiple confirmed legs.
    3. Consider departure/arrival times for multi-leg trips — the next leg must depart at least 60 minutes after the previous leg arrives.
    4. Stay within budget.
    5. When you have booked all necessary tickets and reached your destination, call "finish".

    Output ONLY valid JSON with NO additional text, NO markdown, NO explanation:
    {"command": "search_trains|check_availability|book_ticket|finish", "source_stn": "...", "dest_stn": "...", "train_no": "..."}
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

def build_user_prompt(step: int, obs_dict: dict, history: List[str]) -> str:
    """Build context string from observation."""
    parts = [f"Step: {step}"]
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

    if obs_dict.get("booked_itinerary"):
        parts.append("Booked Itinerary:")
        for b in obs_dict["booked_itinerary"]:
            parts.append(f"  {b['train_no']}: {b['source']}→{b['dest']}, ₹{b['price']} ({b['status']})")

    parts.append(f"Current Location: {obs_dict.get('current_location', 'N/A')}")
    parts.append(f"Wallet Balance: ₹{obs_dict.get('wallet_balance', 0):.0f}")
    
    if history:
        history_block = "\n".join(history[-4:])
        parts.append(f"\nPrevious steps:\n{history_block}")

    parts.append("\nSend your next action (JSON only).")
    return "\n".join(parts)

def parse_action(response_text: str) -> Action:
    text = response_text.strip()
    
    # Qwen frequently outputs markdown code blocks; we must strip them
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
        # Open the connection
        await env.connect()
        obs = await env.reset(seed=42, task_id=TASK_ID)
        
        for step in range(1, MAX_STEPS + 1):
            if obs.done:
                break

            user_prompt = build_user_prompt(step, obs.model_dump(), history)
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
            history.append(f"Step {step}: {action_str} -> reward {reward:+.2f}")

            if done:
                break

        # The final reward returned by our specific environment is the complete [0, 1] score
        score = rewards[-1] if rewards else 0.0
        score = min(max(score, 0.0), 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as e:
        print(f"[DEBUG] Fatal episode error: {e}", flush=True)
    
    finally:
        try:
            await env.close()
        except Exception as e:
            print(f"[DEBUG] env.close() error: {e}", flush=True)
        
        # 4. Log End
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

if __name__ == "__main__":
    asyncio.run(main())
