"""
Baseline inference script for IRCTC Dynamic Train Routing.
Runs an LLM agent against all 3 tasks via the deployed OpenEnv API.
Emits structured [START], [STEP], [END] logs for automated evaluation.
"""

import os
import json
import re

from openai import OpenAI
from server.models import Action
from client import IRCTCEnv

# ── Environment Variables ──
HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("OPENAI_API_KEY", "")
API_BASE_URL = os.environ.get(
    "API_BASE_URL", "https://api-inference.huggingface.co/v1/"
)
MODEL_NAME = os.environ.get(
    "MODEL_NAME", "meta-llama/Meta-Llama-3-70B-Instruct"
)
# URL of your deployed Hugging Face Space (e.g., "https://username-irctc.hf.space")
ENV_URL = os.environ.get("ENV_URL", "http://localhost:7860")

# ── LLM Client Setup ──
client = OpenAI(
    api_key=HF_TOKEN,
    base_url=API_BASE_URL,
)

# ── System Prompt for the Agent ──
SYSTEM_PROMPT = """You are an IRCTC Train Booking Agent.
Your goal: get from the source station to the destination station, under budget, with CONFIRMED (CNF) tickets.

Rules:
1. Use search_trains to discover available trains between stations.
2. If a direct train is Waitlisted (WL), search for intermediate stations to split the journey into multiple confirmed legs.
3. Consider departure/arrival times for multi-leg trips — the next leg must depart at least 60 minutes after the previous leg arrives.
4. Stay within budget.
5. When you have booked all necessary tickets and reached your destination, call "finish".

Output ONLY valid JSON with NO additional text, NO markdown, NO explanation:
{"command": "search_trains|check_availability|book_ticket|finish", "source_stn": "...", "dest_stn": "...", "train_no": "..."}

Only include fields relevant to the command:
- search_trains: requires source_stn and dest_stn
- check_availability: requires train_no
- book_ticket: requires train_no
- finish: no additional fields needed"""


def build_context_message(obs_dict: dict) -> str:
    """Build a concise context string from the observation for the LLM."""
    parts = [obs_dict.get("message", "")]

    if obs_dict.get("search_results"):
        parts.append("\nSearch Results:")
        for t in obs_dict["search_results"]:
            status_info = t.get("status", "")
            if status_info == "WL" and "wl_confirm_prob" in t:
                status_info += f" (confirm prob: {t['wl_confirm_prob']})"
            parts.append(
                f"  Train {t['train_no']} ({t['name']}): "
                f"{t['source']}→{t['dest']}, ₹{t['price']}, "
                f"Status: {status_info}, "
                f"Depart: {t.get('depart_time', 'N/A')}, "
                f"Arrive: {t.get('arrive_time', 'N/A')}"
            )

    if obs_dict.get("booked_itinerary"):
        parts.append("\nBooked Itinerary:")
        for b in obs_dict["booked_itinerary"]:
            parts.append(
                f"  {b['train_no']}: {b['source']}→{b['dest']}, "
                f"₹{b['price']} ({b['status']})"
            )

    parts.append(f"\nCurrent Location: {obs_dict.get('current_location', 'N/A')}")
    parts.append(f"Wallet Balance: ₹{obs_dict.get('wallet_balance', 0):.0f}")

    return "\n".join(parts)


def parse_action(response_text: str) -> Action:
    """Parse the LLM response into an Action, handling common formatting issues."""
    text = response_text.strip()

    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.startswith("```")]
        text = "\n".join(lines).strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # Use DOTALL to catch multiline JSON block if formatting is messy
        json_match = re.search(r'\{.*\}', text, flags=re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group())
            except json.JSONDecodeError:
                return Action(command="finish")
        else:
            return Action(command="finish")

    return Action(
        command=data.get("command", "finish"),
        source_stn=data.get("source_stn"),
        dest_stn=data.get("dest_stn"),
        train_no=data.get("train_no"),
    )


def run_inference():
    """Run the baseline agent on all 3 tasks via the WebSocket API."""
    task_scores = {}

    print("[START]", flush=True)

    # Use the synchronous WebSocket client context manager
    with IRCTCEnv(base_url=ENV_URL).sync() as env:
        for task_id in [1, 2, 3]:
            obs = env.reset(seed=42, task_id=task_id)
            obs_dict = obs.model_dump()
            step_count = 0
            conversation_history = []

            print(
                f"[STEP] task_id={task_id} | type=reset | "
                f"message={obs_dict['message']}",
                flush=True,
            )

            while not obs.done:
                step_count += 1
                context = build_context_message(obs_dict)

                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                ]
                # Keep last 10 turns to stay within context limits
                for h in conversation_history[-10:]:
                    messages.append(h)
                messages.append({"role": "user", "content": context})

                try:
                    response = client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=messages,
                        temperature=0.1,
                        max_tokens=256,
                    )
                    response_text = response.choices[0].message.content or ""
                    action = parse_action(response_text)
                except Exception as e:
                    print(
                        f"[STEP] task_id={task_id} | step={step_count} | "
                        f"error=LLM call failed: {str(e)[:100]} | "
                        f"fallback=finish",
                        flush=True,
                    )
                    action = Action(command="finish")

                conversation_history.append({"role": "user", "content": context})
                conversation_history.append(
                    {"role": "assistant", "content": action.model_dump_json()}
                )

                action_dict = action.model_dump(exclude_none=True)

                try:
                    obs = env.step(action)
                    obs_dict = obs.model_dump()
                except Exception as e:
                    print(
                        f"[STEP] task_id={task_id} | step={step_count} | "
                        f"error=step failed: {str(e)[:100]}",
                        flush=True,
                    )
                    break

                print(
                    f"[STEP] task_id={task_id} | step={step_count} | "
                    f"action={json.dumps(action_dict)} | "
                    f"reward={obs.reward} | done={obs.done}",
                    flush=True,
                )

                if obs.done:
                    task_scores[task_id] = obs.reward
                    print(
                        f"[STEP] task_id={task_id} | type=final | "
                        f"reward={obs.reward} | steps={step_count}",
                        flush=True,
                    )
                    break

                if step_count > 35:
                    task_scores[task_id] = 0.0
                    print(
                        f"[STEP] task_id={task_id} | type=timeout | steps={step_count}",
                        flush=True,
                    )
                    break

    total = sum(task_scores.values())
    avg = total / len(task_scores) if task_scores else 0.0
    print(
        f"[END] scores={json.dumps(task_scores)} | "
        f"total={total:.4f} | average={avg:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    run_inference()
