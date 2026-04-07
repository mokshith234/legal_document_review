"""
inference.py
------------
Hackathon submission inference script.

Requirements met:
  - Named inference.py, placed at repo root
  - Uses OpenAI client for all LLM calls
  - Reads API_BASE_URL, MODEL_NAME, HF_TOKEN from environment
  - Runs all 3 tasks with graders, scores in 0.0-1.0
  - Completes well within the 20-minute runtime limit

Usage:
  python inference.py

Environment variables required:
  API_BASE_URL   - The API endpoint for the LLM (OpenAI-compatible)
  MODEL_NAME     - The model identifier to use for inference
  HF_TOKEN       - Your API key (used as OpenAI API key)
"""

import os
import json
import requests
from openai import OpenAI

# ---------------------------------------------------------------------------
# Required environment variables
# ---------------------------------------------------------------------------
API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:11434/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN     = os.environ.get("HF_TOKEN", "")

# The environment server (runs in same container via Docker CMD)
ENV_BASE = os.environ.get("ENV_BASE_URL", "http://localhost:8000")

# ---------------------------------------------------------------------------
# OpenAI-compatible client (required by hackathon rules)
# ---------------------------------------------------------------------------
client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN if HF_TOKEN else "no-key",
)

TASKS = ["clause_classifier", "risk_spotter", "contract_redliner"]

ACTION_TYPE_MAP = {
    "clause_classifier": "classify",
    "risk_spotter":      "flag_risks",
    "contract_redliner": "redline",
}

SYSTEM_PROMPT = """You are an expert legal document AI assistant specializing in contract review.
You analyze contracts precisely and respond only with what is asked."""


def call_llm(user_prompt: str) -> str:
    """Call the LLM using the OpenAI client. All LLM calls go through here."""
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_prompt},
        ],
        max_tokens=1024,
        temperature=0.2,
    )
    return response.choices[0].message.content.strip()


def build_prompt(task_id: str, obs: dict) -> str:
    """Build a task-specific prompt from the observation."""

    if task_id == "clause_classifier":
        return f"""{obs['instructions']}

Contract clause to classify:
\"\"\"
{obs['document_text']}
\"\"\"

Respond with ONLY the clause type label — nothing else. One word or short phrase."""

    elif task_id == "risk_spotter":
        return f"""{obs['instructions']}

Contract section:
\"\"\"
{obs['document_text']}
\"\"\"

List each risk on a new line. Be specific — cite the clause language and explain why it is risky."""

    elif task_id == "contract_redliner":
        policy = obs.get("context", {}).get("policy_brief", "")
        return f"""{obs['instructions']}

Policy brief:
{policy}

Contract to redline:
\"\"\"
{obs['document_text']}
\"\"\"

For each required change, write:
- Section: <section name>
- Issue: <why this violates policy>
- Original: <exact original text>
- Redline: <proposed replacement text>"""

    return obs['instructions'] + "\n\n" + obs['document_text']


def build_metadata(task_id: str, llm_output: str) -> dict:
    """
    Parse LLM output into structured metadata for best grader scoring.
    Metadata is optional but boosts scores significantly.
    """
    if task_id == "risk_spotter":
        # Split lines into a risks list, filter blanks
        risks = [line.strip("- •*").strip()
                 for line in llm_output.split("\n")
                 if line.strip() and len(line.strip()) > 10]
        return {"risks": risks}

    elif task_id == "contract_redliner":
        # Parse structured edit blocks
        edits = []
        current = {}
        for line in llm_output.split("\n"):
            line = line.strip()
            if line.startswith("- Section:") or line.startswith("Section:"):
                if current:
                    edits.append(current)
                current = {"section": line.split(":", 1)[-1].strip()}
            elif line.startswith("- Issue:") or line.startswith("Issue:"):
                current["issue"] = line.split(":", 1)[-1].strip()
            elif line.startswith("- Original:") or line.startswith("Original:"):
                current["original"] = line.split(":", 1)[-1].strip()
            elif line.startswith("- Redline:") or line.startswith("Redline:"):
                current["redline"] = line.split(":", 1)[-1].strip()
        if current:
            edits.append(current)
        return {"edits": edits} if edits else {}

    return {}


def run_task(task_id: str) -> float:
    """Run one full episode for a task. Returns score in [0.0, 1.0]."""
    print(f"\n{'='*50}")
    print(f"Running task: {task_id}")
    print(f"{'='*50}")

    # 1. Reset environment
    reset_resp = requests.post(
        f"{ENV_BASE}/reset",
        json={"task_id": task_id},
        timeout=30,
    )
    reset_resp.raise_for_status()
    data = reset_resp.json()
    session_id = data["session_id"]
    obs = data["observation"]

    print(f"Session: {session_id}")
    print(f"Document preview: {obs['document_text'][:120]}...")

    # 2. Build prompt and call LLM
    prompt = build_prompt(task_id, obs)
    print(f"Calling LLM ({MODEL_NAME})...")
    llm_output = call_llm(prompt)
    print(f"LLM output preview: {llm_output[:200]}")

    # 3. Build structured action
    action = {
        "action_type": ACTION_TYPE_MAP[task_id],
        "content":     llm_output,
        "metadata":    build_metadata(task_id, llm_output),
    }

    # 4. Submit action to environment
    step_resp = requests.post(
        f"{ENV_BASE}/step",
        json={"action": action, "session_id": session_id},
        timeout=30,
    )
    step_resp.raise_for_status()
    result = step_resp.json()

    score    = result["reward"]["score"]
    feedback = result["reward"].get("feedback", "")
    breakdown = result["reward"].get("breakdown", {})

    print(f"Score:     {score:.4f}")
    print(f"Breakdown: {breakdown}")
    print(f"Feedback:  {feedback[:200]}")

    # Validate score is in required range
    assert 0.0 <= score <= 1.0, f"Score out of range for {task_id}: {score}"

    return score


def main():
    print("[START]")
    print(f"API_BASE_URL={API_BASE_URL}")
    print(f"MODEL_NAME={MODEL_NAME}")
    print(f"ENV_BASE={ENV_BASE}")
    print("[END]")
    print()

    scores = {}

    for task_id in TASKS:
        try:
            print(f"[START] task={task_id}")
            score = run_task(task_id)
            scores[task_id] = score
            print(f"[STEP] task={task_id} score={score:.4f}")
            print(f"[END] task={task_id}")
        except Exception as e:
            print(f"ERROR on task {task_id}: {e}")
            raise  # re-raise so validator catches it

    print()
    print("[START]")
    print("FINAL_SCORES:")
    for task_id, score in scores.items():
        status = "PASS" if 0.0 <= score <= 1.0 else "FAIL"
        print(f"  {task_id}: {score:.4f} [{status}]")

    overall = sum(scores.values()) / len(scores)
    print(f"Overall mean: {overall:.4f}")
    print("[END]")
    print()

    results = {
        "scores": scores,
        "overall_mean": round(overall, 4),
        "model": MODEL_NAME,
        "tasks_run": len(scores),
    }
    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("results.json written.")


if __name__ == "__main__":
    main()
