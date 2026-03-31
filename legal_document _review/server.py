"""
server.py
---------
Phase 5 — FastAPI Server

Exposes the OpenEnv spec endpoints defined in openenv.yaml:
  POST /reset    → start new episode, returns Observation
  POST /step     → submit action, returns StepResponse
  GET  /state    → current environment state
  GET  /tasks    → list all available tasks + schemas
  POST /grader   → direct grader access (no episode needed)
  GET  /baseline → sample baseline agent response for a task

The server is stateful per-session via a session_id returned in responses.
Multiple agents can run in parallel — each gets its own LegalEnv instance
stored in the SESSION_REGISTRY dict.

SESSION MANAGEMENT:
  - Sessions are created on /reset.
  - Session ID is in every response's info dict and response headers.
  - Stale sessions (> 1 hour) are pruned on each /reset call.
  - Pass session_id in the request body to resume a session.

RUNNING:
  uvicorn server:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations
import time
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from env.legal_env import LegalEnv, TASK_IDS
from env.models import Action, Observation, StepResponse, EnvironmentState, TaskDescriptor

app = FastAPI(
    title="Legal Document Review OpenEnv",
    description="AI benchmark environment for contract review tasks.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Session registry
# ---------------------------------------------------------------------------

SESSION_REGISTRY: Dict[str, Dict[str, Any]] = {}
SESSION_TTL = 3600  # 1 hour


def _get_or_create_session(session_id: Optional[str]) -> tuple[LegalEnv, str]:
    """Return (env, session_id). Creates new session if none provided."""
    _prune_stale_sessions()

    if session_id and session_id in SESSION_REGISTRY:
        entry = SESSION_REGISTRY[session_id]
        entry["last_used"] = time.time()
        return entry["env"], session_id

    # New session
    env = LegalEnv()
    sid = env.state().session_id
    SESSION_REGISTRY[sid] = {"env": env, "created_at": time.time(), "last_used": time.time()}
    return env, sid


def _prune_stale_sessions() -> None:
    now = time.time()
    stale = [sid for sid, entry in SESSION_REGISTRY.items()
             if now - entry["last_used"] > SESSION_TTL]
    for sid in stale:
        del SESSION_REGISTRY[sid]


# ---------------------------------------------------------------------------
# Request/response schemas for endpoints
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    task_id: Optional[str] = None
    doc_id: Optional[str] = None
    session_id: Optional[str] = None


class StepRequest(BaseModel):
    action: Action
    session_id: Optional[str] = None


class GraderRequest(BaseModel):
    task_id: str
    action: Action
    doc_id: Optional[str] = None


class ResetResponse(BaseModel):
    observation: Observation
    session_id: str
    message: str = "Episode started. Call POST /step to submit actions."


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/", tags=["meta"])
def root():
    return {
        "name": "Legal Document Review OpenEnv",
        "version": "1.0.0",
        "tasks": TASK_IDS,
        "endpoints": ["/reset", "/step", "/state", "/tasks", "/grader", "/baseline"],
        "docs": "/docs",
    }


@app.post("/reset", response_model=ResetResponse, tags=["env"])
def reset(req: ResetRequest):
    """
    Start a new episode. Returns the first Observation and a session_id.

    Optionally pin to a specific task_id and/or doc_id.
    Pass an existing session_id to reuse the same LegalEnv instance.
    """
    env, sid = _get_or_create_session(req.session_id)
    try:
        obs = env.reset(task_id=req.task_id, doc_id=req.doc_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return ResetResponse(observation=obs, session_id=sid)


@app.post("/step", response_model=StepResponse, tags=["env"])
def step(req: StepRequest):
    """
    Submit an action and receive the next Observation + Reward.

    Requires an active session (call /reset first).
    The response.done flag indicates whether the episode is complete.
    """
    if not req.session_id or req.session_id not in SESSION_REGISTRY:
        raise HTTPException(
            status_code=400,
            detail="Invalid or missing session_id. Call POST /reset first.",
        )

    env = SESSION_REGISTRY[req.session_id]["env"]
    SESSION_REGISTRY[req.session_id]["last_used"] = time.time()

    try:
        response = env.step(req.action)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return response


@app.get("/state", response_model=EnvironmentState, tags=["env"])
def state(session_id: str):
    """
    Return the full internal environment state for debugging/checkpointing.
    """
    if session_id not in SESSION_REGISTRY:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
    env = SESSION_REGISTRY[session_id]["env"]
    return env.state()


@app.get("/tasks", response_model=List[TaskDescriptor], tags=["meta"])
def tasks():
    """List all available tasks with descriptions and action schemas."""
    return LegalEnv.tasks()


@app.post("/grader", tags=["utility"])
def grader(req: GraderRequest):
    """
    Direct grader access — grade an action without starting a full episode.
    Useful for development, testing, and prompt engineering.
    """
    # Spin up a fresh env just to get a sample + grader routing
    env = LegalEnv()
    try:
        env.reset(task_id=req.task_id, doc_id=req.doc_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    try:
        result = env._grade(req.action)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Grader error: {e}")

    return {
        "task_id": req.task_id,
        "score": result.score,
        "breakdown": result.breakdown,
        "feedback": result.feedback,
        "done": result.done,
    }


@app.get("/baseline", tags=["utility"])
def baseline(task_id: Optional[str] = None):
    """
    Return a sample baseline agent response for a given task.
    Useful for testing the environment and understanding scoring.
    """
    baselines = {
        "clause_classifier": {
            "task_id": "clause_classifier",
            "description": "Simple keyword-matching baseline",
            "example_action": {
                "action_type": "classify",
                "content": "indemnification",
                "metadata": {},
            },
            "expected_score_range": "0.0 – 1.0 (exact: 1.0, near-miss: 0.5, wrong: 0.0)",
        },
        "risk_spotter": {
            "task_id": "risk_spotter",
            "description": "Enumerate risks by section",
            "example_action": {
                "action_type": "flag_risks",
                "content": "Section 8: One-sided indemnification.\nSection 9: $100 liability cap is unreasonably low.",
                "metadata": {
                    "risks": [
                        "One-sided indemnification — Client bears all burden",
                        "Liability cap of $100 is dangerously low",
                    ]
                },
            },
            "expected_score_range": "0.0 – 1.0 (F1 of weighted precision/recall)",
        },
        "contract_redliner": {
            "task_id": "contract_redliner",
            "description": "Propose specific contract edits",
            "example_action": {
                "action_type": "redline",
                "content": "Change payment terms from 7 days to 30 days.",
                "metadata": {
                    "edits": [
                        {
                            "section": "Section 2 - Payment",
                            "issue": "Net 7 is too aggressive",
                            "original": "within 7 days of receipt",
                            "redline": "within thirty (30) days of receipt",
                        }
                    ]
                },
            },
            "expected_score_range": "0.0 – 1.0 (mean weighted edit quality)",
        },
    }

    if task_id:
        if task_id not in baselines:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown task '{task_id}'. Choose from: {list(baselines.keys())}",
            )
        return baselines[task_id]

    return {"all_baselines": list(baselines.values())}


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.get("/health", tags=["meta"])
def health():
    return {
        "status": "ok",
        "active_sessions": len(SESSION_REGISTRY),
        "timestamp": time.time(),
    }
