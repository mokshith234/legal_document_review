"""
server.py
---------
Legal Document Review OpenEnv — FastAPI Server

Endpoints:
  POST /reset      → start new episode, returns Observation
  POST /step       → submit action, returns StepResponse
  GET  /state      → current environment state
  GET  /tasks      → list all available tasks + schemas
  POST /grader     → direct grader access (no episode needed)
  GET  /baseline   → sample baseline agent response for a task
  POST /upload     → upload a custom document for any task
  GET  /documents  → list all uploaded custom documents
  GET  /health     → health check
"""

from __future__ import annotations
import time
import uuid
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from env.legal_env import LegalEnv, TASK_IDS
from env.models import Action, Observation, StepResponse, EnvironmentState, TaskDescriptor

app = FastAPI(
    title="Legal Document Review OpenEnv",
    description="AI benchmark environment for contract review tasks. Upload your own documents via POST /upload.",
    version="1.1.0",
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

# ---------------------------------------------------------------------------
# Custom document registry
# In-memory store: { doc_id -> sample dict }
# ---------------------------------------------------------------------------

CUSTOM_DOCS: Dict[str, Dict[str, Any]] = {}


def _get_or_create_session(session_id: Optional[str]) -> tuple[LegalEnv, str]:
    """Return (env, session_id). Creates new session if none provided."""
    _prune_stale_sessions()

    if session_id and session_id in SESSION_REGISTRY:
        entry = SESSION_REGISTRY[session_id]
        entry["last_used"] = time.time()
        return entry["env"], session_id

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
# Request / response schemas
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
    done: bool = False
    message: str = "Episode started. Call POST /step to submit actions."


# ---------------------------------------------------------------------------
# Upload schemas
# ---------------------------------------------------------------------------

class UploadDocumentRequest(BaseModel):
    task_id: str = Field(
        ...,
        description="Which task this document is for: clause_classifier | risk_spotter | contract_redliner"
    )
    document_text: str = Field(
        ...,
        min_length=50,
        description="The full contract or clause text to analyse"
    )
    title: Optional[str] = Field(
        default=None,
        description="Optional title for the document"
    )
    # Optional ground truth — if provided, scoring uses it; otherwise scores on AI quality
    ground_truth: Optional[Dict[str, Any]] = Field(
        default=None,
        description=(
            "Optional ground truth for scoring. "
            "clause_classifier: {'label': 'indemnification'} | "
            "risk_spotter: {'risks': ['risk 1', 'risk 2']} | "
            "contract_redliner: {'redlines': [...], 'policy_brief': '...'}"
        )
    )


class UploadDocumentResponse(BaseModel):
    doc_id: str
    task_id: str
    title: str
    message: str
    usage: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/", tags=["meta"])
def root():
    return {
        "name": "Legal Document Review OpenEnv",
        "version": "1.1.0",
        "tasks": TASK_IDS,
        "endpoints": ["/reset", "/step", "/state", "/tasks", "/grader", "/baseline", "/upload", "/documents"],
        "docs": "/docs",
    }


@app.post("/reset", response_model=ResetResponse, tags=["env"])
def reset(req: ResetRequest):
    """
    Start a new episode. Returns the first Observation and a session_id.

    To use a custom uploaded document, pass its doc_id here.
    """
    env, sid = _get_or_create_session(req.session_id)

    # If doc_id refers to a custom uploaded doc, inject it into the env
    if req.doc_id and req.doc_id in CUSTOM_DOCS:
        custom = CUSTOM_DOCS[req.doc_id]
        try:
            obs = env.reset_with_custom_doc(
                task_id=custom["task_id"],
                sample=custom["sample"],
            )
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
        return ResetResponse(observation=obs, session_id=sid, done=False)

    # Default: use built-in documents
    try:
        obs = env.reset(task_id=req.task_id, doc_id=req.doc_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return ResetResponse(observation=obs, session_id=sid, done=False)


@app.post("/step", response_model=StepResponse, tags=["env"])
def step(req: StepRequest):
    """
    Submit an action and receive the next Observation + Reward.
    Requires an active session (call /reset first).
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
    """Return full internal environment state for debugging/checkpointing."""
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
    """
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


# ---------------------------------------------------------------------------
# Upload endpoint
# ---------------------------------------------------------------------------

@app.post("/upload", response_model=UploadDocumentResponse, tags=["documents"])
def upload_document(req: UploadDocumentRequest):
    """
    Upload a custom document to use in the environment.

    After uploading, use the returned doc_id in POST /reset:
        POST /reset  {"task_id": "risk_spotter", "doc_id": "<returned doc_id>"}

    Supported task_ids: clause_classifier | risk_spotter | contract_redliner

    Ground truth is optional:
    - If provided → scoring compares your AI's answer against it (accurate scoring)
    - If not provided → scoring uses heuristic analysis (approximate scoring)

    Example for clause_classifier:
        {
          "task_id": "clause_classifier",
          "document_text": "Either party may terminate this agreement with 30 days notice.",
          "title": "My Termination Clause",
          "ground_truth": {"label": "termination"}
        }

    Example for risk_spotter:
        {
          "task_id": "risk_spotter",
          "document_text": "Client shall indemnify Vendor for all losses...",
          "title": "My Indemnification Section",
          "ground_truth": {"risks": ["one-sided indemnification", "no liability cap"]}
        }

    Example for contract_redliner:
        {
          "task_id": "contract_redliner",
          "document_text": "Payment due within 7 days. Vendor not liable for any damages.",
          "title": "My Service Agreement",
          "ground_truth": {
            "policy_brief": "Payment terms must be Net 30. Liability cap must be 2x contract value.",
            "redlines": [
              {"section": "Payment", "issue": "Too short", "original": "7 days", "redline": "30 days"}
            ]
          }
        }
    """
    if req.task_id not in TASK_IDS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid task_id '{req.task_id}'. Must be one of: {TASK_IDS}"
        )

    # Generate a unique doc_id
    doc_id = f"custom_{req.task_id[:4]}_{uuid.uuid4().hex[:8]}"
    title  = req.title or f"Custom {req.task_id.replace('_', ' ').title()} Document"

    # Build the internal sample dict based on task type
    gt = req.ground_truth or {}

    if req.task_id == "clause_classifier":
        sample = {
            "id":         doc_id,
            "clause":     req.document_text,
            "label":      gt.get("label", "unknown"),
            "difficulty": "custom",
        }

    elif req.task_id == "risk_spotter":
        sample = {
            "id":                   doc_id,
            "contract_text":        req.document_text,
            "section_title":        title,
            "ground_truth_risks":   gt.get("risks", []),
            "difficulty":           "custom",
        }

    elif req.task_id == "contract_redliner":
        sample = {
            "id":                     doc_id,
            "contract_text":          req.document_text,
            "contract_title":         title,
            "policy_brief":           gt.get("policy_brief", "Review this contract for standard commercial terms."),
            "ground_truth_redlines":  gt.get("redlines", []),
            "difficulty":             "custom",
        }

    CUSTOM_DOCS[doc_id] = {
        "task_id":    req.task_id,
        "title":      title,
        "sample":     sample,
        "uploaded_at": time.time(),
        "has_ground_truth": bool(gt),
    }

    return UploadDocumentResponse(
        doc_id=doc_id,
        task_id=req.task_id,
        title=title,
        message=f"Document uploaded successfully. Use doc_id '{doc_id}' in POST /reset.",
        usage=f'POST /reset  {{"task_id": "{req.task_id}", "doc_id": "{doc_id}"}}',
    )


@app.get("/documents", tags=["documents"])
def list_documents():
    """
    List all custom uploaded documents currently in memory.
    Note: documents are cleared when the server restarts.
    """
    if not CUSTOM_DOCS:
        return {
            "custom_documents": [],
            "count": 0,
            "note": "No custom documents uploaded yet. Use POST /upload to add your own."
        }

    docs = [
        {
            "doc_id":           doc_id,
            "task_id":          info["task_id"],
            "title":            info["title"],
            "has_ground_truth": info["has_ground_truth"],
            "uploaded_at":      info["uploaded_at"],
        }
        for doc_id, info in CUSTOM_DOCS.items()
    ]
    return {
        "custom_documents": docs,
        "count": len(docs),
    }


@app.get("/baseline", tags=["utility"])
def baseline(task_id: Optional[str] = None):
    """Return a sample baseline agent response for a given task."""
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


@app.get("/health", tags=["meta"])
def health():
    return {
        "status": "ok",
        "active_sessions": len(SESSION_REGISTRY),
        "custom_documents": len(CUSTOM_DOCS),
        "timestamp": time.time(),
    }
