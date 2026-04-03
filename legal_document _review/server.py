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

SESSION_REGISTRY: Dict[str, Dict[str, Any]] = {}
SESSION_TTL = 3600

CUSTOM_DOCS: Dict[str, Dict[str, Any]] = {}


def _get_or_create_session(session_id: Optional[str]) -> tuple[LegalEnv, str]:
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


class UploadDocumentRequest(BaseModel):
    task_id: str = Field(..., description="clause_classifier | risk_spotter | contract_redliner")
    document_text: str = Field(..., min_length=50, description="The full contract or clause text")
    title: Optional[str] = Field(default=None, description="Optional title for the document")
    ground_truth: Optional[Dict[str, Any]] = Field(default=None, description=(
        "Optional ground truth. "
        "clause_classifier: {'label': 'indemnification'} | "
        "risk_spotter: {'risks': ['risk 1', 'risk 2']} | "
        "contract_redliner: {'redlines': [...], 'policy_brief': '...'}"
    ))


class UploadDocumentResponse(BaseModel):
    doc_id: str
    task_id: str
    title: str
    message: str
    usage: str


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
    """Start a new episode. Pass doc_id from /upload to use your own document."""
    env, sid = _get_or_create_session(req.session_id)
    if req.doc_id and req.doc_id in CUSTOM_DOCS:
        custom = CUSTOM_DOCS[req.doc_id]
        try:
            obs = env.reset_with_custom_doc(task_id=custom["task_id"], sample=custom["sample"])
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
        return ResetResponse(observation=obs, session_id=sid)
    try:
        obs = env.reset(task_id=req.task_id, doc_id=req.doc_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return ResetResponse(observation=obs, session_id=sid)


@app.post("/step", response_model=StepResponse, tags=["env"])
def step(req: StepRequest):
    """Submit an action and receive the next Observation + Reward."""
    if not req.session_id or req.session_id not in SESSION_REGISTRY:
        raise HTTPException(status_code=400, detail="Invalid or missing session_id. Call POST /reset first.")
    env = SESSION_REGISTRY[req.session_id]["env"]
    SESSION_REGISTRY[req.session_id]["last_used"] = time.time()
    try:
        response = env.step(req.action)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return response


@app.get("/state", response_model=EnvironmentState, tags=["env"])
def state(session_id: str):
    """Return full internal environment state."""
    if session_id not in SESSION_REGISTRY:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
    return SESSION_REGISTRY[session_id]["env"].state()


@app.get("/tasks", response_model=List[TaskDescriptor], tags=["meta"])
def tasks():
    """List all available tasks."""
    return LegalEnv.tasks()


@app.post("/grader", tags=["utility"])
def grader(req: GraderRequest):
    """Direct grader access without starting a full episode."""
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


@app.post("/upload", response_model=UploadDocumentResponse, tags=["documents"])
def upload_document(req: UploadDocumentRequest):
    """
    Upload your own contract document to test against.

    After uploading, use the returned doc_id in POST /reset:
        {"task_id": "risk_spotter", "doc_id": "<returned doc_id>"}

    Example body:
        {
          "task_id": "risk_spotter",
          "document_text": "Client shall indemnify Vendor for all losses without limit...",
          "title": "My Contract",
          "ground_truth": {"risks": ["unlimited indemnification", "no liability cap"]}
        }
    """
    if req.task_id not in TASK_IDS:
        raise HTTPException(status_code=400, detail=f"Invalid task_id. Must be one of: {TASK_IDS}")

    doc_id = f"custom_{req.task_id[:4]}_{uuid.uuid4().hex[:8]}"
    title  = req.title or f"Custom {req.task_id.replace('_', ' ').title()} Document"
    gt     = req.ground_truth or {}

    if req.task_id == "clause_classifier":
        sample = {
            "id": doc_id,
            "clause": req.document_text,
            "label": gt.get("label", "unknown"),
            "difficulty": "custom",
        }
    elif req.task_id == "risk_spotter":
        sample = {
            "id": doc_id,
            "contract_text": req.document_text,
            "section_title": title,
            "ground_truth_risks": gt.get("risks", []),
            "difficulty": "custom",
        }
    elif req.task_id == "contract_redliner":
        sample = {
            "id": doc_id,
            "contract_text": req.document_text,
            "contract_title": title,
            "policy_brief": gt.get("policy_brief", "Review this contract for standard commercial terms."),
            "ground_truth_redlines": gt.get("redlines", []),
            "difficulty": "custom",
        }

    CUSTOM_DOCS[doc_id] = {
        "task_id": req.task_id,
        "title": title,
        "sample": sample,
        "uploaded_at": time.time(),
        "has_ground_truth": bool(gt),
    }

    return UploadDocumentResponse(
        doc_id=doc_id,
        task_id=req.task_id,
        title=title,
        message=f"Document uploaded. Use doc_id '{doc_id}' in POST /reset.",
        usage=f'POST /reset  {{"task_id": "{req.task_id}", "doc_id": "{doc_id}"}}',
    )


@app.get("/documents", tags=["documents"])
def list_documents():
    """List all custom uploaded documents in memory."""
    if not CUSTOM_DOCS:
        return {"custom_documents": [], "count": 0, "note": "No documents uploaded yet. Use POST /upload."}
    docs = [
        {"doc_id": doc_id, "task_id": info["task_id"], "title": info["title"],
         "has_ground_truth": info["has_ground_truth"], "uploaded_at": info["uploaded_at"]}
        for doc_id, info in CUSTOM_DOCS.items()
    ]
    return {"custom_documents": docs, "count": len(docs)}


@app.get("/baseline", tags=["utility"])
def baseline(task_id: Optional[str] = None):
    """Return a sample baseline agent response for a given task."""
    baselines = {
        "clause_classifier": {
            "task_id": "clause_classifier",
            "example_action": {"action_type": "classify", "content": "indemnification", "metadata": {}},
            "expected_score_range": "0.0 – 1.0",
        },
        "risk_spotter": {
            "task_id": "risk_spotter",
            "example_action": {
                "action_type": "flag_risks",
                "content": "Section 8: One-sided indemnification.\nSection 9: $100 liability cap is unreasonably low.",
                "metadata": {"risks": ["One-sided indemnification", "Liability cap of $100 is dangerously low"]},
            },
            "expected_score_range": "0.0 – 1.0",
        },
        "contract_redliner": {
            "task_id": "contract_redliner",
            "example_action": {
                "action_type": "redline",
                "content": "Change payment terms from 7 days to 30 days.",
                "metadata": {"edits": [{"section": "Section 2", "issue": "Net 7 too aggressive",
                                         "original": "within 7 days", "redline": "within thirty (30) days"}]},
            },
            "expected_score_range": "0.0 – 1.0",
        },
    }
    if task_id:
        if task_id not in baselines:
            raise HTTPException(status_code=400, detail=f"Unknown task '{task_id}'. Choose from: {list(baselines.keys())}")
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
