"""
env/models.py
-------------
Typed Pydantic models for the OpenEnv spec.

WHY THIS FILE EXISTS:
  The OpenEnv spec requires typed Observation, Action, and Reward models.
  Every endpoint — step(), reset(), state() — speaks these types.
  Pydantic gives us automatic validation + JSON serialization for free.
  This is the contract between the environment and any agent.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Literal
from pydantic import BaseModel, Field
import time


# ---------------------------------------------------------------------------
# ACTION — what the agent sends IN
# ---------------------------------------------------------------------------

class Action(BaseModel):
    """
    The structured action an agent submits to the environment.
    
    action_type: What kind of response this is
      - classify  → agent is labeling a clause type
      - flag_risks → agent is listing risky clauses
      - redline    → agent is proposing contract edits
      - skip       → agent explicitly passes (penalized)
    
    content: Free-text response from the agent (the actual answer)
    
    metadata: Optional structured data. For flag_risks, agent can pass
      {"risks": ["clause 3 is one-sided", "no liability cap"]}
      For redline, agent can pass {"edits": [{"original": "...", "proposed": "..."}]}
    """
    action_type: Literal["classify", "flag_risks", "redline", "skip"]
    content: str = Field(..., min_length=1, description="Agent's response text")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# OBSERVATION — what the agent sees OUT
# ---------------------------------------------------------------------------

class Observation(BaseModel):
    """
    What the environment returns to the agent after reset() or step().
    
    task_id: Which of the 3 tasks is active
    document_text: The contract text the agent must work with
    instructions: Clear natural-language task instructions
    context: Extra task-specific data (e.g. policy brief for redlining)
    step_count: How many steps taken so far this episode
    max_steps: Step limit — agent must finish before this
    """
    task_id: str
    document_text: str
    instructions: str
    context: Dict[str, Any] = Field(default_factory=dict)
    step_count: int = 0
    max_steps: int = 5


# ---------------------------------------------------------------------------
# REWARD — scoring signal returned after step()
# ---------------------------------------------------------------------------

class Reward(BaseModel):
    """
    Structured reward returned alongside every step.
    
    score: Float 0.0–1.0. The shaped reward for this step.
    breakdown: Dict explaining how the score was computed.
      e.g. {"precision": 0.8, "recall": 0.6, "hallucination_penalty": -0.1}
    done: Whether the episode is complete.
    feedback: Human-readable explanation so agents can learn from it.
    """
    score: float = Field(..., ge=0.0, le=1.0)
    breakdown: Dict[str, float] = Field(default_factory=dict)
    done: bool = False
    feedback: str = ""


# ---------------------------------------------------------------------------
# STEP RESPONSE — the full return from step()
# ---------------------------------------------------------------------------

class StepResponse(BaseModel):
    """
    The complete response from a step() call.
    Bundles observation + reward + episode metadata.
    """
    observation: Observation
    reward: Reward
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# ENVIRONMENT STATE — what state() returns
# ---------------------------------------------------------------------------

class EnvironmentState(BaseModel):
    """
    Full internal state of the environment.
    Useful for debugging, checkpointing, and the /state endpoint.
    """
    session_id: str
    task_id: str
    step_count: int
    max_steps: int
    episode_done: bool
    cumulative_score: float
    current_document_id: str
    created_at: float = Field(default_factory=time.time)
    history: List[Dict[str, Any]] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# TASK DESCRIPTOR — returned by /tasks endpoint
# ---------------------------------------------------------------------------

class TaskDescriptor(BaseModel):
    """
    Metadata about a task, returned by GET /tasks.
    Tells agents what the task is and what action schema to use.
    """
    id: str
    name: str
    difficulty: Literal["easy", "medium", "hard"]
    description: str
    action_schema: Dict[str, Any]
    expected_score_range: Dict[str, float]
