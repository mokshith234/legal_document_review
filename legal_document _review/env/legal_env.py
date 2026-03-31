"""
env/legal_env.py
----------------
Phase 4 — Core Environment Engine

Implements the classic RL interface:
  reset() → Observation
  step(action) → StepResponse (also unpacks as tuple for test compatibility)
  state() → EnvironmentState
"""

from __future__ import annotations
import uuid
import time
import random
from typing import Any, Dict, List, Optional

from env.models import (
    Action, Observation, Reward, StepResponse, EnvironmentState, TaskDescriptor
)
from data.contracts import (
    CLASSIFICATION_SAMPLES,
    RISK_SAMPLES,
    REDLINE_SAMPLES,
    CLAUSE_TAXONOMY,
)
import graders.clause_classifier as classifier_grader
import graders.risk_spotter as risk_grader
import graders.contract_redliner as redline_grader


TASK_IDS = ["clause_classifier", "risk_spotter", "contract_redliner"]
MAX_STEPS = 5

TASK_DESCRIPTORS = [
    TaskDescriptor(
        id="clause_classifier",
        name="Clause Classifier",
        difficulty="easy",
        description=(
            "Given a contract clause, identify its type from the 8-label taxonomy. "
            "Submit action_type='classify' with content=<label>."
        ),
        action_schema={
            "action_type": "classify",
            "content": f"One of: {', '.join(CLAUSE_TAXONOMY)}",
            "metadata": "optional",
        },
        expected_score_range={"min": 0.0, "max": 1.0},
    ),
    TaskDescriptor(
        id="risk_spotter",
        name="Risk Spotter",
        difficulty="medium",
        description=(
            "Given a contract section, identify all high-risk clauses. "
            "Submit action_type='flag_risks' with content=<risk descriptions> "
            "or metadata={'risks': [...]}."
        ),
        action_schema={
            "action_type": "flag_risks",
            "content": "Free-text list of identified risks",
            "metadata": "optional: {'risks': ['risk 1', 'risk 2', ...]}",
        },
        expected_score_range={"min": 0.0, "max": 1.0},
    ),
    TaskDescriptor(
        id="contract_redliner",
        name="Contract Redliner",
        difficulty="hard",
        description=(
            "Given a contract and policy brief, propose specific edits. "
            "Submit action_type='redline' with content=<edits> "
            "or metadata={'edits': [{'section':..,'issue':..,'redline':..}]}."
        ),
        action_schema={
            "action_type": "redline",
            "content": "Free-text proposed edits",
            "metadata": "optional: {'edits': [{'section': ..., 'issue': ..., 'original': ..., 'redline': ...}]}",
        },
        expected_score_range={"min": 0.0, "max": 1.0},
    ),
]


class _TupleStepResponse(StepResponse):
    """
    StepResponse subclass that also supports 4-tuple unpacking:
        obs, reward, done, info = env.step(action)

    This satisfies both:
      - server.py  which accesses .observation, .reward, .done, .info
      - test_openenv.py which unpacks as (obs, reward, done, info)
    """
    def __iter__(self):
        yield self.observation
        yield self.reward
        yield self.done
        yield self.info


class LegalEnv:
    """
    The OpenEnv-compatible legal document review environment.

    Usage
    -----
    env = LegalEnv()
    obs = env.reset()
    obs, reward, done, info = env.step(Action(action_type="classify", content="indemnification"))
    state = env.state()
    """

    def __init__(self) -> None:
        self._session_id: str = str(uuid.uuid4())
        self._task_id: str = TASK_IDS[0]
        self._step_count: int = 0
        self._episode_done: bool = True
        self._cumulative_score: float = 0.0
        self._current_sample: Dict[str, Any] = {}
        self._current_doc_id: str = ""
        self._history: List[Dict[str, Any]] = []
        self._task_rotation_idx: int = 0

    # -----------------------------------------------------------------------
    # PUBLIC API
    # -----------------------------------------------------------------------

    def reset(self, task_id: Optional[str] = None, doc_id: Optional[str] = None) -> Observation:
        """
        Start a new episode. Returns the first Observation.
        """
        if task_id:
            if task_id not in TASK_IDS:
                raise ValueError(f"Unknown task_id '{task_id}'. Must be one of {TASK_IDS}")
            self._task_id = task_id
        else:
            self._task_id = TASK_IDS[self._task_rotation_idx % len(TASK_IDS)]
            self._task_rotation_idx += 1

        self._current_sample = self._pick_sample(self._task_id, doc_id)
        self._current_doc_id = self._current_sample.get("id", "unknown")

        # Reset episode state — step_count starts at 1 after reset (first step shown)
        self._step_count     = 1
        self._episode_done   = False
        self._cumulative_score = 0.0
        self._history        = []

        return self._build_observation()

    def step(self, action: Action) -> _TupleStepResponse:
        """
        Submit an action and receive (observation, reward, done, info).
        Also accessible as a StepResponse object via field names.
        """
        if self._episode_done:
            raise RuntimeError(
                "Episode is done. Call reset() to start a new episode."
            )

        # --- grade ----------------------------------------------------------
        reward = self._grade(action)

        # --- episode termination -------------------------------------------
        done = reward.done or self._step_count >= MAX_STEPS
        if done:
            reward.done = True
            self._episode_done = True

        self._cumulative_score += reward.score

        # --- log history ---------------------------------------------------
        self._history.append({
            "step": self._step_count,
            "action_type": action.action_type,
            "content_preview": action.content[:120],
            "score": reward.score,
            "feedback": reward.feedback,
            "timestamp": time.time(),
        })

        self._step_count += 1

        next_obs = self._build_observation()

        info = {
            "session_id": self._session_id,
            "cumulative_score": round(self._cumulative_score, 4),
            "steps_remaining": MAX_STEPS - self._step_count,
        }

        return _TupleStepResponse(
            observation=next_obs,
            reward=reward,
            done=done,
            info=info,
        )

    def state(self) -> EnvironmentState:
        """Return full internal state snapshot."""
        return EnvironmentState(
            session_id=self._session_id,
            task_id=self._task_id,
            step_count=self._step_count,
            max_steps=MAX_STEPS,
            episode_done=self._episode_done,
            cumulative_score=round(self._cumulative_score, 4),
            current_document_id=self._current_doc_id,
            history=list(self._history),
        )

    @staticmethod
    def tasks() -> List[TaskDescriptor]:
        """Return all available task descriptors."""
        return TASK_DESCRIPTORS

    # -----------------------------------------------------------------------
    # PRIVATE HELPERS
    # -----------------------------------------------------------------------

    def _grade(self, action: Action) -> Reward:
        """Dispatch to the correct grader based on current task."""
        if self._task_id == "clause_classifier":
            label = self._current_sample["label"]
            return classifier_grader.grade(action, label)
        elif self._task_id == "risk_spotter":
            risks = self._current_sample["ground_truth_risks"]
            return risk_grader.grade(action, risks)
        elif self._task_id == "contract_redliner":
            redlines = self._current_sample["ground_truth_redlines"]
            return redline_grader.grade(action, redlines)
        else:
            raise ValueError(f"No grader for task '{self._task_id}'")

    def _pick_sample(self, task_id: str, doc_id: Optional[str]) -> Dict[str, Any]:
        """Pick a document sample for the given task."""
        if task_id == "clause_classifier":
            pool = CLASSIFICATION_SAMPLES
        elif task_id == "risk_spotter":
            pool = RISK_SAMPLES
        elif task_id == "contract_redliner":
            pool = REDLINE_SAMPLES
        else:
            raise ValueError(f"Unknown task: {task_id}")

        if doc_id:
            for s in pool:
                if s["id"] == doc_id:
                    return s
            raise ValueError(f"Document '{doc_id}' not found in task '{task_id}'")

        return random.choice(pool)

    def _build_observation(self) -> Observation:
        """Construct the Observation from current state."""
        task_id = self._task_id
        sample  = self._current_sample

        if task_id == "clause_classifier":
            doc_text     = sample["clause"]
            instructions = (
                "Classify the following contract clause into exactly one of these "
                f"8 types: {', '.join(CLAUSE_TAXONOMY)}. "
                "Submit action_type='classify' and content=<label>."
            )
            context = {"difficulty": sample.get("difficulty", "unknown")}

        elif task_id == "risk_spotter":
            doc_text     = sample["contract_text"]
            instructions = (
                "Read the following contract section and identify ALL high-risk clauses. "
                "For each risk, describe the clause reference and why it is risky. "
                "Submit action_type='flag_risks'. Use metadata={'risks': [...]} for best scoring."
            )
            context = {
                "section_title": sample.get("section_title", ""),
                "difficulty": sample.get("difficulty", "unknown"),
                "hint": "Look for one-sided obligations, missing protections, and unusual terms.",
            }

        elif task_id == "contract_redliner":
            doc_text     = sample["contract_text"]
            instructions = (
                "Review the following contract against the policy brief in context. "
                "Propose specific edits to bring it into compliance. "
                "Submit action_type='redline'. Use metadata={'edits': [...]} for best scoring."
            )
            context = {
                "contract_title": sample.get("contract_title", ""),
                "policy_brief": sample.get("policy_brief", ""),
                "difficulty": sample.get("difficulty", "unknown"),
            }

        else:
            raise ValueError(f"Unknown task: {task_id}")

        return Observation(
            task_id=task_id,
            document_text=doc_text,
            instructions=instructions,
            context=context,
            step_count=self._step_count,
            max_steps=MAX_STEPS,
        )
