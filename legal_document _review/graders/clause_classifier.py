"""
graders/clause_classifier.py
-----------------------------
Phase 2 — Task 1 Grader: Clause Classifier

SCORING LOGIC:
  1.0  — exact match (agent label == ground truth label)
  0.5  — near-miss (agent label is in the near-miss list for this label)
  0.0  — wrong answer
  -0.1 — skip penalty (agent sent action_type="skip")

Normalization: labels are lowercased and stripped before comparison
so "  Indemnification  " still scores 1.0.

WHY PARTIAL CREDIT?
  Indemnification vs LimitationOfLiability are genuinely confusing.
  Penalizing those the same as a random wrong answer would make the
  reward signal noisy. Near-miss credit is the industry practice.
"""

from __future__ import annotations
from typing import Tuple
from data.contracts import CLAUSE_NEAR_MISSES, CLAUSE_TAXONOMY
from env.models import Action, Reward


SKIP_PENALTY = -0.1   # applied when agent skips intentionally
EXACT_SCORE  =  1.0
NEAR_SCORE   =  0.5
WRONG_SCORE  =  0.0


def grade(action: Action, ground_truth_label: str) -> Reward:
    """
    Grade a clause classification action.

    Parameters
    ----------
    action : Action
        The agent's submitted action. action.content should be one of
        the 8 labels in CLAUSE_TAXONOMY (case-insensitive).
    ground_truth_label : str
        The correct label from contracts.CLASSIFICATION_SAMPLES.

    Returns
    -------
    Reward
        Structured reward with score, breakdown, done flag, and feedback.
    """
    # --- skip ---------------------------------------------------------------
    if action.action_type == "skip":
        return Reward(
            score=0.0,
            breakdown={"base": 0.0, "skip_penalty": SKIP_PENALTY},
            done=True,
            feedback=(
                "You skipped this clause. Skipping is penalized — "
                f"the correct label was '{ground_truth_label}'."
            ),
        )

    # --- normalise ----------------------------------------------------------
    predicted = action.content.strip().lower().replace(" ", "_").replace("-", "_")
    truth      = ground_truth_label.strip().lower()

    # --- validate predicted label is in taxonomy ----------------------------
    if predicted not in CLAUSE_TAXONOMY:
        close = _closest_taxonomy_label(predicted)
        return Reward(
            score=WRONG_SCORE,
            breakdown={"base": 0.0, "invalid_label_penalty": 0.0},
            done=True,
            feedback=(
                f"'{predicted}' is not a valid taxonomy label. "
                f"Did you mean '{close}'? "
                f"Valid labels: {', '.join(CLAUSE_TAXONOMY)}. "
                f"Correct answer was '{truth}'."
            ),
        )

    # --- exact match --------------------------------------------------------
    if predicted == truth:
        return Reward(
            score=EXACT_SCORE,
            breakdown={"exact_match": 1.0},
            done=True,
            feedback=f"✓ Correct! The clause is '{truth}'.",
        )

    # --- near-miss ----------------------------------------------------------
    near_misses = CLAUSE_NEAR_MISSES.get(truth, [])
    if predicted in near_misses:
        return Reward(
            score=NEAR_SCORE,
            breakdown={"near_miss": NEAR_SCORE},
            done=True,
            feedback=(
                f"Partial credit. You predicted '{predicted}' but the correct "
                f"label is '{truth}'. These two are commonly confused — "
                f"review the distinction carefully."
            ),
        )

    # --- wrong --------------------------------------------------------------
    return Reward(
        score=WRONG_SCORE,
        breakdown={"wrong": 0.0},
        done=True,
        feedback=(
            f"✗ Incorrect. You predicted '{predicted}', "
            f"but the correct label is '{truth}'."
        ),
    )


def _closest_taxonomy_label(predicted: str) -> str:
    """
    Simple substring heuristic to suggest a valid label when the agent
    submits something close but not exact (e.g. 'indemnify' → 'indemnification').
    """
    for label in CLAUSE_TAXONOMY:
        if predicted[:6] in label or label[:6] in predicted:
            return label
    return CLAUSE_TAXONOMY[0]  # fallback
