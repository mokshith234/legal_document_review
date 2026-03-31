"""
graders/risk_spotter.py
-----------------------
Phase 3 — Task 2 Grader: Risk Spotter

SCORING LOGIC:
  Score = weighted F1 of (precision, recall) over risks found.

  precision = matched_risks / agent_risks_submitted
  recall    = matched_risks / total_ground_truth_risks
  f1        = 2 * (P * R) / (P + R)

  Severity weighting:
    critical → weight 3
    high     → weight 2
    medium   → weight 1

  Hallucination penalty: -0.05 per fabricated risk that has no
    semantic overlap with any ground truth risk.

  Skip penalty: score floored to 0.0, feedback notes the skip.

MATCHING:
  We use keyword overlap (Jaccard on word tokens) rather than
  exact string match — agents phrase risks in their own words.
  Threshold ≥ 0.15 Jaccard counts as a match.
"""

from __future__ import annotations
import re
from typing import Any, Dict, List, Tuple
from env.models import Action, Reward


HALLUCINATION_PENALTY = 0.05
SKIP_PENALTY          = 0.2
JACCARD_THRESHOLD     = 0.15

SEVERITY_WEIGHTS = {"critical": 3.0, "high": 2.0, "medium": 1.0, "low": 0.5}


def grade(action: Action, ground_truth_risks: List[Dict[str, Any]]) -> Reward:
    """
    Grade a risk-spotting action.

    Parameters
    ----------
    action : Action
        action.content  — free-text listing of risks (one per line or sentence).
        action.metadata — optional {"risks": ["risk text 1", "risk text 2"]}
                          If provided, metadata.risks is used for precision scoring.
    ground_truth_risks : list
        From contracts.RISK_SAMPLES[n]["ground_truth_risks"].

    Returns
    -------
    Reward
    """
    if action.action_type == "skip":
        return Reward(
            score=0.0,
            breakdown={"skip_penalty": -SKIP_PENALTY},
            done=True,
            feedback=(
                f"You skipped risk spotting. There were "
                f"{len(ground_truth_risks)} risks to find."
            ),
        )

    # --- parse agent risks --------------------------------------------------
    agent_risks = _parse_agent_risks(action)
    if not agent_risks:
        return Reward(
            score=0.0,
            breakdown={"no_risks_submitted": 0.0},
            done=True,
            feedback="No risks were identified in your response.",
        )

    # --- match agent risks against ground truth -----------------------------
    total_gt_weight = sum(
        SEVERITY_WEIGHTS.get(r.get("severity", "medium"), 1.0)
        for r in ground_truth_risks
    )

    matched_weight  = 0.0
    matched_count   = 0
    unmatched_agent = []
    matched_gt_ids  = set()

    for a_risk in agent_risks:
        best_score, best_gt = _best_match(a_risk, ground_truth_risks, matched_gt_ids)
        if best_score >= JACCARD_THRESHOLD and best_gt is not None:
            weight = SEVERITY_WEIGHTS.get(best_gt.get("severity", "medium"), 1.0)
            matched_weight += weight
            matched_count  += 1
            matched_gt_ids.add(best_gt["risk_id"])
        else:
            unmatched_agent.append(a_risk)

    # --- precision / recall / F1 ------------------------------------------
    precision = matched_count / len(agent_risks) if agent_risks else 0.0
    recall    = matched_weight / total_gt_weight if total_gt_weight > 0 else 0.0
    f1        = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0 else 0.0
    )

    # --- hallucination penalty ----------------------------------------------
    hallucination_count   = len(unmatched_agent)
    hallucination_penalty = min(hallucination_count * HALLUCINATION_PENALTY, 0.3)
    final_score           = max(0.0, f1 - hallucination_penalty)

    # --- feedback -----------------------------------------------------------
    missed = [r for r in ground_truth_risks if r["risk_id"] not in matched_gt_ids]
    feedback_parts = [
        f"You identified {matched_count}/{len(ground_truth_risks)} risks.",
        f"Precision: {precision:.2f}  Recall: {recall:.2f}  F1: {f1:.2f}.",
    ]
    if missed:
        feedback_parts.append(
            "Missed risks: " + "; ".join(r["risk"] for r in missed[:3])
            + ("..." if len(missed) > 3 else "")
        )
    if hallucination_count:
        feedback_parts.append(
            f"{hallucination_count} submitted risk(s) did not match any ground-truth risk "
            f"(hallucination penalty: -{hallucination_penalty:.2f})."
        )

    return Reward(
        score=round(final_score, 4),
        breakdown={
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "hallucination_penalty": -round(hallucination_penalty, 4),
            "final": round(final_score, 4),
        },
        done=True,
        feedback=" ".join(feedback_parts),
    )


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _parse_agent_risks(action: Action) -> List[str]:
    """
    Extract a list of risk strings from the action.
    Priority: metadata["risks"] list → content split by newlines/bullets.
    """
    if action.metadata and "risks" in action.metadata:
        raw = action.metadata["risks"]
        if isinstance(raw, list):
            return [str(r).strip() for r in raw if str(r).strip()]

    # Split content on newlines or bullet-like delimiters
    lines = re.split(r"\n|•|-\s+|\d+\.\s+", action.content)
    return [l.strip() for l in lines if len(l.strip()) > 10]


def _tokenize(text: str) -> set:
    """Lowercase word tokens, removing stopwords."""
    stopwords = {"the", "a", "an", "is", "are", "of", "to", "in", "and",
                 "or", "for", "with", "that", "this", "it", "its", "any",
                 "all", "be", "by", "at", "no", "not", "on", "as", "such"}
    tokens = set(re.findall(r"\b[a-z]{3,}\b", text.lower()))
    return tokens - stopwords


def _jaccard(a: str, b: str) -> float:
    ta, tb = _tokenize(a), _tokenize(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _best_match(
    agent_risk: str,
    ground_truth_risks: List[Dict[str, Any]],
    already_matched: set,
) -> Tuple[float, Any]:
    best_score = 0.0
    best_gt    = None
    for gt in ground_truth_risks:
        if gt["risk_id"] in already_matched:
            continue
        score = _jaccard(agent_risk, gt["risk"])
        if score > best_score:
            best_score = score
            best_gt    = gt
    return best_score, best_gt
