"""
graders/contract_redliner.py
-----------------------------
Phase 3 — Task 3 Grader: Contract Redliner

SCORING LOGIC:
  Each ground-truth redline has three components to match:
    1. Section identification  (correct section cited?)     → 0.2 weight
    2. Issue description       (did agent identify problem?) → 0.3 weight
    3. Proposed redline text   (is the fix directionally right?) → 0.5 weight

  Per-redline score = weighted average of the three components.
  Final score = mean over all ground-truth redlines.

  Hallucination penalty: -0.05 per proposed edit that invents a
    problem not present in the contract or policy brief.

  Coverage bonus: +0.05 if agent found ALL redlines (rare).

MATCHING uses Jaccard overlap on word tokens, same as risk_spotter.
  Thresholds differ per component — issue matching is lenient (0.1),
  redline text matching is stricter (0.2).
"""

from __future__ import annotations
import re
from typing import Any, Dict, List, Tuple
from env.models import Action, Reward


HALLUCINATION_PENALTY = 0.05
COVERAGE_BONUS        = 0.05

WEIGHTS = {"section": 0.2, "issue": 0.3, "redline": 0.5}
THRESHOLDS = {"section": 0.1, "issue": 0.10, "redline": 0.20}


def grade(action: Action, ground_truth_redlines: List[Dict[str, Any]]) -> Reward:
    """
    Grade a contract redlining action.

    Parameters
    ----------
    action : Action
        action.content  — free text of proposed edits.
        action.metadata — optional {"edits": [{"section": ..., "issue": ...,
                            "original": ..., "redline": ...}]}
    ground_truth_redlines : list
        From contracts.REDLINE_SAMPLES[n]["ground_truth_redlines"].

    Returns
    -------
    Reward
    """
    if action.action_type == "skip":
        return Reward(
            score=0.0,
            breakdown={"skip": 0.0},
            done=True,
            feedback=(
                f"You skipped redlining. There were "
                f"{len(ground_truth_redlines)} required edits."
            ),
        )

    # --- parse agent edits --------------------------------------------------
    agent_edits = _parse_agent_edits(action)
    if not agent_edits:
        return Reward(
            score=0.0,
            breakdown={"no_edits": 0.0},
            done=True,
            feedback="No redline edits were found in your response.",
        )

    # --- score each ground-truth redline ------------------------------------
    redline_scores     = []
    matched_agent_idxs = set()

    for gt in ground_truth_redlines:
        best_score, best_idx = _best_matching_edit(gt, agent_edits, matched_agent_idxs)
        redline_scores.append(best_score)
        if best_idx is not None:
            matched_agent_idxs.add(best_idx)

    coverage      = len(matched_agent_idxs)
    mean_score    = sum(redline_scores) / len(redline_scores) if redline_scores else 0.0

    # --- hallucination penalty (agent edits with no GT match) ---------------
    unmatched_agent    = len(agent_edits) - len(matched_agent_idxs)
    hallucination_pen  = min(unmatched_agent * HALLUCINATION_PENALTY, 0.25)

    # --- coverage bonus -------------------------------------------------------
    bonus = COVERAGE_BONUS if coverage == len(ground_truth_redlines) else 0.0

    final_score = max(0.0, min(1.0, mean_score - hallucination_pen + bonus))

    # --- feedback ------------------------------------------------------------
    missed_sections = [
        gt["section"]
        for gt, s in zip(ground_truth_redlines, redline_scores)
        if s < 0.3
    ]
    feedback_parts = [
        f"Matched {coverage}/{len(ground_truth_redlines)} required redlines.",
        f"Average redline quality: {mean_score:.2f}.",
    ]
    if missed_sections:
        feedback_parts.append(
            "Weak/missing: " + ", ".join(missed_sections[:4])
            + ("..." if len(missed_sections) > 4 else "")
        )
    if hallucination_pen:
        feedback_parts.append(
            f"{unmatched_agent} edit(s) appeared fabricated "
            f"(penalty: -{hallucination_pen:.2f})."
        )
    if bonus:
        feedback_parts.append("✓ Full coverage bonus applied (+0.05).")

    return Reward(
        score=round(final_score, 4),
        breakdown={
            "mean_redline_quality": round(mean_score, 4),
            "hallucination_penalty": -round(hallucination_pen, 4),
            "coverage_bonus": round(bonus, 4),
            "redlines_matched": coverage,
            "total_required": len(ground_truth_redlines),
        },
        done=True,
        feedback=" ".join(feedback_parts),
    )


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _parse_agent_edits(action: Action) -> List[Dict[str, str]]:
    """
    Try metadata["edits"] first, then fall back to content parsing.
    Each edit dict needs at least some of: section, issue, original, redline.
    """
    if action.metadata and "edits" in action.metadata:
        raw = action.metadata["edits"]
        if isinstance(raw, list) and raw:
            return [e for e in raw if isinstance(e, dict)]

    # Fallback: treat each paragraph or numbered item as a separate edit
    chunks = re.split(r"\n{2,}|\d+\.\s+", action.content)
    edits = []
    for chunk in chunks:
        chunk = chunk.strip()
        if len(chunk) > 20:
            edits.append({"redline": chunk, "issue": chunk, "section": ""})
    return edits


def _tokenize(text: str) -> set:
    stopwords = {"the", "a", "an", "is", "are", "of", "to", "in", "and",
                 "or", "for", "with", "that", "this", "it", "its", "any",
                 "all", "be", "by", "at", "no", "not", "on", "as", "such",
                 "shall", "may", "party", "parties"}
    return set(re.findall(r"\b[a-z]{3,}\b", text.lower())) - stopwords


def _jaccard(a: str, b: str) -> float:
    ta, tb = _tokenize(a), _tokenize(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _score_edit_against_gt(
    agent_edit: Dict[str, str],
    gt: Dict[str, str],
) -> float:
    """Return weighted component score for one (agent_edit, ground_truth) pair."""
    # Section match
    sec_score = (
        _jaccard(agent_edit.get("section", ""), gt.get("section", ""))
        >= THRESHOLDS["section"]
    ) * 1.0

    # Issue/problem identification match
    agent_issue = agent_edit.get("issue", agent_edit.get("redline", ""))
    issue_score = min(
        _jaccard(agent_issue, gt.get("issue", "")) / THRESHOLDS["issue"], 1.0
    )

    # Redline text quality
    agent_redline = agent_edit.get("redline", agent_edit.get("content", ""))
    redline_score = min(
        _jaccard(agent_redline, gt.get("redline", "")) / THRESHOLDS["redline"], 1.0
    )

    return (
        WEIGHTS["section"]  * sec_score
        + WEIGHTS["issue"]  * issue_score
        + WEIGHTS["redline"] * redline_score
    )


def _best_matching_edit(
    gt: Dict[str, str],
    agent_edits: List[Dict[str, str]],
    already_used: set,
) -> Tuple[float, Any]:
    best_score = 0.0
    best_idx   = None
    for i, edit in enumerate(agent_edits):
        if i in already_used:
            continue
        s = _score_edit_against_gt(edit, gt)
        if s > best_score:
            best_score = s
            best_idx   = i
    return best_score, best_idx
