"""
scripts/baseline_agent.py
--------------------------
Phase 6 — Baseline Agent

A simple rule-based agent that demonstrates how to interact with the
LegalEnv API. This is the reference implementation for the OpenEnv spec.

THREE STRATEGIES (one per task):

  1. clause_classifier — keyword matching against taxonomy labels.
     Looks for discriminating keywords in the clause text.

  2. risk_spotter — heuristic sentence scanner.
     Flags sentences containing "red flag" legal patterns:
     - extremely low liability caps
     - one-sided obligations
     - broad unilateral rights
     - missing protections

  3. contract_redliner — policy diff agent.
     Reads the policy brief, extracts requirements, checks contract
     text for violations, proposes standard language substitutions.

USAGE:
  # Run against live server (default: localhost:8000)
  python scripts/baseline_agent.py --server http://localhost:8000

  # Run in direct mode (imports env directly, no server needed)
  python scripts/baseline_agent.py --direct

  # Run a specific task
  python scripts/baseline_agent.py --direct --task risk_spotter

OUTPUT:
  Prints a score report for each task to stdout.
"""

from __future__ import annotations
import argparse
import json
import re
import sys
import time
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Constants — keyword dictionaries for classifier baseline
# ---------------------------------------------------------------------------

CLAUSE_KEYWORDS: Dict[str, List[str]] = {
    "indemnification":         ["indemnif", "hold harmless", "defend"],
    "limitation_of_liability": ["limitation", "liability", "shall not exceed", "cap"],
    "confidentiality":         ["confidential", "nondisclosure", "non-disclosure", "secret"],
    "termination":             ["terminat", "cancel", "expire", "surviv"],
    "intellectual_property":   ["intellectual property", "work made for hire", "assign", "patent", "copyright"],
    "governing_law":           ["governed by", "governing law", "jurisdiction", "courts"],
    "force_majeure":           ["force majeure", "act of god", "beyond.*control", "pandemic"],
    "payment_terms":           ["payment", "invoice", "interest", "overdue", "net 30", "net 7"],
}

RISK_PATTERNS = [
    (r"\$\s*(?:0|100|zero)\b",                    "Unreasonably low liability cap"),
    (r"sole\s+discretion",                         "Unilateral sole discretion clause"),
    (r"at any time",                               "Unqualified 'at any time' right — no notice required"),
    (r"sell.{0,40}data|data.{0,40}sell",           "Data selling rights — major privacy risk"),
    (r"ownership.{0,40}data|data.{0,40}ownership", "Data ownership transfer to vendor"),
    (r"audit.{0,30}any time|any time.{0,30}audit", "Broad audit rights with no notice requirement"),
    (r"without notice|no notice",                  "Right exercisable without notice"),
    (r"150%|200%|triple|punitive",                 "Punitive fee or penalty clause"),
    (r"180.day|one hundred eighty",                "Unusually long notice period — vendor lock-in risk"),
    (r"as.is|as is",                               "Complete warranty disclaimer — no vendor accountability"),
    (r"immediately\s+(?:suspend|terminat)",        "Right to immediately suspend/terminate without cure period"),
    (r"machine learning|train.{0,20}model",        "Data used for ML training without consent"),
]

POLICY_SUBSTITUTIONS = [
    {
        "pattern": r"\b7\s+days?\b",
        "issue": "Net 7 payment terms — policy requires Net 30",
        "original_hint": "7 days",
        "redline": "thirty (30) days of receipt",
        "section_hint": "Payment",
    },
    {
        "pattern": r"5%\s*per\s*month",
        "issue": "5%/month interest rate — policy max is 1.5%",
        "original_hint": "5% per month",
        "redline": "1.5% per month or the maximum permitted by applicable law, whichever is less",
        "section_hint": "Late Interest",
    },
    {
        "pattern": r"immediately\s+suspend",
        "issue": "Immediate suspension without notice — policy requires 30 days notice",
        "original_hint": "immediately suspend services",
        "redline": "suspend services upon thirty (30) days written notice if an undisputed invoice remains unpaid",
        "section_hint": "Suspension",
    },
    {
        "pattern": r"owned\s+solely\s+by\s+vendor|property\s+of\s+vendor",
        "issue": "Vendor IP ownership — policy requires Client to own all deliverables",
        "original_hint": "owned solely by Vendor",
        "redline": "works made for hire owned solely by Client; Vendor assigns all rights to Client",
        "section_hint": "IP Ownership",
    },
    {
        "pattern": r"one\s*\(1\)\s*year|1.year.{0,15}confidential",
        "issue": "1-year confidentiality — policy minimum is 3 years",
        "original_hint": "one (1) year",
        "redline": "three (3) years from the date of disclosure",
        "section_hint": "Confidentiality",
    },
    {
        "pattern": r"cayman\s+islands",
        "issue": "Cayman Islands jurisdiction — policy requires US jurisdiction",
        "original_hint": "Cayman Islands",
        "redline": "State of Delaware; arbitration in Wilmington, Delaware under AAA rules",
        "section_hint": "Governing Law",
    },
    {
        "pattern": r"as.is|provided as.is",
        "issue": "As-is warranty disclaimer — policy requires 90-day workmanship warranty",
        "original_hint": "provided AS-IS",
        "redline": "Vendor warrants Services will be performed professionally and Deliverables will conform to SOW for ninety (90) days post-delivery",
        "section_hint": "Warranty",
    },
]


# ---------------------------------------------------------------------------
# Baseline Agent
# ---------------------------------------------------------------------------

class BaselineAgent:
    """Rule-based baseline agent. Demonstrates the OpenEnv interaction loop."""

    def run_direct(self, task_id: Optional[str] = None) -> Dict[str, Any]:
        """Run directly against the Python env (no HTTP server needed)."""
        sys.path.insert(0, ".")
        from env.legal_env import LegalEnv
        from env.models import Action

        env = LegalEnv()
        results = []

        task_ids = [task_id] if task_id else ["clause_classifier", "risk_spotter", "contract_redliner"]

        for tid in task_ids:
            obs = env.reset(task_id=tid)
            print(f"\n{'='*60}")
            print(f"TASK: {tid}")
            print(f"DOC:  {obs.document_text[:200]}...")
            print(f"{'='*60}")

            action = self._decide(obs)
            print(f"\nACTION: {action.action_type}")
            print(f"CONTENT: {action.content[:300]}")
            if action.metadata:
                print(f"METADATA: {json.dumps(action.metadata, indent=2)[:400]}")

            response = env.step(action)
            reward = response.reward

            print(f"\nSCORE:    {reward.score:.4f}")
            print(f"BREAKDOWN: {reward.breakdown}")
            print(f"FEEDBACK: {reward.feedback}")

            results.append({
                "task_id": tid,
                "score": reward.score,
                "breakdown": reward.breakdown,
                "feedback": reward.feedback,
            })

        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        total = 0.0
        for r in results:
            print(f"  {r['task_id']:25s}  score={r['score']:.4f}")
            total += r["score"]
        print(f"  {'MEAN':25s}  score={total/len(results):.4f}")
        return results

    def run_http(self, server_url: str, task_id: Optional[str] = None) -> Dict[str, Any]:
        """Run against a live FastAPI server."""
        try:
            import requests
        except ImportError:
            print("requests not installed. Run: pip install requests")
            sys.exit(1)

        results = []
        task_ids = [task_id] if task_id else ["clause_classifier", "risk_spotter", "contract_redliner"]

        for tid in task_ids:
            print(f"\n{'='*60}")
            print(f"TASK: {tid}")

            # Reset
            reset_resp = requests.post(
                f"{server_url}/reset",
                json={"task_id": tid},
                timeout=10,
            )
            reset_resp.raise_for_status()
            reset_data = reset_resp.json()
            sid = reset_data["session_id"]
            obs_data = reset_data["observation"]

            print(f"SESSION: {sid}")
            print(f"DOC: {obs_data['document_text'][:200]}...")

            # Build a minimal Observation-like dict for _decide
            class _Obs:
                task_id = obs_data["task_id"]
                document_text = obs_data["document_text"]
                context = obs_data.get("context", {})
                instructions = obs_data.get("instructions", "")

            action = self._decide(_Obs())

            # Step
            step_resp = requests.post(
                f"{server_url}/step",
                json={
                    "session_id": sid,
                    "action": {
                        "action_type": action.action_type,
                        "content": action.content,
                        "metadata": action.metadata or {},
                    },
                },
                timeout=10,
            )
            step_resp.raise_for_status()
            step_data = step_resp.json()
            reward = step_data["reward"]

            print(f"SCORE:    {reward['score']:.4f}")
            print(f"FEEDBACK: {reward['feedback']}")

            results.append({
                "task_id": tid,
                "score": reward["score"],
                "feedback": reward["feedback"],
            })

        return results

    # -----------------------------------------------------------------------
    # Core decision logic
    # -----------------------------------------------------------------------

    def _decide(self, obs: Any):
        """Route to task-specific strategy."""
        from env.models import Action
        task = obs.task_id

        if task == "clause_classifier":
            label = self._classify_clause(obs.document_text)
            return Action(action_type="classify", content=label, metadata={})

        elif task == "risk_spotter":
            risks = self._spot_risks(obs.document_text)
            content = "\n".join(f"- {r}" for r in risks)
            return Action(
                action_type="flag_risks",
                content=content,
                metadata={"risks": risks},
            )

        elif task == "contract_redliner":
            policy = obs.context.get("policy_brief", "")
            edits = self._redline_contract(obs.document_text, policy)
            content = "\n\n".join(
                f"{e['section']}: {e['issue']}\nRedline: {e['redline']}"
                for e in edits
            )
            return Action(
                action_type="redline",
                content=content,
                metadata={"edits": edits},
            )

        else:
            from env.models import Action
            return Action(action_type="skip", content="unknown task")

    def _classify_clause(self, text: str) -> str:
        """Keyword-matching classifier. Returns best taxonomy label."""
        text_lower = text.lower()
        scores: Dict[str, int] = {}

        for label, keywords in CLAUSE_KEYWORDS.items():
            score = sum(
                1 for kw in keywords
                if re.search(kw, text_lower)
            )
            if score:
                scores[label] = score

        if not scores:
            return "governing_law"  # low-confidence fallback

        return max(scores, key=lambda k: scores[k])

    def _spot_risks(self, text: str) -> List[str]:
        """Heuristic pattern scanner for risky clauses."""
        risks = []
        text_lower = text.lower()

        for pattern, description in RISK_PATTERNS:
            if re.search(pattern, text_lower):
                # Try to grab context sentence
                match = re.search(r"[^.!?]*" + pattern + r"[^.!?]*[.!?]", text_lower)
                if match:
                    excerpt = match.group(0).strip()[:120]
                    risks.append(f"{description} — \"{excerpt}\"")
                else:
                    risks.append(description)

        return risks if risks else ["No significant risks identified by baseline scanner."]

    def _redline_contract(self, contract: str, policy: str) -> List[Dict[str, str]]:
        """Match contract text against policy requirements and propose edits."""
        edits = []
        contract_lower = contract.lower()

        for sub in POLICY_SUBSTITUTIONS:
            if re.search(sub["pattern"], contract_lower):
                # Find the original sentence for context
                match = re.search(
                    r"[^.\n]*" + sub["pattern"] + r"[^.\n]*",
                    contract,
                    re.IGNORECASE,
                )
                original = match.group(0).strip() if match else sub["original_hint"]
                edits.append({
                    "section": sub["section_hint"],
                    "issue": sub["issue"],
                    "original": original,
                    "redline": sub["redline"],
                })

        return edits if edits else [{"section": "N/A", "issue": "No policy violations detected", "original": "", "redline": ""}]


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Baseline agent for the Legal Document Review OpenEnv"
    )
    parser.add_argument(
        "--direct", action="store_true",
        help="Run directly against Python env (no server needed)"
    )
    parser.add_argument(
        "--server", default="http://localhost:8000",
        help="Server URL for HTTP mode (default: http://localhost:8000)"
    )
    parser.add_argument(
        "--task", default=None,
        choices=["clause_classifier", "risk_spotter", "contract_redliner"],
        help="Run a specific task only",
    )
    args = parser.parse_args()

    agent = BaselineAgent()

    if args.direct:
        agent.run_direct(task_id=args.task)
    else:
        print(f"Connecting to server: {args.server}")
        agent.run_http(server_url=args.server, task_id=args.task)


if __name__ == "__main__":
    main()
