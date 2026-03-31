"""
scripts/eval/run_eval.py
------------------------
Eval Harness — runs any agent across every document sample and
produces a structured score report.

WHAT IT DOES:
  - Iterates all samples in all 3 tasks (exhaustive coverage)
  - Runs the agent's decide() method for each
  - Collects score, breakdown, feedback per sample
  - Computes per-task and overall aggregate statistics
  - Writes a JSON results file + prints a human-readable summary table

BUILT-IN AGENTS:
  --agent baseline   Rule-based keyword/heuristic agent (scripts/baseline_agent.py)
  --agent random     Random valid-action agent (sanity floor)
  --agent oracle     Submits ground-truth answers (ceiling score — should score ~1.0)

USAGE:
  # Run all agents, all tasks
  python scripts/eval/run_eval.py --agent baseline
  python scripts/eval/run_eval.py --agent oracle
  python scripts/eval/run_eval.py --agent random

  # Specific task only
  python scripts/eval/run_eval.py --agent baseline --task clause_classifier

  # Save results to file
  python scripts/eval/run_eval.py --agent baseline --output results/baseline.json

OUTPUT:
  ┌─────────────────────────┬──────────┬────────┬────────┬────────┐
  │ Task                    │ Samples  │  Mean  │  Min   │  Max   │
  ├─────────────────────────┼──────────┼────────┼────────┼────────┤
  │ clause_classifier       │    10    │  0.850 │  0.500 │  1.000 │
  │ risk_spotter            │     2    │  0.612 │  0.580 │  0.643 │
  │ contract_redliner       │     1    │  0.541 │  0.541 │  0.541 │
  ├─────────────────────────┼──────────┼────────┼────────┼────────┤
  │ OVERALL                 │    13    │  0.774 │        │        │
  └─────────────────────────┴──────────┴────────┴────────┴────────┘
"""

from __future__ import annotations
import argparse
import json
import os
import random as stdlib_random
import sys
import time
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

# Stub pydantic if not installed
try:
    import pydantic
except ImportError:
    import types
    pydantic_mod = types.ModuleType("pydantic")
    class _BaseModel:
        def __init__(self, **kw):
            for k, v in kw.items(): setattr(self, k, v)
    pydantic_mod.BaseModel = _BaseModel
    pydantic_mod.Field = lambda *a, **kw: None
    sys.modules["pydantic"] = pydantic_mod

from env.legal_env import LegalEnv
from env.models import Action
from data.contracts import (
    CLASSIFICATION_SAMPLES, RISK_SAMPLES, REDLINE_SAMPLES, CLAUSE_TAXONOMY
)


# ---------------------------------------------------------------------------
# Agent definitions
# ---------------------------------------------------------------------------

class OracleAgent:
    """
    Submits the ground-truth answer for every sample.
    Provides the theoretical score ceiling — should approach 1.0 on all tasks.
    """
    name = "oracle"

    def decide(self, obs, sample: Dict[str, Any]) -> Action:
        task = obs.task_id

        if task == "clause_classifier":
            return Action(
                action_type="classify",
                content=sample["label"],
                metadata={},
            )

        elif task == "risk_spotter":
            risks = [r["risk"] for r in sample["ground_truth_risks"]]
            return Action(
                action_type="flag_risks",
                content="\n".join(risks),
                metadata={"risks": risks},
            )

        elif task == "contract_redliner":
            edits = [
                {
                    "section": r["section"],
                    "issue": r["issue"],
                    "original": r["original"],
                    "redline": r["redline"],
                }
                for r in sample["ground_truth_redlines"]
            ]
            return Action(
                action_type="redline",
                content="\n".join(e["redline"] for e in edits),
                metadata={"edits": edits},
            )

        return Action(action_type="skip", content="unknown task")


class RandomAgent:
    """
    Submits random valid actions. Provides the random baseline floor.
    Expected score: ~1/8 for classifier, near-0 for others.
    """
    name = "random"

    def decide(self, obs, sample: Dict[str, Any]) -> Action:
        task = obs.task_id

        if task == "clause_classifier":
            return Action(
                action_type="classify",
                content=stdlib_random.choice(CLAUSE_TAXONOMY),
                metadata={},
            )
        elif task == "risk_spotter":
            fake_risks = [
                "The contract contains unusual provisions.",
                "One party has disproportionate obligations.",
            ]
            return Action(
                action_type="flag_risks",
                content="\n".join(fake_risks),
                metadata={"risks": fake_risks},
            )
        elif task == "contract_redliner":
            return Action(
                action_type="redline",
                content="Standard terms should be applied throughout.",
                metadata={"edits": [
                    {"section": "General", "issue": "Non-standard terms",
                     "original": "...", "redline": "Standard language should apply."}
                ]},
            )
        return Action(action_type="skip", content="unknown")


class BaselineAgentWrapper:
    """Wraps the rule-based baseline agent from scripts/baseline_agent.py."""
    name = "baseline"

    def __init__(self):
        from scripts.baseline_agent import BaselineAgent
        self._agent = BaselineAgent()

    def decide(self, obs, sample: Dict[str, Any]) -> Action:
        return self._agent._decide(obs)


AGENT_REGISTRY = {
    "oracle": OracleAgent,
    "random": RandomAgent,
    "baseline": BaselineAgentWrapper,
}


# ---------------------------------------------------------------------------
# Eval runner
# ---------------------------------------------------------------------------

TASK_SAMPLES = {
    "clause_classifier": CLASSIFICATION_SAMPLES,
    "risk_spotter": RISK_SAMPLES,
    "contract_redliner": REDLINE_SAMPLES,
}


def run_eval(
    agent_name: str,
    task_filter: Optional[str] = None,
    seed: int = 42,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run a full evaluation of the named agent across all samples.

    Returns a results dict with per-sample and aggregate scores.
    """
    stdlib_random.seed(seed)

    if agent_name not in AGENT_REGISTRY:
        raise ValueError(f"Unknown agent '{agent_name}'. Choose from: {list(AGENT_REGISTRY)}")

    agent = AGENT_REGISTRY[agent_name]()
    env = LegalEnv()

    results = {
        "agent": agent_name,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "seed": seed,
        "tasks": {},
    }

    task_ids = [task_filter] if task_filter else list(TASK_SAMPLES.keys())

    for task_id in task_ids:
        samples = TASK_SAMPLES[task_id]
        task_results = []

        for sample in samples:
            obs = env.reset(task_id=task_id, doc_id=sample["id"])
            action = agent.decide(obs, sample)

            try:
                response = env.step(action)
                reward = response.reward
            except Exception as e:
                reward_data = {"score": 0.0, "breakdown": {}, "feedback": str(e)}
                task_results.append({
                    "sample_id": sample["id"],
                    "difficulty": sample.get("difficulty", "unknown"),
                    **reward_data,
                })
                continue

            task_results.append({
                "sample_id": sample["id"],
                "difficulty": sample.get("difficulty", "unknown"),
                "score": reward.score,
                "breakdown": reward.breakdown,
                "feedback": reward.feedback,
                "action_type": action.action_type,
                "content_preview": action.content[:100],
            })

            if verbose:
                diff = sample.get("difficulty", "?")
                print(f"  [{task_id}] {sample['id']:12s} ({diff:6s})  score={reward.score:.4f}  {reward.feedback[:60]}")

        scores = [r["score"] for r in task_results]
        results["tasks"][task_id] = {
            "samples": task_results,
            "n": len(scores),
            "mean": round(sum(scores) / len(scores), 4) if scores else 0.0,
            "min": round(min(scores), 4) if scores else 0.0,
            "max": round(max(scores), 4) if scores else 0.0,
            "by_difficulty": _group_by_difficulty(task_results),
        }

    # Overall aggregate
    all_scores = [
        r["score"]
        for td in results["tasks"].values()
        for r in td["samples"]
    ]
    results["overall"] = {
        "n": len(all_scores),
        "mean": round(sum(all_scores) / len(all_scores), 4) if all_scores else 0.0,
        "min": round(min(all_scores), 4) if all_scores else 0.0,
        "max": round(max(all_scores), 4) if all_scores else 0.0,
    }

    return results


def _group_by_difficulty(task_results: List[Dict]) -> Dict[str, float]:
    groups: Dict[str, List[float]] = {}
    for r in task_results:
        d = r.get("difficulty", "unknown")
        groups.setdefault(d, []).append(r["score"])
    return {
        d: round(sum(v) / len(v), 4)
        for d, v in groups.items()
    }


def print_summary(results: Dict[str, Any]) -> None:
    w = 25
    print()
    print(f"  Agent: {results['agent']}   |   {results['timestamp']}")
    print()
    print(f"  {'Task':<{w}} {'Samples':>8}  {'Mean':>7}  {'Min':>7}  {'Max':>7}")
    print(f"  {'-'*w} {'-'*8}  {'-'*7}  {'-'*7}  {'-'*7}")
    for task_id, td in results["tasks"].items():
        print(f"  {task_id:<{w}} {td['n']:>8}  {td['mean']:>7.4f}  {td['min']:>7.4f}  {td['max']:>7.4f}")
        for diff, score in sorted(td["by_difficulty"].items()):
            print(f"    {'  └─ ' + diff:<{w-2}} {'':>8}  {score:>7.4f}")
    print(f"  {'-'*w} {'-'*8}  {'-'*7}  {'-'*7}  {'-'*7}")
    ov = results["overall"]
    print(f"  {'OVERALL':<{w}} {ov['n']:>8}  {ov['mean']:>7.4f}  {ov['min']:>7.4f}  {ov['max']:>7.4f}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Legal OpenEnv Eval Harness")
    parser.add_argument("--agent", default="baseline",
                        choices=list(AGENT_REGISTRY),
                        help="Which agent to evaluate")
    parser.add_argument("--task", default=None,
                        choices=["clause_classifier", "risk_spotter", "contract_redliner"],
                        help="Evaluate a single task only")
    parser.add_argument("--output", default=None,
                        help="Path to write JSON results (e.g. results/baseline.json)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress per-sample output")
    args = parser.parse_args()

    print(f"\nRunning eval: agent={args.agent}" + (f" task={args.task}" if args.task else ""))
    print("=" * 60)

    results = run_eval(
        agent_name=args.agent,
        task_filter=args.task,
        seed=args.seed,
        verbose=not args.quiet,
    )

    print_summary(results)

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: {args.output}")

    return results


if __name__ == "__main__":
    main()
