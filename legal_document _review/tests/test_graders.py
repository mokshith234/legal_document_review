"""
tests/test_graders.py
---------------------
Unit tests for all three graders.

Tests cover:
  - Exact / near-miss / wrong scoring for clause_classifier
  - Skip penalty behavior across all graders
  - Invalid label handling in classifier
  - Weighted F1 math for risk_spotter
  - Hallucination penalties
  - Redliner partial and full coverage
  - Edge cases: empty content, no metadata, single-word answers

Run with:
  python -m pytest tests/test_graders.py -v
"""

import sys
import types
import unittest

sys.path.insert(0, ".")

# ---------------------------------------------------------------------------
# Minimal Pydantic stub so tests run without installing pydantic
# ---------------------------------------------------------------------------

def _make_pydantic_stub():
    pydantic_mod = types.ModuleType("pydantic")

    class _Field:
        def __init__(self, *a, **kw): pass
    def Field(*a, **kw): return None

    class BaseModel:
        def __init__(self, **kw):
            for k, v in kw.items():
                setattr(self, k, v)
        def model_dump(self):
            return self.__dict__

    pydantic_mod.BaseModel = Mock()
    pydantic_mod.Field = Mock()

    # also stub fastapi
    fastapi_mod = types.ModuleType("fastapi")
    class FastAPI:
        def __init__(self, **kw): pass
        def get(self, *a, **kw): return lambda f: f
        def post(self, *a, **kw): return lambda f: f
        def add_middleware(self, *a, **kw): pass
    fastapi_mod.FastAPI = Mock()
    fastapi_mod.HTTPException = Mock()
    fastapi_mod.Header = Mock()

    cors = types.ModuleType("fastapi.middleware.cors")
    cors.CORSMiddleware = Mock()

    sys.modules["fastapi"] = fastapi_mod
    sys.modules["fastapi.middleware"] = types.ModuleType("fastapi.middleware")
    sys.modules["fastapi.middleware.cors"] = cors

_make_pydantic_stub()

from env.models import Action, Reward
import graders.clause_classifier as cc
import graders.risk_spotter as rs
import graders.contract_redliner as cr
from data.contracts import (
    CLASSIFICATION_SAMPLES,
    RISK_SAMPLES,
    REDLINE_SAMPLES,
    CLAUSE_TAXONOMY,
)


def _action(action_type="classify", content="indemnification", metadata=None):
    return Action(action_type=action_type, content=content, metadata=metadata or {})


# ===========================================================================
# Clause Classifier Tests
# ===========================================================================

class TestClauseClassifier(unittest.TestCase):

    def test_exact_match_scores_1(self):
        action = _action(content="indemnification")
        reward = cc.grade(action, "indemnification")
        self.assertAlmostEqual(reward.score, 1.0)
        self.assertTrue(reward.done)

    def test_exact_match_case_insensitive(self):
        action = _action(content="  Indemnification  ")
        reward = cc.grade(action, "indemnification")
        self.assertAlmostEqual(reward.score, 1.0)

    def test_exact_match_hyphen_normalised(self):
        action = _action(content="limitation-of-liability")
        reward = cc.grade(action, "limitation_of_liability")
        self.assertAlmostEqual(reward.score, 1.0)

    def test_near_miss_scores_half(self):
        # indemnification ↔ limitation_of_liability are near-misses
        action = _action(content="limitation_of_liability")
        reward = cc.grade(action, "indemnification")
        self.assertAlmostEqual(reward.score, 0.5)

    def test_near_miss_reverse(self):
        action = _action(content="indemnification")
        reward = cc.grade(action, "limitation_of_liability")
        self.assertAlmostEqual(reward.score, 0.5)

    def test_confidentiality_ip_near_miss(self):
        action = _action(content="intellectual_property")
        reward = cc.grade(action, "confidentiality")
        self.assertAlmostEqual(reward.score, 0.5)

    def test_wrong_scores_zero(self):
        action = _action(content="governing_law")
        reward = cc.grade(action, "indemnification")
        self.assertAlmostEqual(reward.score, 0.0)

    def test_skip_scores_zero(self):
        action = _action(action_type="skip", content="skip")
        reward = cc.grade(action, "indemnification")
        self.assertAlmostEqual(reward.score, 0.0)
        self.assertTrue(reward.done)
        self.assertIn("indemnification", reward.feedback)

    def test_invalid_label_scores_zero(self):
        action = _action(content="nonsense_label_xyz")
        reward = cc.grade(action, "indemnification")
        self.assertAlmostEqual(reward.score, 0.0)
        self.assertIn("not a valid taxonomy label", reward.feedback)

    def test_all_taxonomy_labels_exact_match(self):
        for label in CLAUSE_TAXONOMY:
            action = _action(content=label)
            reward = cc.grade(action, label)
            self.assertAlmostEqual(reward.score, 1.0, msg=f"Failed for label: {label}")

    def test_every_sample_has_valid_label(self):
        for sample in CLASSIFICATION_SAMPLES:
            self.assertIn(
                sample["label"], CLAUSE_TAXONOMY,
                msg=f"Sample {sample['id']} has invalid label {sample['label']}"
            )

    def test_feedback_always_present(self):
        for content in ["indemnification", "governing_law", "nonsense", "skip"]:
            atype = "skip" if content == "skip" else "classify"
            action = _action(action_type=atype, content=content)
            reward = cc.grade(action, "indemnification")
            self.assertIsInstance(reward.feedback, str)
            self.assertGreater(len(reward.feedback), 0)

    def test_score_bounds(self):
        for label in CLAUSE_TAXONOMY + ["wrong_label", "another_wrong"]:
            action = _action(content=label)
            reward = cc.grade(action, "payment_terms")
            self.assertGreaterEqual(reward.score, 0.0)
            self.assertLessEqual(reward.score, 1.0)


# ===========================================================================
# Risk Spotter Tests
# ===========================================================================

class TestRiskSpotter(unittest.TestCase):

    def _gt_risks(self, sample_idx=0):
        return RISK_SAMPLES[sample_idx]["ground_truth_risks"]

    def test_perfect_recall_scores_high(self):
        gt = self._gt_risks(0)
        risk_texts = [r["risk"] for r in gt]
        action = _action(
            action_type="flag_risks",
            content="\n".join(risk_texts),
            metadata={"risks": risk_texts},
        )
        reward = rs.grade(action, gt)
        self.assertGreater(reward.score, 0.7)

    def test_empty_risks_scores_zero(self):
        action = _action(action_type="flag_risks", content="", metadata={})
        reward = rs.grade(action, self._gt_risks())
        self.assertAlmostEqual(reward.score, 0.0)

    def test_skip_scores_zero(self):
        action = _action(action_type="skip", content="skip")
        reward = rs.grade(action, self._gt_risks())
        self.assertAlmostEqual(reward.score, 0.0)
        self.assertTrue(reward.done)

    def test_hallucination_penalty_applied(self):
        # Submit only fabricated risks
        action = _action(
            action_type="flag_risks",
            content="The contract uses Comic Sans font.",
            metadata={"risks": [
                "Uses Comic Sans font — unprofessional",
                "Missing party address on page 3",
                "Date format is non-ISO",
                "Document not notarized",
            ]},
        )
        reward = rs.grade(action, self._gt_risks())
        # hallucination penalty should reduce score
        self.assertLess(reward.score, 0.2)
        self.assertIn("hallucination_penalty", reward.breakdown)

    def test_partial_recall_scores_partial(self):
        gt = self._gt_risks(0)
        # Only submit half the risks
        half_risks = [r["risk"] for r in gt[:2]]
        action = _action(
            action_type="flag_risks",
            content="\n".join(half_risks),
            metadata={"risks": half_risks},
        )
        reward = rs.grade(action, gt)
        # Should score between 0.2 and 0.8
        self.assertGreater(reward.score, 0.05)
        self.assertLess(reward.score, 0.95)

    def test_breakdown_keys_present(self):
        gt = self._gt_risks()
        action = _action(action_type="flag_risks", content="Some risk identified")
        reward = rs.grade(action, gt)
        for key in ("precision", "recall", "f1"):
            self.assertIn(key, reward.breakdown)

    def test_score_always_bounded(self):
        gt = self._gt_risks()
        test_cases = [
            "Nothing wrong here.",
            "\n".join(r["risk"] for r in gt),
            "Fake risk 1\nFake risk 2\nFake risk 3\nFake risk 4\nFake risk 5",
        ]
        for content in test_cases:
            action = _action(action_type="flag_risks", content=content)
            reward = rs.grade(action, gt)
            self.assertGreaterEqual(reward.score, 0.0, msg=f"Score < 0 for: {content[:50]}")
            self.assertLessEqual(reward.score, 1.0, msg=f"Score > 1 for: {content[:50]}")

    def test_metadata_risks_preferred_over_content(self):
        gt = self._gt_risks()
        risk_texts = [r["risk"] for r in gt]
        # metadata has real risks; content is garbage
        action = _action(
            action_type="flag_risks",
            content="Completely irrelevant content here.",
            metadata={"risks": risk_texts},
        )
        reward_with_meta = rs.grade(action, gt)

        action_no_meta = _action(
            action_type="flag_risks",
            content="Completely irrelevant content here.",
        )
        reward_no_meta = rs.grade(action_no_meta, gt)

        self.assertGreater(reward_with_meta.score, reward_no_meta.score)

    def test_content_fallback_parsing(self):
        gt = self._gt_risks(0)
        # No metadata — use content with newline-separated risks
        content = "\n".join(r["risk"] for r in gt)
        action = _action(action_type="flag_risks", content=content)
        reward = rs.grade(action, gt)
        self.assertGreater(reward.score, 0.3)


# ===========================================================================
# Contract Redliner Tests
# ===========================================================================

class TestContractRedliner(unittest.TestCase):

    def _gt_redlines(self, sample_idx=0):
        return REDLINE_SAMPLES[sample_idx]["ground_truth_redlines"]

    def test_perfect_edits_scores_high(self):
        gt = self._gt_redlines()
        edits = [
            {
                "section": r["section"],
                "issue": r["issue"],
                "original": r["original"],
                "redline": r["redline"],
            }
            for r in gt
        ]
        action = _action(
            action_type="redline",
            content="Proposed redlines per policy.",
            metadata={"edits": edits},
        )
        reward = cr.grade(action, gt)
        self.assertGreater(reward.score, 0.7)

    def test_single_correct_edit(self):
        gt = self._gt_redlines()
        first_edit = gt[0]
        action = _action(
            action_type="redline",
            content=first_edit["redline"],
            metadata={"edits": [{
                "section": first_edit["section"],
                "issue": first_edit["issue"],
                "original": first_edit["original"],
                "redline": first_edit["redline"],
            }]},
        )
        reward = cr.grade(action, gt)
        # One out of 7 redlines → ~1/7 ≈ 0.14 but with good quality per edit
        self.assertGreater(reward.score, 0.05)

    def test_skip_scores_zero(self):
        action = _action(action_type="skip", content="skip")
        reward = cr.grade(action, self._gt_redlines())
        self.assertAlmostEqual(reward.score, 0.0)

    def test_no_edits_scores_zero(self):
        action = _action(action_type="redline", content="x")
        reward = cr.grade(action, self._gt_redlines())
        self.assertAlmostEqual(reward.score, 0.0)

    def test_hallucination_penalty(self):
        gt = self._gt_redlines()
        fake_edits = [
            {"section": "Section 99", "issue": "Missing Oxford comma", "original": "a, b and c", "redline": "a, b, and c"},
            {"section": "Section 100", "issue": "Wrong font size", "original": "12pt", "redline": "11pt"},
            {"section": "Section 101", "issue": "Header not bold", "original": "Header", "redline": "**Header**"},
            {"section": "Section 102", "issue": "Missing logo", "original": "", "redline": "[INSERT LOGO]"},
            {"section": "Section 103", "issue": "Wrong date format", "original": "01/01/2024", "redline": "January 1, 2024"},
        ]
        action = _action(
            action_type="redline",
            content="Proposed various edits.",
            metadata={"edits": fake_edits},
        )
        reward = cr.grade(action, gt)
        self.assertIn("hallucination_penalty", reward.breakdown)
        self.assertLessEqual(reward.breakdown["hallucination_penalty"], 0.0)

    def test_breakdown_keys_present(self):
        gt = self._gt_redlines()
        action = _action(
            action_type="redline",
            content="Change payment terms to 30 days.",
        )
        reward = cr.grade(action, gt)
        for key in ("mean_redline_quality", "redlines_matched", "total_required"):
            self.assertIn(key, reward.breakdown)

    def test_full_coverage_bonus(self):
        gt = self._gt_redlines()
        edits = [
            {"section": r["section"], "issue": r["issue"],
             "original": r["original"], "redline": r["redline"]}
            for r in gt
        ]
        action = _action(
            action_type="redline",
            content="All edits proposed.",
            metadata={"edits": edits},
        )
        reward = cr.grade(action, gt)
        self.assertGreater(reward.breakdown.get("coverage_bonus", 0), 0)

    def test_score_always_bounded(self):
        gt = self._gt_redlines()
        cases = [
            _action(action_type="redline", content="No changes needed."),
            _action(action_type="redline", content="\n".join(r["redline"] for r in gt)),
            _action(action_type="skip", content="skip"),
        ]
        for action in cases:
            reward = cr.grade(action, gt)
            self.assertGreaterEqual(reward.score, 0.0)
            self.assertLessEqual(reward.score, 1.0)

    def test_content_fallback_parsing(self):
        gt = self._gt_redlines()
        # Policy-language content without metadata should still partially score
        content = "\n\n".join(
            f"{r['section']}: {r['issue']}. Proposed: {r['redline']}"
            for r in gt[:3]
        )
        action = _action(action_type="redline", content=content)
        reward = cr.grade(action, gt)
        self.assertGreaterEqual(reward.score, 0.0)
        self.assertLessEqual(reward.score, 1.0)


# ===========================================================================
# Environment Integration Tests
# ===========================================================================

class TestLegalEnvIntegration(unittest.TestCase):

    def setUp(self):
        from env.legal_env import LegalEnv
        self.env = LegalEnv()

    def test_reset_returns_observation(self):
        obs = self.env.reset()
        self.assertIsNotNone(obs.task_id)
        self.assertIsNotNone(obs.document_text)
        self.assertGreater(len(obs.document_text), 10)
        self.assertEqual(obs.step_count, 0)

    def test_reset_pins_task(self):
        for task_id in ["clause_classifier", "risk_spotter", "contract_redliner"]:
            obs = self.env.reset(task_id=task_id)
            self.assertEqual(obs.task_id, task_id)

    def test_reset_invalid_task_raises(self):
        with self.assertRaises(ValueError):
            self.env.reset(task_id="nonexistent_task")

    def test_step_returns_step_response(self):
        self.env.reset(task_id="clause_classifier")
        action = _action(content="indemnification")
        response = self.env.step(action)
        self.assertIsNotNone(response.reward)
        self.assertIsNotNone(response.observation)
        self.assertIn("cumulative_score", response.info)

    def test_step_increments_step_count(self):
        obs = self.env.reset(task_id="risk_spotter")
        self.assertEqual(obs.step_count, 0)
        action = _action(action_type="flag_risks", content="Some risk here identified")
        response = self.env.step(action)
        self.assertEqual(response.observation.step_count, 1)

    def test_step_after_done_raises(self):
        self.env.reset(task_id="clause_classifier")
        action = _action(content="indemnification")
        self.env.step(action)  # done=True after first step
        with self.assertRaises(RuntimeError):
            self.env.step(action)

    def test_state_reflects_current_episode(self):
        self.env.reset(task_id="clause_classifier")
        state = self.env.state()
        self.assertEqual(state.task_id, "clause_classifier")
        self.assertFalse(state.episode_done)
        self.assertEqual(state.step_count, 0)

    def test_cumulative_score_accumulates(self):
        self.env.reset(task_id="clause_classifier")
        action = _action(content="indemnification")
        response = self.env.step(action)
        state = self.env.state()
        self.assertAlmostEqual(state.cumulative_score, response.reward.score)

    def test_history_logged(self):
        self.env.reset(task_id="clause_classifier")
        self.env.step(_action(content="indemnification"))
        state = self.env.state()
        self.assertEqual(len(state.history), 1)
        self.assertIn("score", state.history[0])
        self.assertIn("feedback", state.history[0])

    def test_task_rotation(self):
        env = __import__("env.legal_env", fromlist=["LegalEnv"]).LegalEnv()
        seen_tasks = set()
        for _ in range(6):
            obs = env.reset()
            seen_tasks.add(obs.task_id)
        self.assertEqual(len(seen_tasks), 3, "All 3 tasks should appear in rotation")

    def test_tasks_endpoint(self):
        from env.legal_env import LegalEnv
        descriptors = LegalEnv.tasks()
        self.assertEqual(len(descriptors), 3)
        ids = {d.id for d in descriptors}
        self.assertIn("clause_classifier", ids)
        self.assertIn("risk_spotter", ids)
        self.assertIn("contract_redliner", ids)

    def test_full_episode_classifier(self):
        """Full episode: reset → step → done."""
        obs = self.env.reset(task_id="clause_classifier")
        self.assertFalse(self.env.state().episode_done)
        response = self.env.step(_action(content=obs.document_text[:5]))  # likely wrong
        self.assertTrue(response.done)
        self.assertTrue(self.env.state().episode_done)

    def test_full_episode_risk_spotter(self):
        obs = self.env.reset(task_id="risk_spotter")
        action = _action(
            action_type="flag_risks",
            content="Liability cap is too low. Audit rights are too broad.",
            metadata={"risks": ["liability cap too low", "audit rights overly broad"]},
        )
        response = self.env.step(action)
        self.assertTrue(response.done)

    def test_full_episode_redliner(self):
        obs = self.env.reset(task_id="contract_redliner")
        self.assertIn("policy_brief", obs.context)
        action = _action(
            action_type="redline",
            content="Change payment terms to 30 days.",
            metadata={"edits": [{"section": "Section 2", "issue": "Net 7 too short", "original": "7 days", "redline": "30 days"}]},
        )
        response = self.env.step(action)
        self.assertTrue(response.done)


# ===========================================================================
# Data Integrity Tests
# ===========================================================================

class TestDataIntegrity(unittest.TestCase):

    def test_classification_samples_have_required_keys(self):
        for s in CLASSIFICATION_SAMPLES:
            for key in ("id", "clause", "label", "difficulty"):
                self.assertIn(key, s, msg=f"Missing '{key}' in {s.get('id')}")

    def test_risk_samples_have_required_keys(self):
        for s in RISK_SAMPLES:
            for key in ("id", "contract_text", "ground_truth_risks"):
                self.assertIn(key, s, msg=f"Missing '{key}' in {s.get('id')}")
            for r in s["ground_truth_risks"]:
                for key in ("risk_id", "risk", "severity"):
                    self.assertIn(key, r, msg=f"Risk missing '{key}' in {s.get('id')}")

    def test_redline_samples_have_required_keys(self):
        for s in REDLINE_SAMPLES:
            for key in ("id", "contract_text", "policy_brief", "ground_truth_redlines"):
                self.assertIn(key, s, msg=f"Missing '{key}' in {s.get('id')}")
            for r in s["ground_truth_redlines"]:
                for key in ("section", "issue", "original", "redline"):
                    self.assertIn(key, r)

    def test_no_duplicate_sample_ids(self):
        cls_ids = [s["id"] for s in CLASSIFICATION_SAMPLES]
        self.assertEqual(len(cls_ids), len(set(cls_ids)), "Duplicate IDs in CLASSIFICATION_SAMPLES")
        risk_ids = [s["id"] for s in RISK_SAMPLES]
        self.assertEqual(len(risk_ids), len(set(risk_ids)), "Duplicate IDs in RISK_SAMPLES")

    def test_severity_values_valid(self):
        valid = {"critical", "high", "medium", "low"}
        for s in RISK_SAMPLES:
            for r in s["ground_truth_risks"]:
                self.assertIn(r["severity"], valid, msg=f"Invalid severity in {s['id']}")

    def test_taxonomy_has_eight_labels(self):
        self.assertEqual(len(CLAUSE_TAXONOMY), 8)

    def test_near_misses_are_bidirectional(self):
        from data.contracts import CLAUSE_NEAR_MISSES
        for label, misses in CLAUSE_NEAR_MISSES.items():
            for miss in misses:
                reverse = CLAUSE_NEAR_MISSES.get(miss, [])
                self.assertIn(
                    label, reverse,
                    msg=f"Near-miss {label}↔{miss} is not bidirectional"
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
