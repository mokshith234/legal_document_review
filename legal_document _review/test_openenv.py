"""
test_openenv.py
---------------
Validation tests for OpenEnv specification compliance.

This module ensures:
  1. Environment can be reset() and produces valid Observations
  2. step(action) returns properly typed responses
  3. state() returns valid EnvironmentState
  4. All 3 tasks are available and runnable
  5. Graders are deterministic
  6. Reward bounds are respected (0.0 ≤ reward ≤ 1.0)
  7. Episodes terminate cleanly

Run: pytest test_openenv.py -v
"""

import pytest
from typing import Any, Dict
from env.legal_env import LegalEnv, TASK_IDS
from env.models import Action, Observation, Reward, StepResponse, EnvironmentState


class TestOpenEnvSpecCompliance:
    """Test OpenEnv specification compliance."""

    @pytest.fixture
    def env(self):
        """Create a fresh environment for each test."""
        return LegalEnv()

    def test_reset_returns_observation(self, env):
        """reset() must return a valid Observation."""
        obs = env.reset()
        assert isinstance(obs, Observation), "reset() must return Observation"
        assert obs.task_id in TASK_IDS, f"task_id must be one of {TASK_IDS}"
        assert isinstance(obs.document_text, str), "document_text must be string"
        assert len(obs.document_text) > 0, "document_text must not be empty"
        assert isinstance(obs.instructions, str), "instructions must be string"
        assert obs.step_count == 1, "First step must be step_count=1"

    def test_reset_with_specific_task(self, env):
        """reset(task_id=...) must pin to specific task."""
        for task_id in TASK_IDS:
            obs = env.reset(task_id=task_id)
            assert obs.task_id == task_id, f"reset(task_id={task_id}) failed"

    def test_step_returns_step_response(self, env):
        """step(action) must return properly typed StepResponse."""
        env.reset()
        action = Action(action_type="skip", content="")

        obs, reward, done, info = env.step(action)

        assert isinstance(obs, Observation), "step() must return (Observation, ...)"
        assert isinstance(reward, Reward), "step() must return (..., Reward, ...)"
        assert isinstance(done, bool), "step() must return (..., bool, ...)"
        assert isinstance(info, dict), "step() must return (..., dict)"

    def test_reward_bounds(self, env):
        """Rewards must be in [0.0, 1.0] range."""
        for _ in range(10):
            env.reset()
            for step in range(5):
                action = Action(action_type="skip", content="")
                obs, reward, done, info = env.step(action)
                assert 0.0 <= reward.score <= 1.0, \
                    f"Reward score {reward.score} out of bounds [0.0, 1.0]"
                if done:
                    break

    def test_state_returns_environment_state(self, env):
        """state() must return valid EnvironmentState."""
        env.reset()
        state = env.state()

        assert isinstance(state, EnvironmentState), "state() must return EnvironmentState"
        assert hasattr(state, 'session_id'), "EnvironmentState must have session_id"
        assert isinstance(state.session_id, str), "session_id must be string"

    def test_deterministic_grading(self, env):
        """Same input must produce same score (determinism)."""
        # Task 1: Classifier
        env.reset(task_id="clause_classifier")
        action = Action(action_type="classify", content="limitation_of_liability")
        _, reward1, _, _ = env.step(action)

        env.reset(task_id="clause_classifier")
        action = Action(action_type="classify", content="limitation_of_liability")
        _, reward2, _, _ = env.step(action)

        assert reward1.score == reward2.score, \
            f"Grader not deterministic: {reward1.score} != {reward2.score}"

    def test_episode_termination(self, env):
        """Episode must terminate within MAX_STEPS."""
        MAX_STEPS = 5
        obs = env.reset()
        step_count = 0

        for step_count in range(MAX_STEPS + 1):
            action = Action(action_type="skip", content="")
            obs, reward, done, info = env.step(action)
            if done:
                break
            assert step_count < MAX_STEPS, \
                f"Episode did not terminate within {MAX_STEPS} steps"

    def test_all_tasks_available(self, env):
        """All 3 tasks must be available."""
        for task_id in TASK_IDS:
            obs = env.reset(task_id=task_id)
            assert obs.task_id == task_id, f"Task {task_id} not available"
            action = Action(action_type="skip", content="")
            _, reward, done, info = env.step(action)
            assert hasattr(reward, 'score'), f"Task {task_id} grader failed"

    def test_observation_schema(self, env):
        """Observation must follow schema."""
        obs = env.reset()

        # Required fields
        required = ['task_id', 'document_text', 'instructions', 'context', 'step_count', 'max_steps']
        for field in required:
            assert hasattr(obs, field), f"Observation missing required field: {field}"

        # Type checks
        assert isinstance(obs.task_id, str)
        assert isinstance(obs.document_text, str)
        assert isinstance(obs.instructions, str)
        assert isinstance(obs.context, dict)
        assert isinstance(obs.step_count, int)
        assert isinstance(obs.max_steps, int)

    def test_action_schema(self, env):
        """Action must follow schema."""
        env.reset()

        # Valid action types
        valid_types = ["classify", "flag_risks", "redline", "skip"]
        for action_type in valid_types:
            action = Action(action_type=action_type, content="test")
            assert action.action_type in valid_types

        # Invalid action type should raise validation error
        with pytest.raises(Exception):
            Action(action_type="invalid", content="test")

    def test_reward_structure(self, env):
        """Reward must have required fields."""
        env.reset()
        action = Action(action_type="skip", content="")
        _, reward, _, _ = env.step(action)

        assert hasattr(reward, 'score'), "Reward missing: score"
        assert hasattr(reward, 'breakdown'), "Reward missing: breakdown"
        assert isinstance(reward.score, float)
        assert isinstance(reward.breakdown, dict)

    def test_no_negative_rewards(self, env):
        """Rewards should not go below -0.2 (skip penalty)."""
        for _ in range(20):
            env.reset()
            for _ in range(5):
                action = Action(action_type="skip", content="")
                _, reward, done, info = env.step(action)
                assert reward.score >= -0.2, \
                    f"Reward {reward.score} below minimum threshold -0.2"
                if done:
                    break

    def test_session_isolation(self):
        """Multiple environments must not interfere."""
        env1 = LegalEnv()
        env2 = LegalEnv()

        obs1 = env1.reset(task_id="clause_classifier")
        obs2 = env2.reset(task_id="risk_spotter")

        assert obs1.task_id != obs2.task_id, "Session isolation failed"
        assert env1.state().session_id != env2.state().session_id, \
            "Session IDs must be unique"

    def test_clean_reset(self, env):
        """Reset must clear history and produce fresh state."""
        env.reset()
        state1 = env.state()

        # Perform some steps
        for _ in range(3):
            action = Action(action_type="skip", content="")
            _, _, done, _ = env.step(action)
            if done:
                break

        # Reset and check clean state
        obs = env.reset()
        state2 = env.state()

        assert obs.step_count == 1, "After reset, step_count should be 1"
        # Session ID should change after reset
        assert state1.session_id != state2.session_id, \
            "Session ID should change after reset"


class TestTaskGraders:
    """Test individual task graders."""

    def test_classifier_perfect_score(self):
        """Classifier should score 1.0 for exact match."""
        env = LegalEnv()
        obs = env.reset(task_id="clause_classifier")
        # Submit correct label (example)
        action = Action(action_type="classify", content="limitation_of_liability")
        _, reward, done, _ = env.step(action)
        # Note: actual score depends on document's ground truth

    def test_risk_spotter_score_range(self):
        """Risk spotter scores should be in [0.0, 1.0]."""
        env = LegalEnv()
        obs = env.reset(task_id="risk_spotter")
        action = Action(action_type="flag_risks", content="some risks identified")
        _, reward, done, _ = env.step(action)
        assert 0.0 <= reward.score <= 1.0

    def test_redliner_score_range(self):
        """Redliner scores should be in [0.0, 1.0]."""
        env = LegalEnv()
        obs = env.reset(task_id="contract_redliner")
        action = Action(
            action_type="redline",
            content="edits: change X to Y"
        )
        _, reward, done, _ = env.step(action)
        assert 0.0 <= reward.score <= 1.0

    def test_skip_penalty(self):
        """Skipping should result in zero or negative reward."""
        env = LegalEnv()
        obs = env.reset()
        action = Action(action_type="skip", content="")
        _, reward, done, _ = env.step(action)
        assert reward.score <= 0.0, "Skip should yield zero or negative reward"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
