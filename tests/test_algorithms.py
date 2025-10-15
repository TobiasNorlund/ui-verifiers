"""Tests for RL algorithms."""

import pytest
from unittest.mock import Mock
from src.learner.algorithms.rejection_sampling import RejectionSampling
from src.learner.algorithms.ppo import PPO
from src.data_utils.trajectory import Trajectory


class TestRejectionSampling:
    """Test cases for RejectionSampling algorithm."""

    @pytest.fixture
    def algorithm(self):
        """Create rejection sampling algorithm."""
        config = {
            'reward_threshold': 0.5
        }
        return RejectionSampling(config)

    @pytest.fixture
    def trajectories(self):
        """Create sample trajectories with varying rewards."""
        return [
            Trajectory(
                observations=[{}, {}],
                actions=['a1'],
                rewards=[0.8]  # High reward
            ),
            Trajectory(
                observations=[{}, {}],
                actions=['a2'],
                rewards=[0.3]  # Low reward
            ),
            Trajectory(
                observations=[{}, {}],
                actions=['a3'],
                rewards=[0.9]  # High reward
            )
        ]

    def test_initialization(self):
        """Test algorithm initialization."""
        config = {'reward_threshold': 0.7}
        algo = RejectionSampling(config)

        assert algo.reward_threshold == 0.7

    def test_filter_trajectories(self, algorithm, trajectories):
        """Test trajectory filtering by reward threshold."""
        filtered = algorithm._filter_trajectories(trajectories)

        # Should keep only trajectories with reward >= 0.5
        assert len(filtered) == 2
        assert all(sum(t.rewards) / len(t.rewards) >= 0.5 for t in filtered)

    def test_train_step_with_accepted_trajectories(self, algorithm, trajectories):
        """Test training step with accepted trajectories."""
        mock_model = Mock()
        metrics = algorithm.train_step(trajectories, mock_model)

        assert metrics['num_trajectories'] == 3
        assert metrics['num_accepted'] == 2
        assert metrics['acceptance_rate'] > 0

    def test_train_step_with_no_accepted_trajectories(self, algorithm):
        """Test training step when no trajectories meet threshold."""
        low_reward_trajs = [
            Trajectory(
                observations=[{}, {}],
                actions=['a'],
                rewards=[0.1]
            )
        ]

        mock_model = Mock()
        metrics = algorithm.train_step(low_reward_trajs, mock_model)

        assert metrics['num_accepted'] == 0
        assert metrics['acceptance_rate'] == 0.0


class TestPPO:
    """Test cases for PPO algorithm."""

    @pytest.fixture
    def algorithm(self):
        """Create PPO algorithm."""
        config = {
            'clip_epsilon': 0.2,
            'value_coef': 0.5,
            'entropy_coef': 0.01,
            'num_epochs': 2,
            'batch_size': 32
        }
        return PPO(config)

    @pytest.fixture
    def trajectories(self):
        """Create sample trajectories."""
        return [
            Trajectory(
                observations=[{}, {}, {}],
                actions=['a1', 'a2'],
                rewards=[0.5, 0.5]
            )
        ]

    def test_initialization(self):
        """Test algorithm initialization."""
        config = {
            'clip_epsilon': 0.3,
            'value_coef': 0.6,
            'entropy_coef': 0.02
        }
        algo = PPO(config)

        assert algo.clip_epsilon == 0.3
        assert algo.value_coef == 0.6
        assert algo.entropy_coef == 0.02

    def test_train_step(self, algorithm, trajectories):
        """Test PPO training step."""
        mock_model = Mock()
        metrics = algorithm.train_step(trajectories, mock_model)

        assert 'loss' in metrics
        assert 'num_trajectories' in metrics
        assert metrics['num_trajectories'] == 1

    def test_compute_advantages(self, algorithm, trajectories):
        """Test GAE advantage computation."""
        result = algorithm.compute_advantages(trajectories)

        assert 'advantages' in result
        assert 'returns' in result
