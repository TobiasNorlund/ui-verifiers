"""Tests for Trajectory class."""

import pytest
from src.data_utils.trajectory import Trajectory


class TestTrajectory:
    """Test cases for Trajectory class."""

    def test_trajectory_creation(self):
        """Test basic trajectory creation."""
        observations = [{'state': i} for i in range(5)]
        actions = ['action1', 'action2', 'action3', 'action4']
        rewards = [0.1, 0.2, 0.3, 0.4]

        trajectory = Trajectory(
            observations=observations,
            actions=actions,
            rewards=rewards
        )

        assert len(trajectory) == 4
        assert trajectory.observations == observations
        assert trajectory.actions == actions
        assert trajectory.rewards == rewards

    def test_trajectory_length(self):
        """Test trajectory length calculation."""
        trajectory = Trajectory(
            observations=[{}, {}, {}],
            actions=['a1', 'a2'],
            rewards=[1.0, 2.0]
        )

        assert len(trajectory) == 2

    def test_trajectory_to_dict(self):
        """Test trajectory serialization to dict."""
        trajectory = Trajectory(
            observations=[{'obs': 1}],
            actions=['action'],
            rewards=[1.0],
            metadata={'test': True}
        )

        data = trajectory.to_dict()

        assert 'observations' in data
        assert 'actions' in data
        assert 'rewards' in data
        assert 'metadata' in data
        assert data['metadata']['test'] is True

    def test_trajectory_from_dict(self):
        """Test trajectory deserialization from dict."""
        data = {
            'observations': [{'obs': 1}, {'obs': 2}],
            'actions': ['action1'],
            'rewards': [0.5],
            'metadata': {'task': 'test'}
        }

        trajectory = Trajectory.from_dict(data)

        assert len(trajectory) == 1
        assert trajectory.observations == data['observations']
        assert trajectory.actions == data['actions']
        assert trajectory.rewards == data['rewards']
        assert trajectory.metadata == data['metadata']

    def test_trajectory_without_metadata(self):
        """Test trajectory creation without metadata."""
        trajectory = Trajectory(
            observations=[{}, {}],
            actions=['a'],
            rewards=[1.0]
        )

        assert trajectory.metadata is None
