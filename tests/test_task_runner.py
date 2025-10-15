"""Tests for TaskRunner class."""

import pytest
from unittest.mock import Mock, MagicMock
from src.actor.task_runner import TaskRunner
from src.data_utils.trajectory import Trajectory


class TestTaskRunner:
    """Test cases for TaskRunner class."""

    @pytest.fixture
    def mock_vlm(self):
        """Create a mock VLM wrapper."""
        vlm = Mock()
        vlm.generate_action.return_value = "click(100, 200)"
        return vlm

    @pytest.fixture
    def mock_env(self):
        """Create a mock environment client."""
        env = Mock()
        env.reset.return_value = {'state': 'initial'}
        env.step.return_value = (
            {'state': 'next'},
            0.5,
            False,
            {}
        )
        return env

    def test_task_runner_initialization(self, mock_vlm, mock_env):
        """Test TaskRunner initialization."""
        runner = TaskRunner(
            vlm_wrapper=mock_vlm,
            env_client=mock_env,
            max_steps=10
        )

        assert runner.vlm == mock_vlm
        assert runner.env == mock_env
        assert runner.max_steps == 10

    def test_run_episode(self, mock_vlm, mock_env):
        """Test running a single episode."""
        # Setup environment to finish after 3 steps
        mock_env.step.side_effect = [
            ({'state': 's1'}, 0.1, False, {}),
            ({'state': 's2'}, 0.2, False, {}),
            ({'state': 's3'}, 0.3, True, {})  # Done
        ]

        runner = TaskRunner(
            vlm_wrapper=mock_vlm,
            env_client=mock_env
        )

        task = {'description': 'test task'}
        trajectory = runner.run_episode(task)

        assert isinstance(trajectory, Trajectory)
        assert len(trajectory) == 3
        assert len(trajectory.observations) == 4  # Initial + 3 steps
        assert trajectory.metadata['task'] == task

    def test_run_episode_max_steps(self, mock_vlm, mock_env):
        """Test episode termination at max steps."""
        # Environment never finishes
        mock_env.step.return_value = (
            {'state': 'next'},
            0.5,
            False,  # Never done
            {}
        )

        runner = TaskRunner(
            vlm_wrapper=mock_vlm,
            env_client=mock_env,
            max_steps=5
        )

        task = {'description': 'test task'}
        trajectory = runner.run_episode(task)

        # Should stop at max_steps
        assert len(trajectory) == 5

    def test_run_batch(self, mock_vlm, mock_env):
        """Test running multiple episodes in batch."""
        mock_env.step.return_value = (
            {'state': 'next'},
            0.5,
            True,  # Finish immediately
            {}
        )

        runner = TaskRunner(
            vlm_wrapper=mock_vlm,
            env_client=mock_env
        )

        tasks = [
            {'description': 'task1'},
            {'description': 'task2'},
            {'description': 'task3'}
        ]

        trajectories = runner.run_batch(tasks)

        assert len(trajectories) == 3
        assert all(isinstance(t, Trajectory) for t in trajectories)
