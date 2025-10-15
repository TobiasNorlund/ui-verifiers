"""Integration tests for the full training pipeline."""

import pytest
from unittest.mock import Mock, MagicMock, patch
from pathlib import Path

from src.learner.trainer import Trainer
from src.learner.algorithms.rejection_sampling import RejectionSampling
from src.actor.task_runner import TaskRunner
from src.data_utils.trajectory import Trajectory


class TestTrainingPipeline:
    """Integration tests for the complete training pipeline."""

    @pytest.fixture
    def mock_vlm(self):
        """Create a mock VLM wrapper."""
        vlm = Mock()
        vlm.generate_action.return_value = "click(100, 200)"
        return vlm

    @pytest.fixture
    def mock_env(self):
        """Create a mock environment."""
        env = Mock()
        env.reset.return_value = {'state': 'initial'}
        env.step.return_value = (
            {'state': 'next'},
            0.8,  # High reward for rejection sampling
            True,
            {}
        )
        return env

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return {
            'reward_threshold': 0.5,
            'checkpoint_interval': 100,
            'training': {
                'num_iterations': 2
            }
        }

    def test_end_to_end_data_collection_and_training(
        self,
        mock_vlm,
        mock_env,
        config,
        tmp_path
    ):
        """Test complete pipeline from data collection to training."""
        # Create task runner for data collection
        task_runner = TaskRunner(
            vlm_wrapper=mock_vlm,
            env_client=mock_env,
            max_steps=5
        )

        # Collect trajectories
        tasks = [
            {'description': 'task1'},
            {'description': 'task2'}
        ]
        trajectories = task_runner.run_batch(tasks)

        assert len(trajectories) == 2

        # Create algorithm and trainer
        algorithm = RejectionSampling(config)
        trainer = Trainer(
            algorithm=algorithm,
            vlm_wrapper=mock_vlm,
            config=config,
            checkpoint_dir=tmp_path
        )

        # Train on collected trajectories
        metrics = trainer.train(trajectories, num_epochs=1)

        assert 'epoch_0' in metrics
        assert trainer.epoch == 1

    def test_multiple_training_iterations(
        self,
        mock_vlm,
        mock_env,
        config,
        tmp_path
    ):
        """Test multiple training iterations."""
        task_runner = TaskRunner(
            vlm_wrapper=mock_vlm,
            env_client=mock_env,
            max_steps=3
        )

        algorithm = RejectionSampling(config)
        trainer = Trainer(
            algorithm=algorithm,
            vlm_wrapper=mock_vlm,
            config=config,
            checkpoint_dir=tmp_path
        )

        # Simulate multiple iterations
        for iteration in range(2):
            # Collect data
            tasks = [{'description': f'task{i}'} for i in range(3)]
            trajectories = task_runner.run_batch(tasks)

            # Train
            metrics = trainer.train(trajectories, num_epochs=1)

            assert metrics is not None

        assert trainer.epoch == 2


class TestCollationIntegration:
    """Integration tests for data collation."""

    def test_collate_multiple_trajectories(self):
        """Test collating multiple trajectories."""
        from src.data_utils.collation import collate_trajectories

        trajectories = [
            Trajectory(
                observations=[{'obs': 0}, {'obs': 1}],
                actions=['a1'],
                rewards=[0.5]
            ),
            Trajectory(
                observations=[{'obs': 2}, {'obs': 3}, {'obs': 4}],
                actions=['a2', 'a3'],
                rewards=[0.6, 0.7]
            )
        ]

        batch = collate_trajectories(trajectories)

        assert len(batch['observations']) == 5
        assert len(batch['actions']) == 3
        assert len(batch['rewards']) == 3
        assert batch['lengths'] == [1, 2]

    def test_collate_empty_trajectories(self):
        """Test that collating empty list raises error."""
        from src.data_utils.collation import collate_trajectories

        with pytest.raises(ValueError):
            collate_trajectories([])
