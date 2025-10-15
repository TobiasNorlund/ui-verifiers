"""Tests for Trainer class."""

import pytest
from unittest.mock import Mock, MagicMock
from pathlib import Path
from src.learner.trainer import Trainer
from src.data_utils.trajectory import Trajectory


class TestTrainer:
    """Test cases for Trainer class."""

    @pytest.fixture
    def mock_algorithm(self):
        """Create a mock algorithm."""
        algo = Mock()
        algo.train_step.return_value = {'loss': 0.5}
        return algo

    @pytest.fixture
    def mock_vlm(self):
        """Create a mock VLM wrapper."""
        return Mock()

    @pytest.fixture
    def sample_config(self):
        """Create a sample configuration."""
        return {
            'checkpoint_interval': 5,
            'training': {
                'num_iterations': 10
            }
        }

    @pytest.fixture
    def sample_trajectories(self):
        """Create sample trajectories for testing."""
        return [
            Trajectory(
                observations=[{}, {}, {}],
                actions=['a1', 'a2'],
                rewards=[0.5, 0.5]
            )
        ]

    def test_trainer_initialization(self, mock_algorithm, mock_vlm, sample_config, tmp_path):
        """Test Trainer initialization."""
        trainer = Trainer(
            algorithm=mock_algorithm,
            vlm_wrapper=mock_vlm,
            config=sample_config,
            checkpoint_dir=tmp_path
        )

        assert trainer.algorithm == mock_algorithm
        assert trainer.vlm == mock_vlm
        assert trainer.config == sample_config
        assert trainer.global_step == 0
        assert trainer.epoch == 0

    def test_train_single_epoch(self, mock_algorithm, mock_vlm, sample_config, sample_trajectories, tmp_path):
        """Test training for a single epoch."""
        trainer = Trainer(
            algorithm=mock_algorithm,
            vlm_wrapper=mock_vlm,
            config=sample_config,
            checkpoint_dir=tmp_path
        )

        metrics = trainer.train(sample_trajectories, num_epochs=1)

        assert 'epoch_0' in metrics
        assert mock_algorithm.train_step.called
        assert trainer.epoch == 1

    def test_train_multiple_epochs(self, mock_algorithm, mock_vlm, sample_config, sample_trajectories, tmp_path):
        """Test training for multiple epochs."""
        trainer = Trainer(
            algorithm=mock_algorithm,
            vlm_wrapper=mock_vlm,
            config=sample_config,
            checkpoint_dir=tmp_path
        )

        metrics = trainer.train(sample_trajectories, num_epochs=3)

        assert len(metrics) == 3
        assert mock_algorithm.train_step.call_count == 3
        assert trainer.epoch == 3

    def test_checkpoint_saving(self, mock_algorithm, mock_vlm, sample_config, tmp_path):
        """Test checkpoint saving."""
        trainer = Trainer(
            algorithm=mock_algorithm,
            vlm_wrapper=mock_vlm,
            config=sample_config,
            checkpoint_dir=tmp_path
        )

        trainer.save_checkpoint(name='test_checkpoint')
        # Checkpoint saving is a TODO, so we just test it doesn't error

    def test_checkpoint_loading(self, mock_algorithm, mock_vlm, sample_config, tmp_path):
        """Test checkpoint loading."""
        trainer = Trainer(
            algorithm=mock_algorithm,
            vlm_wrapper=mock_vlm,
            config=sample_config,
            checkpoint_dir=tmp_path
        )

        checkpoint_path = tmp_path / 'checkpoint_test'
        checkpoint_path.mkdir()

        trainer.load_checkpoint(checkpoint_path)
        # Checkpoint loading is a TODO, so we just test it doesn't error
