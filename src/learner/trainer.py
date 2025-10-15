"""Trainer class for coordinating the training loop."""

from typing import List, Dict, Any, Optional
from pathlib import Path
import logging
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW

from ..data_utils.trajectory import Trajectory
from ..data_utils.collation import collate_trajectories
from .algorithms.base import Algorithm

logger = logging.getLogger(__name__)


class Trainer:
    """
    Learner component: Trains the VLM using trajectories from actors.

    Design decisions:
    1. Uses composition (Algorithm class) for flexibility
    2. Waits for batch_size trajectories before training
    3. Delegates algorithm-specific logic to Algorithm class
    4. Handles model checkpointing and logging

    Responsibilities:
    - Collect trajectories from queue until batch is ready
    - Process trajectories with algorithm
    - Run training step
    - Save checkpoints
    - Log metrics
    """

    def __init__(
        self,
        model: nn.Module,
        algorithm: Algorithm,
        optimizer: Optional[torch.optim.Optimizer] = None,
        batch_size: int = 32,
        learning_rate: float = 1e-5,
        checkpoint_dir: Optional[Path] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Args:
            model: VLM to train
            algorithm: Algorithm to use (RejectionSampling, PPO, etc.)
            optimizer: Optional optimizer (creates AdamW if None)
            batch_size: Number of trajectories per training batch
            learning_rate: Learning rate
            checkpoint_dir: Directory to save checkpoints
            device: Device for training
        """
        self.model = model.to(device)
        self.algorithm = algorithm
        self.batch_size = batch_size
        self.device = device

        # Create optimizer if not provided
        if optimizer is None:
            self.optimizer = AdamW(model.parameters(), lr=learning_rate)
        else:
            self.optimizer = optimizer

        # Setup checkpointing
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        if self.checkpoint_dir:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Training state
        self.step = 0
        self.trajectories_seen = 0

        logger.info(f"Trainer initialized with {algorithm.__class__.__name__}")

    def train_step(self, trajectories: List[Trajectory]) -> Dict[str, float]:
        """
        Run one training step on a batch of trajectories.

        Args:
            trajectories: Batch of trajectories to train on

        Returns:
            Dict of training metrics
        """
        self.model.train()

        # Let algorithm process trajectories (filter, compute advantages, etc.)
        processed_trajectories = self.algorithm.process_trajectories(trajectories)

        if len(processed_trajectories) == 0:
            logger.warning("No trajectories after algorithm processing, skipping training step")
            return {'loss': 0.0, 'trajectories_used': 0}

        # Collate into batch
        batch = collate_trajectories(processed_trajectories)

        # Move batch to device
        batch = self._batch_to_device(batch)

        # Compute loss
        loss = self.algorithm.compute_loss(self.model, batch)

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()

        # Optional: gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        # Update weights
        self.optimizer.step()

        # Update state
        self.step += 1
        self.trajectories_seen += len(processed_trajectories)

        # Metrics
        metrics = {
            'loss': loss.item(),
            'trajectories_used': len(processed_trajectories),
            'trajectories_filtered': len(trajectories) - len(processed_trajectories),
            'avg_trajectory_length': float(np.mean(batch['trajectory_lengths'])),
            'avg_trajectory_reward': float(np.mean([t.total_reward() for t in processed_trajectories]))
        }

        logger.info(f"Step {self.step}: loss={metrics['loss']:.4f}, "
                   f"trajs={metrics['trajectories_used']}")

        return metrics

    def _batch_to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Move batch tensors to training device."""
        device_batch = {}
        for key, value in batch.items():
            if isinstance(value, np.ndarray):
                device_batch[key] = torch.from_numpy(value).to(self.device)
            elif isinstance(value, torch.Tensor):
                device_batch[key] = value.to(self.device)
            else:
                device_batch[key] = value
        return device_batch

    def save_checkpoint(self, filename: Optional[str] = None):
        """
        Save model checkpoint.

        Args:
            filename: Optional filename, uses step number if None
        """
        if self.checkpoint_dir is None:
            logger.warning("No checkpoint_dir configured, skipping save")
            return

        if filename is None:
            filename = f"checkpoint_step_{self.step}.pt"

        filepath = self.checkpoint_dir / filename

        checkpoint = {
            'step': self.step,
            'trajectories_seen': self.trajectories_seen,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }

        try:
            torch.save(checkpoint, filepath)
            logger.info(f"Saved checkpoint to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")

    def load_checkpoint(self, filepath: Path):
        """Load model checkpoint."""
        try:
            checkpoint = torch.load(filepath, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.step = checkpoint['step']
            self.trajectories_seen = checkpoint['trajectories_seen']
            logger.info(f"Loaded checkpoint from {filepath} at step {self.step}")
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")

    def run_training_loop(self, num_iterations: int):
        """
        Run the complete training loop.

        Args:
            num_iterations: Number of training iterations
        """
        for iteration in range(num_iterations):
            logger.info(f"Training iteration {iteration + 1}/{num_iterations}")

            # TODO: Implement full training loop with data collection
            # 1. Collect trajectories using actors
            # 2. Train on collected data
            # 3. Log metrics
            # 4. Save checkpoints

            self.global_step += 1
