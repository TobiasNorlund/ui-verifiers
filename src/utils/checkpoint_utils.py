"""Checkpoint utilities for saving and loading model states."""

from pathlib import Path
from typing import Dict, Any, Optional
import json
import shutil


class CheckpointManager:
    """
    Manages model checkpoints during training.

    Handles saving, loading, and cleanup of checkpoints.
    """

    def __init__(
        self,
        checkpoint_dir: Path,
        max_to_keep: int = 5
    ):
        """
        Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory for checkpoints
            max_to_keep: Maximum number of checkpoints to keep
        """
        self.checkpoint_dir = checkpoint_dir
        self.max_to_keep = max_to_keep
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.checkpoints = []

    def save(
        self,
        model_state: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
        name: Optional[str] = None
    ) -> Path:
        """
        Save a checkpoint.

        Args:
            model_state: Model state dictionary
            metadata: Optional metadata to save
            name: Optional checkpoint name

        Returns:
            Path to saved checkpoint
        """
        if name is None:
            name = f"checkpoint_{len(self.checkpoints)}"

        checkpoint_path = self.checkpoint_dir / name
        checkpoint_path.mkdir(exist_ok=True)

        # Save model state
        # TODO: Implement actual model state saving

        # Save metadata
        if metadata:
            metadata_file = checkpoint_path / 'metadata.json'
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)

        # Track checkpoint
        self.checkpoints.append(checkpoint_path)

        # Clean up old checkpoints
        self._cleanup_old_checkpoints()

        return checkpoint_path

    def load(self, checkpoint_path: Path) -> Dict[str, Any]:
        """
        Load a checkpoint.

        Args:
            checkpoint_path: Path to checkpoint directory

        Returns:
            Dictionary containing model state and metadata
        """
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        # TODO: Implement actual model state loading

        # Load metadata
        metadata = {}
        metadata_file = checkpoint_path / 'metadata.json'
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)

        return {
            'model_state': {},
            'metadata': metadata
        }

    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints beyond max_to_keep."""
        if len(self.checkpoints) > self.max_to_keep:
            old_checkpoints = self.checkpoints[:-self.max_to_keep]
            for checkpoint in old_checkpoints:
                if checkpoint.exists():
                    shutil.rmtree(checkpoint)
            self.checkpoints = self.checkpoints[-self.max_to_keep:]

    def list_checkpoints(self) -> list:
        """
        List all available checkpoints.

        Returns:
            List of checkpoint paths
        """
        return list(self.checkpoint_dir.glob('checkpoint_*'))
