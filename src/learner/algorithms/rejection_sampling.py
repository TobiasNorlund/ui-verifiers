"""Rejection sampling algorithm implementation."""

from typing import List, Dict, Any, Optional
import logging
import torch
import torch.nn as nn
from .base import Algorithm
from ...data_utils.trajectory import Trajectory

logger = logging.getLogger(__name__)


class RejectionSampling(Algorithm):
    """
    Rejection Sampling: Only train on trajectories with positive total reward.
    Uses supervised learning (behavior cloning) on successful episodes.

    Design: Simplest possible RL algorithm. Great starting point.
    """

    def __init__(self, reward_threshold: float = 0.0):
        """
        Args:
            reward_threshold: Minimum total reward to keep trajectory
        """
        self.reward_threshold = reward_threshold

    def process_trajectories(self, trajectories: List[Trajectory]) -> List[Trajectory]:
        """Filter out trajectories with low total reward."""
        filtered = [t for t in trajectories if t.total_reward() > self.reward_threshold]
        logger.info(f"Rejection sampling: kept {len(filtered)}/{len(trajectories)} trajectories")
        return filtered

    def compute_loss(
        self,
        model: nn.Module,
        batch: Dict[str, Any],
        model_outputs: Optional[Dict[str, torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        Supervised learning loss on successful trajectories.

        Assumes model outputs action logits that can be compared to ground truth actions.
        """
        if model_outputs is None:
            # Run model inference
            # Note: Actual implementation depends on your VLM's interface
            model_outputs = model(
                images=batch['images'],
                prompts=batch['prompts']
            )

        # Extract action logits from model output
        # This is a placeholder - adjust based on your VLM's actual output format
        action_logits = model_outputs['action_logits']  # [B*T, action_dim]

        # Convert ground truth actions to tensor
        # Placeholder - adjust based on your action space
        target_actions = self._actions_to_tensor(batch['actions'])

        # Cross-entropy loss for action prediction
        loss = nn.functional.cross_entropy(action_logits, target_actions)

        return loss

    def _actions_to_tensor(self, actions: List[Dict[str, Any]]) -> torch.Tensor:
        """
        Convert action dicts to tensor format.

        Placeholder implementation - you'll need to implement this based on
        your specific action space representation.
        """
        # Example: If actions are {"type": "click", "x": 100, "y": 200}
        # You might discretize x,y into bins and create a single action index
        raise NotImplementedError("Implement based on your action space")
