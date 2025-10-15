"""Collation utilities for batching trajectories."""

from typing import List, Dict, Any
import numpy as np
from .trajectory import Trajectory


def collate_trajectories(trajectories: List[Trajectory]) -> Dict[str, Any]:
    """
    Collate multiple trajectories into a single training batch.

    Design: Standalone function (not part of Trajectory class) for flexibility.
    Can be easily customized without modifying Trajectory.

    Args:
        trajectories: List of trajectories to batch together

    Returns:
        Dict containing batched data ready for model training:
        - images: [B*T, H, W, C] stacked observations
        - actions: [B*T] flattened actions
        - rewards: [B*T] flattened rewards
        - prompts: [B*T] flattened prompts
        - trajectory_lengths: [B] length of each trajectory (for masking if needed)
        - trajectory_indices: [B*T] which trajectory each timestep belongs to
    """
    if not trajectories:
        raise ValueError("Cannot collate empty list of trajectories")

    all_observations = []
    all_actions = []
    all_rewards = []
    all_prompts = []
    trajectory_lengths = []
    trajectory_indices = []

    for traj_idx, traj in enumerate(trajectories):
        all_observations.extend(traj.observations)
        all_actions.extend(traj.actions)
        all_rewards.extend(traj.rewards)
        all_prompts.extend(traj.prompts)
        trajectory_lengths.append(len(traj))
        trajectory_indices.extend([traj_idx] * len(traj))

    return {
        'images': np.stack(all_observations) if all_observations else np.array([]),
        'actions': all_actions,
        'rewards': np.array(all_rewards),
        'prompts': all_prompts,
        'trajectory_lengths': trajectory_lengths,
        'trajectory_indices': trajectory_indices
    }
