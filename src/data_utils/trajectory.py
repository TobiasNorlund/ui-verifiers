"""Trajectory data structure for storing episode information."""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import numpy as np


@dataclass
class Trajectory:
    """
    Represents a single episode trajectory.

    Design: Simple dataclass that just holds episode data.
    No logic here - keeps it easy to serialize/deserialize.
    """
    observations: List[np.ndarray]  # Screenshots [T, H, W, C]
    actions: List[Dict[str, Any]]   # Action dicts [T], e.g. {"type": "click", "x": 100, "y": 200}
    rewards: List[float]            # Rewards [T]
    prompts: List[str]              # Task prompts [T]
    metadata: Dict[str, Any] = field(default_factory=dict)  # task_id, success, timestamps, etc.

    def __len__(self) -> int:
        """Return the length of the trajectory."""
        return len(self.observations)

    def total_reward(self) -> float:
        """Convenience method for filtering."""
        return sum(self.rewards)

    def to_dict(self) -> Dict[str, Any]:
        """For serialization."""
        return {
            'observations': np.stack(self.observations) if self.observations else np.array([]),
            'actions': self.actions,
            'rewards': np.array(self.rewards),
            'prompts': self.prompts,
            'metadata': self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Trajectory':
        """For deserialization."""
        return cls(
            observations=list(data['observations']),
            actions=data['actions'],
            rewards=list(data['rewards']),
            prompts=data['prompts'],
            metadata=data['metadata']
        )
