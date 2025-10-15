"""Pool of UI environment instances for parallel data collection."""

from typing import List, Dict, Any, Optional
from ..actor.env_client import EnvClient


class EnvPool:
    """
    Manages a pool of UI environment instances.

    Coordinates multiple environment instances for parallel
    trajectory collection across different actors.
    """

    def __init__(
        self,
        env_urls: List[str],
        max_concurrent: int = 10
    ):
        """
        Initialize environment pool.

        Args:
            env_urls: List of environment instance URLs
            max_concurrent: Maximum concurrent environments
        """
        self.env_urls = env_urls
        self.max_concurrent = max_concurrent

        # Create environment clients
        self.envs = [EnvClient(url) for url in env_urls[:max_concurrent]]
        self.available_envs = list(range(len(self.envs)))
        self.busy_envs = []

    def acquire(self) -> Optional[EnvClient]:
        """
        Acquire an available environment from the pool.

        Returns:
            EnvClient instance or None if none available
        """
        if not self.available_envs:
            return None

        env_idx = self.available_envs.pop(0)
        self.busy_envs.append(env_idx)
        return self.envs[env_idx]

    def release(self, env: EnvClient):
        """
        Release an environment back to the pool.

        Args:
            env: EnvClient to release
        """
        # Find environment index
        env_idx = None
        for idx, pool_env in enumerate(self.envs):
            if pool_env is env:
                env_idx = idx
                break

        if env_idx is not None and env_idx in self.busy_envs:
            self.busy_envs.remove(env_idx)
            self.available_envs.append(env_idx)

    def available_count(self) -> int:
        """Return number of available environments."""
        return len(self.available_envs)

    def close_all(self):
        """Close all environment connections."""
        for env in self.envs:
            env.close()
