"""Distributed training coordinator for multi-VM setups."""

from typing import List, Dict, Any, Optional
from .vm_manager import VMManager
from .env_pool import EnvPool


class DistributedRunner:
    """
    Coordinates distributed training across multiple VMs.

    Manages the distribution of tasks to actor VMs and
    aggregates results for the central learner.
    """

    def __init__(
        self,
        vm_manager: VMManager,
        env_pool: EnvPool,
        config: Dict[str, Any]
    ):
        """
        Initialize distributed runner.

        Args:
            vm_manager: VM manager instance
            env_pool: Pool of environments
            config: Configuration dictionary
        """
        self.vm_manager = vm_manager
        self.env_pool = env_pool
        self.config = config

        self.num_actors = config.get('num_actors', 4)
        self.batch_size = config.get('batch_size', 32)

    def distribute_tasks(
        self,
        tasks: List[Dict[str, Any]]
    ) -> List[str]:
        """
        Distribute tasks to actor VMs.

        Args:
            tasks: List of task specifications

        Returns:
            List of job IDs for tracking
        """
        # TODO: Implement task distribution to VMs
        job_ids = []

        for i, task in enumerate(tasks):
            job_id = f"job_{i}"
            # Send task to available actor VM
            job_ids.append(job_id)

        return job_ids

    def collect_results(
        self,
        job_ids: List[str],
        timeout: int = 300
    ) -> List[Any]:
        """
        Collect results from distributed actors.

        Args:
            job_ids: List of job IDs to collect
            timeout: Timeout in seconds

        Returns:
            List of results from actors
        """
        # TODO: Implement result collection from VMs
        results = []

        for job_id in job_ids:
            # Fetch result for job_id
            result = None  # Placeholder
            results.append(result)

        return results

    def scale_actors(self, target_count: int):
        """
        Scale the number of actor VMs.

        Args:
            target_count: Desired number of actors
        """
        self.vm_manager.scale_vms(target_count)
        self.num_actors = target_count

    def shutdown(self):
        """Shutdown all distributed resources."""
        self.env_pool.close_all()
        # Clean up VMs if needed
        print("Shutting down distributed runner")
