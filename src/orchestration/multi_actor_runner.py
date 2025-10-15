"""Multi-actor runner for managing multiple TaskRunner instances."""

from typing import List, Dict, Any, Optional
from pathlib import Path
import logging
import queue
import threading
import torch.nn as nn
from ..actor.task_runner import TaskRunner

logger = logging.getLogger(__name__)


class MultiActorRunner:
    """
    Manages multiple TaskRunner instances for parallel data collection.

    Design:
    - Each TaskRunner runs in its own thread
    - Each TaskRunner connects to a different UI environment instance
    - All TaskRunners share the same model and trajectory queue
    - Automatic load balancing and fault recovery
    """

    def __init__(
        self,
        model: nn.Module,
        trajectory_queue: queue.Queue,
        ui_env_urls: List[str],
        task_prompt: str,
        max_steps_per_episode: int = 50,
        screenshot_size: tuple = (224, 224),
        data_dir: Optional[Path] = None,
        device: str = "cuda",
        action_format: str = "text"
    ):
        """
        Initialize multi-actor runner.

        Args:
            model: Shared VLM for inference
            trajectory_queue: Shared queue for trajectories
            ui_env_urls: List of UI environment URLs (one per actor)
            task_prompt: Task description
            max_steps_per_episode: Max steps per episode
            screenshot_size: Screenshot size
            data_dir: Optional data directory
            device: Device for inference
            action_format: Action format for decoding
        """
        self.model = model
        self.trajectory_queue = trajectory_queue
        self.ui_env_urls = ui_env_urls
        self.task_prompt = task_prompt
        self.max_steps_per_episode = max_steps_per_episode
        self.screenshot_size = screenshot_size
        self.data_dir = Path(data_dir) if data_dir else None
        self.device = device
        self.action_format = action_format

        # Create TaskRunner instances
        self.task_runners: List[TaskRunner] = []
        self.actor_threads: List[threading.Thread] = []
        self.running = False

        # Statistics
        self.episodes_per_actor: List[int] = [0] * len(ui_env_urls)
        self.errors_per_actor: List[int] = [0] * len(ui_env_urls)

        logger.info(f"MultiActorRunner initialized with {len(ui_env_urls)} actors")

    def start(self, num_episodes_per_actor: Optional[int] = None):
        """
        Start all TaskRunner instances.

        Args:
            num_episodes_per_actor: Episodes per actor (None = infinite)
        """
        if self.running:
            logger.warning("MultiActorRunner already running")
            return

        self.running = True

        # Create and start TaskRunners
        for idx, ui_env_url in enumerate(self.ui_env_urls):
            # Create actor-specific data directory
            actor_data_dir = None
            if self.data_dir:
                actor_data_dir = self.data_dir / f"actor_{idx}"
                actor_data_dir.mkdir(parents=True, exist_ok=True)

            # Create TaskRunner
            task_runner = TaskRunner(
                ui_env_url=ui_env_url,
                model=self.model,
                trajectory_queue=self.trajectory_queue,
                task_prompt=self.task_prompt,
                max_steps_per_episode=self.max_steps_per_episode,
                screenshot_size=self.screenshot_size,
                data_dir=actor_data_dir,
                device=self.device,
                action_format=self.action_format
            )

            self.task_runners.append(task_runner)

            # Create and start thread
            thread = threading.Thread(
                target=self._run_actor,
                args=(idx, task_runner, num_episodes_per_actor),
                daemon=True,
                name=f"Actor-{idx}"
            )

            self.actor_threads.append(thread)
            thread.start()

            logger.info(f"Started actor {idx} on {ui_env_url}")

        logger.info(f"All {len(self.task_runners)} actors started")

    def _run_actor(
        self,
        actor_id: int,
        task_runner: TaskRunner,
        num_episodes: Optional[int]
    ):
        """
        Run a single actor with error handling.

        Args:
            actor_id: Actor identifier
            task_runner: TaskRunner instance
            num_episodes: Number of episodes (None = infinite)
        """
        try:
            episode = 0
            while self.running and (num_episodes is None or episode < num_episodes):
                try:
                    # Run episode
                    trajectory = task_runner.run_episode()

                    # Add to queue
                    self.trajectory_queue.put(trajectory)

                    # Update statistics
                    self.episodes_per_actor[actor_id] += 1
                    episode += 1

                    logger.debug(f"Actor {actor_id} completed episode {episode}")

                except Exception as e:
                    logger.error(f"Actor {actor_id} episode error: {e}", exc_info=True)
                    self.errors_per_actor[actor_id] += 1

                    # If too many errors, stop this actor
                    if self.errors_per_actor[actor_id] > 10:
                        logger.error(f"Actor {actor_id} exceeded error threshold, stopping")
                        break

        except Exception as e:
            logger.error(f"Actor {actor_id} fatal error: {e}", exc_info=True)

        finally:
            logger.info(f"Actor {actor_id} stopped after {self.episodes_per_actor[actor_id]} episodes")

    def stop(self):
        """Stop all actors gracefully."""
        logger.info("Stopping all actors...")
        self.running = False

        # Wait for threads to finish (with timeout)
        for idx, thread in enumerate(self.actor_threads):
            thread.join(timeout=5.0)
            if thread.is_alive():
                logger.warning(f"Actor {idx} thread did not stop gracefully")

        logger.info("All actors stopped")

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics for all actors."""
        return {
            'num_actors': len(self.task_runners),
            'total_episodes': sum(self.episodes_per_actor),
            'episodes_per_actor': self.episodes_per_actor,
            'total_errors': sum(self.errors_per_actor),
            'errors_per_actor': self.errors_per_actor,
            'active_threads': sum(1 for t in self.actor_threads if t.is_alive())
        }

    def is_running(self) -> bool:
        """Check if any actors are still running."""
        return any(thread.is_alive() for thread in self.actor_threads)

    def wait_for_completion(self, timeout: Optional[float] = None):
        """
        Wait for all actors to complete.

        Args:
            timeout: Maximum time to wait (None = infinite)
        """
        for thread in self.actor_threads:
            thread.join(timeout=timeout)

    def restart_actor(self, actor_id: int):
        """
        Restart a specific actor (useful for fault recovery).

        Args:
            actor_id: Actor to restart
        """
        if actor_id >= len(self.task_runners):
            logger.error(f"Invalid actor_id: {actor_id}")
            return

        # Stop old thread if still running
        if self.actor_threads[actor_id].is_alive():
            logger.warning(f"Actor {actor_id} thread still alive, cannot restart yet")
            return

        logger.info(f"Restarting actor {actor_id}")

        # Create new thread
        thread = threading.Thread(
            target=self._run_actor,
            args=(actor_id, self.task_runners[actor_id], None),
            daemon=True,
            name=f"Actor-{actor_id}-restarted"
        )

        self.actor_threads[actor_id] = thread
        thread.start()

        # Reset statistics
        self.errors_per_actor[actor_id] = 0


class ActorPool:
    """
    Alternative pattern: Pool of actors with dynamic task assignment.

    Use this when you have more potential UI sessions than concurrent actors,
    and want to dynamically assign sessions to available actors.
    """

    def __init__(
        self,
        num_actors: int,
        model: nn.Module,
        trajectory_queue: queue.Queue,
        task_queue: queue.Queue,  # Queue of (ui_env_url, task_prompt) tuples
        **task_runner_kwargs
    ):
        """
        Initialize actor pool.

        Args:
            num_actors: Number of concurrent actors
            model: Shared VLM
            trajectory_queue: Shared trajectory queue
            task_queue: Queue of tasks to process
            **task_runner_kwargs: Additional TaskRunner arguments
        """
        self.num_actors = num_actors
        self.model = model
        self.trajectory_queue = trajectory_queue
        self.task_queue = task_queue
        self.task_runner_kwargs = task_runner_kwargs

        self.workers: List[threading.Thread] = []
        self.running = False

        logger.info(f"ActorPool initialized with {num_actors} workers")

    def start(self):
        """Start all worker threads."""
        self.running = True

        for worker_id in range(self.num_actors):
            thread = threading.Thread(
                target=self._worker,
                args=(worker_id,),
                daemon=True,
                name=f"Worker-{worker_id}"
            )
            self.workers.append(thread)
            thread.start()

        logger.info(f"ActorPool started with {self.num_actors} workers")

    def _worker(self, worker_id: int):
        """Worker thread that processes tasks from the queue."""
        logger.info(f"Worker {worker_id} started")

        while self.running:
            try:
                # Get task from queue (with timeout to check running flag)
                try:
                    ui_env_url, task_prompt = self.task_queue.get(timeout=1.0)
                except queue.Empty:
                    continue

                logger.info(f"Worker {worker_id} processing task: {task_prompt[:50]}...")

                # Create temporary TaskRunner for this task
                task_runner = TaskRunner(
                    ui_env_url=ui_env_url,
                    model=self.model,
                    trajectory_queue=self.trajectory_queue,
                    task_prompt=task_prompt,
                    **self.task_runner_kwargs
                )

                # Run episode
                trajectory = task_runner.run_episode()

                # Add to trajectory queue
                self.trajectory_queue.put(trajectory)

                # Mark task as done
                self.task_queue.task_done()

                logger.info(f"Worker {worker_id} completed task")

            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}", exc_info=True)

        logger.info(f"Worker {worker_id} stopped")

    def stop(self):
        """Stop all workers."""
        logger.info("Stopping ActorPool...")
        self.running = False

        for thread in self.workers:
            thread.join(timeout=5.0)

        logger.info("ActorPool stopped")
