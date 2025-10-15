#!/usr/bin/env python3
"""
Example: Multi-actor training with parallel data collection

This example demonstrates how to use multiple TaskRunner instances
for parallel trajectory collection from multiple UI environments.
"""

import sys
from pathlib import Path
import queue
import torch
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.vlm_wrapper import VLMWrapper
from src.learner.trainer import Trainer
from src.learner.algorithms.rejection_sampling import RejectionSampling
from src.orchestration.multi_actor_runner import MultiActorRunner
from src.utils.logging_utils import setup_logger

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def train_with_multiple_actors():
    """Train with multiple actors collecting data in parallel."""

    print("=" * 70)
    print(" Multi-Actor Training Example")
    print("=" * 70)

    # Configuration
    num_actors = 4
    batch_size = 16
    num_training_steps = 100
    checkpoint_every = 25

    # UI environment URLs (one per actor)
    # In production, these would be different UI environment instances
    ui_env_urls = [
        f"http://localhost:{8000 + i}" for i in range(num_actors)
    ]

    print(f"\nConfiguration:")
    print(f"  Number of actors: {num_actors}")
    print(f"  Batch size: {batch_size}")
    print(f"  Training steps: {num_training_steps}")
    print(f"  UI environments: {ui_env_urls}")

    # Step 1: Initialize model
    print("\n[1/5] Initializing VLM...")
    vlm = VLMWrapper(
        model_name="Qwen/Qwen2-VL-2B-Instruct",
        device="cuda" if torch.cuda.is_available() else "cpu",
        torch_dtype=torch.float16,
        use_lora=True,
        lora_config={
            "r": 16,
            "lora_alpha": 32,
            "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
            "lora_dropout": 0.05
        }
    )
    print(f"  Model loaded: {vlm.get_trainable_parameters() / 1e6:.1f}M trainable params")

    # Step 2: Create algorithm
    print("\n[2/5] Setting up algorithm...")
    algorithm = RejectionSampling(reward_threshold=0.5)

    # Step 3: Create trainer
    print("\n[3/5] Initializing trainer...")
    checkpoint_dir = Path("experiments/multi_actor/checkpoints")
    trainer = Trainer(
        model=vlm,
        algorithm=algorithm,
        batch_size=batch_size,
        learning_rate=1e-5,
        checkpoint_dir=checkpoint_dir
    )

    # Step 4: Setup multi-actor data collection
    print("\n[4/5] Setting up multi-actor runner...")
    trajectory_queue = queue.Queue(maxsize=batch_size * 5)

    task_prompt = """You are a UI automation agent. Given a screenshot and task, generate the next action.

Available actions:
- click(x, y): Click at coordinates
- type("text"): Type text
- scroll(direction, amount): Scroll

Task: Click the login button
Action:"""

    multi_actor = MultiActorRunner(
        model=vlm,
        trajectory_queue=trajectory_queue,
        ui_env_urls=ui_env_urls,
        task_prompt=task_prompt,
        max_steps_per_episode=50,
        data_dir=Path("experiments/multi_actor/trajectories"),
        action_format="text"
    )

    # Start actors
    multi_actor.start()
    print(f"  Started {num_actors} actors")

    # Step 5: Training loop
    print("\n[5/5] Starting training loop...")
    print("-" * 70)

    try:
        for step in range(num_training_steps):
            # Collect batch of trajectories
            trajectories = []
            print(f"\nStep {step + 1}/{num_training_steps}: Collecting {batch_size} trajectories...")

            for i in range(batch_size):
                traj = trajectory_queue.get()  # Blocks until trajectory available
                trajectories.append(traj)
                print(f"  [{i+1}/{batch_size}] Collected trajectory (reward: {traj.total_reward():.2f})")

            # Train
            metrics = trainer.train_step(trajectories)

            print(f"\nTraining metrics:")
            print(f"  Loss: {metrics['loss']:.4f}")
            print(f"  Trajectories used: {metrics['trajectories_used']}/{len(trajectories)}")
            print(f"  Avg trajectory reward: {metrics['avg_trajectory_reward']:.2f}")

            # Actor statistics
            stats = multi_actor.get_statistics()
            print(f"\nActor statistics:")
            print(f"  Total episodes: {stats['total_episodes']}")
            print(f"  Episodes per actor: {stats['episodes_per_actor']}")
            print(f"  Total errors: {stats['total_errors']}")
            print(f"  Active actors: {stats['active_threads']}")

            # Checkpoint
            if (step + 1) % checkpoint_every == 0:
                trainer.save_checkpoint()
                print(f"\n  Checkpoint saved at step {step + 1}")

        print("\n" + "=" * 70)
        print(" Training completed successfully!")
        print("=" * 70)

        # Final statistics
        final_stats = multi_actor.get_statistics()
        print(f"\nFinal statistics:")
        print(f"  Total episodes collected: {final_stats['total_episodes']}")
        print(f"  Total errors: {final_stats['total_errors']}")
        print(f"  Average episodes per actor: {final_stats['total_episodes'] / num_actors:.1f}")

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")

    except Exception as e:
        logger.error(f"Training error: {e}", exc_info=True)

    finally:
        # Stop actors
        print("\nStopping actors...")
        multi_actor.stop()
        print("Actors stopped")

        # Save final checkpoint
        trainer.save_checkpoint("final_checkpoint.pt")
        print("Final checkpoint saved")


def train_with_worker_pool():
    """
    Train with worker pool pattern (for diverse tasks).

    This pattern is useful when you have many different tasks
    and want workers to dynamically pick them up.
    """
    from src.orchestration.multi_actor_runner import ActorPool

    print("=" * 70)
    print(" Worker Pool Training Example")
    print("=" * 70)

    # Initialize model
    vlm = VLMWrapper(
        model_name="Qwen/Qwen2-VL-2B-Instruct",
        device="cuda" if torch.cuda.is_available() else "cpu",
        torch_dtype=torch.float16,
        use_lora=True
    )

    # Create queues
    task_queue = queue.Queue()
    trajectory_queue = queue.Queue()

    # Add diverse tasks
    tasks = [
        ("http://localhost:8000", "Click the login button"),
        ("http://localhost:8001", "Fill out the form"),
        ("http://localhost:8002", "Navigate to settings"),
        ("http://localhost:8003", "Search for items"),
        ("http://localhost:8004", "Add item to cart"),
    ]

    for task in tasks:
        task_queue.put(task)

    print(f"\nAdded {len(tasks)} tasks to queue")

    # Create worker pool
    num_workers = 3
    actor_pool = ActorPool(
        num_actors=num_workers,
        model=vlm,
        trajectory_queue=trajectory_queue,
        task_queue=task_queue,
        max_steps_per_episode=50,
        action_format="text"
    )

    print(f"Starting {num_workers} workers...")
    actor_pool.start()

    # Collect results
    print("\nCollecting trajectories as workers complete tasks...")
    trajectories = []

    try:
        while len(trajectories) < len(tasks):
            traj = trajectory_queue.get(timeout=60)  # 60 second timeout
            trajectories.append(traj)
            print(f"  Collected {len(trajectories)}/{len(tasks)} trajectories")

        print(f"\nAll {len(trajectories)} trajectories collected!")

    except queue.Empty:
        print("\nTimeout waiting for trajectories")

    finally:
        actor_pool.stop()
        print("Workers stopped")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Multi-actor training example")
    parser.add_argument(
        "--mode",
        choices=["fixed", "pool"],
        default="fixed",
        help="Training mode: 'fixed' for fixed actors, 'pool' for worker pool"
    )

    args = parser.parse_args()

    if args.mode == "fixed":
        train_with_multiple_actors()
    else:
        train_with_worker_pool()
