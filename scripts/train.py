#!/usr/bin/env python3
"""Main training script for UI VLM training."""

import argparse
import yaml
from pathlib import Path
import sys
import queue
import threading
import torch.nn as nn
from typing import Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.learner.trainer import Trainer
from src.learner.algorithms.rejection_sampling import RejectionSampling
from src.learner.algorithms.ppo import PPO
from src.learner.algorithms.base import Algorithm
from src.models.vlm_wrapper import VLMWrapper
from src.actor.task_runner import TaskRunner
from src.actor.env_client import EnvClient
from src.utils.logging_utils import setup_logger, MetricsLogger


def load_config(config_path: Path) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_algorithm(config: dict):
    """Create algorithm instance based on config."""
    algo_name = config['training']['algorithm']

    if algo_name == 'rejection_sampling':
        reward_threshold = config.get('algorithm', {}).get('reward_threshold', 0.0)
        return RejectionSampling(reward_threshold=reward_threshold)

    elif algo_name == 'ppo':
        algo_config = config.get('algorithm', {})
        return PPO(
            clip_epsilon=algo_config.get('clip_epsilon', 0.2),
            gamma=algo_config.get('gamma', 0.99),
            gae_lambda=algo_config.get('gae_lambda', 0.95)
        )

    else:
        raise ValueError(f"Unknown algorithm: {algo_name}")


def train_with_orchestration(
    model: nn.Module,
    ui_env_url: str,
    task_prompt: str,
    algorithm: Algorithm,
    num_training_steps: int = 1000,
    batch_size: int = 32,
    num_actors: int = 1,
    checkpoint_every: int = 100,
    checkpoint_dir: Optional[Path] = None,
    data_dir: Optional[Path] = None
):
    """
    Main training loop that orchestrates actors and learner.

    Design: Simple single-threaded version to start. Can be extended to:
    - Multiple TaskRunner threads/processes for parallel collection
    - Distributed training with multiple Trainers
    - Asynchronous updates

    Args:
        model: VLM to train
        ui_env_url: URL of UI environment service
        task_prompt: Task description
        algorithm: RL algorithm to use
        num_training_steps: Number of training steps
        batch_size: Trajectories per training batch
        num_actors: Number of parallel actors (currently only 1 supported)
        checkpoint_every: Save checkpoint every N steps
        checkpoint_dir: Where to save checkpoints
        data_dir: Where to save raw trajectories
    """
    # Setup
    logger = setup_logger('train_orchestration')
    logger.info("Initializing training...")

    # Create trajectory queue
    # maxsize limits memory usage (actors will block if queue is full)
    trajectory_queue = queue.Queue(maxsize=batch_size * 5)

    # Create trainer
    trainer = Trainer(
        model=model,
        algorithm=algorithm,
        batch_size=batch_size,
        checkpoint_dir=checkpoint_dir
    )

    # Create task runner(s)
    # Note: Currently only supports 1 actor in same process
    # For multiple actors, use threading.Thread or multiprocessing.Process
    task_runner = TaskRunner(
        ui_env_url=ui_env_url,
        model=model,  # Shared reference (same model)
        trajectory_queue=trajectory_queue,
        task_prompt=task_prompt,
        data_dir=data_dir
    )

    # Start actor in separate thread
    actor_thread = threading.Thread(
        target=task_runner.run,
        kwargs={'num_episodes': None},  # Infinite
        daemon=True  # Thread dies when main thread exits
    )
    actor_thread.start()
    logger.info("Actor thread started")

    # Training loop
    logger.info("Starting training loop...")

    try:
        for step in range(num_training_steps):
            # Collect batch of trajectories
            trajectories = []
            logger.info(f"Waiting for {batch_size} trajectories...")

            while len(trajectories) < batch_size:
                trajectory = trajectory_queue.get()  # Blocks until available
                trajectories.append(trajectory)
                logger.debug(f"Collected trajectory {len(trajectories)}/{batch_size}")

            # Train
            metrics = trainer.train_step(trajectories)

            # Log
            logger.info(f"Training step {step+1}/{num_training_steps}: {metrics}")

            # Checkpoint
            if (step + 1) % checkpoint_every == 0:
                trainer.save_checkpoint()

            # Optional: Sync model weights to actor
            # For now, actor uses same model reference so weights are automatically synced
            # If using separate inference model, sync here:
            # task_runner.sync_model()

        logger.info("Training complete!")

    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training error: {e}", exc_info=True)
    finally:
        # Save final checkpoint
        trainer.save_checkpoint("final_checkpoint.pt")


def main():
    """Main training entry point."""
    parser = argparse.ArgumentParser(description='Train UI VLM model')
    parser.add_argument(
        '--config',
        type=str,
        default='config/default_config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--experiment-name',
        type=str,
        default='default',
        help='Name of the experiment'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume from'
    )
    args = parser.parse_args()

    # Load configuration
    config_path = Path(args.config)
    config = load_config(config_path)

    # Setup logging
    log_dir = Path(config['logging']['log_dir'])
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(
        'train',
        log_file=log_dir / 'train.log'
    )
    logger.info(f"Starting training with experiment: {args.experiment_name}")
    logger.info(f"Configuration: {config}")

    # Initialize metrics logger
    metrics_logger = MetricsLogger(log_dir)

    # Create VLM wrapper with HuggingFace model
    logger.info(f"Initializing VLM: {config['model']['name']}")

    # Parse torch dtype
    import torch
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32
    }
    torch_dtype = dtype_map.get(config['model'].get('torch_dtype', 'float16'), torch.float16)

    vlm = VLMWrapper(
        model_name=config['model']['name'],
        device=config['model'].get('device', 'cuda'),
        torch_dtype=torch_dtype,
        use_lora=config['model'].get('use_lora', False),
        lora_config=config['model'].get('lora_config'),
        max_new_tokens=config['model'].get('max_new_tokens', 128),
        temperature=config['model'].get('temperature', 0.7),
        do_sample=config['model'].get('do_sample', True)
    )

    # Optionally freeze vision encoder
    if config['model'].get('freeze_vision_encoder', False):
        vlm.freeze_vision_encoder()

    # Create algorithm
    logger.info(f"Creating algorithm: {config['training']['algorithm']}")
    algorithm = create_algorithm(config)

    # Create trainer
    checkpoint_dir = Path(config['logging']['checkpoint_dir'])
    trainer = Trainer(
        model=vlm,
        algorithm=algorithm,
        batch_size=config['training'].get('batch_size', 32),
        learning_rate=config.get('algorithm', {}).get('learning_rate', 1e-5),
        checkpoint_dir=checkpoint_dir
    )

    # Resume from checkpoint if specified
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(Path(args.resume))

    # Run training loop
    num_iterations = config['training']['num_iterations']
    logger.info(f"Starting training loop for {num_iterations} iterations")

    try:
        trainer.run_training_loop(num_iterations)
        logger.info("Training completed successfully")
    except Exception as e:
        logger.error(f"Training failed with error: {e}", exc_info=True)
        raise


if __name__ == '__main__':
    main()
