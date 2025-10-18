#!/usr/bin/env python3
"""
Training script for Qwen2.5-VL-3B-Instruct with LoRA fine-tuning.

This script demonstrates:
- Loading Qwen2.5-VL with HuggingFace
- Flexible LoRA targeting options for experimentation
- Integration with ActorPoolManager and Trainer
- Full Actor-Learner training loop

Usage:
    # Default: attention+mlp targeting (recommended)
    python examples/train_qwen25_vl.py --vm-url http://VM_IP:8000

    # Experiment with attention-only LoRA
    python examples/train_qwen25_vl.py --vm-url http://VM_IP:8000 --lora-target attention

    # Experiment with MLP-only LoRA
    python examples/train_qwen25_vl.py --vm-url http://VM_IP:8000 --lora-target mlp

    # Use all linear layers
    python examples/train_qwen25_vl.py --vm-url http://VM_IP:8000 --lora-target all-linear

    # Custom targeting
    python examples/train_qwen25_vl.py --vm-url http://VM_IP:8000 --lora-target custom --lora-custom-modules q_proj k_proj gate_proj

    # Full fine-tuning (no LoRA)
    python examples/train_qwen25_vl.py --vm-url http://VM_IP:8000 --no-lora
"""

import argparse
import logging
import sys
import queue
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.vlm_wrapper import VLMWrapper
from src.learner.trainer import Trainer
from src.learner.algorithms.rejection_sampling import RejectionSampling
from src.orchestration import ActorPoolManager


def setup_logging(verbose: bool = False, log_file: str = None):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO

    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S',
        handlers=handlers
    )

    # Reduce noise from third-party libraries
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('transformers').setLevel(logging.INFO)


def main():
    parser = argparse.ArgumentParser(
        description="Train Qwen2.5-VL-3B-Instruct with LoRA fine-tuning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # VM Configuration
    vm_group = parser.add_argument_group('VM Configuration')
    vm_group.add_argument(
        "--vm-url",
        type=str,
        required=True,
        help="URL of ui-verifiers VM (e.g., http://34.123.45.67:8000)"
    )
    vm_group.add_argument(
        "--num-actors",
        type=int,
        default=2,
        help="Target number of concurrent actors (default: 2)"
    )
    vm_group.add_argument(
        "--max-concurrent-per-vm",
        type=int,
        default=2,
        help="Max concurrent sessions per VM - memory limit (default: 2)"
    )
    vm_group.add_argument(
        "--session-type",
        type=str,
        default="simple_data_entry",
        help="Type of session to create (default: simple_data_entry)"
    )

    # Model Configuration
    model_group = parser.add_argument_group('Model Configuration')
    model_group.add_argument(
        "--model-name",
        type=str,
        default="Qwen/Qwen2.5-VL-3B-Instruct",
        help="HuggingFace model identifier (default: Qwen/Qwen2.5-VL-3B-Instruct)"
    )
    model_group.add_argument(
        "--no-lora",
        action="store_true",
        help="Disable LoRA (full fine-tuning)"
    )

    # LoRA Configuration
    lora_group = parser.add_argument_group('LoRA Configuration')
    lora_group.add_argument(
        "--lora-target",
        type=str,
        default="attention+mlp",
        choices=["attention", "mlp", "attention+mlp", "all-linear", "custom"],
        help="LoRA targeting strategy (default: attention+mlp - recommended)"
    )
    lora_group.add_argument(
        "--lora-custom-modules",
        type=str,
        nargs="+",
        help="Custom module names (only used with --lora-target custom)"
    )
    lora_group.add_argument(
        "--lora-rank",
        type=int,
        default=32,
        help="LoRA rank (default: 32)"
    )
    lora_group.add_argument(
        "--lora-alpha",
        type=int,
        default=32,
        help="LoRA alpha (default: 32)"
    )
    lora_group.add_argument(
        "--lora-dropout",
        type=float,
        default=0.05,
        help="LoRA dropout (default: 0.05)"
    )

    # Training Configuration
    train_group = parser.add_argument_group('Training Configuration')
    train_group.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Number of trajectories per training batch (default: 4)"
    )
    train_group.add_argument(
        "--num-steps",
        type=int,
        default=100,
        help="Number of training steps (default: 100)"
    )
    train_group.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Learning rate (auto-detected: 2e-4 for LoRA, 1e-5 for full fine-tuning)"
    )
    train_group.add_argument(
        "--max-steps-per-episode",
        type=int,
        default=50,
        help="Maximum steps per episode (default: 50)"
    )
    train_group.add_argument(
        "--reward-threshold",
        type=float,
        default=0.0,
        help="Rejection sampling reward threshold (default: 0.0)"
    )

    # Checkpointing
    checkpoint_group = parser.add_argument_group('Checkpointing')
    checkpoint_group.add_argument(
        "--checkpoint-dir",
        type=str,
        default="./checkpoints",
        help="Directory to save checkpoints (default: ./checkpoints)"
    )
    checkpoint_group.add_argument(
        "--save-every",
        type=int,
        default=10,
        help="Save checkpoint every N steps (default: 10)"
    )

    # Logging
    log_group = parser.add_argument_group('Logging')
    log_group.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose (DEBUG) logging"
    )
    log_group.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Optional log file path"
    )

    args = parser.parse_args()

    # Validation
    if args.lora_target == "custom" and not args.lora_custom_modules:
        parser.error("--lora-custom-modules required when using --lora-target custom")

    # Setup logging
    setup_logging(verbose=args.verbose, log_file=args.log_file)
    logger = logging.getLogger(__name__)

    # Print configuration
    logger.info("=" * 60)
    logger.info("Qwen2.5-VL Training with LoRA")
    logger.info("=" * 60)
    logger.info(f"Model: {args.model_name}")
    logger.info(f"LoRA enabled: {not args.no_lora}")
    if not args.no_lora:
        logger.info(f"LoRA targeting: {args.lora_target}")
        if args.lora_target == "custom":
            logger.info(f"Custom modules: {args.lora_custom_modules}")
        logger.info(f"LoRA rank: {args.lora_rank}")
        logger.info(f"LoRA alpha: {args.lora_alpha}")
    logger.info(f"VM URL: {args.vm_url}")
    logger.info(f"Target concurrent actors: {args.num_actors}")
    logger.info(f"Max concurrent per VM: {args.max_concurrent_per_vm}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Training steps: {args.num_steps}")
    logger.info(f"Learning rate: {args.learning_rate or 'auto-detected'}")
    logger.info(f"Checkpoint dir: {args.checkpoint_dir}")
    logger.info("=" * 60)

    # Create checkpoint directory
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Load VLM with LoRA
    logger.info("\nLoading Qwen2.5-VL model...")

    lora_config = None
    if not args.no_lora:
        lora_config = {
            "r": args.lora_rank,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
        }
        if args.lora_target == "custom":
            lora_config["custom_modules"] = args.lora_custom_modules

    try:
        model = VLMWrapper(
            model_name=args.model_name,
            use_lora=not args.no_lora,
            lora_config=lora_config,
            lora_target_option=args.lora_target if not args.no_lora else None,
            max_new_tokens=128,
            temperature=0.7,
            do_sample=True
        )
        logger.info("Model loaded successfully!")
    except Exception as e:
        logger.error(f"Failed to load model: {e}", exc_info=True)
        return 1

    # Create shared trajectory queue
    trajectory_queue = queue.Queue()

    # Create algorithm (Rejection Sampling)
    logger.info("\nCreating training algorithm...")
    algorithm = RejectionSampling(reward_threshold=args.reward_threshold)

    # Create Trainer
    logger.info("Creating Trainer...")
    trainer = Trainer(
        model=model,
        algorithm=algorithm,
        trajectory_queue=trajectory_queue,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,  # Will auto-detect if None
        checkpoint_dir=checkpoint_dir
    )

    # Create ActorPoolManager
    logger.info("\nCreating ActorPoolManager...")
    actor_pool = ActorPoolManager(
        target_concurrent_actors=args.num_actors,
        vm_urls=[args.vm_url],  # Can support multiple VMs
        max_concurrent_per_vm=args.max_concurrent_per_vm,
        model=model,
        trajectory_queue=trajectory_queue,
        task_prompt="Complete the data entry task",
        session_type=args.session_type,
        max_steps_per_episode=args.max_steps_per_episode,
        action_format="json"
    )

    # Start actor pool
    logger.info("\nStarting actor pool...")
    actor_pool.start()

    # Run training loop
    logger.info("\nStarting training...")
    start_time = datetime.now()

    try:
        metrics = trainer.train(
            num_steps=args.num_steps,
            save_every=args.save_every
        )
    except KeyboardInterrupt:
        logger.info("\nTraining interrupted by user")
    except Exception as e:
        logger.error(f"\nTraining failed: {e}", exc_info=True)
    finally:
        # Always stop actor pool gracefully
        logger.info("\nStopping actor pool...")
        actor_pool.stop(timeout=30)

    # Training complete
    elapsed = (datetime.now() - start_time).total_seconds()

    logger.info("\n" + "=" * 60)
    logger.info("TRAINING SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Training time: {elapsed:.1f}s ({elapsed/60:.1f}m)")
    logger.info(f"Training steps completed: {trainer.step}")
    logger.info(f"Total trajectories processed: {trainer.trajectories_seen}")

    # Actor pool stats
    pool_stats = actor_pool.get_stats()
    logger.info(f"Episodes collected: {pool_stats['total_episodes_collected']}")
    logger.info(f"Total actors spawned: {pool_stats['total_actors_spawned']}")
    logger.info(f"Actors crashed: {pool_stats['actors_crashed']}")

    if trainer.step > 0:
        logger.info(f"Average trajectories per step: {trainer.trajectories_seen / trainer.step:.1f}")
        logger.info(f"Steps per second: {trainer.step / elapsed:.2f}")

    # Final checkpoint
    logger.info("\nSaving final checkpoint...")
    trainer.save_checkpoint(filename="final_checkpoint.pt")

    # Save model
    final_model_path = checkpoint_dir / "final_model"
    logger.info(f"Saving final model to {final_model_path}...")
    model.save_pretrained(str(final_model_path))

    logger.info("\n✓ Training complete!")
    logger.info(f"Model saved to: {final_model_path}")
    logger.info(f"Checkpoints saved to: {checkpoint_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
