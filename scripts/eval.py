#!/usr/bin/env python3
"""Evaluation script for trained UI VLM models."""

import argparse
import yaml
from pathlib import Path
import sys
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.vlm_wrapper import VLMWrapper
from src.actor.task_runner import TaskRunner
from src.actor.env_client import EnvClient
from src.parsers.action_decoder import ActionDecoder
from src.utils.logging_utils import setup_logger


def load_config(config_path: Path) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_tasks(tasks_file: Path) -> list:
    """Load evaluation tasks from JSON file."""
    with open(tasks_file, 'r') as f:
        tasks = json.load(f)
    return tasks


def evaluate_model(
    vlm: VLMWrapper,
    tasks: list,
    config: dict,
    logger
) -> dict:
    """
    Evaluate model on tasks.

    Args:
        vlm: VLM wrapper
        tasks: List of evaluation tasks
        config: Configuration dictionary
        logger: Logger instance

    Returns:
        Dictionary of evaluation metrics
    """
    # Create environment client
    # TODO: Get actual environment URL from config
    env = EnvClient(env_url="http://localhost:8000")

    # Create action decoder
    action_decoder = ActionDecoder()

    # Create task runner
    task_runner = TaskRunner(
        vlm_wrapper=vlm,
        env_client=env,
        action_decoder=action_decoder,
        max_steps=config['actor']['max_steps_per_episode']
    )

    # Run evaluation
    results = []
    total_reward = 0.0
    success_count = 0

    for i, task in enumerate(tasks):
        logger.info(f"Evaluating task {i+1}/{len(tasks)}: {task.get('description', 'N/A')}")

        try:
            trajectory = task_runner.run_episode(task)

            # Calculate metrics
            episode_reward = sum(trajectory.rewards)
            success = episode_reward > 0  # Simple success criterion

            total_reward += episode_reward
            if success:
                success_count += 1

            results.append({
                'task_id': i,
                'task': task,
                'reward': episode_reward,
                'steps': len(trajectory),
                'success': success
            })

            logger.info(f"Task {i+1} - Reward: {episode_reward:.2f}, Steps: {len(trajectory)}, Success: {success}")

        except Exception as e:
            logger.error(f"Error evaluating task {i+1}: {e}", exc_info=True)
            results.append({
                'task_id': i,
                'task': task,
                'error': str(e)
            })

    # Calculate aggregate metrics
    num_tasks = len(tasks)
    avg_reward = total_reward / num_tasks if num_tasks > 0 else 0
    success_rate = success_count / num_tasks if num_tasks > 0 else 0

    metrics = {
        'num_tasks': num_tasks,
        'total_reward': total_reward,
        'avg_reward': avg_reward,
        'success_count': success_count,
        'success_rate': success_rate,
        'results': results
    }

    return metrics


def main():
    """Main evaluation entry point."""
    parser = argparse.ArgumentParser(description='Evaluate trained UI VLM model')
    parser.add_argument(
        '--config',
        type=str,
        default='config/default_config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--tasks',
        type=str,
        required=True,
        help='Path to evaluation tasks JSON file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='evaluation_results.json',
        help='Output file for results'
    )
    args = parser.parse_args()

    # Load configuration
    config = load_config(Path(args.config))

    # Setup logging
    logger = setup_logger('eval')
    logger.info(f"Starting evaluation with checkpoint: {args.checkpoint}")

    # Load tasks
    tasks = load_tasks(Path(args.tasks))
    logger.info(f"Loaded {len(tasks)} evaluation tasks")

    # Create VLM wrapper
    vlm = VLMWrapper(
        model_name=config['model']['name'],
        config=config['model']
    )

    # TODO: Load checkpoint weights into VLM
    logger.info(f"Loading checkpoint from {args.checkpoint}")

    # Run evaluation
    metrics = evaluate_model(vlm, tasks, config, logger)

    # Log results
    logger.info("=" * 50)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 50)
    logger.info(f"Number of tasks: {metrics['num_tasks']}")
    logger.info(f"Average reward: {metrics['avg_reward']:.4f}")
    logger.info(f"Success rate: {metrics['success_rate']:.2%}")
    logger.info(f"Total reward: {metrics['total_reward']:.2f}")

    # Save results
    output_path = Path(args.output)
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"Results saved to {output_path}")


if __name__ == '__main__':
    main()
