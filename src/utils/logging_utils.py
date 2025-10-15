"""Logging utilities for training and evaluation."""

import logging
from pathlib import Path
from typing import Dict, Any, Optional
import json
from datetime import datetime


def setup_logger(
    name: str,
    log_file: Optional[Path] = None,
    level: int = logging.INFO
) -> logging.Logger:
    """
    Set up a logger with console and file handlers.

    Args:
        name: Logger name
        log_file: Optional log file path
        level: Logging level

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)

    # File handler
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(console_format)
        logger.addHandler(file_handler)

    return logger


class MetricsLogger:
    """Logger for training metrics."""

    def __init__(self, log_dir: Path):
        """
        Initialize metrics logger.

        Args:
            log_dir: Directory for saving metrics
        """
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.metrics_file = log_dir / 'metrics.jsonl'
        self.current_step = 0

    def log(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """
        Log metrics for a training step.

        Args:
            metrics: Dictionary of metrics
            step: Optional step number
        """
        if step is None:
            step = self.current_step
            self.current_step += 1

        log_entry = {
            'step': step,
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics
        }

        with open(self.metrics_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')

    def load_metrics(self) -> list:
        """
        Load all logged metrics.

        Returns:
            List of metric entries
        """
        if not self.metrics_file.exists():
            return []

        metrics = []
        with open(self.metrics_file, 'r') as f:
            for line in f:
                metrics.append(json.loads(line))

        return metrics
