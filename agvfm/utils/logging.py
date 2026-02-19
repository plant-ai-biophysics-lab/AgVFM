"""Logging utilities for experiments."""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


def setup_logging(
    log_dir: Path,
    experiment_name: str,
    level: int = logging.INFO,
) -> Path:
    """
    Set up logging for an experiment.
    
    Args:
        log_dir: Directory to save log files
        experiment_name: Name of the experiment
        level: Logging level
        
    Returns:
        Path to the log file
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"{experiment_name}_{timestamp}.log"
    
    # Configure logging
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout),
        ],
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging to {log_file}")
    
    return log_file


def log_experiment_start(experiment_name: str, config: dict):
    """Log experiment start with configuration."""
    logger = logging.getLogger(__name__)
    logger.info("=" * 70)
    logger.info(f"Starting experiment: {experiment_name}")
    logger.info("=" * 70)
    logger.info(f"Configuration:")
    for key, value in config.items():
        logger.info(f"  {key}: {value}")
    logger.info("=" * 70)


def log_experiment_complete(experiment_name: str, results_file: Path):
    """Log experiment completion."""
    logger = logging.getLogger(__name__)
    logger.info("=" * 70)
    logger.info(f"Experiment complete: {experiment_name}")
    logger.info(f"Results saved to: {results_file}")
    logger.info("=" * 70)
