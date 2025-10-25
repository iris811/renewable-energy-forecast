"""
Logging configuration for the renewable energy forecast system
"""
import os
from loguru import logger
import yaml


def setup_logger(config_path='configs/config.yaml'):
    """
    Setup logger with configuration

    Args:
        config_path: Path to configuration file
    """
    # Load config
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            log_config = config.get('logging', {})
    else:
        log_config = {
            'level': 'INFO',
            'file': {'enabled': True, 'path': './logs/app.log'}
        }

    # Remove default handler
    logger.remove()

    # Console handler
    logger.add(
        sink=lambda msg: print(msg, end=''),
        level=log_config.get('level', 'INFO'),
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>"
    )

    # File handler
    if log_config.get('file', {}).get('enabled', True):
        log_path = log_config['file'].get('path', './logs/app.log')
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

        logger.add(
            sink=log_path,
            level=log_config.get('level', 'INFO'),
            rotation="10 MB",
            retention="30 days",
            encoding='utf-8',
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function} - {message}"
        )

    return logger


# Create default logger instance
log = setup_logger()
