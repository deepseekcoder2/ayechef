"""Logging Utilities for Aye Chef
=======================================

Centralized logging configuration and utilities.
Replaces scattered print() statements with structured logging.

⚠️ MANDATORY FOR ALL NEW CODE ⚠️
See: docs/logging/LOGGING_STANDARDS.md

Usage:
    from tools.logging_utils import get_logger
    
    logger = get_logger(__name__)
    logger.info("Operation completed successfully")
    logger.error("Operation failed")

Standards:
    - Backend/operational code: MUST use logger
    - User-facing output: Use print() for CLI/UI
    - Log levels: CRITICAL, ERROR, WARNING, INFO, DEBUG
    - Configuration: config.py lines 464-500
    - Location: logs/aye_chef.log (10MB rotation, 5 backups)
"""

import sys
import os
# Add parent directory to path for imports from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
import logging.config
from config import LOGGING_CONFIG


def setup_logging():
    """
    Initialize logging configuration once.
    
    Thread-safe and idempotent - safe to call multiple times.
    """
    try:
        from config import DATA_DIR
        os.makedirs(str(DATA_DIR / "logs"), exist_ok=True)
        logging.config.dictConfig(LOGGING_CONFIG)
    except Exception as e:
        print(f"Warning: Logging setup failed: {e}")


def get_logger(name: str) -> logging.Logger:
    """
    Get logger for module.
    
    Args:
        name: Module name (use __name__)
    
    Returns:
        Configured logger instance
    
    Example:
        logger = get_logger(__name__)
        logger.info("Starting process")
    """
    setup_logging()
    return logging.getLogger(name)


# Log level mapping for emoji/color prefixes
LOG_LEVEL_MAPPING = {
    "✅": logging.INFO,      # Success messages
    "⚠️": logging.WARNING,   # Warnings
    "❌": logging.ERROR,     # Errors
    "🔍": logging.DEBUG,     # Debug/info
    "🔧": logging.DEBUG,     # Setup info
    "📄": logging.DEBUG,     # File operations
    "📊": logging.INFO,      # Statistics
    "🚀": logging.INFO,      # Process starts
    "💾": logging.DEBUG,     # Data operations
}


def log_with_emoji(logger: logging.Logger, message: str):
    """
    Log message with appropriate level based on emoji prefix.
    
    Args:
        logger: Logger instance
        message: Message with optional emoji prefix
    
    Example:
        log_with_emoji(logger, "✅ Operation successful")
        # Logs at INFO level
    """
    # Extract first 2 characters (emoji can be 1-2 chars)
    emoji = message[:2].strip() if len(message) >= 2 else None
    
    # Check if it's a known emoji
    level = LOG_LEVEL_MAPPING.get(emoji, logging.INFO)
    logger.log(level, message)

