import logging
import os
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler
from backend.utils.config import config


def _create_logger(name: str, log_file: str, level: str = None) -> logging.Logger:
    """
    Factory function to create a configured logger.
    
    Features:
    - Console output with colors (via ANSI codes)
    - Rotating file handler (10MB max, 5 backups)
    - Structured format with timestamps
    
    Args:
        name: Logger name
        log_file: Log file name (inside LOG_DIR)
        level: Override log level (defaults to config.LOG_LEVEL)
        
    Returns:
        Configured Logger instance
    """
    logger = logging.getLogger(name)
    
    # Prevent duplicate handlers on reload
    if logger.handlers:
        return logger
    
    log_level = getattr(logging, (level or config.LOG_LEVEL).upper(), logging.INFO)
    logger.setLevel(log_level)
    
    # ── Formatter ─────────────────────────────────────────────────────────────
    file_formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    console_formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)-15s | %(message)s",
        datefmt="%H:%M:%S"
    )
    
    # ── Console Handler ───────────────────────────────────────────────────────
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # ── File Handler ──────────────────────────────────────────────────────────
    os.makedirs(config.LOG_DIR, exist_ok=True)
    log_path = os.path.join(config.LOG_DIR, log_file)
    
    file_handler = RotatingFileHandler(
        log_path,
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5,
        encoding="utf-8"
    )
    file_handler.setLevel(log_level)
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    return logger


# Public function for custom logger creation
def create_logger(name: str, log_file: str, level: str = None) -> logging.Logger:
    """Create a custom logger for any module"""
    return _create_logger(name, log_file, level)


# ── Specialized Loggers ────────────────────────────────────────────────────────
# Each subsystem gets its own logger and log file for easy debugging

app_logger = _create_logger(
    name="digital_twin.app",
    log_file="app.log"
)

ws_logger = _create_logger(
    name="digital_twin.websocket",
    log_file="websocket.log"
)

db_logger = _create_logger(
    name="digital_twin.database",
    log_file="database.log"
)

ai_logger = _create_logger(
    name="digital_twin.ai",
    log_file="ai.log"
)

data_logger = _create_logger(
    name="digital_twin.data",
    log_file="data.log"
)

api_logger = _create_logger(
    name="digital_twin.api",
    log_file="api.log"
)