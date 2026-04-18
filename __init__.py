"""Backend package entrypoint.

This module exposes the core backend package components used by the
application and simplifies imports across the project.
"""

from .utils.config import config
from .database.db_manager import db, DatabaseManager
from .ai_engine.digital_twin import DigitalTwinEngine, SensorReading, MachineState
from .utils.logger import (
    app_logger,
    api_logger,
    ai_logger,
    data_logger,
    ws_logger,
    create_logger,
)

__all__ = [
    "config",
    "db",
    "DatabaseManager",
    "DigitalTwinEngine",
    "SensorReading",
    "MachineState",
    "app_logger",
    "api_logger",
    "ai_logger",
    "data_logger",
    "ws_logger",
    "create_logger",
]
