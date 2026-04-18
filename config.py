import os
from typing import Dict, List
from dotenv import load_dotenv

load_dotenv()


class Config:
    """
    Central configuration for the Digital Twin AI System.
    
    All values can be overridden via environment variables.
    This makes the system production-ready with zero code changes.
    
    Environment Variables:
    - DATABASE_PATH: SQLite DB file path
    - STREAM_INTERVAL: Seconds between sensor readings
    - WS_HEARTBEAT_INTERVAL: WebSocket ping interval
    - LOG_LEVEL: Logging verbosity
    - HOST / PORT: API server binding
    """
    
    # ── Server ────────────────────────────────────────────────────────────────
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))
    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"
    
    # ── Database ──────────────────────────────────────────────────────────────
    DATABASE_PATH: str = os.getenv("DATABASE_PATH", "digital_twin.db")
    DB_CLEANUP_DAYS: int = int(os.getenv("DB_CLEANUP_DAYS", "7"))
    
    # ── Streaming ─────────────────────────────────────────────────────────────
    STREAM_INTERVAL: float = float(os.getenv("STREAM_INTERVAL", "2.0"))  # seconds
    WS_HEARTBEAT_INTERVAL: float = float(os.getenv("WS_HEARTBEAT_INTERVAL", "30.0"))
    
    # ── Logging ───────────────────────────────────────────────────────────────
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_DIR: str = os.getenv("LOG_DIR", "logs")
    
    # ── Machine Registry ──────────────────────────────────────────────────────
    MACHINE_IDS: List[str] = [
        "CNC_MILL_01",
        "CONVEYOR_02",
        "HYDRAULIC_PRESS_03",
        "ROBOT_ARM_04",
        "COMPRESSOR_05"
    ]
    
    MACHINE_NAMES: Dict[str, str] = {
        "CNC_MILL_01":          "CNC Milling Machine #1",
        "CONVEYOR_02":          "Main Assembly Conveyor #2",
        "HYDRAULIC_PRESS_03":   "Hydraulic Press Unit #3",
        "ROBOT_ARM_04":         "Robotic Assembly Arm #4",
        "COMPRESSOR_05":        "Air Compressor System #5"
    }
    
    MACHINE_TYPES: Dict[str, str] = {
        "CNC_MILL_01":          "cnc_mill",
        "CONVEYOR_02":          "conveyor",
        "HYDRAULIC_PRESS_03":   "hydraulic_press",
        "ROBOT_ARM_04":         "robot_arm",
        "COMPRESSOR_05":        "compressor"
    }
    
    # ── AI Thresholds ─────────────────────────────────────────────────────────
    # Risk scoring weights (must sum to 1.0)
    RISK_WEIGHTS: Dict[str, float] = {
        "temperature":      0.25,
        "vibration":        0.25,
        "pressure":         0.15,
        "current":          0.15,
        "oil_level":        0.10,
        "rpm":              0.05,
        "voltage":          0.05
    }
    
    # Anomaly detection thresholds (sigma levels)
    ANOMALY_SIGMA_THRESHOLD: float = float(os.getenv("ANOMALY_SIGMA", "2.5"))
    
    # Risk level thresholds (0-100 score)
    RISK_THRESHOLDS: Dict[str, float] = {
        "CRITICAL": 80.0,
        "HIGH":     60.0,
        "MEDIUM":   40.0,
        "LOW":      0.0
    }
    
    # ── Self-Healing ──────────────────────────────────────────────────────────
    HEALING_ENABLED: bool = os.getenv("HEALING_ENABLED", "true").lower() == "true"
    HEALING_COOLDOWN_SECONDS: int = int(os.getenv("HEALING_COOLDOWN", "60"))
    AUTO_HEAL_THRESHOLD: float = float(os.getenv("AUTO_HEAL_THRESHOLD", "60.0"))
    
    # ── Kalman Filter ─────────────────────────────────────────────────────────
    KALMAN_PROCESS_NOISE: float = float(os.getenv("KALMAN_PROCESS_NOISE", "0.1"))
    KALMAN_MEASUREMENT_NOISE: float = float(os.getenv("KALMAN_MEASUREMENT_NOISE", "1.0"))
    
    # ── EWMA (Exponential Weighted Moving Average) ────────────────────────────
    EWMA_ALPHA: float = float(os.getenv("EWMA_ALPHA", "0.3"))
    
    # ── API ───────────────────────────────────────────────────────────────────
    API_TITLE: str = "Digital Twin AI System"
    API_VERSION: str = "1.0.0"
    API_DESCRIPTION: str = """
    🤖 **Industrial Digital Twin AI System**
    
    Real-time machine health monitoring with:
    - **Kalman Filter** noise reduction
    - **Isolation Forest** anomaly detection  
    - **LSTM** predictive maintenance
    - **Self-Healing** autonomous responses
    - **WebSocket** live sensor streams
    """
    
    # CORS origins (comma-separated in env)
    CORS_ORIGINS: List[str] = os.getenv(
        "CORS_ORIGINS", 
        "http://localhost:3000,http://localhost:8080,http://localhost:5173"
    ).split(",")
    
    # ── Retention ─────────────────────────────────────────────────────────────
    MAX_READINGS_PER_MACHINE: int = int(os.getenv("MAX_READINGS", "10000"))
    HEALTH_SNAPSHOT_INTERVAL: int = int(os.getenv("SNAPSHOT_INTERVAL", "300"))  # 5 min
    
    # ── Alert Queue ───────────────────────────────────────────────────────────
    ALERT_QUEUE_MAX_SIZE: int = int(os.getenv("ALERT_QUEUE_SIZE", "1000"))
    ALERT_DEDUP_WINDOW: int = int(os.getenv("ALERT_DEDUP_WINDOW", "30"))  # seconds
    
    def __repr__(self) -> str:
        return (
            f"Config(machines={len(self.MACHINE_IDS)}, "
            f"stream_interval={self.STREAM_INTERVAL}s, "
            f"port={self.PORT})"
        )


# Global singleton
config = Config()