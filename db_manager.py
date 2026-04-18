import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import aiosqlite
from backend.utils.logger import db_logger
from backend.utils.config import config
from backend.database.optimization import init_database_optimization, optimized_db


def convert_numpy_types(obj):
    """
    Recursively convert NumPy and other non-JSON-serializable types to native Python types.
    """
    if isinstance(obj, dict):
        return {key: convert_numpy_types(val) for key, val in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_types(item) for item in obj]
    elif hasattr(obj, 'item'):  # NumPy scalar types
        return obj.item()
    elif isinstance(obj, (bool, type(None))):
        return obj
    elif isinstance(obj, (int, float, str)):
        return obj
    else:
        try:
            return float(obj) if hasattr(obj, '__float__') else str(obj)
        except (ValueError, TypeError):
            return str(obj)


class DatabaseManager:
    """
    Async SQLite Database Manager.
    
    Handles all persistence for the Digital Twin AI system:
    - Sensor readings (time-series)
    - Alerts and incidents
    - Machine health history
    - Healing action logs
    - System statistics
    
    Uses aiosqlite for non-blocking async operations,
    ensuring DB writes never block the AI processing pipeline.
    
    Schema:
    ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
    │  sensor_readings│  │     alerts      │  │  healing_logs   │
    ├─────────────────┤  ├─────────────────┤  ├─────────────────┤
    │ id              │  │ id              │  │ id              │
    │ reading_id      │  │ alert_id        │  │ action_id       │
    │ machine_id      │  │ machine_id      │  │ machine_id      │
    │ timestamp       │  │ timestamp       │  │ timestamp       │
    │ sensors_json    │  │ risk_level      │  │ action_type     │
    │ analysis_json   │  │ alert_type      │  │ result          │
    └─────────────────┘  │ message         │  │ details_json    │
                         │ resolved        │  └─────────────────┘
                         └─────────────────┘
    """
    
    def __init__(self):
        self.db_path = config.DATABASE_PATH
        self._connection: Optional[aiosqlite.Connection] = None
        self._initialized = False
        db_logger.info(f"🗄 DatabaseManager created: {self.db_path}")
    
    async def initialize(self):
        """
        Initialize database — create tables if they don't exist.
        Called once at application startup.
        """
        db_logger.info("🔧 Initializing database schema...")

        async with aiosqlite.connect(self.db_path) as conn:
            await conn.executescript(self._get_schema_sql())
            await conn.commit()

        self._initialized = True

        # Initialize optimization layer (disabled for now due to schema issues)
        # await init_database_optimization(self.db_path)

        db_logger.info("✅ Database initialized successfully")
    
    def _get_schema_sql(self) -> str:
        """Return SQL for creating all tables"""
        return """
        -- Enable WAL mode for better concurrent performance
        PRAGMA journal_mode=WAL;
        PRAGMA synchronous=NORMAL;
        
        -- Sensor readings table (time-series data)
        CREATE TABLE IF NOT EXISTS sensor_readings (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            reading_id      TEXT NOT NULL UNIQUE,
            machine_id      TEXT NOT NULL,
            timestamp       TEXT NOT NULL,
            sensors_json    TEXT NOT NULL,    -- JSON blob of sensor values
            analysis_json   TEXT,             -- JSON blob of AI analysis
            created_at      TEXT DEFAULT (datetime('now'))
        );
        
        -- Index for fast time-range queries
        CREATE INDEX IF NOT EXISTS idx_readings_machine_time 
        ON sensor_readings(machine_id, timestamp DESC);
        
        CREATE INDEX IF NOT EXISTS idx_readings_timestamp 
        ON sensor_readings(timestamp DESC);
        
        -- Alerts table
        CREATE TABLE IF NOT EXISTS alerts (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            alert_id        TEXT NOT NULL UNIQUE,
            machine_id      TEXT NOT NULL,
            timestamp       TEXT NOT NULL,
            risk_level      TEXT NOT NULL,    -- CRITICAL/HIGH/MEDIUM/LOW
            alert_type      TEXT,             -- bearing_wear, overheating, etc.
            message         TEXT,
            sensors_json    TEXT,             -- Sensor values at time of alert
            details_json    TEXT,             -- Full alert details
            resolved        INTEGER DEFAULT 0,-- 0=active, 1=resolved
            resolved_at     TEXT,
            created_at      TEXT DEFAULT (datetime('now'))
        );
        
        CREATE INDEX IF NOT EXISTS idx_alerts_machine 
        ON alerts(machine_id, timestamp DESC);
        
        CREATE INDEX IF NOT EXISTS idx_alerts_risk 
        ON alerts(risk_level, resolved);
        
        -- Healing actions log
        CREATE TABLE IF NOT EXISTS healing_logs (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            action_id       TEXT NOT NULL UNIQUE,
            machine_id      TEXT NOT NULL,
            timestamp       TEXT NOT NULL,
            alert_id        TEXT,             -- Related alert
            action_type     TEXT NOT NULL,    -- Type of healing action
            success         INTEGER DEFAULT 1,-- 1=success, 0=failed
            details_json    TEXT,             -- Full action details
            created_at      TEXT DEFAULT (datetime('now'))
        );
        
        CREATE INDEX IF NOT EXISTS idx_healing_machine 
        ON healing_logs(machine_id, timestamp DESC);
        
        -- Machine health history (snapshot per hour)
        CREATE TABLE IF NOT EXISTS health_snapshots (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            machine_id      TEXT NOT NULL,
            timestamp       TEXT NOT NULL,
            health_score    REAL,             -- 0-100
            risk_level      TEXT,
            state           TEXT,             -- operational/degraded/critical
            metrics_json    TEXT,             -- Health metrics snapshot
            created_at      TEXT DEFAULT (datetime('now'))
        );
        
        CREATE INDEX IF NOT EXISTS idx_health_machine 
        ON health_snapshots(machine_id, timestamp DESC);
        
        -- System events log
        CREATE TABLE IF NOT EXISTS system_events (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            event_type      TEXT NOT NULL,    -- startup/shutdown/error/etc.
            message         TEXT,
            details_json    TEXT,
            created_at      TEXT DEFAULT (datetime('now'))
        );
        """
    
    async def save_reading(self, reading_data: Dict) -> bool:
        """
        Persist a sensor reading to the database.
        
        Args:
            reading_data: Dict containing reading_id, machine_id,
                         timestamp, sensors, analysis
                         
        Returns:
            True if saved successfully
        """
        try:
            # Convert NumPy types to JSON-serializable types
            sensors_data = convert_numpy_types(reading_data.get("sensors", {}))
            analysis_data = convert_numpy_types(reading_data.get("analysis", {}))
            
            async with aiosqlite.connect(self.db_path) as conn:
                await conn.execute(
                    """
                    INSERT OR IGNORE INTO sensor_readings 
                    (reading_id, machine_id, timestamp, sensors_json, analysis_json)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        reading_data.get("reading_id"),
                        reading_data.get("machine_id"),
                        reading_data.get("timestamp"),
                        json.dumps(sensors_data),
                        json.dumps(analysis_data)
                    )
                )
                await conn.commit()
            
            return True
            
        except Exception as e:
            db_logger.error(f"Failed to save reading: {e}")
            return False
    
    async def save_alert(self, alert_data: Dict) -> bool:
        """
        Persist an alert to the database.
        
        Args:
            alert_data: Alert dict with alert_id, machine_id, risk_level, etc.
            
        Returns:
            True if saved successfully
        """
        try:
            alert_data = convert_numpy_types(alert_data)
            async with aiosqlite.connect(self.db_path) as conn:
                await conn.execute(
                    """
                    INSERT OR IGNORE INTO alerts
                    (alert_id, machine_id, timestamp, risk_level, 
                     alert_type, message, sensors_json, details_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        alert_data.get("alert_id"),
                        alert_data.get("machine_id"),
                        alert_data.get("timestamp", datetime.utcnow().isoformat()),
                        alert_data.get("risk_level", "MEDIUM"),
                        alert_data.get("alert_type"),
                        alert_data.get("message"),
                        json.dumps(alert_data.get("sensors", {})),
                        json.dumps(alert_data)
                    )
                )
                await conn.commit()
            
            db_logger.info(
                f"💾 Alert saved: {alert_data.get('alert_id')} | "
                f"Risk: {alert_data.get('risk_level')}"
            )
            return True
            
        except Exception as e:
            db_logger.error(f"Failed to save alert: {e}")
            return False

    async def create_alert(self, alert_data: Dict) -> bool:
        """
        Compatibility wrapper for legacy code paths that called create_alert.
        """
        return await self.save_alert(alert_data)
    
    async def save_healing_action(self, healing_data: Dict) -> bool:
        """
        Log a self-healing action to the database.
        
        Args:
            healing_data: Healing action details
            
        Returns:
            True if saved successfully
        """
        try:
            async with aiosqlite.connect(self.db_path) as conn:
                await conn.execute(
                    """
                    INSERT OR IGNORE INTO healing_logs
                    (action_id, machine_id, timestamp, alert_id, 
                     action_type, success, details_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        healing_data.get("action_id"),
                        healing_data.get("machine_id"),
                        healing_data.get("timestamp", datetime.utcnow().isoformat()),
                        healing_data.get("alert_id"),
                        healing_data.get("action_type"),
                        1 if healing_data.get("success", True) else 0,
                        json.dumps(healing_data)
                    )
                )
                await conn.commit()
            
            return True
            
        except Exception as e:
            db_logger.error(f"Failed to save healing action: {e}")
            return False
    
    async def get_recent_readings(
        self,
        machine_id: str,
        limit: int = 100,
        hours: int = 1
    ) -> List[Dict]:
        """
        Fetch recent sensor readings for a machine.

        Args:
            machine_id: Machine identifier
            limit: Maximum readings to return
            hours: Time window in hours

        Returns:
            List of reading dicts, newest first
        """
        if optimized_db:
            # Use optimized cached query
            readings = await optimized_db.get_machine_data_cached(machine_id, hours)
            return readings[:limit]

        # Fallback to direct query
        try:
            cutoff = (datetime.utcnow() - timedelta(hours=hours)).isoformat()

            async with aiosqlite.connect(self.db_path) as conn:
                conn.row_factory = aiosqlite.Row

                async with conn.execute(
                    """
                    SELECT reading_id, machine_id, timestamp, sensors_json, analysis_json
                    FROM sensor_readings
                    WHERE machine_id = ? AND timestamp >= ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                    """,
                    (machine_id, cutoff, limit)
                ) as cursor:
                    rows = await cursor.fetchall()

            return [dict(row) for row in rows]

        except Exception as e:
            db_logger.error(f"Failed to fetch readings: {e}")
            return []
        try:
            cutoff = (datetime.utcnow() - timedelta(hours=hours)).isoformat()
            
            async with aiosqlite.connect(self.db_path) as conn:
                conn.row_factory = aiosqlite.Row
                
                async with conn.execute(
                    """
                    SELECT reading_id, machine_id, timestamp, 
                           sensors_json, analysis_json
                    FROM sensor_readings
                    WHERE machine_id = ? AND timestamp >= ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                    """,
                    (machine_id, cutoff, limit)
                ) as cursor:
                    rows = await cursor.fetchall()
            
            return [
                {
                    "reading_id": row["reading_id"],
                    "machine_id": row["machine_id"],
                    "timestamp": row["timestamp"],
                    "sensors": json.loads(row["sensors_json"] or "{}"),
                    "analysis": json.loads(row["analysis_json"] or "{}")
                }
                for row in rows
            ]
            
        except Exception as e:
            db_logger.error(f"Failed to get readings: {e}")
            return []
    
    async def get_alerts(
        self,
        machine_id: Optional[str] = None,
        risk_level: Optional[str] = None,
        resolved: Optional[bool] = None,
        limit: int = 50,
        hours: int = 24
    ) -> List[Dict]:
        """
        Fetch alerts with optional filtering.

        Args:
            machine_id: Filter by machine (None = all)
            risk_level: Filter by risk level (None = all)
            resolved: Filter by resolution status (None = all)
            limit: Maximum alerts to return
            hours: Time window in hours

        Returns:
            List of alert dicts, newest first
        """
        if optimized_db:
            # Use optimized cached query
            alerts = await optimized_db.get_alerts_cached(machine_id, risk_level)
            return alerts[:limit]

        # Fallback to direct query
        try:
            cutoff = (datetime.utcnow() - timedelta(hours=hours)).isoformat()

            conditions = ["timestamp >= ?"]
            params = [cutoff]

            if machine_id:
                conditions.append("machine_id = ?")
                params.append(machine_id)

            if risk_level:
                conditions.append("risk_level = ?")
                params.append(risk_level.upper())

            if resolved is not None:
                conditions.append("resolved = ?")
                params.append(1 if resolved else 0)

            where_clause = " AND ".join(conditions)
            params.append(limit)

            async with aiosqlite.connect(self.db_path) as conn:
                conn.row_factory = aiosqlite.Row

                async with conn.execute(
                    f"""
                    SELECT alert_id, machine_id, timestamp, risk_level,
                           alert_type, message, sensors_json, details_json,
                           resolved, resolved_at
                    FROM alerts
                    WHERE {where_clause}
                    ORDER BY
                        CASE risk_level
                            WHEN 'CRITICAL' THEN 1
                            WHEN 'HIGH' THEN 2
                            WHEN 'MEDIUM' THEN 3
                            ELSE 4
                        END,
                        timestamp DESC
                    LIMIT ?
                    """,
                    params
                ) as cursor:
                    rows = await cursor.fetchall()

            return [
                {
                    "alert_id": row["alert_id"],
                    "machine_id": row["machine_id"],
                    "timestamp": row["timestamp"],
                    "risk_level": row["risk_level"],
                    "alert_type": row["alert_type"],
                    "message": row["message"],
                    "sensors": json.loads(row["sensors_json"] or "{}"),
                    "details": json.loads(row["details_json"] or "{}"),
                    "resolved": bool(row["resolved"]),
                    "resolved_at": row["resolved_at"]
                }
                for row in rows
            ]

        except Exception as e:
            db_logger.error(f"Failed to get alerts: {e}")
            return []
    
    async def resolve_alert(self, alert_id: str) -> bool:
        """
        Mark an alert as resolved.
        
        Args:
            alert_id: Alert to resolve
            
        Returns:
            True if resolved successfully
        """
        try:
            async with aiosqlite.connect(self.db_path) as conn:
                await conn.execute(
                    """
                    UPDATE alerts 
                    SET resolved = 1, resolved_at = ?
                    WHERE alert_id = ?
                    """,
                    (datetime.utcnow().isoformat(), alert_id)
                )
                await conn.commit()
            
            db_logger.info(f"✅ Alert resolved: {alert_id}")
            return True
            
        except Exception as e:
            db_logger.error(f"Failed to resolve alert: {e}")
            return False
    
    async def get_machine_health_history(
        self,
        machine_id: str,
        hours: int = 24
    ) -> List[Dict]:
        """
        Fetch health score history for a machine.
        
        Args:
            machine_id: Target machine
            hours: How far back to look
            
        Returns:
            List of health snapshot dicts
        """
        try:
            cutoff = (datetime.utcnow() - timedelta(hours=hours)).isoformat()
            
            async with aiosqlite.connect(self.db_path) as conn:
                conn.row_factory = aiosqlite.Row
                
                async with conn.execute(
                    """
                    SELECT machine_id, timestamp, health_score, 
                           risk_level, state, metrics_json
                    FROM health_snapshots
                    WHERE machine_id = ? AND timestamp >= ?
                    ORDER BY timestamp ASC
                    """,
                    (machine_id, cutoff)
                ) as cursor:
                    rows = await cursor.fetchall()
            
            return [
                {
                    "machine_id": row["machine_id"],
                    "timestamp": row["timestamp"],
                    "health_score": row["health_score"],
                    "risk_level": row["risk_level"],
                    "state": row["state"],
                    "metrics": json.loads(row["metrics_json"] or "{}")
                }
                for row in rows
            ]
            
        except Exception as e:
            db_logger.error(f"Failed to get health history: {e}")
            return []
    
    async def save_health_snapshot(self, snapshot_data: Dict) -> bool:
        """
        Save a health snapshot for trending analysis.
        
        Args:
            snapshot_data: Health snapshot dict
            
        Returns:
            True if saved successfully
        """
        try:
            async with aiosqlite.connect(self.db_path) as conn:
                await conn.execute(
                    """
                    INSERT INTO health_snapshots
                    (machine_id, timestamp, health_score, risk_level, state, metrics_json)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        snapshot_data.get("machine_id"),
                        snapshot_data.get("timestamp", datetime.utcnow().isoformat()),
                        snapshot_data.get("health_score"),
                        snapshot_data.get("risk_level"),
                        snapshot_data.get("state"),
                        json.dumps(snapshot_data.get("metrics", {}))
                    )
                )
                await conn.commit()
            
            return True
            
        except Exception as e:
            db_logger.error(f"Failed to save health snapshot: {e}")
            return False
    
    async def get_system_stats(self) -> Dict:
        """
        Fetch system-wide statistics from the database.
        
        Returns:
            Dict with aggregate counts and metrics
        """
        try:
            async with aiosqlite.connect(self.db_path) as conn:
                # Total readings
                async with conn.execute(
                    "SELECT COUNT(*) as cnt FROM sensor_readings"
                ) as cur:
                    total_readings = (await cur.fetchone())[0]
                
                # Total alerts by risk level
                async with conn.execute(
                    """
                    SELECT risk_level, COUNT(*) as cnt 
                    FROM alerts 
                    GROUP BY risk_level
                    """
                ) as cur:
                    alert_counts = {
                        row[0]: row[1] for row in await cur.fetchall()
                    }
                
                # Unresolved alerts
                async with conn.execute(
                    "SELECT COUNT(*) FROM alerts WHERE resolved = 0"
                ) as cur:
                    unresolved = (await cur.fetchone())[0]
                
                # Total healing actions
                async with conn.execute(
                    "SELECT COUNT(*) FROM healing_logs WHERE success = 1"
                ) as cur:
                    healing_count = (await cur.fetchone())[0]
                
                # Readings in last hour
                cutoff = (datetime.utcnow() - timedelta(hours=1)).isoformat()
                async with conn.execute(
                    "SELECT COUNT(*) FROM sensor_readings WHERE timestamp >= ?",
                    (cutoff,)
                ) as cur:
                    recent_readings = (await cur.fetchone())[0]
            
            return {
                "total_readings": total_readings,
                "recent_readings_1h": recent_readings,
                "alerts_by_risk": alert_counts,
                "unresolved_alerts": unresolved,
                "healing_actions": healing_count,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            db_logger.error(f"Failed to get system stats: {e}")
            return {}
    
    async def log_system_event(self, event_type: str, message: str, details: Dict = None):
        """
        Log a system event (startup, shutdown, error, etc.)
        
        Args:
            event_type: Type of event
            message: Human-readable message
            details: Optional additional details
        """
        try:
            async with aiosqlite.connect(self.db_path) as conn:
                await conn.execute(
                    """
                    INSERT INTO system_events (event_type, message, details_json)
                    VALUES (?, ?, ?)
                    """,
                    (event_type, message, json.dumps(details or {}))
                )
                await conn.commit()
        except Exception as e:
            db_logger.error(f"Failed to log system event: {e}")
    
    async def cleanup_old_data(self, days: int = 7):
        """
        Remove old data beyond retention period.
        Called periodically to prevent unbounded DB growth.
        
        Args:
            days: Retention period in days
        """
        cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat()
        
        try:
            async with aiosqlite.connect(self.db_path) as conn:
                # Delete old readings
                result = await conn.execute(
                    "DELETE FROM sensor_readings WHERE timestamp < ?",
                    (cutoff,)
                )
                readings_deleted = result.rowcount
                
                # Delete old resolved alerts
                result = await conn.execute(
                    "DELETE FROM alerts WHERE timestamp < ? AND resolved = 1",
                    (cutoff,)
                )
                alerts_deleted = result.rowcount
                
                # Delete old health snapshots
                result = await conn.execute(
                    "DELETE FROM health_snapshots WHERE timestamp < ?",
                    (cutoff,)
                )
                snapshots_deleted = result.rowcount
                
                await conn.commit()
                
                # Optimize DB after deletion
                await conn.execute("VACUUM")
            
            db_logger.info(
                f"🗑 Cleanup complete: "
                f"{readings_deleted} readings, "
                f"{alerts_deleted} alerts, "
                f"{snapshots_deleted} snapshots deleted"
            )
            
        except Exception as e:
            db_logger.error(f"Cleanup failed: {e}")


# Global singleton instance
db = DatabaseManager()