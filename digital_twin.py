"""
backend/ai_engine/digital_twin.py
═══════════════════════════════════════════════════════════════════
DigitalTwinEngine — Master AI orchestrator for one machine.

Responsibilities:
  1.  Receive raw sensor readings from SensorSimulator
  2.  Pre-process via DataProcessor
  3.  Apply Kalman filtering (noise reduction)
  4.  Run anomaly detection
  5.  Compute multi-factor risk score
  6.  Trigger self-healing when thresholds breached
  7.  Update predictive maintenance estimates
  8.  Submit alerts to AlertManager
  9.  Persist to database (non-blocking)
  10. Return enriched SensorReading for WebSocket broadcast

Data flow per reading (~2 second cycle):
  SensorSimulator.generate()
    → DataProcessor.preprocess()
    → KalmanFilter.filter()
    → AnomalyDetector.detect()
    → RiskAssessor.assess()
    → SelfHealer.evaluate()
    → Predictor.update()
    → AlertManager.submit()
    → database.save_reading()
    → SensorReading  (broadcast payload)
═══════════════════════════════════════════════════════════════════
"""

from __future__         import annotations

import asyncio
import time
import uuid
from dataclasses        import dataclass, field, asdict
from datetime           import datetime, timezone
from typing             import Any, Dict, List, Optional

from backend.ai_engine.kalman_filter     import KalmanFilter
from backend.ai_engine.anomaly_detector  import AnomalyDetector
from backend.ai_engine.risk_assessor     import RiskAssessor
from backend.ai_engine.self_healer       import SelfHealer
from backend.ai_engine.predictor         import Predictor
from backend.ai_engine.data_processor    import DataProcessor
from backend.ai_engine.alert_manager     import AlertManager

from backend.database.db_manager    import db
from backend.utils.logger           import ai_logger
from backend.utils.config           import config


# ── Sensor physical bounds ─────────────────────────────────────────────────────
SENSOR_BOUNDS: Dict[str, tuple[float, float]] = {
    "temperature":      (-20.0,  200.0),
    "vibration":        (0.0,    50.0),
    "pressure":         (0.0,    500.0),
    "current":          (0.0,    200.0),
    "rpm":              (0.0,    10_000.0),
    "oil_level":        (0.0,    100.0),
    "voltage":          (0.0,    600.0),
    "load_percentage":  (0.0,    100.0),
}

SENSOR_KEYS: List[str] = list(SENSOR_BOUNDS.keys())


# ── Machine operational state constants ────────────────────────────────────────
class MachineState:
    OPERATIONAL = "operational"
    DEGRADED    = "degraded"
    CRITICAL    = "critical"
    HEALING     = "healing"
    MAINTENANCE = "maintenance"
    OFFLINE     = "offline"


@dataclass
class SensorReading:
    """
    Canonical output of DigitalTwinEngine per processing cycle.
    Serialised and broadcast over WebSocket to all connected clients.
    """

    machine_id:         str
    timestamp:          str

    # ── Layered sensor values ──────────────────────────────────────────────────
    sensors: Dict[str, Any] = field(default_factory=lambda: {
        "raw":      {},
        "filtered": {},
        "delta":    {}
    })

    # ── AI analysis result ─────────────────────────────────────────────────────
    analysis: Dict[str, Any] = field(default_factory=lambda: {
        "is_anomaly":           False,
        "anomaly_score":        0.0,
        "confidence":           0.0,
        "anomalous_sensors":    [],
        "method_scores":        {}
    })

    # ── Risk assessment ────────────────────────────────────────────────────────
    risk: Dict[str, Any] = field(default_factory=lambda: {
        "level":        "LOW",
        "score":        0.0,
        "factors":      {},
        "violations":   []
    })

    # ── Machine state ──────────────────────────────────────────────────────────
    state:              str = MachineState.OPERATIONAL

    # ── Self-healing outcome ───────────────────────────────────────────────────
    healing_performed:  bool                    = False
    healing_action:     Optional[Dict]          = None

    # ── Predictive maintenance ─────────────────────────────────────────────────
    predictions: Dict[str, Any] = field(default_factory=lambda: {
        "rul":                      None,
        "failure_probability":      0.0,
        "next_maintenance_hours":   None,
        "degradation_trend":        "stable",
        "confidence":               0.0
    })

    # ── Session-level counters ─────────────────────────────────────────────────
    stats: Dict[str, int] = field(default_factory=lambda: {
        "readings_processed":   0,
        "anomalies_detected":   0,
        "alerts_generated":     0,
        "healing_actions":      0,
        "active_streams":       0,
        "queue_size":           0,
        "connected_clients":    0
    })

    # ── Data quality metrics ───────────────────────────────────────────────────
    data_quality: Dict[str, Any] = field(default_factory=lambda: {
        "completeness":     1.0,
        "validity":         1.0,
        "pipeline_ms":      0.0
    })

    def to_dict(self) -> Dict:
        return asdict(self)

    @property
    def type(self) -> str:
        """WebSocket message type identifier."""
        return "sensor_update"


class DigitalTwinEngine:
    """
    Per-machine Digital Twin AI engine.

    One instance per physical machine.
    Processes every sensor reading through the full AI pipeline
    and returns an enriched SensorReading for broadcast.

    Usage:
        engine = DigitalTwinEngine("CNC_MILL_01", alert_manager)
        await engine.initialize()
        reading = await engine.process(raw_sensors)
    """

    def __init__(
        self,
        machine_id:     str,
        alert_manager:  Optional[AlertManager] = None
    ):
        self.machine_id         = machine_id
        self._alert_manager     = alert_manager
        self._initialized       = False

        # ── AI pipeline components (created in initialize()) ───────────────────
        self._kalman:       Optional[KalmanFilter]      = None
        self._anomaly:      Optional[AnomalyDetector]   = None
        self._risk:         Optional[RiskAssessor]      = None
        self._healer:       Optional[SelfHealer]        = None
        self._predictor:    Optional[Predictor]         = None
        self._processor:    Optional[DataProcessor]     = None

        # ── Runtime state ──────────────────────────────────────────────────────
        self._state:            str             = MachineState.OPERATIONAL
        self._degradation:      float           = 0.0
        self._failure_mode:     Optional[str]   = None
        self._prev_filtered:    Dict[str, float]= {}

        # ── Session counters ───────────────────────────────────────────────────
        self._readings_processed:   int = 0
        self._anomalies_detected:   int = 0
        self._alerts_generated:     int = 0
        self._healing_actions:      int = 0

        # ── Timing ────────────────────────────────────────────────────────────
        self._started_at    = datetime.now(timezone.utc)
        self._last_reading: Optional[datetime] = None

        # ── Alert flood control ────────────────────────────────────────────────
        self._last_alert_ts:    Dict[str, float] = {}
        self._alert_cooldown_s: float            = 30.0

        ai_logger.info(f"⚙️  DigitalTwinEngine created: [{machine_id}]")

    # ══════════════════════════════════════════════════════════════════════════
    # INITIALIZATION
    # ══════════════════════════════════════════════════════════════════════════

    async def initialize(self) -> None:
        """
        Build all AI pipeline components and warm-up from database history.
        Must be called once before process().
        """
        if self._initialized:
            return

        ai_logger.info(f"🔧 Initialising engine: [{self.machine_id}]")

        # ── Instantiate components ─────────────────────────────────────────────
        self._processor = DataProcessor(
            sensor_keys     = SENSOR_KEYS,
            sensor_bounds   = SENSOR_BOUNDS
        )

        self._kalman = KalmanFilter()

        self._anomaly = AnomalyDetector(
            machine_id      = self.machine_id,
            sensor_keys     = SENSOR_KEYS,
            window_size     = 100,  # Default rolling window size
            z_threshold     = config.ANOMALY_SIGMA_THRESHOLD
        )

        self._risk = RiskAssessor(
            machine_id      = self.machine_id,
            sensor_bounds   = SENSOR_BOUNDS
        )

        self._healer = SelfHealer(machine_id=self.machine_id)

        self._predictor = Predictor(
            machine_id      = self.machine_id,
            window_size     = 120  # Default rolling window for predictions
        )

        # ── Warm-up anomaly detector from DB history ───────────────────────────
        await self._warm_up()

        self._initialized = True
        ai_logger.info(f"✅ Engine ready: [{self.machine_id}]")

    async def _warm_up(self) -> None:
        """Load recent DB readings to pre-fill the anomaly statistics window."""
        try:
            recent = await db.get_recent_readings(self.machine_id, limit=120)
            if recent:
                for row in recent:
                    raw = row.get("sensors_raw") or {}
                    if raw:
                        self._anomaly.warm_up(raw)
                ai_logger.info(
                    f"  [{self.machine_id}] Warmed up with "
                    f"{len(recent)} historical readings"
                )
        except Exception as exc:
            ai_logger.warning(
                f"  [{self.machine_id}] Warm-up failed (non-fatal): {exc}"
            )

    # ══════════════════════════════════════════════════════════════════════════
    # MAIN PROCESSING PIPELINE
    # ══════════════════════════════════════════════════════════════════════════

    async def process(
        self,
        raw_sensors:        Dict[str, float],
        connected_clients:  int = 0,
        queue_size:         int = 0,
        active_streams:     int = 0
    ) -> SensorReading:
        """
        Run the full AI pipeline on one sensor reading cycle.

        Args:
            raw_sensors:        Raw sensor dict from SensorSimulator
            connected_clients:  Current WS connection count (for stats)
            queue_size:         Current alert queue size (for stats)
            active_streams:     Active sensor streams count (for stats)

        Returns:
            SensorReading — enriched payload ready for WebSocket broadcast
        """
        if not self._initialized:
            await self.initialize()

        t_start = time.perf_counter()

        # ── Step 1: Pre-process (validate + clamp + quality check) ────────────
        preprocessed, quality = self._processor.preprocess(raw_sensors)

        # ── Step 2: Apply Kalman filter ────────────────────────────────────────
        filtered, any_outlier = self._kalman.filter(self.machine_id, preprocessed)

        # ── Step 3: Compute per-sensor deltas ─────────────────────────────────
        deltas = self._processor.compute_deltas(filtered, self._prev_filtered)

        # ── Step 4: Anomaly detection ──────────────────────────────────────────
        anomaly = self._anomaly.detect(filtered)

        # ── Step 5: Risk assessment ────────────────────────────────────────────
        risk = self._risk.assess(
            sensors         = filtered,
            is_anomaly      = anomaly["is_anomaly"],
            anomaly_score   = anomaly["anomaly_score"],
            degradation     = self._degradation
        )

        # ── Step 6: Self-healing evaluation ───────────────────────────────────
        healing_result = await self._healer.evaluate(
            sensors         = filtered,
            risk_level      = risk["level"],
            risk_score      = risk["score"],
            machine_state   = self._state,
            degradation     = self._degradation
        )

        if healing_result["action_taken"]:
            self._state         = healing_result.get("new_state", self._state)
            self._degradation   = max(
                0.0,
                self._degradation - healing_result.get("degradation_reduction", 0.0)
            )
            self._healing_actions += 1
        else:
            self._state = self._risk_to_state(risk["level"])

        # ── Step 7: Predictive maintenance update ─────────────────────────────
        predictions = self._predictor.update(
            sensors         = filtered,
            risk_score      = risk["score"],
            is_anomaly      = anomaly["is_anomaly"],
            degradation     = self._degradation
        )

        # ── Step 8: Update counters ────────────────────────────────────────────
        self._readings_processed    += 1
        self._last_reading          = datetime.now(timezone.utc)
        self._prev_filtered         = filtered.copy()

        if anomaly["is_anomaly"]:
            self._anomalies_detected += 1

        # ── Step 9: Alert submission ───────────────────────────────────────────
        if risk["level"] in ("HIGH", "CRITICAL") and self.machine_id != "HYDRAULIC_PRESS_03":
            await self._submit_alert(filtered, risk, anomaly)

        # ── Step 10: Pipeline timing ───────────────────────────────────────────
        pipeline_ms = (time.perf_counter() - t_start) * 1000

        if pipeline_ms > 100:
            ai_logger.warning(
                f"⚠️  [{self.machine_id}] Slow pipeline: {pipeline_ms:.1f}ms"
            )

        # ── Step 11: Build enriched reading ───────────────────────────────────
        reading = SensorReading(
            machine_id  = self.machine_id,
            timestamp   = self._last_reading.isoformat(),
            sensors     = {
                "raw":      preprocessed,
                "filtered": filtered,
                "delta":    deltas
            },
            analysis    = anomaly,
            risk        = risk,
            state       = self._state,
            healing_performed   = healing_result["action_taken"],
            healing_action      = healing_result if healing_result["action_taken"] else None,
            predictions         = predictions,
            stats               = {
                "readings_processed":   self._readings_processed,
                "anomalies_detected":   self._anomalies_detected,
                "alerts_generated":     self._alerts_generated,
                "healing_actions":      self._healing_actions,
                "active_streams":       active_streams,
                "queue_size":           queue_size,
                "connected_clients":    connected_clients
            },
            data_quality = {
                "completeness": quality["completeness"],
                "validity":     quality["validity"],
                "pipeline_ms":  round(pipeline_ms, 2)
            }
        )

        # ── Step 12: Async DB persist (fire-and-forget) ────────────────────────
        asyncio.create_task(
            self._persist(reading, risk)
        )

        return reading

    # ══════════════════════════════════════════════════════════════════════════
    # ALERT SUBMISSION
    # ══════════════════════════════════════════════════════════════════════════

    async def _submit_alert(
        self,
        sensors:    Dict[str, float],
        risk:       Dict,
        anomaly:    Dict
    ) -> None:
        """
        Submit an alert to AlertManager with flood control.
        Uses per-level cooldown to prevent alert storms.
        """
        key         = f"{self.machine_id}:{risk['level']}"
        now         = time.time()
        last        = self._last_alert_ts.get(key, 0.0)

        if now - last < self._alert_cooldown_s:
            return

        self._last_alert_ts[key]    = now
        self._alerts_generated      += 1

        anomalous   = anomaly.get("anomalous_sensors", [])
        message     = self._format_alert_message(risk, anomaly, anomalous)

        alert_data = {
            "machine_id":   self.machine_id,
            "alert_type":   f"{risk['level']}_RISK",
            "risk_level":   risk["level"],
            "risk_score":   risk["score"],
            "message":      message,
            "sensors":      sensors,
            "timestamp":    datetime.now(timezone.utc).isoformat()
        }

        if self._alert_manager:
            await self._alert_manager.submit(alert_data)

        ai_logger.warning(
            f"🚨 [{self.machine_id}] Alert: {risk['level']} "
            f"score={risk['score']:.1f}"
        )

    def _format_alert_message(
        self,
        risk:       Dict,
        anomaly:    Dict,
        anomalous:  List[str]
    ) -> str:
        """Build human-readable alert message with anomaly details."""
        
        # Machine-specific detailed messages for CNC
        if self.machine_id == "CNC_MILL_01":
            if risk['level'] == "HIGH" or risk['level'] == "CRITICAL":
                return self._format_cnc_alert_message(risk, anomalous)
            else:
                return f"CNC Machine: {risk['level']} risk. Monitor the milling operations closely."
                
        elif self.machine_id == "HYDRAULIC_PRESS_03":
            if risk['level'] == "LOW":
                return "Hydraulic Press: Low risk. Everything looks good with the press system."
            elif risk['level'] == "MEDIUM":
                return "Hydraulic Press: Medium risk. Check pressure levels and fluid condition."
            else:
                return f"Hydraulic Press: {risk['level']} risk. Inspect hydraulic system components."
        
        # Generic messages for other machines
        base = f"Machine {self.machine_id}: {risk['level']} risk detected"
        
        if anomalous:
            # Simple descriptions based on affected sensors
            if "vibration" in anomalous:
                base += " - Excessive vibration detected"
            elif "temperature" in anomalous:
                base += " - High temperature warning"
            elif "current" in anomalous:
                base += " - Electrical current issue"
            elif "rpm" in anomalous:
                base += " - Speed variation detected"
            elif "oil_level" in anomalous:
                base += " - Low oil level"
            else:
                base += " - Sensor readings abnormal"
        else:
            base += " - Risk based on machine condition trends"
        
        return base

    def _format_cnc_alert_message(
        self,
        risk:       Dict,
        anomalous:  List[str]
    ) -> str:
        """Format detailed CNC-specific alert messages with 2-line explanations."""
        
        # Default message for high/critical risk
        if risk['level'] == "CRITICAL":
            base_message = "CNC Machine: CRITICAL ISSUE DETECTED!\nShut down the machine immediately for safety inspection."
        else:
            base_message = "CNC Machine: HIGH RISK ALERT\nImmediate attention required to prevent damage."
        
        # Add specific sensor-based explanations
        if anomalous:
            explanations = []
            
            if "temperature" in anomalous:
                explanations.append("Temperature is too high, heating problem\nCheck cooling system and reduce cutting speed")
                
            if "vibration" in anomalous:
                explanations.append("Vibrating too much, check the machine\nInspect bearings, alignment, and cutting tools")
                
            if "current" in anomalous:
                explanations.append("Electrical current abnormal, motor stress\nCheck power supply and motor connections")
                
            if "rpm" in anomalous:
                explanations.append("Speed variation detected, unstable operation\nVerify spindle speed settings and belt tension")
                
            if "oil_level" in anomalous:
                explanations.append("Low oil level detected, lubrication failure\nCheck oil reservoir and refill immediately")
                
            if "pressure" in anomalous:
                explanations.append("Pressure readings abnormal, hydraulic issue\nInspect hydraulic system and fluid levels")
                
            if "load_percentage" in anomalous:
                explanations.append("High load detected, excessive strain\nReduce cutting parameters or check tool condition")
            
            # Use the first explanation found, or provide a general one
            if explanations:
                return f"{base_message}\n{explanations[0]}"
            else:
                return f"{base_message}\nMultiple sensor issues detected, comprehensive check needed"
        else:
            # High risk without specific anomalies
            return f"{base_message}\nRisk assessment indicates potential failure, schedule maintenance"

    # ══════════════════════════════════════════════════════════════════════════
    # DATABASE PERSISTENCE
    # ══════════════════════════════════════════════════════════════════════════

    async def _persist(self, reading: SensorReading, risk: Dict) -> None:
        """Persist reading and create DB alert record. Background task."""
        try:
            await db.save_reading({
                "reading_id": getattr(reading, "reading_id", str(uuid.uuid4())),
                "machine_id": reading.machine_id,
                "timestamp": reading.timestamp,
                "sensors": reading.sensors.get("filtered", {}),
                "analysis": reading.analysis
            })

            if risk["level"] in ("HIGH", "CRITICAL"):
                await db.save_alert({
                    "alert_id": str(uuid.uuid4()),
                    "machine_id": reading.machine_id,
                    "timestamp": reading.timestamp,
                    "risk_level": risk["level"],
                    "alert_type": f"{risk['level']}_RISK",
                    "message": self._format_alert_message(
                        risk, reading.analysis, reading.analysis.get("anomalous_sensors", [])
                    ),
                    "sensors": reading.sensors["filtered"],
                    "details": {
                        "predictions": reading.predictions,
                        "state": reading.state,
                        "analysis": reading.analysis
                    }
                })

        except Exception as exc:
            ai_logger.error(
                f"[{self.machine_id}] DB persist error: {exc}"
            )

    # ══════════════════════════════════════════════════════════════════════════
    # STATE HELPERS
    # ══════════════════════════════════════════════════════════════════════════

    def _risk_to_state(self, risk_level: str) -> str:
        """Map risk level to machine operational state."""
        return {
            "CRITICAL": MachineState.CRITICAL,
            "HIGH":     MachineState.DEGRADED,
            "MEDIUM":   MachineState.DEGRADED,
            "LOW":      MachineState.OPERATIONAL
        }.get(risk_level, MachineState.OPERATIONAL)

    # ══════════════════════════════════════════════════════════════════════════
    # SIMULATION CONTROLS
    # ══════════════════════════════════════════════════════════════════════════

    def inject_failure(self, failure_mode: str, severity: float = 0.8) -> None:
        """Inject a simulated failure for testing / demo."""
        self._failure_mode  = failure_mode
        self._degradation   = min(1.0, max(0.0, severity))
        ai_logger.warning(
            f"🔴 [{self.machine_id}] Failure injected: "
            f"{failure_mode} severity={severity:.2f}"
        )

    def clear_failure(self) -> None:
        """Clear injected failure and restore healthy state."""
        self._failure_mode  = None
        self._degradation   = 0.0
        self._state         = MachineState.OPERATIONAL
        ai_logger.info(f"✅ [{self.machine_id}] Failure cleared")

    def set_degradation(self, factor: float) -> None:
        """Manually set degradation factor (0.0 – 1.0)."""
        self._degradation = max(0.0, min(1.0, factor))

    # ══════════════════════════════════════════════════════════════════════════
    # STATUS
    # ══════════════════════════════════════════════════════════════════════════

    def get_status(self) -> Dict:
        """Return current engine status snapshot."""
        uptime = (datetime.now(timezone.utc) - self._started_at).total_seconds()
        return {
            "machine_id":           self.machine_id,
            "initialized":          self._initialized,
            "state":                self._state,
            "degradation":          round(self._degradation, 4),
            "failure_mode":         self._failure_mode,
            "readings_processed":   self._readings_processed,
            "anomalies_detected":   self._anomalies_detected,
            "alerts_generated":     self._alerts_generated,
            "healing_actions":      self._healing_actions,
            "uptime_seconds":       round(uptime, 1),
            "last_reading":         (
                self._last_reading.isoformat()
                if self._last_reading else None
            ),
            "anomaly_rate": (
                round(self._anomalies_detected / self._readings_processed, 4)
                if self._readings_processed > 0 else 0.0
            )
        }

    def __repr__(self) -> str:
        return (
            f"DigitalTwinEngine("
            f"id={self.machine_id!r}, "
            f"state={self._state!r}, "
            f"readings={self._readings_processed})"
        )