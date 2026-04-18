"""
backend/ai/anomaly_detector.py
═══════════════════════════════════════════════════════════════════
AnomalyDetector — Statistical anomaly detection for sensor data.

Methods:
  1. Z-score detection  — flags readings > N std devs from running mean
  2. IQR detection      — interquartile range outlier detection
  3. Rate-of-change     — flags sudden spikes in delta values
  4. Composite score    — weighted combination of all methods

Returns a score 0.0 → 1.0 (0 = normal, 1 = certain anomaly)
and a binary is_anomaly flag when score > threshold.
═══════════════════════════════════════════════════════════════════
"""

from __future__     import annotations

import math
import statistics
from collections    import deque
from typing         import Dict, List, Optional, Deque

from backend.utils.logger   import ai_logger


# ── Per-sensor anomaly thresholds ──────────────────────────────────────────────
SENSOR_Z_THRESHOLDS: Dict[str, float] = {
    "temperature":      3.0,
    "vibration":        2.5,    # More sensitive (early bearing wear)
    "pressure":         3.0,
    "current":          2.8,
    "rpm":              3.5,
    "oil_level":        3.0,
    "voltage":          2.8,
    "load_percentage":  3.5
}

# ── Rate-of-change thresholds (per reading) ────────────────────────────────────
SENSOR_ROC_THRESHOLDS: Dict[str, float] = {
    "temperature":      5.0,    # °C per reading
    "vibration":        2.0,    # mm/s per reading
    "pressure":         20.0,   # bar per reading
    "current":          5.0,    # A per reading
    "rpm":              300.0,  # rpm per reading
    "oil_level":        2.0,    # % per reading
    "voltage":          20.0,   # V per reading
    "load_percentage":  15.0    # % per reading
}

# ── Method weights for composite score ────────────────────────────────────────
METHOD_WEIGHTS = {
    "z_score":      0.45,
    "iqr":          0.35,
    "roc":          0.20
}


class SensorHistory:
    """
    Rolling window of sensor values for statistical analysis.
    Uses a deque for O(1) append + auto-eviction.
    """

    def __init__(self, window_size: int = 100):
        self._window:   Deque[float]    = deque(maxlen=window_size)
        self._prev:     Optional[float] = None

    def push(self, value: float) -> None:
        self._window.append(value)

    @property
    def values(self) -> List[float]:
        return list(self._window)

    @property
    def count(self) -> int:
        return len(self._window)

    @property
    def mean(self) -> float:
        if not self._window:
            return 0.0
        return statistics.mean(self._window)

    @property
    def std(self) -> float:
        if len(self._window) < 2:
            return 0.0
        return statistics.stdev(self._window)

    @property
    def median(self) -> float:
        if not self._window:
            return 0.0
        return statistics.median(self._window)

    def percentile(self, p: float) -> float:
        """Approximate Nth percentile using linear interpolation."""
        values  = sorted(self._window)
        n       = len(values)
        if n == 0:
            return 0.0
        if n == 1:
            return values[0]

        idx = (p / 100) * (n - 1)
        lo  = int(idx)
        hi  = min(lo + 1, n - 1)
        frac = idx - lo

        return values[lo] + frac * (values[hi] - values[lo])


class AnomalyDetector:
    """
    Multi-method statistical anomaly detector.

    Maintains a rolling history window per sensor and applies
    Z-score, IQR and rate-of-change detection methods.

    Usage:
        detector = AnomalyDetector("CNC_MILL_01", sensor_keys, window_size=100)
        result = detector.detect({"temperature": 95.2, "vibration": 8.1, ...})
    """

    def __init__(
        self,
        machine_id:     str,
        sensor_keys:    List[str],
        window_size:    int     = 100,
        z_threshold:    float   = 3.0,
        anomaly_score_threshold: float = 0.5
    ):
        """
        Args:
            machine_id:               Machine identifier (for logging)
            sensor_keys:              List of sensor channel names
            window_size:              History window size (number of readings)
            z_threshold:              Default Z-score threshold (per-sensor overrides apply)
            anomaly_score_threshold:  Composite score above which is_anomaly=True
        """
        self.machine_id             = machine_id
        self._sensor_keys           = sensor_keys
        self._window_size           = window_size
        self._default_z_threshold   = z_threshold
        self._anomaly_threshold     = anomaly_score_threshold

        # ── Per-sensor history ─────────────────────────────────────────────────
        self._histories: Dict[str, SensorHistory] = {
            key: SensorHistory(window_size)
            for key in sensor_keys
        }

        # ── Previous values for rate-of-change ────────────────────────────────
        self._prev: Dict[str, float] = {}

        # ── Stats ──────────────────────────────────────────────────────────────
        self._total_readings    = 0
        self._total_anomalies   = 0

        ai_logger.debug(
            f"AnomalyDetector [{machine_id}] "
            f"window={window_size} z_thresh={z_threshold}"
        )

    # ══════════════════════════════════════════════════════════════════════════
    # WARM-UP
    # ══════════════════════════════════════════════════════════════════════════

    def warm_up(self, sensors: Dict[str, float]) -> None:
        """
        Feed a historical reading to build the statistics window.
        Does not return anomaly results.

        Args:
            sensors: Historical sensor reading dict
        """
        for key, value in sensors.items():
            if key in self._histories:
                self._histories[key].push(float(value))

    # ══════════════════════════════════════════════════════════════════════════
    # MAIN DETECTION
    # ══════════════════════════════════════════════════════════════════════════

    def detect(self, sensors: Dict[str, float]) -> Dict:
        """
        Run anomaly detection on a sensor reading.

        Args:
            sensors: Dict of sensor_key → float value (filtered)

        Returns:
            {
                "is_anomaly":        bool,
                "anomaly_score":     float,   # 0.0 – 1.0
                "confidence":        float,   # based on history depth
                "anomalous_sensors": list,    # which sensors triggered
                "method_scores":     dict,    # per-method contributions
                "sensor_scores":     dict,    # per-sensor anomaly scores
                "z_scores":          dict     # z-scores for reference
            }
        """
        self._total_readings += 1

        sensor_scores:  Dict[str, float] = {}
        z_scores:       Dict[str, float] = {}
        method_scores:  Dict[str, float] = {"z_score": 0.0, "iqr": 0.0, "roc": 0.0}
        anomalous:      List[str]        = []

        # ── Per-sensor analysis ────────────────────────────────────────────────
        for key, value in sensors.items():
            if key not in self._histories:
                continue

            history = self._histories[key]

            if history.count < 5:
                # Not enough data — push and skip
                history.push(value)
                self._prev[key] = value
                sensor_scores[key] = 0.0
                continue

            # Individual method scores for this sensor
            z_score_val,  z_method  = self._z_score_detect(key, value, history)
            iqr_score,    iqr_method = self._iqr_detect(key, value, history)
            roc_score,    roc_method = self._roc_detect(key, value)

            z_scores[key]           = z_score_val

            # Weighted composite for this sensor
            sensor_score = (
                METHOD_WEIGHTS["z_score"] * z_method  +
                METHOD_WEIGHTS["iqr"]     * iqr_score  +
                METHOD_WEIGHTS["roc"]     * roc_score
            )
            sensor_score            = min(1.0, max(0.0, sensor_score))
            sensor_scores[key]      = round(sensor_score, 4)

            if sensor_score > 0.4:
                anomalous.append(key)

            # Push new value to history
            history.push(value)
            self._prev[key] = value

        # ── Global composite score ─────────────────────────────────────────────
        if sensor_scores:
            # Use max of sensor scores (any bad sensor = anomaly)
            # plus a fraction of the mean for breadth
            all_scores      = list(sensor_scores.values())
            max_score       = max(all_scores)
            mean_score      = sum(all_scores) / len(all_scores)
            anomaly_score   = 0.7 * max_score + 0.3 * mean_score
        else:
            anomaly_score   = 0.0

        # ── Method-level scores (aggregated across sensors) ────────────────────
        if sensor_scores:
            for method_key in method_scores:
                method_scores[method_key] = round(
                    sum(sensor_scores.values()) / len(sensor_scores), 4
                )

        # ── Confidence (based on history depth) ───────────────────────────────
        min_history = min(
            (h.count for h in self._histories.values()),
            default=0
        )
        confidence  = min(1.0, min_history / 30)

        is_anomaly  = anomaly_score > self._anomaly_threshold

        if is_anomaly:
            self._total_anomalies += 1
            ai_logger.debug(
                f"⚠️  [{self.machine_id}] Anomaly: score={anomaly_score:.4f} "
                f"sensors={anomalous}"
            )

        return {
            "is_anomaly":           is_anomaly,
            "anomaly_score":        round(anomaly_score, 4),
            "confidence":           round(confidence, 4),
            "anomalous_sensors":    anomalous,
            "method_scores":        method_scores,
            "sensor_scores":        sensor_scores,
            "z_scores":             z_scores
        }

    # ══════════════════════════════════════════════════════════════════════════
    # DETECTION METHODS
    # ══════════════════════════════════════════════════════════════════════════

    def _z_score_detect(
        self,
        key:        str,
        value:      float,
        history:    SensorHistory
    ) -> tuple[float, float]:
        """
        Z-score anomaly detection.
        z = |value - mean| / std_dev

        Args:
            key:     Sensor name
            value:   Current measurement
            history: Sensor history

        Returns:
            (z_score_raw, normalized_score 0-1)
        """
        mean    = history.mean
        std     = history.std

        if std < 1e-9:
            return 0.0, 0.0

        z       = abs(value - mean) / std
        thresh  = SENSOR_Z_THRESHOLDS.get(key, self._default_z_threshold)

        # Normalize: score = 0 at z=0, score = 1 at z = 2×threshold
        score   = min(1.0, z / (2.0 * thresh))

        return round(z, 4), round(score, 4)

    def _iqr_detect(
        self,
        key:        str,
        value:      float,
        history:    SensorHistory
    ) -> tuple[float, float]:
        """
        Interquartile Range anomaly detection.
        Flags values outside [Q1 - k*IQR, Q3 + k*IQR].

        Args:
            key:     Sensor name
            value:   Current measurement
            history: Sensor history

        Returns:
            (iqr_raw, normalized_score 0-1)
        """
        if history.count < 10:
            return 0.0, 0.0

        q1      = history.percentile(25)
        q3      = history.percentile(75)
        iqr     = q3 - q1

        if iqr < 1e-9:
            return 0.0, 0.0

        k   = 1.5
        lo  = q1 - k * iqr
        hi  = q3 + k * iqr

        if value < lo:
            deviation   = (lo - value) / iqr
        elif value > hi:
            deviation   = (value - hi) / iqr
        else:
            return 0.0, 0.0

        score = min(1.0, deviation / 3.0)

        return round(iqr, 4), round(score, 4)

    def _roc_detect(
        self,
        key:    str,
        value:  float
    ) -> tuple[float, float]:
        """
        Rate-of-change anomaly detection.
        Flags sudden spikes larger than the per-sensor threshold.

        Args:
            key:   Sensor name
            value: Current measurement

        Returns:
            (raw_roc, normalized_score 0-1)
        """
        if key not in self._prev:
            return 0.0, 0.0

        roc     = abs(value - self._prev[key])
        thresh  = SENSOR_ROC_THRESHOLDS.get(key, 10.0)

        if roc < thresh * 0.5:
            return roc, 0.0

        score = min(1.0, (roc - thresh * 0.5) / (thresh * 1.5))

        return round(roc, 4), round(score, 4)

    # ══════════════════════════════════════════════════════════════════════════
    # DIAGNOSTICS
    # ══════════════════════════════════════════════════════════════════════════

    def get_statistics(self) -> Dict:
        """
        Get statistical summary for all sensor histories.

        Returns:
            Dict with per-sensor stats (mean, std, min, max, count)
        """
        stats = {}
        for key, history in self._histories.items():
            if history.count == 0:
                continue
            values  = history.values
            stats[key] = {
                "count":    history.count,
                "mean":     round(history.mean, 4),
                "std":      round(history.std, 4),
                "median":   round(history.median, 4),
                "min":      round(min(values), 4),
                "max":      round(max(values), 4),
                "q1":       round(history.percentile(25), 4),
                "q3":       round(history.percentile(75), 4)
            }
        return stats

    def get_anomaly_rate(self) -> float:
        """
        Fraction of readings flagged as anomalies.

        Returns:
            Rate as 0.0 to 1.0
        """
        if self._total_readings == 0:
            return 0.0
        return round(self._total_anomalies / self._total_readings, 4)

    def __repr__(self) -> str:
        return (
            f"AnomalyDetector("
            f"machine={self.machine_id!r}, "
            f"sensors={len(self._sensor_keys)}, "
            f"window={self._window_size}, "
            f"readings={self._total_readings})"
        )