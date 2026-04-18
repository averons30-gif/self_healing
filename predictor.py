"""
backend/ai/predictor.py
═══════════════════════════════════════════════════════════════════
Predictor — Predictive maintenance estimation.

Computes:
  1. Remaining Useful Life (RUL) in hours
  2. Failure probability (0.0 – 1.0)
  3. Degradation trend (improving | stable | worsening)
  4. Next recommended maintenance window (hours from now)

Method:
  - Rolling risk score window → degradation velocity
  - Exponential smoothing of risk trend
  - RUL = (max_degradation - current) / velocity
  - Failure probability via sigmoid of risk score
═══════════════════════════════════════════════════════════════════
"""

from __future__     import annotations

import math
from collections    import deque
from typing         import Dict, List, Optional, Deque

from backend.utils.logger   import ai_logger


# ── Model constants ────────────────────────────────────────────────────────────
SMOOTHING_ALPHA         = 0.15      # Exponential smoothing factor (lower = smoother)
RUL_BASE_HOURS          = 8760.0    # 1 year in hours (new machine RUL)
FAILURE_RISK_THRESHOLD  = 75.0      # Risk score that indicates imminent failure
MAINTENANCE_RISK_TRIGGER = 50.0     # Schedule maintenance above this risk
TREND_WINDOW            = 10        # Readings to determine trend direction
VELOCITY_WINDOW         = 30        # Readings for degradation velocity calc


class Predictor:
    """
    Predictive maintenance model.

    Maintains a rolling window of risk scores to compute:
    - Degradation velocity (how fast things are getting worse)
    - Remaining useful life estimate
    - Failure probability
    - Optimal maintenance timing

    Usage:
        predictor = Predictor("CNC_MILL_01", window_size=200)
        predictions = predictor.update(sensors, risk_score, is_anomaly, degradation)
    """

    def __init__(
        self,
        machine_id:     str,
        window_size:    int     = 200,
        baseline_rul:   float   = RUL_BASE_HOURS
    ):
        """
        Args:
            machine_id:     Machine identifier
            window_size:    Rolling history window size
            baseline_rul:   Baseline RUL for a new machine (hours)
        """
        self.machine_id         = machine_id
        self._window_size       = window_size
        self._baseline_rul      = baseline_rul

        # ── Risk score history ─────────────────────────────────────────────────
        self._risk_history:     Deque[float] = deque(maxlen=window_size)
        self._anomaly_history:  Deque[bool]  = deque(maxlen=window_size)

        # ── Smoothed estimates ─────────────────────────────────────────────────
        self._smoothed_risk:        float = 0.0
        self._smoothed_velocity:    float = 0.0     # Risk score change rate
        self._rul_estimate:         Optional[float] = None
        self._failure_prob:         float = 0.0
        self._trend:                str   = "stable"
        self._next_maintenance:     Optional[float] = None

        # ── Reading count ──────────────────────────────────────────────────────
        self._n:                    int   = 0

        ai_logger.debug(
            f"Predictor created: {machine_id} "
            f"window={window_size} baseline_rul={baseline_rul}h"
        )

    # ══════════════════════════════════════════════════════════════════════════
    # MAIN UPDATE
    # ══════════════════════════════════════════════════════════════════════════

    def update(
        self,
        sensors:        Dict[str, float],
        risk_score:     float,
        is_anomaly:     bool    = False,
        degradation:    float   = 0.0
    ) -> Dict:
        """
        Update the predictive model with a new reading.

        Args:
            sensors:        Current filtered sensor readings
            risk_score:     Current risk score (0-100)
            is_anomaly:     Whether an anomaly was detected
            degradation:    Injected degradation factor (0-1)

        Returns:
            {
                "rul":                      float | None,  # hours
                "failure_probability":      float,         # 0.0 – 1.0
                "next_maintenance_hours":   float | None,
                "degradation_trend":        str,
                "confidence":               float,
                "risk_velocity":            float,
                "smoothed_risk":            float
            }
        """
        self._n += 1

        # ── Update histories ───────────────────────────────────────────────────
        self._risk_history.append(risk_score)
        self._anomaly_history.append(is_anomaly)

        # ── Exponential smoothing of risk ──────────────────────────────────────
        if self._n == 1:
            self._smoothed_risk = risk_score
        else:
            self._smoothed_risk = (
                SMOOTHING_ALPHA * risk_score +
                (1 - SMOOTHING_ALPHA) * self._smoothed_risk
            )

        # ── Compute degradation velocity ───────────────────────────────────────
        velocity = self._compute_velocity()

        # Smooth velocity
        if self._n <= 2:
            self._smoothed_velocity = velocity
        else:
            self._smoothed_velocity = (
                SMOOTHING_ALPHA * velocity +
                (1 - SMOOTHING_ALPHA) * self._smoothed_velocity
            )

        # ── Compute trend ──────────────────────────────────────────────────────
        self._trend = self._compute_trend()

        # ── Failure probability (sigmoid) ──────────────────────────────────────
        # Apply degradation as a multiplier to the risk score
        effective_risk  = self._smoothed_risk * (1.0 + degradation * 0.5)
        effective_risk  = min(100.0, effective_risk)
        self._failure_prob = self._sigmoid_probability(effective_risk)

        # ── RUL estimate ───────────────────────────────────────────────────────
        self._rul_estimate = self._compute_rul(
            effective_risk, self._smoothed_velocity, degradation
        )

        # ── Maintenance window ─────────────────────────────────────────────────
        self._next_maintenance = self._compute_maintenance_window(
            effective_risk, self._rul_estimate
        )

        # ── Confidence (based on history depth) ───────────────────────────────
        confidence = min(1.0, self._n / 50)

        predictions = {
            "rul":                      (
                round(self._rul_estimate, 1)
                if self._rul_estimate is not None else None
            ),
            "failure_probability":      round(self._failure_prob, 4),
            "next_maintenance_hours":   (
                round(self._next_maintenance, 1)
                if self._next_maintenance is not None else None
            ),
            "degradation_trend":        self._trend,
            "confidence":               round(confidence, 4),
            "risk_velocity":            round(self._smoothed_velocity, 4),
            "smoothed_risk":            round(self._smoothed_risk, 2)
        }

        ai_logger.debug(
            f"[{self.machine_id}] Predictor: "
            f"RUL={predictions['rul']}h "
            f"P(fail)={predictions['failure_probability']:.3f} "
            f"trend={self._trend}"
        )

        return predictions

    # ══════════════════════════════════════════════════════════════════════════
    # COMPUTATION METHODS
    # ══════════════════════════════════════════════════════════════════════════

    def _compute_velocity(self) -> float:
        """
        Compute the rate of change of risk score per reading.
        Uses a linear regression slope over the velocity window.

        Returns:
            Risk velocity (positive = worsening, negative = improving)
        """
        window_data = list(self._risk_history)[-VELOCITY_WINDOW:]
        n           = len(window_data)

        if n < 3:
            return 0.0

        # Simple linear regression slope
        x_mean  = (n - 1) / 2.0
        y_mean  = sum(window_data) / n

        num     = sum((i - x_mean) * (window_data[i] - y_mean) for i in range(n))
        den     = sum((i - x_mean) ** 2 for i in range(n))

        if den < 1e-9:
            return 0.0

        slope   = num / den
        return round(slope, 6)

    def _compute_trend(self) -> str:
        """
        Classify the current degradation trend.

        Returns:
            "improving" | "stable" | "worsening"
        """
        window  = list(self._risk_history)[-TREND_WINDOW:]

        if len(window) < 3:
            return "stable"

        # Compare first half vs second half means
        mid     = len(window) // 2
        first   = sum(window[:mid]) / mid
        second  = sum(window[mid:]) / (len(window) - mid)
        delta   = second - first

        if delta > 3.0:
            return "worsening"
        if delta < -3.0:
            return "improving"
        return "stable"

    def _sigmoid_probability(self, risk_score: float) -> float:
        """
        Convert risk score to failure probability using a sigmoid curve.

        Calibrated so that:
          risk=0   → P≈0.01  (near-zero probability)
          risk=50  → P≈0.5   (50% probability)
          risk=75  → P≈0.88  (high probability)
          risk=100 → P≈0.99  (near-certain)

        Args:
            risk_score: 0 – 100

        Returns:
            Failure probability 0.0 – 1.0
        """
        # Shift center to risk=50, scale so slope is appropriate
        x       = (risk_score - 50) / 15.0
        prob    = 1.0 / (1.0 + math.exp(-x))
        return round(prob, 4)

    def _compute_rul(
        self,
        effective_risk: float,
        velocity:       float,
        degradation:    float
    ) -> Optional[float]:
        """
        Estimate Remaining Useful Life in hours.

        Method:
          - If risk is rising (velocity > 0):
              RUL = (FAILURE_RISK_THRESHOLD - current_risk) / velocity * time_per_reading
          - If stable or improving:
              RUL = baseline scaled by current health fraction

        Args:
            effective_risk: Current effective risk score
            velocity:       Risk score change per reading
            degradation:    Degradation factor (0-1)

        Returns:
            Estimated hours or None if insufficient data
        """
        if self._n < 10:
            return None

        # Assume readings every 2 seconds → convert velocity to per-hour
        READINGS_PER_HOUR = 1800.0   # 1 reading / 2s × 3600 s/hr

        if velocity > 0.01:
            # Risk is rising — extrapolate to failure threshold
            risk_to_failure = max(0.0, FAILURE_RISK_THRESHOLD - effective_risk)
            hours_remaining = (risk_to_failure / velocity) / READINGS_PER_HOUR
            rul             = max(0.0, hours_remaining)
        else:
            # Stable or improving — scale baseline by health fraction
            health_fraction = max(0.0, 1.0 - (effective_risk / 100.0))
            rul             = self._baseline_rul * health_fraction

        # Apply degradation penalty
        rul_adjusted = rul * (1.0 - degradation * 0.5)

        return max(0.0, round(rul_adjusted, 1))

    def _compute_maintenance_window(
        self,
        effective_risk: float,
        rul:            Optional[float]
    ) -> Optional[float]:
        """
        Recommend next maintenance window in hours.

        Logic:
          - CRITICAL risk:      Within 24h
          - HIGH risk:          Within 72h
          - MEDIUM risk (>30):  Within 168h (1 week)
          - LOW risk:           Based on RUL / 3 (proactive)

        Args:
            effective_risk: Current risk score
            rul:            Estimated RUL in hours

        Returns:
            Recommended hours until maintenance, or None
        """
        if effective_risk >= FAILURE_RISK_THRESHOLD:
            return 24.0
        if effective_risk >= MAINTENANCE_RISK_TRIGGER:
            return 72.0
        if effective_risk >= 30.0:
            return 168.0

        # Proactive: schedule at 1/3 of RUL
        if rul is not None and rul > 0:
            return round(rul / 3.0, 1)

        return None

    # ══════════════════════════════════════════════════════════════════════════
    # DIAGNOSTICS
    # ══════════════════════════════════════════════════════════════════════════

    def get_risk_history(self, limit: int = 60) -> List[float]:
        """
        Get recent risk score history.

        Args:
            limit: Max entries to return

        Returns:
            List of risk scores (oldest first)
        """
        history = list(self._risk_history)
        return [round(v, 2) for v in history[-limit:]]

    def get_anomaly_rate(self) -> float:
        """
        Fraction of recent readings flagged as anomalies.

        Returns:
            Rate 0.0 – 1.0
        """
        if not self._anomaly_history:
            return 0.0
        return round(sum(self._anomaly_history) / len(self._anomaly_history), 4)

    def get_stats(self) -> Dict:
        """
        Get full predictor statistics.

        Returns:
            Stats dict
        """
        return {
            "machine_id":           self.machine_id,
            "readings":             self._n,
            "smoothed_risk":        round(self._smoothed_risk, 2),
            "velocity":             round(self._smoothed_velocity, 4),
            "trend":                self._trend,
            "rul_hours":            self._rul_estimate,
            "failure_probability":  round(self._failure_prob, 4),
            "anomaly_rate":         self.get_anomaly_rate(),
            "history_depth":        len(self._risk_history)
        }

    def reset(self) -> None:
        """Reset all state (for testing / reinitialization)."""
        self._risk_history.clear()
        self._anomaly_history.clear()
        self._smoothed_risk     = 0.0
        self._smoothed_velocity = 0.0
        self._rul_estimate      = None
        self._failure_prob      = 0.0
        self._trend             = "stable"
        self._n                 = 0
        ai_logger.debug(f"Predictor reset: {self.machine_id}")

    def __repr__(self) -> str:
        return (
            f"Predictor("
            f"machine={self.machine_id!r}, "
            f"readings={self._n}, "
            f"trend={self._trend!r}, "
            f"rul={self._rul_estimate}h)"
        )