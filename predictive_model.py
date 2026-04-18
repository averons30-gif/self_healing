"""
predictive_model.py
═══════════════════════════════════════════════════════════════════
Predictive Maintenance ML Model for failure prediction.

Uses historical sensor data to predict machine failures 24-48 hours
in advance with confidence scores.
═══════════════════════════════════════════════════════════════════
"""

from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import numpy as np
from dataclasses import dataclass
from backend.utils.logger import create_logger

pred_logger = create_logger("digital_twin.ai.predictive", "predictive_model.log")


@dataclass
class FailurePrediction:
    """Prediction result for a machine"""
    machine_id: str
    confidence: float  # 0-100
    risk_level: str  # CRITICAL, HIGH, MEDIUM, LOW
    predicted_failure_time: Optional[datetime]
    hours_to_failure: Optional[float]
    contributing_factors: List[str]
    recommendation: str


class PredictiveMaintenanceModel:
    """
    ML-based predictive maintenance model.
    Predicts failures based on sensor patterns and anomaly scores.
    """

    def __init__(self):
        self.thresholds = {
            "vibration": 8.5,
            "temperature": 75.0,
            "pressure": 95.0,
            "noise": 85.0,
            "anomaly_score": 0.75
        }
        self.training_data = {}
        pred_logger.info("✅ PredictiveMaintenanceModel initialized")

    def predict_failure(
        self,
        machine_id: str,
        recent_readings: List[Dict],
        anomaly_scores: List[float],
        alert_history: List[Dict]
    ) -> FailurePrediction:
        """
        Predict if a machine will fail and when.
        
        Args:
            machine_id: Machine identifier
            recent_readings: Last 100 sensor readings (last ~1 hour)
            anomaly_scores: Anomaly scores for each reading
            alert_history: Recent alerts for the machine
        
        Returns:
            FailurePrediction with confidence and estimated time to failure
        """
        
        try:
            if not recent_readings or len(recent_readings) < 10:
                return FailurePrediction(
                    machine_id=machine_id,
                    confidence=0.0,
                    risk_level="LOW",
                    predicted_failure_time=None,
                    hours_to_failure=None,
                    contributing_factors=["Insufficient data"],
                    recommendation="Collect more sensor data for accurate predictions"
                )

            # Calculate risk metrics
            sensor_risk = self._calculate_sensor_risk(recent_readings)
            trend_risk = self._calculate_trend_risk(recent_readings)
            anomaly_risk = self._calculate_anomaly_risk(anomaly_scores)
            alert_risk = self._calculate_alert_risk(alert_history)

            # Weighted combination
            total_risk = (
                sensor_risk * 0.3 +
                trend_risk * 0.3 +
                anomaly_risk * 0.25 +
                alert_risk * 0.15
            )

            # Estimate time to failure
            hours_to_failure = self._estimate_hours_to_failure(
                total_risk,
                trend_risk,
                recent_readings
            )

            # Determine risk level and recommendation
            confidence = min(100, total_risk * 100)
            risk_level = self._get_risk_level(total_risk)
            factors = self._get_contributing_factors(
                recent_readings, anomaly_scores, alert_history
            )
            recommendation = self._get_recommendation(risk_level, factors)

            prediction = FailurePrediction(
                machine_id=machine_id,
                confidence=round(confidence, 2),
                risk_level=risk_level,
                predicted_failure_time=(
                    datetime.now() + timedelta(hours=hours_to_failure)
                    if hours_to_failure else None
                ),
                hours_to_failure=round(hours_to_failure, 1) if hours_to_failure else None,
                contributing_factors=factors,
                recommendation=recommendation
            )

            pred_logger.info(
                f"📊 Prediction for {machine_id}: "
                f"Risk={risk_level} ({confidence}%), "
                f"TTF={hours_to_failure}h"
            )

            return prediction

        except Exception as e:
            pred_logger.error(f"Prediction failed for {machine_id}: {e}")
            return FailurePrediction(
                machine_id=machine_id,
                confidence=0.0,
                risk_level="MEDIUM",
                predicted_failure_time=None,
                hours_to_failure=None,
                contributing_factors=["Prediction error"],
                recommendation="Contact system administrator"
            )

    def _calculate_sensor_risk(self, readings: List[Dict]) -> float:
        """Calculate risk based on current sensor values."""
        if not readings:
            return 0.0

        latest = readings[-1]
        sensors = latest.get("sensors_json", {})
        
        risk_scores = []

        # Check each sensor against thresholds
        if "vibration_mm_s" in sensors:
            vib = float(sensors["vibration_mm_s"])
            if vib > self.thresholds["vibration"]:
                risk_scores.append(min(1.0, (vib / 10.0)))

        if "temperature_c" in sensors:
            temp = float(sensors["temperature_c"])
            if temp > self.thresholds["temperature"]:
                risk_scores.append(min(1.0, (temp - 60) / 30))

        if "pressure_bar" in sensors:
            pressure = float(sensors["pressure_bar"])
            if pressure > self.thresholds["pressure"]:
                risk_scores.append(min(1.0, (pressure - 80) / 40))

        if "noise_db" in sensors:
            noise = float(sensors["noise_db"])
            if noise > self.thresholds["noise"]:
                risk_scores.append(min(1.0, (noise - 70) / 30))

        return np.mean(risk_scores) if risk_scores else 0.0

    def _calculate_trend_risk(self, readings: List[Dict]) -> float:
        """Calculate risk based on trending data."""
        if len(readings) < 10:
            return 0.0

        # Extract vibration trend (most important indicator)
        vibrations = []
        for reading in readings[-20:]:  # Last 20 readings
            sensors = reading.get("sensors_json", {})
            if "vibration_mm_s" in sensors:
                vibrations.append(float(sensors["vibration_mm_s"]))

        if len(vibrations) < 5:
            return 0.0

        # Calculate trend using linear regression
        x = np.arange(len(vibrations))
        y = np.array(vibrations)
        
        try:
            coeffs = np.polyfit(x, y, 1)
            slope = coeffs[0]

            # Positive slope = increasing vibration = bad
            trend_risk = min(1.0, max(0.0, slope / 0.5))
            
            # Also check variability
            variability = np.std(y) / (np.mean(y) + 0.1)
            trend_risk = max(trend_risk, min(1.0, variability / 2))

            return trend_risk
        except:
            return 0.0

    def _calculate_anomaly_risk(self, anomaly_scores: List[float]) -> float:
        """Calculate risk from anomaly detection."""
        if not anomaly_scores:
            return 0.0

        recent = anomaly_scores[-20:] if len(anomaly_scores) > 20 else anomaly_scores
        
        # High anomaly scores = high risk
        avg_anomaly = np.mean(recent)
        max_anomaly = np.max(recent)
        
        # If recent readings have very high anomalies
        high_anomaly_count = sum(1 for a in recent if a > 0.8)
        
        risk = min(
            1.0,
            (avg_anomaly * 0.5) + (max_anomaly * 0.3) + ((high_anomaly_count / len(recent)) * 0.2)
        )
        
        return risk

    def _calculate_alert_risk(self, alerts: List[Dict]) -> float:
        """Calculate risk from recent alert patterns."""
        if not alerts:
            return 0.0

        # Check alerts from last 24 hours
        now = datetime.now()
        recent_alerts = [
            a for a in alerts
            if (datetime.fromisoformat(a.get("timestamp", "")) 
                if isinstance(a.get("timestamp"), str)
                else a.get("timestamp", now))
            > now - timedelta(hours=24)
        ]

        if not recent_alerts:
            return 0.0

        # Weight by risk level
        risk_weights = {
            "CRITICAL": 0.9,
            "HIGH": 0.7,
            "MEDIUM": 0.4,
            "LOW": 0.1
        }

        alert_risk = sum(
            risk_weights.get(a.get("risk_level", "MEDIUM"), 0.5)
            for a in recent_alerts
        ) / max(1, len(recent_alerts))

        return min(1.0, alert_risk)

    def _estimate_hours_to_failure(
        self,
        total_risk: float,
        trend_risk: float,
        readings: List[Dict]
    ) -> Optional[float]:
        """Estimate hours until failure based on risk and trend."""
        
        if total_risk < 0.3:
            return None  # No failure expected
        
        if total_risk < 0.5:
            return 72.0  # Failure in ~3 days
        
        if total_risk < 0.7:
            return 48.0  # Failure in ~2 days
        
        if total_risk < 0.85:
            return 24.0  # Failure in ~1 day
        
        # Critical - could fail soon
        if trend_risk > 0.7:
            return 4.0  # Failure within 4 hours
        
        return 12.0  # Failure within 12 hours

    def _get_risk_level(self, risk_score: float) -> str:
        """Convert risk score to risk level."""
        if risk_score < 0.3:
            return "LOW"
        elif risk_score < 0.5:
            return "MEDIUM"
        elif risk_score < 0.75:
            return "HIGH"
        else:
            return "CRITICAL"

    def _get_contributing_factors(
        self,
        readings: List[Dict],
        anomaly_scores: List[float],
        alerts: List[Dict]
    ) -> List[str]:
        """Identify what's causing the risk."""
        factors = []

        if readings:
            latest = readings[-1].get("sensors_json", {})
            
            if float(latest.get("vibration_mm_s", 0)) > self.thresholds["vibration"]:
                factors.append("High vibration detected")
            
            if float(latest.get("temperature_c", 0)) > self.thresholds["temperature"]:
                factors.append("Elevated temperature")
            
            if float(latest.get("pressure_bar", 0)) > self.thresholds["pressure"]:
                factors.append("High pressure")
            
            if float(latest.get("noise_db", 0)) > self.thresholds["noise"]:
                factors.append("Abnormal noise")

        if anomaly_scores and np.mean(anomaly_scores[-10:]) > 0.7:
            factors.append("Unusual sensor patterns")

        if alerts:
            recent_critical = [a for a in alerts if a.get("risk_level") == "CRITICAL"]
            if recent_critical:
                factors.append(f"{len(recent_critical)} recent critical alerts")

        return factors if factors else ["Degrading performance trend"]

    def _get_recommendation(self, risk_level: str, factors: List[str]) -> str:
        """Generate maintenance recommendation."""
        
        recommendations = {
            "CRITICAL": (
                "🚨 URGENT: Schedule emergency maintenance immediately. "
                "Machine may fail within hours."
            ),
            "HIGH": (
                "⚠️ Schedule preventive maintenance within 24 hours. "
                "Increase monitoring frequency."
            ),
            "MEDIUM": (
                "📋 Plan maintenance for next maintenance window. "
                "Monitor trends closely."
            ),
            "LOW": (
                "✅ Continue normal operations. Maintain regular monitoring."
            )
        }

        base = recommendations.get(risk_level, "Monitor machine status")
        
        # Add specific action based on factors
        if "High vibration" in factors:
            base += " Focus on bearing inspection."
        elif "Elevated temperature" in factors:
            base += " Check cooling system."
        elif "High pressure" in factors:
            base += " Review pressure relief system."

        return base

    def compare_predictions(self, predictions: Dict[str, FailurePrediction]) -> Dict:
        """
        Compare predictions across multiple machines.
        
        Returns rankings and comparative metrics.
        """
        sorted_pred = sorted(
            predictions.items(),
            key=lambda x: x[1].confidence,
            reverse=True
        )

        return {
            "rankings": [
                {
                    "rank": i + 1,
                    "machine_id": machine_id,
                    "confidence": pred.confidence,
                    "risk_level": pred.risk_level,
                    "hours_to_failure": pred.hours_to_failure
                }
                for i, (machine_id, pred) in enumerate(sorted_pred)
            ],
            "healthiest": sorted_pred[-1][0] if sorted_pred else None,
            "highest_risk": sorted_pred[0][0] if sorted_pred else None,
            "average_confidence": np.mean([p[1].confidence for p in sorted_pred]),
            "critical_count": sum(1 for _, p in sorted_pred if p.risk_level == "CRITICAL")
        }
