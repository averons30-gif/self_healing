import numpy as np
from collections import deque
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from backend.utils.logger import ai_logger

class FailurePredictor:
    """
    Predictive failure analysis using:
    1. Trend extrapolation (linear + polynomial regression)
    2. Remaining Useful Life (RUL) estimation
    3. Scenario simulation (what-if analysis)
    4. Pattern-based failure forecasting
    """
    
    def __init__(self, sequence_length: int = 50):
        self.sequence_length = sequence_length
        self.prediction_horizons = [1, 6, 24, 72]  # hours
        self.sensor_sequences: Dict[str, deque] = {}
        self.timestamp_sequences: Dict[str, deque] = {}
    
    def _init_machine(self, machine_id: str):
        if machine_id not in self.sensor_sequences:
            self.sensor_sequences[machine_id] = deque(maxlen=self.sequence_length)
            self.timestamp_sequences[machine_id] = deque(maxlen=self.sequence_length)
    
    def add_reading(self, machine_id: str, reading: Dict[str, float], timestamp: datetime):
        """Add sensor reading to prediction buffer"""
        self._init_machine(machine_id)
        self.sensor_sequences[machine_id].append(reading)
        self.timestamp_sequences[machine_id].append(timestamp)
    
    def predict_trajectory(
        self, 
        machine_id: str,
        sensor: str
    ) -> Dict[str, Any]:
        """
        Predict future sensor trajectory using polynomial regression.
        Returns prediction for each horizon.
        """
        self._init_machine(machine_id)
        
        sequence = self.sensor_sequences.get(machine_id, deque())
        if len(sequence) < 10:
            return {"predictions": {}, "confidence": 0, "trend_slope": 0}
        
        values = [s.get(sensor, 0) for s in sequence]
        x = np.arange(len(values))
        
        # Fit polynomial (degree 2 for better curve fitting)
        try:
            coeffs = np.polyfit(x, values, deg=2)
            poly = np.poly1d(coeffs)
            
            # Predict future points
            predictions = {}
            steps_per_hour = 3600 / 1  # Assuming 1-second intervals
            
            for hours in self.prediction_horizons:
                future_x = len(values) + hours * steps_per_hour
                predicted_val = float(poly(future_x))
                predictions[f"{hours}h"] = predicted_val
            
            # Compute trend slope (linear component)
            linear_coeffs = np.polyfit(x[-20:], values[-20:], deg=1)
            slope = float(linear_coeffs[0])
            
            # Confidence based on R²
            fitted = poly(x)
            ss_res = np.sum((np.array(values) - fitted) ** 2)
            ss_tot = np.sum((np.array(values) - np.mean(values)) ** 2)
            r2 = 1 - (ss_res / (ss_tot + 1e-10))
            
            return {
                "predictions": predictions,
                "confidence": max(0, min(1, r2)),
                "trend_slope": slope,
                "current_value": values[-1],
                "trend_direction": "rising" if slope > 0.01 else "falling" if slope < -0.01 else "stable"
            }
        except Exception as e:
            return {"predictions": {}, "confidence": 0, "trend_slope": 0}
    
    def simulate_failure_scenarios(
        self,
        machine_id: str,
        baselines: Dict[str, Dict],
        current_readings: Dict[str, float],
        failure_signature: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Simulate future failure scenarios for Digital Twin.
        Each scenario represents a possible failure trajectory.
        """
        scenarios = []
        sensors = ["temperature", "vibration", "rpm", "current"]
        
        # Scenario 1: Normal degradation
        normal_scenario = {
            "scenario_id": "normal_degradation",
            "name": "Normal Wear & Tear",
            "probability": 0.7,
            "estimated_hours": 720,  # 30 days
            "description": "Expected gradual degradation under normal operating conditions",
            "sensor_projections": {}
        }
        
        for sensor in sensors:
            baseline = baselines.get(sensor, {})
            current = current_readings.get(sensor, 0)
            if baseline.get("max"):
                drift_rate = (baseline["max"] - current) / 720
                normal_scenario["sensor_projections"][sensor] = [
                    current + drift_rate * h for h in [0, 24, 72, 168, 720]
                ]
        
        scenarios.append(normal_scenario)
        
        # Scenario 2: Accelerated failure (if anomaly detected)
        if failure_signature:
            accelerated_scenario = self._build_failure_scenario(
                failure_signature, current_readings, baselines
            )
            scenarios.append(accelerated_scenario)
        
        # Scenario 3: Best case (maintenance performed)
        maintenance_scenario = {
            "scenario_id": "post_maintenance",
            "name": "After Maintenance",
            "probability": 0.3,
            "estimated_hours": 2160,  # 90 days
            "description": "Trajectory after preventive maintenance is performed",
            "sensor_projections": {}
        }
        
        for sensor in sensors:
            baseline = baselines.get(sensor, {})
            if baseline.get("mean"):
                maintenance_scenario["sensor_projections"][sensor] = [
                    baseline["mean"] * (1 + 0.05 * i) for i in range(5)
                ]
        
        scenarios.append(maintenance_scenario)
        
        return scenarios
    
    def _build_failure_scenario(
        self, 
        failure_type: str,
        current_readings: Dict[str, float],
        baselines: Dict[str, Dict]
    ) -> Dict:
        """Build specific failure scenario based on signature"""
        
        failure_profiles = {
            "bearing_fault": {
                "name": "Bearing Failure",
                "hours": 48,
                "probability": 0.65,
                "escalation": {"vibration": 3.0, "temperature": 1.8, "rpm": 0.85, "current": 1.2}
            },
            "overheating": {
                "name": "Thermal Runaway",
                "hours": 24,
                "probability": 0.75,
                "escalation": {"temperature": 4.0, "current": 1.5, "rpm": 0.9, "vibration": 1.3}
            },
            "cavitation": {
                "name": "Cavitation Damage",
                "hours": 72,
                "probability": 0.55,
                "escalation": {"vibration": 2.5, "rpm": 0.8, "current": 1.4, "temperature": 1.3}
            },
            "electrical_fault": {
                "name": "Electrical Fault",
                "hours": 12,
                "probability": 0.80,
                "escalation": {"current": 5.0, "temperature": 2.0, "rpm": 0.7, "vibration": 1.5}
            }
        }
        
        profile = failure_profiles.get(failure_type, failure_profiles["bearing_fault"])
        
        scenario = {
            "scenario_id": failure_type,
            "name": profile["name"],
            "probability": profile["probability"],
            "estimated_hours": profile["hours"],
            "description": f"Projected {profile['name']} failure trajectory",
            "sensor_projections": {}
        }
        
        time_points = [0, profile["hours"] // 4, profile["hours"] // 2, 
                       profile["hours"] * 3 // 4, profile["hours"]]
        
        for sensor in ["temperature", "vibration", "rpm", "current"]:
            current = current_readings.get(sensor, 0)
            escalation = profile["escalation"].get(sensor, 1.0)
            
            projections = [
                current * (1 + (escalation - 1) * t / profile["hours"])
                for t in time_points
            ]
            scenario["sensor_projections"][sensor] = projections
        
        return scenario
    
    def compute_rul(
        self,
        machine_id: str,
        risk_score: float,
        trend: str,
        uptime_hours: float
    ) -> Dict[str, Any]:
        """
        Remaining Useful Life estimation.
        Uses risk-weighted exponential model.
        """
        if risk_score < 20:
            rul_hours = 8760  # 1 year
            confidence = "high"
        elif risk_score < 40:
            rul_hours = max(168, 8760 * (1 - risk_score/100) ** 2)
            confidence = "medium"
        elif risk_score < 60:
            rul_hours = max(72, 1000 * (1 - risk_score/100) ** 1.5)
            confidence = "medium"
        elif risk_score < 80:
            rul_hours = max(24, 200 * (1 - risk_score/100))
            confidence = "low"
        else:
            rul_hours = max(1, 50 * (1 - risk_score/100) * 0.5)
            confidence = "low"
        
        trend_modifier = {"degrading": 0.7, "stable": 1.0, "improving": 1.3}
        rul_hours *= trend_modifier.get(trend, 1.0)
        
        predicted_failure = datetime.utcnow() + timedelta(hours=rul_hours)
        
        return {
            "rul_hours": round(rul_hours, 1),
            "predicted_failure_at": predicted_failure.isoformat(),
            "confidence": confidence,
            "recommended_maintenance_by": (
                datetime.utcnow() + timedelta(hours=rul_hours * 0.7)
            ).isoformat()
        }