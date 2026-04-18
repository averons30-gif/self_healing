"""
Advanced Anomaly Detection Visualization Service
Generates correlation heatmaps, anomaly timelines, and trend analysis
"""

from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
import numpy as np
from backend.utils.logger import ai_logger as logger


@dataclass
class AnomalyEvent:
    timestamp: datetime
    machine_id: str
    sensor_name: str
    sensor_value: float
    expected_range: Tuple[float, float]
    anomaly_score: float
    severity: str
    description: str
    
    def to_dict(self):
        return {
            "timestamp": self.timestamp.isoformat(),
            "machine_id": self.machine_id,
            "sensor_name": self.sensor_name,
            "sensor_value": self.sensor_value,
            "expected_range": list(self.expected_range),
            "anomaly_score": float(self.anomaly_score),
            "severity": self.severity,
            "description": self.description
        }


class AnomalyVisualizer:
    def __init__(self):
        self.sensor_history: Dict[str, List] = {}
        self.anomaly_events: List[AnomalyEvent] = []
        self.correlation_matrix = None
        self.sensor_names = ["vibration", "temperature", "pressure", "noise", "humidity"]
    
    async def generate_correlation_heatmap(self, machine_id: str, 
                                          readings: List[Dict], 
                                          time_window_hours: int = 24) -> Dict:
        """Generate correlation matrix between sensors"""
        
        if len(readings) < 10:
            return {"error": "Insufficient data for correlation analysis"}
        
        # Extract sensor values
        sensor_data = {sensor: [] for sensor in self.sensor_names}
        
        for reading in readings[-288:]:  # Last 24h of readings (5min intervals)
            for sensor in self.sensor_names:
                if sensor in reading:
                    sensor_data[sensor].append(float(reading[sensor]))
        
        # Calculate correlation matrix
        correlation_matrix = np.corrcoef([sensor_data[s] for s in self.sensor_names])
        
        # Create heatmap data
        heatmap = {
            "machine_id": machine_id,
            "timestamp": datetime.utcnow().isoformat(),
            "sensors": self.sensor_names,
            "correlations": correlation_matrix.tolist(),
            "strong_correlations": self._extract_strong_correlations(correlation_matrix),
            "insight": self._generate_correlation_insight(correlation_matrix)
        }
        
        logger.info(f"Correlation heatmap generated for {machine_id}")
        return heatmap
    
    async def generate_anomaly_timeline(self, machine_id: str, 
                                       readings: List[Dict],
                                       days: int = 7) -> Dict:
        """Generate anomaly event timeline"""
        
        cutoff_time = datetime.utcnow() - timedelta(days=days)
        timeline_events = []
        
        # Define sensor thresholds
        thresholds = {
            "vibration": (1.5, 4.0),
            "temperature": (35, 75),
            "pressure": (90, 110),
            "noise": (60, 85),
            "humidity": (30, 70)
        }
        
        # Detect anomalies
        for i, reading in enumerate(readings):
            reading_time = datetime.fromisoformat(reading["timestamp"])
            
            if reading_time < cutoff_time:
                continue
            
            for sensor, (min_val, max_val) in thresholds.items():
                if sensor in reading:
                    value = float(reading[sensor])
                    
                    if value < min_val or value > max_val:
                        anomaly_score = max(
                            abs(value - min_val) / (max_val - min_val),
                            abs(value - max_val) / (max_val - min_val)
                        )
                        
                        severity = self._calculate_anomaly_severity(anomaly_score)
                        
                        event = AnomalyEvent(
                            timestamp=reading_time,
                            machine_id=machine_id,
                            sensor_name=sensor,
                            sensor_value=value,
                            expected_range=(min_val, max_val),
                            anomaly_score=min(anomaly_score, 1.0),
                            severity=severity,
                            description=f"{sensor.capitalize()} out of range: {value:.2f} (expected {min_val}-{max_val})"
                        )
                        
                        timeline_events.append(event)
        
        self.anomaly_events.extend(timeline_events)
        
        return {
            "machine_id": machine_id,
            "period_days": days,
            "total_anomalies": len(timeline_events),
            "events": [e.to_dict() for e in sorted(timeline_events, key=lambda x: x.timestamp)],
            "severity_breakdown": self._breakdown_by_severity(timeline_events),
            "top_anomalous_sensors": self._rank_sensors_by_anomaly(timeline_events)
        }
    
    async def generate_trend_analysis(self, machine_id: str, 
                                      readings: List[Dict],
                                      days: int = 30) -> Dict:
        """Generate trend analysis with degradation curves"""
        
        cutoff_time = datetime.utcnow() - timedelta(days=days)
        
        trends = {}
        for sensor in self.sensor_names:
            sensor_readings = []
            timestamps = []
            
            for reading in readings:
                reading_time = datetime.fromisoformat(reading["timestamp"])
                if reading_time >= cutoff_time and sensor in reading:
                    timestamps.append(reading_time)
                    sensor_readings.append(float(reading[sensor]))
            
            if len(sensor_readings) > 2:
                # Calculate trend direction
                trend_direction = self._calculate_trend_direction(sensor_readings)
                degradation_rate = self._calculate_degradation_rate(sensor_readings)
                trend_forecast = self._forecast_trend(sensor_readings)
                
                trends[sensor] = {
                    "current_value": sensor_readings[-1],
                    "average": np.mean(sensor_readings),
                    "min": np.min(sensor_readings),
                    "max": np.max(sensor_readings),
                    "std_dev": np.std(sensor_readings),
                    "trend_direction": trend_direction,
                    "degradation_rate": degradation_rate,
                    "forecast_24h": trend_forecast,
                    "health_score": self._calculate_sensor_health(sensor_readings)
                }
        
        logger.info(f"Trend analysis generated for {machine_id}")
        
        return {
            "machine_id": machine_id,
            "analysis_period_days": days,
            "timestamp": datetime.utcnow().isoformat(),
            "sensor_trends": trends,
            "overall_health": self._calculate_overall_health(trends),
            "risk_alert": self._generate_trend_risk_alert(trends)
        }
    
    def _extract_strong_correlations(self, correlation_matrix) -> List[Dict]:
        """Extract pairs with strong correlation (>0.7)"""
        strong = []
        for i in range(len(self.sensor_names)):
            for j in range(i+1, len(self.sensor_names)):
                corr = abs(correlation_matrix[i, j])
                if corr > 0.7:
                    strong.append({
                        "sensor1": self.sensor_names[i],
                        "sensor2": self.sensor_names[j],
                        "correlation": float(corr)
                    })
        return sorted(strong, key=lambda x: x["correlation"], reverse=True)
    
    def _calculate_anomaly_severity(self, anomaly_score: float) -> str:
        """Classify anomaly severity"""
        if anomaly_score > 0.8:
            return "CRITICAL"
        elif anomaly_score > 0.6:
            return "HIGH"
        elif anomaly_score > 0.4:
            return "MEDIUM"
        else:
            return "LOW"
    
    def _breakdown_by_severity(self, events: List[AnomalyEvent]) -> Dict:
        """Count events by severity"""
        return {
            "CRITICAL": len([e for e in events if e.severity == "CRITICAL"]),
            "HIGH": len([e for e in events if e.severity == "HIGH"]),
            "MEDIUM": len([e for e in events if e.severity == "MEDIUM"]),
            "LOW": len([e for e in events if e.severity == "LOW"])
        }
    
    def _rank_sensors_by_anomaly(self, events: List[AnomalyEvent]) -> List[Dict]:
        """Rank sensors by anomaly frequency"""
        sensor_counts = {}
        for event in events:
            sensor_counts[event.sensor_name] = sensor_counts.get(event.sensor_name, 0) + 1
        
        return sorted([{"sensor": k, "anomaly_count": v} for k, v in sensor_counts.items()],
                     key=lambda x: x["anomaly_count"], reverse=True)
    
    def _calculate_trend_direction(self, values: List[float]) -> str:
        """Calculate if trend is improving, degrading, or stable"""
        if len(values) < 2:
            return "STABLE"
        
        slope = (values[-1] - values[0]) / len(values)
        if slope > 0.05:
            return "INCREASING"
        elif slope < -0.05:
            return "DECREASING"
        else:
            return "STABLE"
    
    def _calculate_degradation_rate(self, values: List[float]) -> float:
        """Calculate rate of degradation per day"""
        if len(values) < 2:
            return 0.0
        return float((values[-1] - values[0]) / max(len(values), 1))
    
    def _forecast_trend(self, values: List[float]) -> float:
        """Simple linear forecast for next 24h"""
        if len(values) < 2:
            return values[-1]
        
        slope = (values[-1] - values[-2])
        return float(values[-1] + slope)
    
    def _calculate_sensor_health(self, values: List[float]) -> int:
        """Health score 0-100 based on stability"""
        if len(values) < 2:
            return 100
        
        std_dev = np.std(values)
        mean_val = np.mean(values)
        
        cv = (std_dev / mean_val) if mean_val != 0 else 0
        health = max(0, min(100, int(100 - (cv * 100))))
        return health
    
    def _calculate_overall_health(self, trends: Dict) -> int:
        """Average health across all sensors"""
        if not trends:
            return 100
        health_scores = [trends[s].get("health_score", 100) for s in trends]
        return int(np.mean(health_scores))
    
    def _generate_trend_risk_alert(self, trends: Dict) -> List[str]:
        """Generate warnings for concerning trends"""
        alerts = []
        
        for sensor, data in trends.items():
            if data["trend_direction"] == "INCREASING" and "vibration" in sensor:
                alerts.append(f"⚠️ {sensor.capitalize()} increasing - potential bearing wear")
            
            if data["trend_direction"] == "INCREASING" and "temperature" in sensor:
                alerts.append(f"⚠️ {sensor.capitalize()} increasing - thermal stress detected")
            
            if data["health_score"] < 50:
                alerts.append(f"🔴 {sensor.capitalize()} health critically low ({data['health_score']}%)")
        
        return alerts
