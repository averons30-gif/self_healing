"""
System Health Trends & Historical Analytics
Tracks machine degradation curves, performance benchmarking, and SLA compliance
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import numpy as np
from backend.utils.logger import ai_logger as logger


@dataclass
class HealthSnapshot:
    machine_id: str
    timestamp: datetime
    health_score: float  # 0-100
    performance_index: float
    degradation_rate: float
    predicted_eol: datetime
    failure_probability: float
    
    def to_dict(self):
        return {
            "machine_id": self.machine_id,
            "timestamp": self.timestamp.isoformat(),
            "health_score": self.health_score,
            "performance_index": self.performance_index,
            "degradation_rate": self.degradation_rate,
            "predicted_eol": self.predicted_eol.isoformat(),
            "failure_probability": self.failure_probability
        }


class HealthTrendsAnalyzer:
    def __init__(self):
        self.health_history: Dict[str, List[HealthSnapshot]] = {}
        self.sla_targets = {
            "uptime_percent": 99.5,
            "max_unplanned_downtime_hours_per_month": 3,
            "max_failure_rate_percent": 0.5
        }
        self.benchmark_standards = {
            "excellent_health": 90,
            "good_health": 70,
            "acceptable_health": 50,
            "poor_health": 30,
            "critical_health": 0
        }
    
    async def calculate_health_score(self, machine_id: str,
                                     current_prediction: Dict,
                                     sensor_readings: List[Dict],
                                     historical_data: Dict = None) -> HealthSnapshot:
        """Calculate comprehensive health score"""
        
        # Factor 1: Prediction-based risk (40%)
        risk_scores = {
            "CRITICAL": 0,
            "HIGH": 20,
            "MEDIUM": 50,
            "LOW": 90
        }
        prediction_health = risk_scores.get(current_prediction.get("risk_level", "MEDIUM"), 50)
        
        # Factor 2: Alert density (20%)
        alert_score = max(0, 100 - current_prediction.get("alert_count", 0) * 5)
        
        # Factor 3: Sensor stability (20%)
        if sensor_readings:
            sensor_score = self._calculate_sensor_stability_score(sensor_readings)
        else:
            sensor_score = 50
        
        # Factor 4: Uptime compliance (20%)
        uptime_score = historical_data.get("uptime_percent", 99) if historical_data else 99
        
        # Weighted average
        health_score = (
            prediction_health * 0.4 +
            alert_score * 0.2 +
            sensor_score * 0.2 +
            uptime_score * 0.2
        )
        
        # Calculate performance index
        confidence = current_prediction.get("confidence", 0.7)
        performance_index = health_score * confidence
        
        # Calculate degradation rate
        degradation_rate = self._calculate_degradation_rate(machine_id)
        
        # Predict end-of-life
        hours_to_failure = current_prediction.get("hours_to_failure", 500)
        predicted_eol = datetime.utcnow() + timedelta(hours=hours_to_failure)
        
        # Calculate failure probability
        failure_probability = 1 - (health_score / 100)
        
        snapshot = HealthSnapshot(
            machine_id=machine_id,
            timestamp=datetime.utcnow(),
            health_score=max(0, min(100, health_score)),
            performance_index=max(0, min(100, performance_index)),
            degradation_rate=degradation_rate,
            predicted_eol=predicted_eol,
            failure_probability=failure_probability
        )
        
        # Store snapshot
        if machine_id not in self.health_history:
            self.health_history[machine_id] = []
        self.health_history[machine_id].append(snapshot)
        
        logger.info(f"Health score calculated: {machine_id} = {health_score:.1f}")
        
        return snapshot
    
    async def get_degradation_curve(self, machine_id: str,
                                   days: int = 90) -> Dict:
        """Get machine degradation trajectory"""
        
        if machine_id not in self.health_history:
            return {"error": "No health history for machine"}
        
        cutoff_time = datetime.utcnow() - timedelta(days=days)
        history = [h for h in self.health_history[machine_id] 
                  if h.timestamp >= cutoff_time]
        
        if not history:
            return {"error": "Insufficient historical data"}
        
        # Sort by timestamp
        history = sorted(history, key=lambda x: x.timestamp)
        
        # Calculate degradation metrics
        start_health = history[0].health_score
        current_health = history[-1].health_score
        degradation_amount = start_health - current_health
        
        days_elapsed = (history[-1].timestamp - history[0].timestamp).days
        daily_degradation_rate = degradation_amount / max(days_elapsed, 1)
        
        # Forecast EOL
        if daily_degradation_rate > 0:
            days_to_critical = 30 / max(daily_degradation_rate, 0.01)
        else:
            days_to_critical = 999
        
        # Calculate trend acceleration
        if len(history) > 4:
            first_half_avg = np.mean([h.health_score for h in history[:len(history)//2]])
            second_half_avg = np.mean([h.health_score for h in history[len(history)//2:]])
            acceleration = (first_half_avg - second_half_avg) / max(first_half_avg, 1)
        else:
            acceleration = 0
        
        return {
            "machine_id": machine_id,
            "analysis_period_days": days,
            "snapshots_count": len(history),
            "start_health": start_health,
            "current_health": current_health,
            "degradation_amount": degradation_amount,
            "daily_degradation_rate": daily_degradation_rate,
            "health_trajectory": [
                {
                    "timestamp": h.timestamp.isoformat(),
                    "health_score": h.health_score,
                    "performance_index": h.performance_index
                } for h in history
            ],
            "forecast": {
                "days_to_critical": max(0, days_to_critical),
                "predicted_critical_date": (datetime.utcnow() + timedelta(days=max(0, days_to_critical))).isoformat(),
                "trend_acceleration": acceleration,
                "trend_status": "ACCELERATING" if acceleration > 0.05 else "STABLE" if acceleration > -0.05 else "IMPROVING"
            }
        }
    
    async def get_performance_benchmarking(self, machine_id: str = None) -> Dict:
        """Compare machine performance against fleet benchmarks"""
        
        if machine_id:
            machines = [machine_id]
        else:
            machines = list(self.health_history.keys())
        
        benchmarks = {}
        
        for mid in machines:
            if mid not in self.health_history or not self.health_history[mid]:
                continue
            
            snapshots = self.health_history[mid]
            latest = snapshots[-1] if snapshots else None
            
            if not latest:
                continue
            
            # Get last 30 days average
            cutoff = datetime.utcnow() - timedelta(days=30)
            recent = [s for s in snapshots if s.timestamp >= cutoff]
            avg_health = np.mean([s.health_score for s in recent]) if recent else latest.health_score
            
            benchmarks[mid] = {
                "current_health": latest.health_score,
                "avg_health_30d": avg_health,
                "performance_rank": 0,  # Will be calculated
                "benchmark_category": self._categorize_health(latest.health_score),
                "degradation_trend": "accelerating" if latest.degradation_rate > 0.5 else "stable" if latest.degradation_rate > 0.1 else "improving"
            }
        
        # Calculate rankings
        sorted_machines = sorted(benchmarks.items(), 
                                key=lambda x: x[1]["current_health"], 
                                reverse=True)
        
        for rank, (mid, data) in enumerate(sorted_machines, 1):
            benchmarks[mid]["performance_rank"] = rank
        
        fleet_avg = np.mean([b["current_health"] for b in benchmarks.values()]) if benchmarks else 0
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "fleet_size": len(benchmarks),
            "fleet_average_health": fleet_avg,
            "benchmarks": benchmarks,
            "top_performers": sorted_machines[:3] if sorted_machines else [],
            "machines_requiring_attention": [mid for mid, data in sorted_machines 
                                            if data["current_health"] < 50]
        }
    
    async def check_sla_compliance(self, machine_id: str,
                                   incident_data: Dict = None) -> Dict:
        """Check SLA compliance metrics"""
        
        if machine_id not in self.health_history:
            return {"error": "No data for SLA analysis"}
        
        # Get last 30 days data
        cutoff = datetime.utcnow() - timedelta(days=30)
        snapshots = [h for h in self.health_history[machine_id] 
                    if h.timestamp >= cutoff]
        
        # Calculate uptime
        avg_health = np.mean([s.health_score for s in snapshots]) if snapshots else 100
        uptime_percent = min(100, avg_health)
        
        # Estimate failure probability
        avg_failure_prob = np.mean([s.failure_probability for s in snapshots]) if snapshots else 0
        estimated_failure_rate = avg_failure_prob * 100
        
        # Compliance checks
        uptime_compliant = uptime_percent >= self.sla_targets["uptime_percent"]
        failure_rate_compliant = estimated_failure_rate <= self.sla_targets["max_failure_rate_percent"]
        
        overall_compliant = uptime_compliant and failure_rate_compliant
        
        compliance_score = (
            (uptime_percent / self.sla_targets["uptime_percent"]) * 50 +
            ((100 - estimated_failure_rate) / 100) * 50
        )
        
        return {
            "machine_id": machine_id,
            "analysis_period": "last_30_days",
            "sla_metrics": {
                "uptime_percent": round(uptime_percent, 2),
                "uptime_target": self.sla_targets["uptime_percent"],
                "uptime_compliant": uptime_compliant,
                "estimated_failure_rate": round(estimated_failure_rate, 2),
                "failure_rate_target": self.sla_targets["max_failure_rate_percent"],
                "failure_rate_compliant": failure_rate_compliant
            },
            "overall_compliance": overall_compliant,
            "compliance_score": round(min(100, compliance_score), 2),
            "recommendations": self._generate_sla_recommendations(uptime_percent, estimated_failure_rate)
        }
    
    async def get_fleet_health_summary(self) -> Dict:
        """Get summary of entire fleet health"""
        
        all_machines = list(self.health_history.keys())
        
        if not all_machines:
            return {
                "fleet_size": 0,
                "fleet_health": 0,
                "machines_by_status": {}
            }
        
        latest_snapshots = []
        for mid in all_machines:
            if self.health_history[mid]:
                latest_snapshots.append(self.health_history[mid][-1])
        
        # Calculate fleet metrics
        health_scores = [s.health_score for s in latest_snapshots]
        fleet_health = np.mean(health_scores)
        
        # Count by category
        by_status = {
            "excellent": len([s for s in latest_snapshots if s.health_score >= 90]),
            "good": len([s for s in latest_snapshots if 70 <= s.health_score < 90]),
            "acceptable": len([s for s in latest_snapshots if 50 <= s.health_score < 70]),
            "poor": len([s for s in latest_snapshots if 30 <= s.health_score < 50]),
            "critical": len([s for s in latest_snapshots if s.health_score < 30])
        }
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "fleet_size": len(all_machines),
            "fleet_average_health": round(fleet_health, 2),
            "machines_by_status": by_status,
            "critical_machines": by_status["critical"],
            "maintenance_urgency": "IMMEDIATE" if by_status["critical"] > 0 else "HIGH" if by_status["poor"] > len(all_machines) * 0.3 else "NORMAL"
        }
    
    def _calculate_sensor_stability_score(self, sensor_readings: List[Dict]) -> float:
        """Calculate stability score from recent sensor readings"""
        
        if not sensor_readings or len(sensor_readings) < 10:
            return 50
        
        recent = sensor_readings[-20:]
        
        stabilities = []
        sensors = ["vibration", "temperature", "pressure", "noise"]
        
        for sensor in sensors:
            values = [float(r.get(sensor, 0)) for r in recent if sensor in r]
            if values:
                std_dev = np.std(values)
                mean_val = np.mean(values)
                cv = (std_dev / mean_val) if mean_val != 0 else 0
                stability = max(0, 100 - (cv * 200))
                stabilities.append(stability)
        
        return np.mean(stabilities) if stabilities else 50
    
    def _calculate_degradation_rate(self, machine_id: str) -> float:
        """Calculate current degradation rate"""
        
        if machine_id not in self.health_history or len(self.health_history[machine_id]) < 2:
            return 0
        
        history = self.health_history[machine_id]
        if len(history) < 2:
            return 0
        
        # Get degradation from last snapshot to previous
        recent = history[-1]
        previous = history[-2]
        
        time_diff = (recent.timestamp - previous.timestamp).total_seconds() / 3600
        health_diff = previous.health_score - recent.health_score
        
        if time_diff > 0:
            return health_diff / time_diff
        else:
            return 0
    
    def _categorize_health(self, health_score: float) -> str:
        """Categorize health into tiers"""
        if health_score >= 90:
            return "EXCELLENT"
        elif health_score >= 70:
            return "GOOD"
        elif health_score >= 50:
            return "ACCEPTABLE"
        elif health_score >= 30:
            return "POOR"
        else:
            return "CRITICAL"
    
    def _generate_sla_recommendations(self, uptime: float, failure_rate: float) -> List[str]:
        """Generate SLA compliance recommendations"""
        
        recommendations = []
        
        if uptime < self.sla_targets["uptime_percent"]:
            recommendations.append(f"⚠️ Uptime below target ({uptime:.2f}% vs {self.sla_targets['uptime_percent']}%) - Increase maintenance frequency")
        
        if failure_rate > self.sla_targets["max_failure_rate_percent"]:
            recommendations.append(f"🔴 Failure rate exceeds target ({failure_rate:.2f}%) - Schedule preventive maintenance immediately")
        
        if len(recommendations) == 0:
            recommendations.append("✅ All SLA targets met - Continue current maintenance schedule")
        
        return recommendations
