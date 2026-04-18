"""
Predictive Maintenance Scheduler
Optimizes maintenance windows and analyzes impact of downtime vs. failure risk
"""

from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from enum import Enum
from backend.utils.logger import ai_logger as logger
import json


class MaintenanceType(str, Enum):
    PREVENTIVE = "PREVENTIVE"
    CORRECTIVE = "CORRECTIVE"
    PREDICTIVE = "PREDICTIVE"


class MaintenancePriority(str, Enum):
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


@dataclass
class MaintenanceWindow:
    machine_id: str
    start_time: datetime
    end_time: datetime
    duration_hours: float
    maintenance_type: MaintenanceType
    priority: MaintenancePriority
    confidence: float
    estimated_cost: float
    risk_mitigation: float
    description: str
    is_optimal: bool = True
    production_hours_saved: float = 0.0
    
    def to_dict(self):
        return {
            "machine_id": self.machine_id,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "duration_hours": self.duration_hours,
            "maintenance_type": self.maintenance_type.value,
            "priority": self.priority.value,
            "confidence": self.confidence,
            "estimated_cost": self.estimated_cost,
            "risk_mitigation": self.risk_mitigation,
            "description": self.description,
            "is_optimal": self.is_optimal,
            "production_hours_saved": self.production_hours_saved
        }


@dataclass
class MaintenanceScheduleImpact:
    window_id: str
    downtime_impact: Dict
    risk_impact: Dict
    cost_benefit_analysis: Dict
    recommendation: str
    
    def to_dict(self):
        return {
            "window_id": self.window_id,
            "downtime_impact": self.downtime_impact,
            "risk_impact": self.risk_impact,
            "cost_benefit_analysis": self.cost_benefit_analysis,
            "recommendation": self.recommendation
        }


class MaintenanceScheduler:
    def __init__(self):
        self.schedules: Dict[str, List[MaintenanceWindow]] = {}
        self.maintenance_history: List[Dict] = []
        
        # Configuration
        self.cost_per_hour = 500  # Cost per maintenance hour
        self.production_revenue_per_hour = 2000  # Revenue lost per production hour
        self.critical_failure_cost = 50000  # Cost of unplanned failure
        self.equipment_lifespan_hours = 20000  # Expected equipment lifespan
    
    async def suggest_maintenance_windows(self, machine_id: str,
                                          prediction: Dict,
                                          operation_schedule: Dict = None) -> List[MaintenanceWindow]:
        """Suggest optimal maintenance windows based on predictions"""
        
        if operation_schedule is None:
            # Default: Mon-Fri 22:00-06:00, full weekends
            operation_schedule = {
                "weekday_production": [(6, 22)],  # 6am-10pm
                "weekend_available": True
            }
        
        windows = []
        hours_to_failure = prediction.get("hours_to_failure", 72)
        risk_level = prediction.get("risk_level", "MEDIUM")
        confidence = prediction.get("confidence", 0.7)
        
        # Determine maintenance urgency
        if hours_to_failure < 24:
            priority = MaintenancePriority.CRITICAL
            maintenance_type = MaintenanceType.CORRECTIVE
            maintenance_duration = 8  # Full shift
        elif hours_to_failure < 72:
            priority = MaintenancePriority.HIGH
            maintenance_type = MaintenanceType.PREDICTIVE
            maintenance_duration = 4
        else:
            priority = MaintenancePriority.MEDIUM
            maintenance_type = MaintenanceType.PREVENTIVE
            maintenance_duration = 2
        
        # Generate candidate windows (next 14 days)
        base_time = datetime.utcnow()
        
        for day_offset in range(14):
            candidate_time = base_time + timedelta(days=day_offset)
            
            # Check available slots
            if day_offset % 7 < 5:  # Weekday
                # Suggest off-peak window (22:00-06:00)
                candidate_time = candidate_time.replace(hour=22, minute=0, second=0)
            else:  # Weekend
                # Suggest midday slot
                candidate_time = candidate_time.replace(hour=12, minute=0, second=0)
            
            # Skip if window too far in future
            if (candidate_time - base_time).days > 14:
                continue
            
            # Skip if past failure time
            if (candidate_time - base_time).total_seconds() / 3600 > hours_to_failure:
                continue
            
            end_time = candidate_time + timedelta(hours=maintenance_duration)
            
            # Calculate metrics
            production_hours_saved = self._calculate_production_hours_saved(
                candidate_time, maintenance_duration, operation_schedule
            )
            
            estimated_cost = maintenance_duration * self.cost_per_hour
            risk_mitigation = self._calculate_risk_mitigation(
                hours_to_failure, (candidate_time - base_time).total_seconds() / 3600
            )
            
            window = MaintenanceWindow(
                machine_id=machine_id,
                start_time=candidate_time,
                end_time=end_time,
                duration_hours=maintenance_duration,
                maintenance_type=maintenance_type,
                priority=priority,
                confidence=confidence,
                estimated_cost=estimated_cost,
                risk_mitigation=risk_mitigation,
                description=f"{maintenance_type.value} maintenance: {prediction.get('contributing_factors', ['system health'])[0]}",
                production_hours_saved=production_hours_saved
            )
            
            windows.append(window)
            
            # Return top 3 windows
            if len(windows) >= 3:
                break
        
        self.schedules[machine_id] = windows
        logger.info(f"Generated {len(windows)} maintenance windows for {machine_id}")
        
        return windows
    
    async def analyze_maintenance_impact(self, window: MaintenanceWindow,
                                        current_risk: float) -> MaintenanceScheduleImpact:
        """Analyze cost/benefit of proposed maintenance window"""
        
        downtime_impact = {
            "maintenance_duration_hours": window.duration_hours,
            "production_hours_lost": window.production_hours_saved,
            "downtime_cost": window.production_hours_saved * self.production_revenue_per_hour,
            "maintenance_cost": window.estimated_cost,
            "total_cost": (window.production_hours_saved * self.production_revenue_per_hour) + window.estimated_cost
        }
        
        risk_impact = {
            "current_failure_risk": current_risk,
            "risk_after_maintenance": max(0, current_risk - (window.risk_mitigation * current_risk)),
            "risk_reduction_percent": window.risk_mitigation * 100,
            "avoided_failure_cost": self.critical_failure_cost * window.risk_mitigation,
            "net_benefit": (self.critical_failure_cost * window.risk_mitigation) - downtime_impact["total_cost"]
        }
        
        cost_benefit_analysis = {
            "maintenance_investment": downtime_impact["total_cost"],
            "risk_mitigation_value": risk_impact["avoided_failure_cost"],
            "net_roi": risk_impact["avoided_failure_cost"] - downtime_impact["total_cost"],
            "roi_percent": ((risk_impact["avoided_failure_cost"] - downtime_impact["total_cost"]) / 
                           downtime_impact["total_cost"] * 100) if downtime_impact["total_cost"] > 0 else 0,
            "payback_period_days": 0.5 if risk_impact["net_benefit"] > 0 else float('inf'),
            "recommendation_confidence": window.confidence
        }
        
        # Generate recommendation
        if cost_benefit_analysis["roi_percent"] > 100:
            recommendation = "🟢 HIGHLY RECOMMENDED - High ROI with significant risk reduction"
        elif cost_benefit_analysis["roi_percent"] > 0:
            recommendation = "🟡 RECOMMENDED - Positive ROI with risk mitigation"
        elif risk_impact["risk_reduction_percent"] > 50:
            recommendation = "🟡 CONSIDER - Risk reduction outweighs cost"
        else:
            recommendation = "🔴 LOW PRIORITY - Minimal benefit, defer if possible"
        
        impact = MaintenanceScheduleImpact(
            window_id=f"{window.machine_id}_{window.start_time.isoformat()}",
            downtime_impact=downtime_impact,
            risk_impact=risk_impact,
            cost_benefit_analysis=cost_benefit_analysis,
            recommendation=recommendation
        )
        
        logger.info(f"Impact analysis: {impact.window_id} - ROI: {cost_benefit_analysis['roi_percent']:.1f}%")
        
        return impact
    
    async def get_maintenance_history(self, machine_id: str = None) -> List[Dict]:
        """Get maintenance history for machine or all machines"""
        
        history = self.maintenance_history
        
        if machine_id:
            history = [h for h in history if h["machine_id"] == machine_id]
        
        return sorted(history, key=lambda x: x["timestamp"], reverse=True)
    
    async def record_maintenance(self, machine_id: str, maintenance_type: MaintenanceType,
                                 duration_hours: float, cost: float,
                                 notes: str = "") -> Dict:
        """Record completed maintenance activity"""
        
        record = {
            "machine_id": machine_id,
            "timestamp": datetime.utcnow().isoformat(),
            "maintenance_type": maintenance_type.value,
            "duration_hours": duration_hours,
            "cost": cost,
            "notes": notes
        }
        
        self.maintenance_history.append(record)
        logger.info(f"Maintenance recorded: {machine_id} - {maintenance_type.value}")
        
        return record
    
    async def get_fleet_maintenance_status(self) -> Dict:
        """Get overall fleet maintenance status"""
        
        total_machines = len(self.schedules)
        machines_needing_maintenance = sum(1 for m, windows in self.schedules.items() if windows)
        critical_maintenance = sum(1 for m, windows in self.schedules.items() 
                                  if any(w.priority == MaintenancePriority.CRITICAL for w in windows))
        
        total_maintenance_cost = sum(
            sum(w.estimated_cost for w in windows)
            for windows in self.schedules.values()
        )
        
        return {
            "total_machines": total_machines,
            "machines_needing_maintenance": machines_needing_maintenance,
            "critical_maintenance_required": critical_maintenance,
            "recommended_maintenance_windows": sum(len(w) for w in self.schedules.values()),
            "total_estimated_maintenance_cost": total_maintenance_cost,
            "recent_maintenance_count": len([h for h in self.maintenance_history 
                                            if datetime.fromisoformat(h["timestamp"]) > 
                                            datetime.utcnow() - timedelta(days=30)])
        }
    
    def _calculate_production_hours_saved(self, maintenance_time: datetime,
                                         duration_hours: float,
                                         operation_schedule: Dict) -> float:
        """Calculate production hours lost during maintenance window"""
        
        # Simplified: if maintenance during production hours, count as lost production
        hour_of_day = maintenance_time.hour
        weekday = maintenance_time.weekday()
        
        production_start, production_end = operation_schedule.get("weekday_production", [(6, 22)])[0]
        
        if weekday >= 5 and operation_schedule.get("weekend_available", True):
            # Weekend maintenance - no production loss
            return 0
        
        if production_start <= hour_of_day < production_end:
            # During production hours - count loss
            return duration_hours * 0.8  # Assume 80% of downtime blocks production
        else:
            # Off-hours - minimal loss
            return duration_hours * 0.2
    
    def _calculate_risk_mitigation(self, hours_to_failure: float,
                                   hours_until_maintenance: float) -> float:
        """Calculate risk reduction from proactive maintenance"""
        
        if hours_to_failure <= 0:
            return 0.95  # Imminent failure - maintenance prevents 95% risk
        
        if hours_until_maintenance > hours_to_failure:
            return 0.1  # Too late - minimal benefit
        
        # Risk mitigation proportional to how much time before failure
        time_buffer = hours_to_failure - hours_until_maintenance
        mitigation = min(0.95, 0.5 + (time_buffer / hours_to_failure) * 0.45)
        
        return mitigation
