from fastapi import APIRouter, HTTPException, Query, Depends
from typing import Dict, Any, List, Optional
from pydantic import BaseModel
from datetime import datetime, timedelta
from backend.database.db_manager import DatabaseManager
import numpy as np

router = APIRouter(
    prefix="/analytics",
    tags=["analytics"],
    responses={503: {"description": "System initializing"}}
)
db = DatabaseManager()

# Global reference to stream_handler (set by main.py)
_stream_handler = None

def set_stream_handler(handler):
    """Set the stream handler instance (called by main.py)"""
    global _stream_handler
    _stream_handler = handler

class SystemHealth(BaseModel):
    """Overall system health metrics"""
    system_health_percentage: float
    average_risk: float
    machines: Dict[str, Dict[str, Any]]
    active_connections: int
    queue_size: int

class RiskTrends(BaseModel):
    """Risk score trends over time"""
    history: List[float]
    current: float
    trend: str
    max: float
    avg: float

@router.get(
    "/system-health",
    response_model=SystemHealth,
    summary="Get System Health",
    description="Retrieve overall health metrics for the entire digital twin system."
)
async def get_system_health():
    """
    Get comprehensive system health overview.

    Returns:
        - system_health_percentage: Overall system health (0-100%)
        - average_risk: Average risk score across all machines
        - machines: Individual machine health metrics
        - active_connections: Number of active WebSocket connections
        - queue_size: Current alert queue size

    Each machine includes:
        - risk_score: Current risk assessment (0-1)
        - health_percentage: Health as percentage (0-100%)
        - status: Operational state
        - predicted_failure_hours: Estimated time to failure
    """
    stream_handler = _stream_handler
    if not stream_handler:
        raise HTTPException(status_code=503, detail="System initializing")

    machine_healths = {}
    total_risk = 0.0

    for mid, state in stream_handler.machine_states.items():
        risk = state.risk_score
        total_risk += risk
        machine_healths[mid] = {
            "risk_score": risk,
            "health_percentage": (1 - risk) * 100,
            "status": state.status.value,
            "predicted_failure_hours": state.predicted_failure_hours
        }

    avg_risk = total_risk / max(len(stream_handler.machine_states), 1)

    return {
        "system_health_percentage": (1 - avg_risk) * 100,
        "average_risk": avg_risk,
        "machines": machine_healths,
        "active_connections": 0,  # Would use manager.get_connection_count()
        "queue_size": stream_handler.alert_queue.size()
    }

@router.get(
    "/risk-trends",
    response_model=Dict[str, RiskTrends],
    summary="Get Risk Trends",
    description="Retrieve risk score trends and analysis for all machines."
)
async def get_risk_trends():
    """
    Get risk score trends over time for trend analysis.

    Returns:
        For each machine:
        - history: Last 100 risk scores
        - current: Most recent risk score
        - trend: "increasing", "decreasing", or "stable"
        - max: Maximum risk score in history
        - avg: Average risk score

    Used for dashboard trend charts and predictive analytics.
    """
    stream_handler = _stream_handler
    if not stream_handler:
        raise HTTPException(status_code=503, detail="System initializing")

    trends = {}
    for mid, history in stream_handler.risk_histories.items():
        recent = history[-100:] if history else []
        trends[mid] = {
            "history": recent,
            "current": recent[-1] if recent else 0,
            "trend": "increasing" if len(recent) > 10 and recent[-1] > recent[-10] else "stable",
            "max": max(recent) if recent else 0,
            "avg": float(np.mean(recent)) if recent else 0
        }
    return trends

@router.get(
    "/performance-metrics",
    summary="Get Performance Metrics",
    description="Retrieve system performance and efficiency metrics."
)
async def get_performance_metrics():
    """
    Get performance metrics for system optimization.

    Returns:
        - processing_times: AI processing latency
        - throughput: Readings processed per second
        - memory_usage: System resource usage
        - error_rates: Processing error statistics
    """
    stream_handler = _stream_handler
    if not stream_handler:
        raise HTTPException(status_code=503, detail="System initializing")

    # Calculate performance metrics
    metrics = {
        "total_readings_processed": sum(len(history) for history in stream_handler.risk_histories.values()),
        "active_streams": len(stream_handler.machine_states),
        "alert_queue_size": stream_handler.alert_queue.size(),
        "system_uptime": "00:00:00",  # Would track actual uptime
        "average_processing_time": 0.002,  # Mock value in seconds
        "memory_usage_mb": 150  # Mock value
    }

    return metrics

@router.get(
    "/predictive-insights",
    summary="Get Predictive Insights",
    description="Retrieve AI-generated predictive maintenance insights."
)
async def get_predictive_insights():
    """
    Get predictive maintenance insights and recommendations.

    Returns:
        - failure_predictions: Upcoming failure predictions
        - maintenance_recommendations: Suggested actions
        - component_analysis: Component health breakdown
        - optimization_opportunities: System improvement suggestions
    """
    stream_handler = _stream_handler
    if not stream_handler:
        raise HTTPException(status_code=503, detail="System initializing")

    insights = {
        "high_risk_machines": [],
        "maintenance_schedule": [],
        "component_health_summary": {},
        "system_efficiency_score": 85.5
    }

    # Identify high-risk machines
    for mid, state in stream_handler.machine_states.items():
        if state.risk_score > 0.7:
            insights["high_risk_machines"].append({
                "machine_id": mid,
                "risk_score": state.risk_score,
                "predicted_failure_hours": state.predicted_failure_hours,
                "recommended_action": "Immediate inspection required"
            })

    return insights

@router.get(
    "/sensor-correlations",
    summary="Get Sensor Correlations",
    description="Analyze correlations between different sensors for root cause analysis."
)
async def get_sensor_correlations(machine_id: str):
    """
    Get correlation analysis between sensors for a specific machine.

    Parameters:
        - machine_id: Machine identifier for analysis

    Returns:
        - correlation_matrix: Sensor correlation coefficients
        - key_relationships: Important sensor relationships
        - anomaly_patterns: Detected correlation anomalies
    """
    stream_handler = _stream_handler
    if not stream_handler:
        raise HTTPException(status_code=503, detail="System initializing")

    if machine_id not in stream_handler.machine_states:
        raise HTTPException(status_code=404, detail=f"Machine {machine_id} not found")

    # Mock correlation analysis (would use actual data analysis)
    correlations = {
        "correlation_matrix": {
            "temperature_vibration": 0.75,
            "pressure_current": 0.62,
            "rpm_load": 0.45
        },
        "key_relationships": [
            "High temperature often correlates with increased vibration",
            "Pressure spikes typically precede current surges"
        ],
        "anomaly_patterns": []
    }

    return correlations