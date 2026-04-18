"""
API Routes for Anomaly Detection Visualization
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from typing import Optional, List
from backend.ai_engine.anomaly_visualizer import AnomalyVisualizer
from backend.database.db_manager import DatabaseManager
from backend.api.middleware.auth import require_viewer

router = APIRouter()

# Global instances
anomaly_visualizer: Optional[AnomalyVisualizer] = None
db = DatabaseManager()


async def get_anomaly_visualizer() -> AnomalyVisualizer:
    """Dependency to get anomaly visualizer"""
    global anomaly_visualizer
    if anomaly_visualizer is None:
        anomaly_visualizer = AnomalyVisualizer()
    return anomaly_visualizer


@router.get("/anomalies/heatmap/{machine_id}", tags=["Anomaly Detection"])
async def get_correlation_heatmap(
    machine_id: str,
    visualizer: AnomalyVisualizer = Depends(get_anomaly_visualizer),
    user = Depends(require_viewer)
):
    """Get sensor correlation heatmap for machine"""
    try:
        readings = await db.get_sensor_readings(machine_id, limit=300)
        heatmap = await visualizer.generate_correlation_heatmap(machine_id, readings)
        return {
            "status": "success",
            "data": heatmap
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/anomalies/timeline/{machine_id}", tags=["Anomaly Detection"])
async def get_anomaly_timeline(
    machine_id: str,
    days: int = Query(7, ge=1, le=90),
    visualizer: AnomalyVisualizer = Depends(get_anomaly_visualizer),
    user = Depends(require_viewer)
):
    """Get anomaly timeline for specified period"""
    try:
        readings = await db.get_sensor_readings(machine_id, limit=1000)
        timeline = await visualizer.generate_anomaly_timeline(machine_id, readings, days)
        return {
            "status": "success",
            "data": timeline
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/anomalies/trends/{machine_id}", tags=["Anomaly Detection"])
async def get_trend_analysis(
    machine_id: str,
    days: int = Query(30, ge=1, le=90),
    visualizer: AnomalyVisualizer = Depends(get_anomaly_visualizer),
    user = Depends(require_viewer)
):
    """Get trend analysis with degradation curves"""
    try:
        readings = await db.get_sensor_readings(machine_id, limit=2000)
        trends = await visualizer.generate_trend_analysis(machine_id, readings, days)
        return {
            "status": "success",
            "data": trends
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/anomalies/summary", tags=["Anomaly Detection"])
async def get_anomaly_summary(
    visualizer: AnomalyVisualizer = Depends(get_anomaly_visualizer),
    user = Depends(require_viewer)
):
    """Get summary of recent anomalies across all machines"""
    try:
        total_anomalies = len(visualizer.anomaly_events)
        recent_events = visualizer.anomaly_events[-50:] if visualizer.anomaly_events else []
        
        critical_count = len([e for e in recent_events if e.severity == "CRITICAL"])
        high_count = len([e for e in recent_events if e.severity == "HIGH"])
        
        return {
            "status": "success",
            "data": {
                "total_anomalies_detected": total_anomalies,
                "recent_anomalies": [e.to_dict() for e in recent_events],
                "critical_count": critical_count,
                "high_severity_count": high_count,
                "most_anomalous_sensor": visualizer.sensor_names[0] if visualizer.sensor_names else "N/A"
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
