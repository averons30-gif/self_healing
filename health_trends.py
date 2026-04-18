"""
API Routes for System Health Trends & Historical Analytics
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from typing import Optional
from backend.ai_engine.health_trends_analyzer import HealthTrendsAnalyzer
from backend.database.db_manager import DatabaseManager
from backend.api.middleware.auth import require_viewer

router = APIRouter()

# Global instances
health_analyzer: Optional[HealthTrendsAnalyzer] = None
db = DatabaseManager()


async def get_health_analyzer() -> HealthTrendsAnalyzer:
    """Dependency to get health trends analyzer"""
    global health_analyzer
    if health_analyzer is None:
        health_analyzer = HealthTrendsAnalyzer()
    return health_analyzer


@router.get("/health/score/{machine_id}", tags=["Health Trends"])
async def get_health_score(
    machine_id: str,
    analyzer: HealthTrendsAnalyzer = Depends(get_health_analyzer),
    user = Depends(require_viewer)
):
    """Get current health score for a machine"""
    try:
        prediction = await db.get_latest_prediction(machine_id)
        sensor_readings = await db.get_sensor_readings(machine_id, limit=100)
        
        if not prediction:
            raise HTTPException(status_code=404, detail="No prediction data for machine")
        
        snapshot = await analyzer.calculate_health_score(machine_id, prediction, sensor_readings)
        
        return {
            "status": "success",
            "data": snapshot.to_dict()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health/degradation/{machine_id}", tags=["Health Trends"])
async def get_degradation_curve(
    machine_id: str,
    days: int = Query(90, ge=7, le=365),
    analyzer: HealthTrendsAnalyzer = Depends(get_health_analyzer),
    user = Depends(require_viewer)
):
    """Get machine degradation trajectory over time"""
    try:
        curve = await analyzer.get_degradation_curve(machine_id, days)
        return {
            "status": "success",
            "data": curve
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health/benchmarking", tags=["Health Trends"])
async def get_performance_benchmarking(
    machine_id: Optional[str] = Query(None),
    analyzer: HealthTrendsAnalyzer = Depends(get_health_analyzer),
    user = Depends(require_viewer)
):
    """Get machine performance against fleet benchmarks"""
    try:
        benchmarks = await analyzer.get_performance_benchmarking(machine_id)
        return {
            "status": "success",
            "data": benchmarks
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health/sla-compliance/{machine_id}", tags=["Health Trends"])
async def check_sla_compliance(
    machine_id: str,
    analyzer: HealthTrendsAnalyzer = Depends(get_health_analyzer),
    user = Depends(require_viewer)
):
    """Check SLA compliance metrics for a machine"""
    try:
        compliance = await analyzer.check_sla_compliance(machine_id)
        return {
            "status": "success",
            "data": compliance
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health/fleet-summary", tags=["Health Trends"])
async def get_fleet_health_summary(
    analyzer: HealthTrendsAnalyzer = Depends(get_health_analyzer),
    user = Depends(require_viewer)
):
    """Get health summary for entire fleet"""
    try:
        summary = await analyzer.get_fleet_health_summary()
        return {
            "status": "success",
            "data": summary
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
