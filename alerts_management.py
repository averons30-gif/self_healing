"""
API Routes for Real-time Alert Management
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from datetime import datetime
from typing import List, Optional
from backend.ai_engine.alert_notification_service import (
    AlertNotificationService, AlertSeverity, AlertStatus
)
from backend.database.db_manager import DatabaseManager
from backend.api.middleware.auth import require_viewer

router = APIRouter()

# Global alert service instance
alert_service: Optional[AlertNotificationService] = None


async def get_alert_service() -> AlertNotificationService:
    """Dependency to get alert service"""
    global alert_service
    if alert_service is None:
        db_manager = DatabaseManager()
        await db_manager.initialize()
        alert_service = AlertNotificationService(db_manager)
    return alert_service


@router.get("/alerts/active", tags=["Alerts"])
async def get_active_alerts(
    machine_id: Optional[str] = Query(None),
    severity: Optional[str] = Query(None),
    service: AlertNotificationService = Depends(get_alert_service),
    user = Depends(require_viewer)
):
    """Get active alerts, optionally filtered by machine or severity"""
    try:
        severity_enum = AlertSeverity[severity] if severity else None
        alerts = await service.get_active_alerts(machine_id, severity_enum)
        return {
            "status": "success",
            "count": len(alerts),
            "alerts": [a.to_dict() for a in alerts]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/alerts/stats", tags=["Alerts"])
async def get_alert_stats(
    service: AlertNotificationService = Depends(get_alert_service),
    user = Depends(require_viewer)
):
    """Get alert statistics and summary"""
    try:
        stats = await service.get_alert_stats()
        return {
            "status": "success",
            "data": stats
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/alerts/acknowledge/{alert_id}", tags=["Alerts"])
async def acknowledge_alert(
    alert_id: str,
    service: AlertNotificationService = Depends(get_alert_service),
    user = Depends(require_viewer)
):
    """Acknowledge an alert"""
    try:
        success = await service.acknowledge_alert(alert_id, user_id=user)
        if success:
            return {"status": "success", "message": f"Alert {alert_id} acknowledged"}
        else:
            raise HTTPException(status_code=404, detail="Alert not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/alerts/dismiss/{alert_id}", tags=["Alerts"])
async def dismiss_alert(
    alert_id: str,
    duration_minutes: int = Query(60),
    service: AlertNotificationService = Depends(get_alert_service),
    user = Depends(require_viewer)
):
    """Dismiss an alert for specified duration"""
    try:
        success = await service.dismiss_alert(alert_id, duration_minutes)
        if success:
            return {
                "status": "success",
                "message": f"Alert dismissed for {duration_minutes} minutes"
            }
        else:
            raise HTTPException(status_code=404, detail="Alert not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/alerts/resolve/{alert_id}", tags=["Alerts"])
async def resolve_alert(
    alert_id: str,
    service: AlertNotificationService = Depends(get_alert_service),
    user = Depends(require_viewer)
):
    """Mark alert as resolved"""
    try:
        success = await service.resolve_alert(alert_id)
        if success:
            return {"status": "success", "message": f"Alert {alert_id} resolved"}
        else:
            raise HTTPException(status_code=404, detail="Alert not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
