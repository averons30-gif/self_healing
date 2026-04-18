from fastapi import APIRouter, HTTPException, Query, Depends
from typing import Optional, List, Dict, Any
from pydantic import BaseModel
from datetime import datetime
from backend.database.db_manager import DatabaseManager
from backend.api.middleware.auth import require_viewer, require_operator, require_admin

router = APIRouter(
    prefix="/alerts",
    tags=["alerts"],
    responses={404: {"description": "Alert not found"}}
)
db = DatabaseManager()

# Global reference to stream_handler (set by main.py)
_stream_handler = None

def set_stream_handler(handler):
    """Set the stream handler instance (called by main.py)"""
    global _stream_handler
    _stream_handler = handler

class Alert(BaseModel):
    """Alert data model"""
    id: str
    machine_id: str
    timestamp: str
    severity: str
    anomaly_type: str
    message: str
    risk_score: float
    acknowledged: bool
    sensor_data: Dict[str, Any]

class AlertStats(BaseModel):
    """Alert statistics summary"""
    total: int
    by_severity: Dict[str, int]
    by_machine: Dict[str, int]
    by_type: Dict[str, int]

@router.get(
    "",
    response_model=Dict[str, Any],
    summary="Get Alerts",
    description="Retrieve alerts from the system with optional filtering by machine."
)
async def get_alerts(
    machine_id: Optional[str] = Query(None, description="Filter by specific machine ID"),
    limit: int = Query(50, description="Maximum number of alerts to return", ge=1, le=500)
):
    """
    Get alerts from the alert management system.

    Parameters:
        - machine_id: Optional filter for specific machine
        - limit: Maximum alerts to return (1-500)

    Returns:
        - alerts: List of alert objects
        - count: Number of alerts returned
        - queue_size: Current size of priority queue
    """
    stream_handler = _stream_handler
    if not stream_handler:
        raise HTTPException(status_code=503, detail="System initializing")

    alerts = await db.get_alerts(machine_id=machine_id, risk_level=None, resolved=None, limit=limit, hours=24)
    for alert in alerts:
        alert["severity"] = alert.get("risk_level")
        alert["anomaly_type"] = alert.get("alert_type")
        alert["sensor_data"] = alert.get("sensors")

    return {
        "alerts": alerts,
        "count": len(alerts),
        "queue_size": stream_handler.alert_queue.size()
    }

@router.post(
    "/{alert_id}/acknowledge",
    summary="Acknowledge Alert",
    description="Mark an alert as acknowledged by an operator."
)
async def acknowledge_alert(alert_id: str):
    """
    Acknowledge an alert to indicate it has been reviewed.

    Parameters:
        - alert_id: Unique identifier of the alert

    Returns:
        - Confirmation of acknowledgment

    Note: Acknowledged alerts remain in history but are marked as reviewed.
    """
    success = db.acknowledge_alert(alert_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Alert {alert_id} not found")

    return {"status": "acknowledged", "alert_id": alert_id}

@router.get(
    "/priority-queue",
    summary="Get Priority Queue",
    description="Retrieve alerts sorted by priority for operator attention."
)
async def get_priority_queue():
    """
    Get alerts sorted by priority from the priority queue.

    Returns:
        - alerts: Priority-sorted list of active alerts
        - count: Number of alerts in queue

    The priority queue ensures critical alerts are presented first.
    """
    stream_handler = _stream_handler
    if not stream_handler:
        raise HTTPException(status_code=503, detail="System initializing")

    alerts = stream_handler.alert_queue.get_all_sorted()
    for alert in alerts:
        alert["severity"] = alert.get("risk_level")
        alert["anomaly_type"] = alert.get("alert_type")
        alert["sensor_data"] = alert.get("sensors")
    return {"alerts": alerts, "count": len(alerts)}

@router.get(
    "/stats",
    response_model=AlertStats,
    summary="Get Alert Statistics",
    description="Retrieve statistical summary of alerts by severity, machine, and type."
)
async def get_alert_stats():
    """
    Get comprehensive statistics about alerts in the system.

    Returns:
        - total: Total number of alerts
        - by_severity: Alert count by severity level (critical, high, medium, low)
        - by_machine: Alert count per machine
        - by_type: Alert count by anomaly type

    Analyzes last 500 alerts for performance.
    """
    all_alerts = await db.get_alerts(limit=500)

    stats = {
        "total": len(all_alerts),
        "by_severity": {},
        "by_machine": {},
        "by_type": {}
    }

    for alert in all_alerts:
        severity = alert.get('severity') or alert.get('risk_level') or 'unknown'
        alert_type = alert.get('anomaly_type') or alert.get('alert_type') or 'unknown'

        alert['severity'] = severity
        alert['anomaly_type'] = alert_type

        stats['by_severity'][severity] = stats['by_severity'].get(severity, 0) + 1
        mid = alert.get('machine_id', 'unknown')
        stats['by_machine'][mid] = stats['by_machine'].get(mid, 0) + 1
        stats['by_type'][alert_type] = stats['by_type'].get(alert_type, 0) + 1

    return stats

@router.delete(
    "/{alert_id}",
    summary="Delete Alert",
    description="Permanently remove an alert from the system."
)
async def delete_alert(alert_id: str, user: Dict = Depends(require_admin)):
    """
    Delete an alert from the database.

    Parameters:
        - alert_id: Unique identifier of the alert to delete

    Returns:
        - Confirmation of deletion

    Warning: This permanently removes the alert. Use with caution.
    """
    success = db.delete_alert(alert_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Alert {alert_id} not found")

    return {"status": "deleted", "alert_id": alert_id}

@router.get(
    "/active",
    summary="Get Active Alerts",
    description="Retrieve only unresolved alerts requiring attention."
)
async def get_active_alerts():
    """
    Get all active (unresolved) alerts.

    Returns:
        - alerts: List of alerts needing operator attention
        - count: Number of active alerts

    Use this endpoint for dashboard alerts panel.
    """
    # Get unresolved alerts from database (most recent first)
    all_alerts = await db.get_alerts(resolved=False, limit=100, hours=24)
    for alert in all_alerts:
        alert["severity"] = alert.get("risk_level")
        alert["anomaly_type"] = alert.get("alert_type")
        alert["sensor_data"] = alert.get("sensors")
    return {
        "status": "success", 
        "data": {
            "alerts": all_alerts,
            "count": len(all_alerts)
        }
    }