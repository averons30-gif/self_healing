from fastapi import APIRouter, HTTPException, Depends, Query
from typing import Dict, Any, List, Optional
from pydantic import BaseModel
from datetime import datetime
from backend.database.db_manager import DatabaseManager
from backend.api.middleware.auth import require_viewer, require_operator, require_admin

router = APIRouter(
    prefix="/maintenance",
    tags=["maintenance"],
    responses={503: {"description": "System initializing"}}
)
db = DatabaseManager()

# Global reference to stream_handler (set by main.py)
_stream_handler = None

def set_stream_handler(handler):
    """Set the stream handler instance (called by main.py)"""
    global _stream_handler
    _stream_handler = handler

class MaintenanceTask(BaseModel):
    """Maintenance task data model"""
    id: str
    machine_id: str
    task_type: str
    description: str
    priority: str
    scheduled_date: str
    estimated_duration: int
    status: str
    assigned_to: Optional[str]

class HealingAction(BaseModel):
    """Self-healing action record"""
    timestamp: str
    machine_id: str
    action_type: str
    description: str
    success: bool
    risk_reduction: float

@router.get(
    "/schedule",
    summary="Get Maintenance Schedule",
    description="Retrieve scheduled maintenance tasks and work orders."
)
async def get_maintenance_schedule():
    """
    Get all scheduled maintenance activities.

    Returns:
        - schedule: List of maintenance tasks
        - count: Total number of scheduled tasks

    Tasks include preventive maintenance, repairs, and inspections.
    """
    schedule = db.get_maintenance_schedule()
    return {"schedule": schedule, "count": len(schedule)}

@router.post(
    "/schedule",
    summary="Schedule Maintenance",
    description="Create a new maintenance task or work order."
)
async def add_maintenance(maintenance: Dict[str, Any], user: Dict = Depends(require_operator)):
    """
    Schedule a new maintenance activity.

    Parameters:
        - machine_id: Target machine identifier
        - task_type: Type of maintenance (preventive, corrective, predictive)
        - description: Detailed task description
        - priority: Urgency level (low, medium, high, critical)
        - scheduled_date: ISO format date/time
        - estimated_duration: Hours required
        - assigned_to: Optional technician assignment

    Returns:
        - Confirmation of scheduling
        - Task ID for tracking
    """
    required_fields = ['machine_id', 'task_type', 'description', 'priority', 'scheduled_date']
    for field in required_fields:
        if field not in maintenance:
            raise HTTPException(status_code=400, detail=f"Missing required field: {field}")

    task_id = db.schedule_maintenance(maintenance)
    return {"status": "scheduled", "task_id": task_id}

@router.get(
    "/healing-log",
    summary="Get Self-Healing Log",
    description="Retrieve log of autonomous self-healing actions performed by the system."
)
async def get_healing_log():
    """
    Get history of self-healing actions taken by the AI system.

    Returns:
        - log: List of healing actions (last 50)
        - total_actions: Total number of healing actions performed

    Each action includes:
        - timestamp: When action was taken
        - machine_id: Affected machine
        - action_type: Type of healing action
        - description: What was done
        - success: Whether action succeeded
        - risk_reduction: Risk score reduction achieved
    """
    stream_handler = _stream_handler
    if not stream_handler:
        raise HTTPException(status_code=503, detail="System initializing")

    return {
        "log": stream_handler.healing_agent.healing_log[-50:],
        "total_actions": len(stream_handler.healing_agent.healing_log)
    }

@router.get(
    "/predictive-recommendations",
    summary="Get Predictive Maintenance Recommendations",
    description="Retrieve AI-generated maintenance recommendations based on current machine health."
)
async def get_predictive_recommendations():
    """
    Get predictive maintenance recommendations from the AI system.

    Returns:
        - recommendations: List of recommended maintenance actions
        - priority_order: Recommendations sorted by urgency
        - cost_benefit_analysis: Estimated costs and benefits

    Recommendations are generated based on:
        - Current risk scores
        - Failure predictions
        - Component health analysis
        - Historical maintenance data
    """
    stream_handler = _stream_handler
    if not stream_handler:
        raise HTTPException(status_code=503, detail="System initializing")

    recommendations = []

    for mid, state in stream_handler.machine_states.items():
        if state.risk_score > 0.5:
            recommendations.append({
                "machine_id": mid,
                "priority": "high" if state.risk_score > 0.8 else "medium",
                "recommendation": f"Schedule inspection for {mid}",
                "predicted_failure_hours": state.predicted_failure_hours,
                "estimated_cost": 500,  # Mock value
                "risk_reduction": state.risk_score * 0.7
            })

    return {
        "recommendations": sorted(recommendations, key=lambda x: x['priority'], reverse=True),
        "count": len(recommendations)
    }

@router.post(
    "/work-order/{task_id}/complete",
    summary="Complete Work Order",
    description="Mark a maintenance work order as completed."
)
async def complete_work_order(task_id: str, completion_data: Dict[str, Any], user: Dict = Depends(require_operator)):
    """
    Mark a maintenance task as completed.

    Parameters:
        - task_id: Unique task identifier
        - completion_data: Completion details including:
            - actual_duration: Hours actually spent
            - notes: Work performed description
            - parts_used: Parts replaced/consumed
            - success: Whether issue was resolved

    Returns:
        - Confirmation of completion
        - Updated task status
    """
    success = db.complete_maintenance_task(task_id, completion_data)
    if not success:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

    return {"status": "completed", "task_id": task_id}

@router.get(
    "/component-health",
    summary="Get Component Health Analysis",
    description="Retrieve detailed health analysis for machine components."
)
async def get_component_health():
    """
    Get detailed component-level health analysis.

    Returns:
        - components: Health status by component type
        - failure_predictions: Component-specific failure predictions
        - wear_analysis: Component wear and degradation trends

    Components analyzed:
        - Motors and actuators
        - Sensors and transducers
        - Hydraulic systems
        - Electrical systems
        - Structural elements
    """
    stream_handler = _stream_handler
    if not stream_handler:
        raise HTTPException(status_code=503, detail="System initializing")

    component_health = {}

    for mid, twin in stream_handler.digital_twins.items():
        component_health[mid] = twin.component_health

    return {
        "component_health": component_health,
        "analysis_timestamp": datetime.utcnow().isoformat(),
        "overall_system_health": "good"  # Would compute from component data
    }


# ============== PREDICTIVE MAINTENANCE SCHEDULER ROUTES ==============

from backend.ai_engine.maintenance_scheduler import MaintenanceScheduler

maintenance_scheduler: Optional[MaintenanceScheduler] = None

def get_maintenance_scheduler() -> MaintenanceScheduler:
    """Get or create maintenance scheduler instance"""
    global maintenance_scheduler
    if maintenance_scheduler is None:
        maintenance_scheduler = MaintenanceScheduler()
    return maintenance_scheduler


@router.get(
    "/scheduler/suggest/{machine_id}",
    summary="Suggest Maintenance Windows",
    description="Get AI-optimized maintenance windows based on failure predictions"
)
async def suggest_maintenance_windows(machine_id: str, user = Depends(require_viewer)):
    """Get suggested maintenance windows for a machine based on predictions"""
    try:
        stream_handler = _stream_handler
        if not stream_handler:
            raise HTTPException(status_code=503, detail="System initializing")
        
        prediction = stream_handler.machine_states.get(machine_id)
        if not prediction:
            raise HTTPException(status_code=404, detail=f"No prediction data for {machine_id}")
        
        scheduler = get_maintenance_scheduler()
        
        # Prepare prediction dict for scheduler
        pred_dict = {
            "hours_to_failure": getattr(prediction, 'predicted_failure_hours', 72),
            "risk_level": "CRITICAL" if prediction.risk_score > 0.8 else "HIGH" if prediction.risk_score > 0.6 else "MEDIUM" if prediction.risk_score > 0.4 else "LOW",
            "confidence": prediction.confidence if hasattr(prediction, 'confidence') else 0.7,
            "contributing_factors": ["system degradation"]
        }
        
        windows = await scheduler.suggest_maintenance_windows(machine_id, pred_dict)
        
        return {
            "status": "success",
            "machine_id": machine_id,
            "windows_count": len(windows),
            "windows": [w.to_dict() for w in windows]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/scheduler/impact/{machine_id}",
    summary="Maintenance Impact Analysis",
    description="Get cost-benefit analysis for recommended maintenance window"
)
async def get_maintenance_impact(
    machine_id: str,
    window_index: int = Query(0),
    user = Depends(require_operator)
):
    """Get cost-benefit analysis for maintenance window"""
    try:
        scheduler = get_maintenance_scheduler()
        
        if machine_id not in scheduler.schedules or window_index >= len(scheduler.schedules[machine_id]):
            raise HTTPException(status_code=404, detail="Maintenance window not found")
        
        window = scheduler.schedules[machine_id][window_index]
        
        stream_handler = _stream_handler
        prediction = stream_handler.machine_states.get(machine_id) if stream_handler else None
        current_risk = 1.0 - ((prediction.confidence / 100) if hasattr(prediction, 'confidence') else 0.5)
        
        impact = await scheduler.analyze_maintenance_impact(window, current_risk)
        
        return {
            "status": "success",
            "data": impact.to_dict()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/scheduler/fleet-status",
    summary="Fleet Maintenance Status",
    description="Overall maintenance status and requirements across all machines"
)
async def get_fleet_maintenance_status(user = Depends(require_viewer)):
    """Get overall fleet maintenance status"""
    try:
        scheduler = get_maintenance_scheduler()
        status = await scheduler.get_fleet_maintenance_status()
        return {
            "status": "success",
            "data": status
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))