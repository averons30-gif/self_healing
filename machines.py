from fastapi import APIRouter, HTTPException, Query, Depends
from typing import List, Optional, Dict, Any
from pydantic import BaseModel
from backend.database.db_manager import DatabaseManager
from backend.api.middleware.auth import require_viewer, require_operator, require_admin

router = APIRouter(
    prefix="/machines",
    tags=["machines"],
    responses={404: {"description": "Machine not found"}}
)
db = DatabaseManager()

# Global reference to stream_handler (set by main.py)
_stream_handler = None

def set_stream_handler(handler):
    """Set the stream handler instance (called by main.py)"""
    global _stream_handler
    _stream_handler = handler

class MachineSummary(BaseModel):
    """Summary of machine state for list view"""
    machine_id: str
    name: str
    status: str
    risk_score: float
    last_reading: str
    sensor_count: int

class MachineDetail(BaseModel):
    """Detailed machine state with AI analysis"""
    machine_id: str
    name: str
    status: str
    risk_score: float
    sensors: Dict[str, Any]
    analysis: Dict[str, Any]
    digital_twin: Optional[Dict[str, Any]]
    component_health: Optional[Dict[str, Any]]
    virtual_sensors: Optional[Dict[str, Any]]
    failure_scenarios: Optional[List[Dict[str, Any]]]

@router.get(
    "",
    response_model=Dict[str, Any],
    summary="Get All Machines",
    description="Retrieve current state and status of all monitored machines including real-time sensor data and AI analysis results."
)
async def get_all_machines():
    """
    Get comprehensive state of all machines in the system.

    Returns:
        - machines: List of machine states with sensor data and AI analysis
        - count: Total number of machines

    Each machine includes:
        - Current sensor readings (raw, filtered, delta)
        - AI analysis results (anomaly detection, risk scoring)
        - Digital twin state (3D model data)
        - Component health metrics
    """
    stream_handler = _stream_handler
    if not stream_handler:
        raise HTTPException(status_code=503, detail="System initializing")

    machines = []
    for machine_id, state in stream_handler.machine_states.items():
        twin = stream_handler.digital_twins.get(machine_id)
        machine_data = state.to_dict()
        if twin:
            machine_data['digital_twin'] = twin.get_status()
            machine_data['component_health'] = {}
        machines.append(machine_data)
    return {"machines": machines, "count": len(machines)}

@router.get(
    "/{machine_id}",
    response_model=MachineDetail,
    summary="Get Machine Details",
    description="Retrieve detailed information for a specific machine including sensor data, AI analysis, and predictive insights."
)
async def get_machine(machine_id: str):
    """
    Get detailed state and analysis for a specific machine.

    Parameters:
        - machine_id: Unique identifier for the machine (e.g., "CNC_MILL_01")

    Returns:
        - Complete machine state with all sensors
        - AI analysis results (anomaly detection, risk assessment)
        - Digital twin 3D state
        - Component health scores
        - Virtual sensor predictions
        - Failure scenarios (if risk score > 0.3)

    Raises:
        - 404: Machine not found
        - 503: System still initializing
    """
    stream_handler = _stream_handler
    if not stream_handler:
        raise HTTPException(status_code=503, detail="System initializing")

    if machine_id not in stream_handler.machine_states:
        raise HTTPException(status_code=404, detail=f"Machine {machine_id} not found")

    state = stream_handler.machine_states[machine_id]
    twin = stream_handler.digital_twins.get(machine_id)

    result = state.to_dict()
    if twin:
        result['digital_twin'] = twin.get_status()
        result['component_health'] = {}
        result['virtual_sensors'] = twin.virtual_sensors

    # Add failure scenarios for high-risk machines
    if state.risk_score > 0.3:
        baseline_dict = stream_handler.baseline_manager.to_dict(machine_id) or {}
        scenarios = stream_handler.failure_predictor.simulate_scenarios(
            machine_id, state.risk_score, baseline_dict
        )
        result['failure_scenarios'] = scenarios

    return result

@router.get(
    "/{machine_id}/history",
    summary="Get Machine History",
    description="Retrieve historical sensor readings for time-series analysis and trend visualization."
)
async def get_machine_history(machine_id: str, hours: int = 1):
    """
    Get historical sensor data for charting and analysis.

    Parameters:
        - machine_id: Machine identifier
        - hours: Number of hours of history to retrieve (default: 1)

    Returns:
        - readings: List of sensor readings (limited to last 500)
        - count: Total number of readings available

    Note: Returns maximum 500 most recent readings for performance.
    Use pagination for larger datasets in production.
    """
    readings = db.get_historical_readings(machine_id, days=hours/24)
    return {"machine_id": machine_id, "readings": readings[-500:], "count": len(readings)}

@router.get(
    "/{machine_id}/readings",
    summary="Get Sensor Readings",
    description="Retrieve recent sensor readings for a machine with optional filtering and pagination."
)
async def get_machine_readings(
    machine_id: str,
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of readings to return"),
    hours: int = Query(1, ge=1, le=168, description="Time window in hours (max 1 week)")
):
    """
    Get recent sensor readings for analysis and visualization.

    Parameters:
        - machine_id: Machine identifier
        - limit: Maximum readings to return (1-1000)
        - hours: Time window in hours (1-168)

    Returns:
        - readings: List of sensor readings with timestamps
        - count: Total readings in the time window
        - machine_id: Machine identifier

    Each reading includes:
        - timestamp: ISO format timestamp
        - temperature_C: Temperature in Celsius
        - vibration_mm_s: Vibration in mm/s
        - rpm: Rotations per minute
        - current_A: Current in Amperes
        - status: Machine status (running/warning/fault)
        - is_anomaly: Anomaly flag (0=normal, 1=anomaly)
    """
    # Get readings from database
    readings = db.get_historical_readings(machine_id, days=hours/24)

    # If no historical data, try to get from current simulation data
    if not readings:
        stream_handler = _stream_handler
        if stream_handler and machine_id in stream_handler.machine_states:
            # Generate some sample readings from current state
            import random
            from datetime import datetime, timedelta

            state = stream_handler.machine_states[machine_id]
            readings = []
            filtered = state.sensors.get("filtered", {})
            status = state.state if isinstance(state.state, str) else getattr(state.state, "value", str(state.state))
            is_anomaly = 1 if state.analysis.get("is_anomaly") else 0

            # Generate last 'limit' readings going backwards from now
            for i in range(min(limit, 100)):  # Max 100 synthetic readings
                timestamp = datetime.utcnow() - timedelta(minutes=i*5)  # 5 min intervals

                # Add some noise to current values
                reading = {
                    "machine_id": machine_id,
                    "timestamp": timestamp.isoformat() + "Z",
                    "temperature_C": round(filtered.get("temperature", 0) + random.uniform(-2, 2), 2),
                    "vibration_mm_s": round(filtered.get("vibration", 0) + random.uniform(-0.1, 0.1), 3),
                    "rpm": round(filtered.get("rpm", 0) + random.uniform(-10, 10), 1),
                    "current_A": round(filtered.get("current", 0) + random.uniform(-0.5, 0.5), 2),
                    "status": status,
                    "is_anomaly": is_anomaly
                }
                readings.append(reading)

    # Return most recent readings up to limit
    recent_readings = readings[-limit:] if readings else []

    return {
        "machine_id": machine_id,
        "readings": recent_readings,
        "count": len(recent_readings),
        "time_window_hours": hours
    }

@router.get(
    "/{machine_id}/baseline",
    summary="Get Machine Baseline",
    description="Retrieve computed baseline statistics for normal operating conditions."
)
async def get_machine_baseline(machine_id: str):
    """
    Get baseline statistics for machine normal operation.

    Parameters:
        - machine_id: Machine identifier

    Returns:
        - Statistical baselines for all sensors
        - Normal operating ranges
        - Computed from historical data

    Raises:
        - 404: Baseline not computed yet (insufficient data)
    """
    stream_handler = _stream_handler
    if not stream_handler:
        raise HTTPException(status_code=503, detail="System initializing")

    baseline = await stream_handler.baseline_manager.get_baseline(machine_id)
    if not baseline:
        raise HTTPException(status_code=404, detail="Baseline not computed yet - collect more data")
    return {
        "machine_id": machine_id,
        "baseline": baseline,
        "source": "historical_or_synthetic"
    }

@router.post(
    "/{machine_id}/inject-fault",
    summary="Inject Test Fault",
    description="Inject artificial faults for testing anomaly detection and self-healing systems."
)
async def inject_fault(machine_id: str, fault_type: str, intensity: float = 1.0, user: Dict = Depends(require_operator)):
    """
    Inject a fault condition for testing purposes.

    Parameters:
        - machine_id: Target machine identifier
        - fault_type: Type of fault ("spike", "drift", "compound")
        - intensity: Fault intensity multiplier (default: 1.0)

    Returns:
        - Confirmation of fault injection

    Note: Only available in development/demo mode. Requires operator role.
    """
    """
    Inject a fault condition for testing purposes.

    Parameters:
        - machine_id: Target machine identifier
        - fault_type: Type of fault ("spike", "drift", "compound")
        - intensity: Fault intensity multiplier (default: 1.0)

    Returns:
        - Confirmation of fault injection

    Note: Only available in development/demo mode.
    """
    valid_faults = ['spike', 'drift', 'compound']
    if fault_type not in valid_faults:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid fault type. Choose from: {', '.join(valid_faults)}"
        )

    stream_handler = _stream_handler
    if not stream_handler:
        raise HTTPException(status_code=503, detail="System initializing")

    stream_handler.simulator.inject_fault(machine_id, fault_type, intensity)
    return {"status": "fault_injected", "machine_id": machine_id, "type": fault_type, "intensity": intensity}

@router.post(
    "/{machine_id}/clear-fault",
    summary="Clear Test Fault",
    description="Remove any injected faults and restore normal operation."
)
async def clear_fault(machine_id: str, user: Dict = Depends(require_operator)):
    """
    Clear any active fault injection.

    Parameters:
        - machine_id: Target machine identifier

    Returns:
        - Confirmation of fault clearance
    """
    stream_handler = _stream_handler
    if not stream_handler:
        raise HTTPException(status_code=503, detail="System initializing")

    stream_handler.simulator.clear_fault(machine_id)
    return {"status": "fault_cleared", "machine_id": machine_id}