"""
predictions.py
═══════════════════════════════════════════════════════════════════
API routes for predictive maintenance and analytics.
═══════════════════════════════════════════════════════════════════
"""

from fastapi import APIRouter, HTTPException, Depends, Query
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from backend.database.db_manager import DatabaseManager
from backend.ai_engine.predictive_model import PredictiveMaintenanceModel
from backend.api.middleware.auth import require_viewer
from backend.utils.logger import create_logger

pred_logger = create_logger("digital_twin.api.predictions", "predictions.log")

router = APIRouter(
    prefix="/predictions",
    tags=["predictions"],
    responses={404: {"description": "Not found"}}
)

db = DatabaseManager()
model = PredictiveMaintenanceModel()


@router.get("/machine/{machine_id}")
async def predict_machine_failure(
    machine_id: str,
    user: Dict = Depends(require_viewer)
) -> Dict[str, Any]:
    """
    Predict if a machine will fail and provide failure timeline.
    
    Returns prediction with confidence score and contributing factors.
    """
    try:
        # Get recent sensor readings (last 1 hour = ~100 readings)
        readings = await db.get_recent_readings(machine_id, limit=100, hours=1)
        
        if not readings:
            raise HTTPException(status_code=404, detail="No readings found for machine")

        # Get anomaly scores from recent readings
        anomaly_scores = [
            float(r.get("analysis_json", {}).get("anomaly_score", 0.0))
            for r in readings
            if r.get("analysis_json")
        ]

        # Get recent alerts
        alerts = await db.get_machine_alerts(machine_id, limit=50)

        # Generate prediction
        prediction = model.predict_failure(
            machine_id=machine_id,
            recent_readings=readings,
            anomaly_scores=anomaly_scores,
            alert_history=alerts
        )

        pred_logger.info(f"📊 Prediction generated for {machine_id}")

        return {
            "machine_id": prediction.machine_id,
            "confidence": prediction.confidence,
            "risk_level": prediction.risk_level,
            "predicted_failure_time": (
                prediction.predicted_failure_time.isoformat()
                if prediction.predicted_failure_time
                else None
            ),
            "hours_to_failure": prediction.hours_to_failure,
            "contributing_factors": prediction.contributing_factors,
            "recommendation": prediction.recommendation,
            "generated_at": datetime.now().isoformat()
        }

    except HTTPException as e:
        raise e
    except Exception as e:
        pred_logger.error(f"Prediction error for {machine_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@router.get("/all")
async def predict_all_machines(
    user: Dict = Depends(require_viewer)
) -> Dict[str, Any]:
    """
    Get predictions for all machines with rankings.
    """
    try:
        machines = await db.get_all_machines()
        predictions = {}

        for machine in machines:
            machine_id = machine.get("machine_id")
            
            readings = await db.get_recent_readings(machine_id, limit=100, hours=1)
            if not readings:
                continue

            anomaly_scores = [
                float(r.get("analysis_json", {}).get("anomaly_score", 0.0))
                for r in readings
                if r.get("analysis_json")
            ]

            alerts = await db.get_machine_alerts(machine_id, limit=50)

            prediction = model.predict_failure(
                machine_id=machine_id,
                recent_readings=readings,
                anomaly_scores=anomaly_scores,
                alert_history=alerts
            )

            predictions[machine_id] = prediction

        # Get comparative rankings
        comparison = model.compare_predictions(predictions)

        pred_logger.info(f"📊 Generated predictions for {len(predictions)} machines")

        return {
            "predictions": {
                mid: {
                    "confidence": pred.confidence,
                    "risk_level": pred.risk_level,
                    "hours_to_failure": pred.hours_to_failure,
                    "contributing_factors": pred.contributing_factors,
                    "recommendation": pred.recommendation
                }
                for mid, pred in predictions.items()
            },
            "comparison": comparison,
            "generated_at": datetime.now().isoformat()
        }

    except Exception as e:
        pred_logger.error(f"All predictions error: {e}")
        raise HTTPException(status_code=500, detail=f"Predictions failed: {str(e)}")


@router.get("/compare")
async def compare_machines(
    machines: str = Query(..., description="Comma-separated machine IDs"),
    user: Dict = Depends(require_viewer)
) -> Dict[str, Any]:
    """
    Compare health status and predictions of multiple machines.
    
    Example: /compare?machines=CNC_MILL_01,CONVEYOR_02
    """
    try:
        machine_list = [m.strip() for m in machines.split(",")]
        
        comparison_data = {}

        for machine_id in machine_list:
            readings = await db.get_recent_readings(machine_id, limit=100, hours=1)
            if not readings:
                continue

            latest = readings[-1]
            sensors = latest.get("sensors_json", {})

            anomaly_scores = [
                float(r.get("analysis_json", {}).get("anomaly_score", 0.0))
                for r in readings
                if r.get("analysis_json")
            ]

            alerts = await db.get_machine_alerts(machine_id, limit=20)

            prediction = model.predict_failure(
                machine_id=machine_id,
                recent_readings=readings,
                anomaly_scores=anomaly_scores,
                alert_history=alerts
            )

            comparison_data[machine_id] = {
                "latest_sensors": sensors,
                "prediction": {
                    "confidence": prediction.confidence,
                    "risk_level": prediction.risk_level,
                    "hours_to_failure": prediction.hours_to_failure
                },
                "alert_count": len(alerts),
                "critical_alerts": sum(1 for a in alerts if a.get("risk_level") == "CRITICAL")
            }

        pred_logger.info(f"📊 Comparison generated for {len(machine_list)} machines")

        return {
            "comparison": comparison_data,
            "generated_at": datetime.now().isoformat()
        }

    except Exception as e:
        pred_logger.error(f"Comparison error: {e}")
        raise HTTPException(status_code=500, detail=f"Comparison failed: {str(e)}")


@router.get("/executive-summary")
async def generate_executive_summary(
    user: Dict = Depends(require_viewer)
) -> Dict[str, Any]:
    """
    Generate executive summary report with KPIs.
    """
    try:
        machines = await db.get_all_machines()
        predictions = {}
        all_alerts = []

        # Gather data for all machines
        for machine in machines:
            machine_id = machine.get("machine_id")
            
            readings = await db.get_recent_readings(machine_id, limit=100, hours=24)
            if not readings:
                continue

            anomaly_scores = [
                float(r.get("analysis_json", {}).get("anomaly_score", 0.0))
                for r in readings
                if r.get("analysis_json")
            ]

            alerts = await db.get_machine_alerts(machine_id, limit=100)
            all_alerts.extend(alerts)

            prediction = model.predict_failure(
                machine_id=machine_id,
                recent_readings=readings,
                anomaly_scores=anomaly_scores,
                alert_history=alerts
            )

            predictions[machine_id] = prediction

        # Calculate KPIs
        critical_machines = sum(1 for p in predictions.values() if p.risk_level == "CRITICAL")
        high_risk = sum(1 for p in predictions.values() if p.risk_level == "HIGH")
        avg_confidence = sum(p.confidence for p in predictions.values()) / max(1, len(predictions))
        critical_alerts = sum(1 for a in all_alerts if a.get("risk_level") == "CRITICAL")

        # Estimate system uptime
        system_uptime = ((len(machines) - critical_machines) / max(1, len(machines))) * 100

        pred_logger.info("📊 Executive summary generated")

        return {
            "summary": {
                "total_machines": len(machines),
                "critical_risk": critical_machines,
                "high_risk": high_risk,
                "estimated_uptime_percent": round(system_uptime, 2),
                "system_health_score": round(100 - avg_confidence, 2),
                "critical_alerts_24h": critical_alerts
            },
            "predictions": predictions,
            "key_actions": [
                f"Address {critical_machines} critical machines immediately" if critical_machines > 0 else None,
                f"Plan maintenance for {high_risk} high-risk machines" if high_risk > 0 else None,
                "Continue monitoring all systems" if critical_machines == 0 else None
            ],
            "generated_at": datetime.now().isoformat()
        }

    except Exception as e:
        pred_logger.error(f"Executive summary error: {e}")
        raise HTTPException(status_code=500, detail=f"Summary generation failed: {str(e)}")
