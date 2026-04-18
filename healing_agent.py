import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import uuid
from backend.models.alert import Alert, AlertSeverity, HealingAction
from backend.models.machine import MachineState
from backend.utils.logger import ai_logger
from backend.utils.config import config

class HealingAgent:
    """
    Autonomous Self-Healing Agent.
    
    Implements healing protocols for each failure type:
    1. Thermal Management Protocol
    2. Vibration Dampening Protocol
    3. Load Balancing Protocol
    4. Electrical Protection Protocol
    5. Preventive Shutdown Protocol
    """
    
    # Healing playbooks
    HEALING_PLAYBOOKS = {
        "overheating": {
            "actions": [
                {"type": "reduce_load", "description": "Reduce operational load by 20%", "delay": 0},
                {"type": "increase_cooling", "description": "Activate emergency cooling system", "delay": 5},
                {"type": "alert_maintenance", "description": "Notify maintenance team", "delay": 10},
                {"type": "controlled_slowdown", "description": "Gradually reduce RPM to 60%", "delay": 30}
            ],
            "recovery_threshold": 65.0,  # Temperature that indicates recovery
            "escalation_threshold": 90.0,
            "cooldown_minutes": 30
        },
        "bearing_fault": {
            "actions": [
                {"type": "lubrication_check", "description": "Trigger lubrication system", "delay": 0},
                {"type": "vibration_monitoring", "description": "Increase monitoring frequency to 10Hz", "delay": 5},
                {"type": "load_reduction", "description": "Reduce bearing load by 15%", "delay": 10},
                {"type": "schedule_inspection", "description": "Schedule bearing inspection within 24h", "delay": 15}
            ],
            "cooldown_minutes": 60
        },
        "electrical_fault": {
            "actions": [
                {"type": "current_limit", "description": "Engage current limiter", "delay": 0},
                {"type": "power_cycle_check", "description": "Check power supply stability", "delay": 5},
                {"type": "isolate_phase", "description": "Isolate affected electrical phase", "delay": 10},
                {"type": "emergency_stop", "description": "Initiate controlled emergency stop", "delay": 20}
            ],
            "cooldown_minutes": 120
        },
        "cavitation": {
            "actions": [
                {"type": "flow_adjustment", "description": "Adjust flow rate by +15%", "delay": 0},
                {"type": "pressure_check", "description": "Check inlet pressure sensors", "delay": 5},
                {"type": "priming_cycle", "description": "Run priming cycle", "delay": 15}
            ],
            "cooldown_minutes": 45
        },
        "general_anomaly": {
            "actions": [
                {"type": "enhanced_monitoring", "description": "Switch to high-frequency monitoring", "delay": 0},
                {"type": "parameter_adjustment", "description": "Adjust operating parameters to safe range", "delay": 10},
                {"type": "alert_operator", "description": "Send alert to on-duty operator", "delay": 15}
            ],
            "cooldown_minutes": 20
        }
    }
    
    def __init__(self):
        self.active_healings: Dict[str, Dict] = {}
        self.healing_history: Dict[str, List] = {}
        self.cooldown_tracker: Dict[str, datetime] = {}
    
    async def execute_healing(
        self,
        alert: Alert,
        machine_id: str,
        failure_signature: Optional[str],
        risk_score: float
    ) -> List[HealingAction]:
        """
        Execute autonomous healing protocol for detected failure.
        Returns list of healing actions taken.
        """
        
        if not config.AUTO_HEALING_ENABLED:
            return []
        
        # Check cooldown
        cooldown_key = f"{machine_id}_{failure_signature or 'general'}"
        if self._in_cooldown(cooldown_key):
            ai_logger.info(f"Healing in cooldown for {machine_id}")
            return []
        
        # Don't over-heal low risk situations
        if risk_score < 35 and alert.severity not in [AlertSeverity.HIGH, AlertSeverity.CRITICAL]:
            return []
        
        # Select playbook
        playbook_key = failure_signature or "general_anomaly"
        if playbook_key not in self.HEALING_PLAYBOOKS:
            playbook_key = "general_anomaly"
        
        playbook = self.HEALING_PLAYBOOKS[playbook_key]
        
        healing_actions = []
        
        # Execute actions from playbook
        for action_spec in playbook["actions"]:
            # In real system, these would control actual machine actuators
            # Here we simulate the action execution
            action = await self._execute_action(
                machine_id=machine_id,
                action_type=action_spec["type"],
                description=action_spec["description"],
                risk_score=risk_score
            )
            healing_actions.append(action)
            
            # Small delay between actions
            await asyncio.sleep(0.1)
        
        # Set cooldown
        self._set_cooldown(cooldown_key, playbook.get("cooldown_minutes", 30))
        
        # Record healing history
        if machine_id not in self.healing_history:
            self.healing_history[machine_id] = []
        
        self.healing_history[machine_id].append({
            "timestamp": datetime.utcnow().isoformat(),
            "failure_type": failure_signature,
            "risk_score": risk_score,
            "actions_taken": len(healing_actions),
            "alert_id": alert.alert_id
        })
        
        ai_logger.info(
            f"🔧 Self-healing executed for {machine_id}: "
            f"{len(healing_actions)} actions, protocol: {playbook_key}"
        )
        
        return healing_actions
    
    async def _execute_action(
        self,
        machine_id: str,
        action_type: str,
        description: str,
        risk_score: float
    ) -> HealingAction:
        """Execute a single healing action"""
        action = HealingAction(
            action_type=action_type,
            description=description
        )
        
        try:
            # Simulate action execution
            # In production: integrate with SCADA/PLC systems
            action_handlers = {
                "reduce_load": self._reduce_load,
                "increase_cooling": self._increase_cooling,
                "alert_maintenance": self._alert_maintenance,
                "controlled_slowdown": self._controlled_slowdown,
                "lubrication_check": self._lubrication_check,
                "current_limit": self._current_limit,
                "flow_adjustment": self._flow_adjustment,
                "enhanced_monitoring": self._enhanced_monitoring,
                "emergency_stop": self._emergency_stop,
                "parameter_adjustment": self._parameter_adjustment,
                "alert_operator": self._alert_operator,
                "schedule_inspection": self._schedule_inspection,
                "priming_cycle": self._priming_cycle,
                "vibration_monitoring": self._vibration_monitoring,
                "load_reduction": self._load_reduction,
                "power_cycle_check": self._power_cycle_check,
                "isolate_phase": self._isolate_phase,
                "pressure_check": self._pressure_check
            }
            
            handler = action_handlers.get(action_type, self._generic_action)
            result = await handler(machine_id, risk_score)
            
            action.success = True
            action.result_message = result
            
        except Exception as e:
            action.success = False
            action.result_message = f"Action failed: {str(e)}"
            ai_logger.error(f"Healing action {action_type} failed: {e}")
        
        return action
    
    def _in_cooldown(self, key: str) -> bool:
        if key not in self.cooldown_tracker:
            return False
        return datetime.utcnow() < self.cooldown_tracker[key]
    
    def _set_cooldown(self, key: str, minutes: int):
        self.cooldown_tracker[key] = datetime.utcnow() + timedelta(minutes=minutes)
    
    # Action handlers (simulate real machine control)
    async def _reduce_load(self, machine_id: str, risk: float) -> str:
        reduction = min(30, int(risk / 3))
        return f"Load reduced by {reduction}% on {machine_id}"
    
    async def _increase_cooling(self, machine_id: str, risk: float) -> str:
        return f"Emergency cooling activated on {machine_id}, coolant flow +40%"
    
    async def _alert_maintenance(self, machine_id: str, risk: float) -> str:
        return f"Maintenance team notified for {machine_id} via SMS/Email"
    
    async def _controlled_slowdown(self, machine_id: str, risk: float) -> str:
        target_rpm_pct = max(50, 100 - int(risk * 0.5))
        return f"RPM reduced to {target_rpm_pct}% on {machine_id}"
    
    async def _lubrication_check(self, machine_id: str, risk: float) -> str:
        return f"Lubrication cycle triggered on {machine_id}"
    
    async def _current_limit(self, machine_id: str, risk: float) -> str:
        limit = max(60, 100 - int(risk * 0.4))
        return f"Current limiter set to {limit}% on {machine_id}"
    
    async def _flow_adjustment(self, machine_id: str, risk: float) -> str:
        return f"Flow rate adjusted +15% to prevent cavitation on {machine_id}"
    
    async def _enhanced_monitoring(self, machine_id: str, risk: float) -> str:
        return f"Monitoring frequency increased to 10Hz on {machine_id}"
    
    async def _emergency_stop(self, machine_id: str, risk: float) -> str:
        return f"Controlled emergency stop initiated on {machine_id}"
    
    async def _parameter_adjustment(self, machine_id: str, risk: float) -> str:
        return f"Operating parameters adjusted to safe envelope on {machine_id}"
    
    async def _alert_operator(self, machine_id: str, risk: float) -> str:
        return f"On-duty operator alerted for {machine_id}"
    
    async def _schedule_inspection(self, machine_id: str, risk: float) -> str:
        return f"Inspection scheduled within 24h for {machine_id}"
    
    async def _priming_cycle(self, machine_id: str, risk: float) -> str:
        return f"Priming cycle initiated on {machine_id}"
    
    async def _vibration_monitoring(self, machine_id: str, risk: float) -> str:
        return f"Vibration sensor sampling rate increased on {machine_id}"
    
    async def _load_reduction(self, machine_id: str, risk: float) -> str:
        return f"Bearing load reduced 15% on {machine_id}"
    
    async def _power_cycle_check(self, machine_id: str, risk: float) -> str:
        return f"Power supply stability check completed on {machine_id}"
    
    async def _isolate_phase(self, machine_id: str, risk: float) -> str:
        return f"Affected electrical phase isolated on {machine_id}"
    
    async def _pressure_check(self, machine_id: str, risk: float) -> str:
        return f"Inlet pressure verified on {machine_id}"
    
    async def _generic_action(self, machine_id: str, risk: float) -> str:
        return f"Generic healing action applied on {machine_id}"
    
    def generate_explanation(
        self,
        alert: Alert,
        detection_result: Dict,
        risk_result: Dict,
        healing_actions: List[HealingAction]
    ) -> str:
        """
        Generate human-readable explanation of the alert and healing actions.
        This is the 'Conscious Layer' narrative.
        """
        severity = alert.severity.value.upper()
        machine = alert.machine_id
        anomaly_type = detection_result.get("anomaly_type", "unknown")
        failure_sig = detection_result.get("failure_signature")
        risk = risk_result.get("risk_score", 0)
        trend = risk_result.get("trend", "stable")
        
        # Build explanation
        parts = []
        
        # Opening
        parts.append(
            f"🚨 {severity} ALERT on {machine}: Detected {anomaly_type.value if hasattr(anomaly_type, 'value') else anomaly_type} anomaly "
            f"with {risk:.0f}% risk score."
        )
        
        # Sensor details
        z_scores = detection_result.get("z_scores", {})
        high_z_sensors = [(s, z) for s, z in z_scores.items() if abs(z) > 2.0]
        if high_z_sensors:
            sensor_desc = ", ".join([
                f"{s} ({'+' if z > 0 else ''}{z:.1f}σ)"
                for s, z in sorted(high_z_sensors, key=lambda x: abs(x[1]), reverse=True)
            ])
            parts.append(f"Sensors deviating from baseline: {sensor_desc}.")
        
        # Failure signature
        if failure_sig:
            signature_explanations = {
                "bearing_fault": "Pattern matches bearing degradation — elevated vibration with temperature rise indicates mechanical wear.",
                "overheating": "Thermal runaway pattern detected — excessive current draw is generating heat beyond safe limits.",
                "cavitation": "Cavitation signature detected — fluid dynamics disruption causing resonant vibration in the pump.",
                "electrical_fault": "Electrical fault pattern — current spike with RPM instability suggests power delivery issue."
            }
            parts.append(
                signature_explanations.get(failure_sig, f"Matches '{failure_sig}' failure pattern from historical database.")
            )
        
        # Trend
        trend_desc = {
            "degrading": "⬆️ Conditions are WORSENING — immediate attention required.",
            "stable": "➡️ Conditions are stable — continued monitoring advised.",
            "improving": "⬇️ Conditions are improving — healing protocols appear effective."
        }
        parts.append(trend_desc.get(trend, ""))
        
        # RUL
        rul = risk_result.get("estimated_failure_hours")
        if rul:
            if rul < 24:
                parts.append(f"⏰ CRITICAL: Estimated {rul:.0f} hours to potential failure. IMMEDIATE ACTION REQUIRED.")
            elif rul < 72:
                parts.append(f"⏰ WARNING: Estimated {rul:.0f} hours ({rul/24:.1f} days) to potential failure.")
            else:
                parts.append(f"⏱️ Estimated {rul/24:.0f} days until maintenance required.")
        
        # Healing actions
        if healing_actions:
            successful = [a for a in healing_actions if a.success]
            parts.append(
                f"🔧 AUTO-HEALING: {len(successful)}/{len(healing_actions)} recovery actions executed automatically."
            )
            if successful:
                parts.append(f"Primary action: {successful[0].description}")
        
        return " ".join(p for p in parts if p)
    
    def get_recommendation(
        self,
        failure_signature: Optional[str],
        risk_score: float,
        severity: AlertSeverity
    ) -> str:
        """Generate specific maintenance recommendation"""
        recommendations = {
            "bearing_fault": (
                "Schedule bearing inspection and lubrication service within 24-48 hours. "
                "Check alignment and balance. Replace bearing if vibration >8mm/s RMS."
            ),
            "overheating": (
                "Verify cooling system efficiency. Check coolant levels and flow rate. "
                "Inspect heat exchanger for fouling. Reduce load until temperature normalizes."
            ),
            "electrical_fault": (
                "Inspect power supply and motor windings. Check for loose connections, "
                "insulation degradation, or phase imbalance. Test motor impedance."
            ),
            "cavitation": (
                "Check inlet conditions and suction head. Verify valve positions. "
                "Inspect impeller for erosion damage. Adjust pump curve operating point."
            )
        }
        
        base_rec = recommendations.get(failure_signature, 
            "Perform general inspection. Check all mechanical connections and lubrication points."
        )
        
        if severity == AlertSeverity.CRITICAL:
            return f"🚨 URGENT: {base_rec} Stop machine if risk increases above 90%."
        elif severity == AlertSeverity.HIGH:
            return f"⚠️ HIGH PRIORITY: {base_rec}"
        else:
            return f"📋 RECOMMENDED: {base_rec}"