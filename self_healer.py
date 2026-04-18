"""
backend/ai/self_healer.py
═══════════════════════════════════════════════════════════════════
SelfHealer — Automated corrective action engine.

Evaluates current machine state and triggers healing actions
when risk thresholds are exceeded.

Healing actions:
  1. reduce_load           — throttle machine throughput
  2. adjust_parameters     — modify operating setpoints
  3. increase_cooling      — boost coolant / fan speed
  4. lubrication_cycle     — trigger oil circulation
  5. speed_reduction       — reduce RPM to safe range
  6. emergency_shutdown    — stop machine (last resort)
  7. maintenance_flag      — flag for human maintenance

Cooldown logic prevents repeated actions for the same condition.
═══════════════════════════════════════════════════════════════════
"""

from __future__     import annotations

import time
from typing         import Dict, List, Optional, Any
from datetime       import datetime, timezone

from backend.utils.logger   import ai_logger
from backend.utils.config   import config


# ── Healing action definitions ─────────────────────────────────────────────────
HEALING_ACTIONS = {
    "reduce_load": {
        "description":          "Reduce machine load by 20%",
        "triggers":             ["HIGH", "CRITICAL"],
        "degradation_reduction": 0.05,
        "cooldown_s":           120
    },
    "increase_cooling": {
        "description":          "Increase cooling system output",
        "triggers":             ["HIGH", "CRITICAL"],
        "degradation_reduction": 0.08,
        "cooldown_s":           90
    },
    "lubrication_cycle": {
        "description":          "Run automatic lubrication cycle",
        "triggers":             ["MEDIUM", "HIGH", "CRITICAL"],
        "degradation_reduction": 0.12,
        "cooldown_s":           300
    },
    "speed_reduction": {
        "description":          "Reduce operating speed by 15%",
        "triggers":             ["HIGH", "CRITICAL"],
        "degradation_reduction": 0.06,
        "cooldown_s":           150
    },
    "adjust_parameters": {
        "description":          "Adjust operating parameters to safe range",
        "triggers":             ["MEDIUM", "HIGH"],
        "degradation_reduction": 0.03,
        "cooldown_s":           60
    },
    "emergency_shutdown": {
        "description":          "Emergency machine shutdown",
        "triggers":             ["CRITICAL"],
        "degradation_reduction": 0.0,
        "cooldown_s":           600
    },
    "maintenance_flag": {
        "description":          "Flag machine for scheduled maintenance",
        "triggers":             ["HIGH", "CRITICAL"],
        "degradation_reduction": 0.0,
        "cooldown_s":           3600
    }
}

# ── Healing decision matrix ────────────────────────────────────────────────────
# Maps (risk_level, primary_symptom) → [action_priority_list]
HEALING_MATRIX: Dict[str, Dict[str, List[str]]] = {
    "CRITICAL": {
        "temperature":      ["increase_cooling", "reduce_load", "emergency_shutdown"],
        "vibration":        ["speed_reduction", "reduce_load", "emergency_shutdown"],
        "pressure":         ["reduce_load", "speed_reduction", "emergency_shutdown"],
        "oil_level":        ["lubrication_cycle", "emergency_shutdown"],
        "current":          ["reduce_load", "speed_reduction", "emergency_shutdown"],
        "general":          ["reduce_load", "emergency_shutdown", "maintenance_flag"]
    },
    "HIGH": {
        "temperature":      ["increase_cooling", "reduce_load"],
        "vibration":        ["speed_reduction", "lubrication_cycle"],
        "pressure":         ["reduce_load", "adjust_parameters"],
        "oil_level":        ["lubrication_cycle", "maintenance_flag"],
        "current":          ["reduce_load", "adjust_parameters"],
        "general":          ["adjust_parameters", "reduce_load", "maintenance_flag"]
    },
    "MEDIUM": {
        "general":          ["adjust_parameters", "lubrication_cycle"]
    }
}


class SelfHealer:
    """
    Automated self-healing engine for machine anomalies.

    Evaluates current conditions, selects appropriate healing actions,
    applies cooldowns to prevent action storms, and returns
    the action taken (or None if no action warranted).

    Usage:
        healer = SelfHealer("CNC_MILL_01")
        result = await healer.evaluate(
            sensors, risk_level, risk_score, machine_state, degradation
        )
    """

    def __init__(self, machine_id: str):
        """
        Args:
            machine_id: Machine identifier
        """
        self.machine_id         = machine_id
        self._action_history:   List[Dict] = []
        self._cooldowns:        Dict[str, float] = {}   # action → last_fired_ts
        self._total_actions:    int = 0
        self._enabled:          bool = True

        ai_logger.debug(f"SelfHealer created for: {machine_id}")

    # ══════════════════════════════════════════════════════════════════════════
    # MAIN EVALUATION
    # ══════════════════════════════════════════════════════════════════════════

    async def evaluate(
        self,
        sensors:        Dict[str, float],
        risk_level:     str,
        risk_score:     float,
        machine_state:  str,
        degradation:    float = 0.0
    ) -> Dict:
        """
        Evaluate current conditions and trigger healing if warranted.

        Args:
            sensors:        Current filtered sensor readings
            risk_level:     CRITICAL | HIGH | MEDIUM | LOW
            risk_score:     Numeric risk score (0-100)
            machine_state:  Current machine operational state
            degradation:    Current degradation factor (0-1)

        Returns:
            {
                "action_taken":         bool,
                "action_type":          str | None,
                "description":          str,
                "new_state":            str,
                "degradation_reduction": float,
                "parameters":           dict,
                "timestamp":            str
            }
        """
        NO_ACTION = {
            "action_taken":             False,
            "action_type":              None,
            "description":              "No healing required",
            "new_state":                machine_state,
            "degradation_reduction":    0.0,
            "parameters":               {},
            "timestamp":                datetime.now(timezone.utc).isoformat()
        }

        # ── Skip if healing disabled or low risk ───────────────────────────────
        if not self._enabled:
            return NO_ACTION

        if risk_level == "LOW":
            return NO_ACTION

        if risk_level == "MEDIUM" and risk_score < 30:
            return NO_ACTION

        # ── Identify primary symptom ───────────────────────────────────────────
        primary = self._identify_primary_symptom(sensors, risk_level)

        # ── Get candidate actions from decision matrix ─────────────────────────
        candidates = (
            HEALING_MATRIX.get(risk_level, {}).get(primary, []) or
            HEALING_MATRIX.get(risk_level, {}).get("general", [])
        )

        if not candidates:
            return NO_ACTION

        # ── Select first available action (respects cooldown) ─────────────────
        selected_action = None
        for action_type in candidates:
            if self._can_execute(action_type):
                selected_action = action_type
                break

        if not selected_action:
            ai_logger.debug(
                f"[{self.machine_id}] Healing: all actions on cooldown"
            )
            return NO_ACTION

        # ── Execute action ─────────────────────────────────────────────────────
        action_def      = HEALING_ACTIONS[selected_action]
        parameters      = self._build_parameters(selected_action, sensors, risk_score)
        new_state       = self._determine_post_healing_state(
            selected_action, machine_state
        )

        # Record cooldown
        self._cooldowns[selected_action]    = time.time()
        self._total_actions                 += 1

        result = {
            "action_taken":             True,
            "action_type":              selected_action,
            "description":              action_def["description"],
            "new_state":                new_state,
            "degradation_reduction":    action_def["degradation_reduction"],
            "parameters":               parameters,
            "timestamp":                datetime.now(timezone.utc).isoformat()
        }

        # Store in history
        self._action_history.append({
            **result,
            "risk_level":   risk_level,
            "risk_score":   risk_score,
            "primary":      primary
        })

        # Keep last 50 actions
        if len(self._action_history) > 50:
            self._action_history = self._action_history[-50:]

        ai_logger.info(
            f"🔧 [{self.machine_id}] Healing: {selected_action} "
            f"(risk={risk_level}, score={risk_score:.1f})"
        )

        return result

    # ══════════════════════════════════════════════════════════════════════════
    # HELPERS
    # ══════════════════════════════════════════════════════════════════════════

    def _identify_primary_symptom(
        self,
        sensors:    Dict[str, float],
        risk_level: str
    ) -> str:
        """
        Identify the primary concerning sensor to choose the right healing path.

        Uses threshold-based severity ranking.

        Args:
            sensors:    Current sensor readings
            risk_level: Current risk level

        Returns:
            Sensor key string or "general"
        """
        # Sensor severity scores (higher = more concerning)
        severity: Dict[str, float] = {}

        thresholds = {
            "temperature":      (80, 95),
            "vibration":        (6, 10),
            "pressure":         (200, 250),
            "current":          (40, 50),
            "oil_level":        (30, 15),    # inverted
        }

        for key, (warn, crit) in thresholds.items():
            value = sensors.get(key)
            if value is None:
                continue

            if key == "oil_level":
                if value < crit:
                    severity[key] = 2.0
                elif value < warn:
                    severity[key] = 1.0
            else:
                if value > crit:
                    severity[key] = 2.0
                elif value > warn:
                    severity[key] = 1.0

        if not severity:
            return "general"

        # Return the most severe sensor
        return max(severity, key=severity.__getitem__)

    def _can_execute(self, action_type: str) -> bool:
        """
        Check if an action is off cooldown.

        Args:
            action_type: Action identifier

        Returns:
            True if action can be executed
        """
        if action_type not in HEALING_ACTIONS:
            return False

        last_fired  = self._cooldowns.get(action_type, 0.0)
        cooldown_s  = HEALING_ACTIONS[action_type]["cooldown_s"]

        return (time.time() - last_fired) >= cooldown_s

    def _build_parameters(
        self,
        action_type:    str,
        sensors:        Dict[str, float],
        risk_score:     float
    ) -> Dict[str, Any]:
        """
        Build action-specific parameters based on current conditions.

        Args:
            action_type: Action to parameterize
            sensors:     Current sensor readings
            risk_score:  Current risk score

        Returns:
            Dict of action parameters
        """
        params: Dict[str, Any] = {
            "risk_score_at_trigger": round(risk_score, 2)
        }

        if action_type == "reduce_load":
            reduction = 0.15 + (risk_score / 100) * 0.25
            params["load_reduction_pct"]    = round(reduction * 100, 1)
            params["target_load"]           = round(
                max(20, sensors.get("load_percentage", 100) * (1 - reduction)),
                1
            )

        elif action_type == "increase_cooling":
            params["cooling_boost_pct"]     = 30 if risk_score > 75 else 15
            params["duration_minutes"]      = 10

        elif action_type == "speed_reduction":
            current_rpm = sensors.get("rpm", 1000)
            params["current_rpm"]           = current_rpm
            params["target_rpm"]            = round(current_rpm * 0.85, 0)
            params["reduction_pct"]         = 15

        elif action_type == "lubrication_cycle":
            params["cycle_duration_s"]      = 30
            params["oil_level_before"]      = sensors.get("oil_level", 50)

        elif action_type == "adjust_parameters":
            params["adjustment_type"]       = "conservative"
            params["temperature_setpoint"]  = 75
            params["pressure_setpoint"]     = sensors.get("pressure", 100) * 0.9

        elif action_type == "emergency_shutdown":
            params["reason"]                = f"Risk score {risk_score:.1f} exceeded limit"
            params["restart_clearance"]     = "MANUAL_REQUIRED"

        elif action_type == "maintenance_flag":
            params["priority"]              = "HIGH" if risk_score > 75 else "MEDIUM"
            params["recommended_actions"]   = ["inspect_bearings", "check_lubrication"]

        return params

    def _determine_post_healing_state(
        self,
        action_type:    str,
        current_state:  str
    ) -> str:
        """
        Determine machine state after healing action.

        Args:
            action_type:    Action taken
            current_state:  State before healing

        Returns:
            New machine state string
        """
        state_map = {
            "emergency_shutdown":   "offline",
            "maintenance_flag":     "maintenance",
            "reduce_load":          "degraded",
            "speed_reduction":      "degraded",
            "increase_cooling":     "healing",
            "lubrication_cycle":    "healing",
            "adjust_parameters":    "operational"
        }

        return state_map.get(action_type, current_state)

    # ══════════════════════════════════════════════════════════════════════════
    # CONTROLS
    # ══════════════════════════════════════════════════════════════════════════

    def enable(self) -> None:
        """Enable automatic healing."""
        self._enabled = True
        ai_logger.info(f"[{self.machine_id}] Self-healing ENABLED")

    def disable(self) -> None:
        """Disable automatic healing (manual mode)."""
        self._enabled = False
        ai_logger.info(f"[{self.machine_id}] Self-healing DISABLED")

    def clear_cooldowns(self) -> None:
        """Clear all cooldowns (allow all actions immediately)."""
        self._cooldowns.clear()
        ai_logger.debug(f"[{self.machine_id}] All cooldowns cleared")

    # ══════════════════════════════════════════════════════════════════════════
    # DIAGNOSTICS
    # ══════════════════════════════════════════════════════════════════════════

    def get_action_history(self, limit: int = 20) -> List[Dict]:
        """
        Get recent healing action history.

        Args:
            limit: Max entries to return

        Returns:
            List of action dicts (newest first)
        """
        return list(reversed(self._action_history[-limit:]))

    def get_cooldown_status(self) -> Dict[str, float]:
        """
        Get remaining cooldown seconds for each action.

        Returns:
            Dict of action_type → remaining_seconds (0 if ready)
        """
        now     = time.time()
        status  = {}

        for action_type, action_def in HEALING_ACTIONS.items():
            last_fired  = self._cooldowns.get(action_type, 0.0)
            elapsed     = now - last_fired
            remaining   = max(0.0, action_def["cooldown_s"] - elapsed)
            status[action_type] = round(remaining, 1)

        return status

    def get_stats(self) -> Dict:
        """
        Get healer statistics.

        Returns:
            Stats dict with counts and history summary
        """
        action_counts: Dict[str, int] = {}
        for entry in self._action_history:
            at = entry.get("action_type", "unknown")
            action_counts[at] = action_counts.get(at, 0) + 1

        return {
            "machine_id":       self.machine_id,
            "enabled":          self._enabled,
            "total_actions":    self._total_actions,
            "action_counts":    action_counts,
            "cooldowns":        self.get_cooldown_status()
        }

    def __repr__(self) -> str:
        return (
            f"SelfHealer("
            f"machine={self.machine_id!r}, "
            f"actions={self._total_actions}, "
            f"enabled={self._enabled})"
        )