"""
backend/ai_engine/sensor_simulator.py
═══════════════════════════════════════════════════════════════════
SensorSimulator — Realistic industrial sensor data generator.

Simulates real physical phenomena:
  - Thermal drift (temperature rises under load)
  - Vibration patterns (normal + bearing-wear harmonics)
  - Pressure fluctuations (process variation)
  - Electrical noise (current + voltage)
  - Degradation progression (aging effects)
  - Correlated sensor behaviour (hot machine → higher current)
  - Sudden fault injection (spike events)
  - Recovery after healing actions

Machine type profiles control baseline sensor ranges.
═══════════════════════════════════════════════════════════════════
"""

from __future__     import annotations

import math
import random
from datetime       import datetime, timezone
from typing         import Dict, List, Optional, Tuple

from backend.utils.logger   import ai_logger


# ── Machine type profiles ──────────────────────────────────────────────────────
MACHINE_PROFILES: Dict[str, Dict] = {
    "cnc_mill": {
        "temperature":      {"base": 45.0,  "range": 25.0,  "noise": 0.8},
        "vibration":        {"base": 2.5,   "range": 2.0,   "noise": 0.3},
        "pressure":         {"base": 120.0, "range": 30.0,  "noise": 2.0},
        "current":          {"base": 25.0,  "range": 10.0,  "noise": 0.5},
        "rpm":              {"base": 2500.0,"range": 500.0, "noise": 20.0},
        "oil_level":        {"base": 80.0,  "range": 5.0,   "noise": 0.2},
        "voltage":          {"base": 380.0, "range": 10.0,  "noise": 1.5},
        "load_percentage":  {"base": 60.0,  "range": 25.0,  "noise": 2.0}
    },
    "conveyor": {
        "temperature":      {"base": 35.0,  "range": 15.0,  "noise": 0.5},
        "vibration":        {"base": 1.5,   "range": 1.0,   "noise": 0.2},
        "pressure":         {"base": 80.0,  "range": 20.0,  "noise": 1.5},
        "current":          {"base": 15.0,  "range": 8.0,   "noise": 0.4},
        "rpm":              {"base": 500.0, "range": 100.0, "noise": 10.0},
        "oil_level":        {"base": 85.0,  "range": 3.0,   "noise": 0.1},
        "voltage":          {"base": 380.0, "range": 8.0,   "noise": 1.0},
        "load_percentage":  {"base": 50.0,  "range": 20.0,  "noise": 1.5}
    },
    "hydraulic_press": {
        "temperature":      {"base": 55.0,  "range": 20.0,  "noise": 1.0},
        "vibration":        {"base": 4.0,   "range": 3.0,   "noise": 0.5},
        "pressure":         {"base": 200.0, "range": 50.0,  "noise": 3.0},
        "current":          {"base": 35.0,  "range": 15.0,  "noise": 0.8},
        "rpm":              {"base": 1800.0,"range": 300.0, "noise": 15.0},
        "oil_level":        {"base": 75.0,  "range": 8.0,   "noise": 0.3},
        "voltage":          {"base": 380.0, "range": 12.0,  "noise": 2.0},
        "load_percentage":  {"base": 70.0,  "range": 20.0,  "noise": 2.0}
    },
    "robot_arm": {
        "temperature":      {"base": 40.0,  "range": 18.0,  "noise": 0.6},
        "vibration":        {"base": 3.0,   "range": 2.5,   "noise": 0.4},
        "pressure":         {"base": 90.0,  "range": 25.0,  "noise": 2.0},
        "current":          {"base": 20.0,  "range": 12.0,  "noise": 0.6},
        "rpm":              {"base": 3000.0,"range": 600.0, "noise": 25.0},
        "oil_level":        {"base": 82.0,  "range": 4.0,   "noise": 0.2},
        "voltage":          {"base": 380.0, "range": 10.0,  "noise": 1.2},
        "load_percentage":  {"base": 55.0,  "range": 30.0,  "noise": 2.5}
    },
    "compressor": {
        "temperature":      {"base": 60.0,  "range": 30.0,  "noise": 1.2},
        "vibration":        {"base": 5.0,   "range": 4.0,   "noise": 0.6},
        "pressure":         {"base": 250.0, "range": 60.0,  "noise": 4.0},
        "current":          {"base": 40.0,  "range": 18.0,  "noise": 1.0},
        "rpm":              {"base": 1500.0,"range": 200.0, "noise": 12.0},
        "oil_level":        {"base": 70.0,  "range": 10.0,  "noise": 0.4},
        "voltage":          {"base": 380.0, "range": 15.0,  "noise": 2.0},
        "load_percentage":  {"base": 75.0,  "range": 20.0,  "noise": 2.0}
    }
}

# ── Machine ID → profile mapping ───────────────────────────────────────────────
MACHINE_TYPE_MAP: Dict[str, str] = {
    "CNC_MILL_01":          "cnc_mill",
    "CONVEYOR_02":          "conveyor",
    "HYDRAULIC_PRESS_03":   "hydraulic_press",
    "ROBOT_ARM_04":         "robot_arm",
    "COMPRESSOR_05":        "compressor"
}

# ── Failure mode sensor impact multipliers ─────────────────────────────────────
FAILURE_IMPACTS: Dict[str, Dict[str, float]] = {
    "bearing_wear": {
        "vibration":    3.5,
        "temperature":  1.4,
        "current":      1.2
    },
    "overheating": {
        "temperature":  2.0,
        "current":      1.3,
        "oil_level":    0.8
    },
    "hydraulic_leak": {
        "pressure":     0.5,
        "oil_level":    0.6,
        "temperature":  1.2
    },
    "electrical_fault": {
        "current":      2.5,
        "voltage":      0.7,
        "temperature":  1.3
    },
    "lubrication_loss": {
        "oil_level":    0.3,
        "vibration":    2.0,
        "temperature":  1.5
    }
}


class SensorSimulator:
    """
    Realistic sensor data simulator for one machine.

    Uses:
    - Machine type profile for baseline values
    - Sinusoidal process cycles (load variation)
    - Gaussian noise per sensor
    - Degradation-based drift
    - Failure mode impact multipliers
    - Correlated sensor physics
    - Random spike events

    Usage:
        sim = SensorSimulator("CNC_MILL_01")
        raw = sim.generate()
        # → {"temperature": 78.3, "vibration": 3.7, ...}
    """

    def __init__(
        self,
        machine_id:     str,
        seed:           Optional[int] = None
    ):
        """
        Args:
            machine_id: Machine identifier (maps to profile)
            seed:       Optional random seed for reproducibility
        """
        self.machine_id = machine_id

        # ── Resolve machine type profile ───────────────────────────────────────
        machine_type    = MACHINE_TYPE_MAP.get(machine_id, "cnc_mill")
        self._profile   = MACHINE_PROFILES[machine_type]

        # ── Simulation state ───────────────────────────────────────────────────
        self._step:             int     = 0
        self._degradation:      float   = 0.0
        self._failure_mode:     Optional[str] = None
        self._failure_severity: float   = 0.0
        self._healing:          bool    = False

        # ── Internal continuous states (slow-moving) ───────────────────────────
        self._thermal_state:    float   = 0.0      # 0 = cold, 1 = at temp
        self._load_state:       float   = 0.5      # current load level 0-1
        self._oil_consumed:     float   = 0.0      # cumulative oil consumption

        # ── Spike injection ────────────────────────────────────────────────────
        self._spike_countdown:  int     = random.randint(300, 800)
        self._in_spike:         bool    = False
        self._spike_remaining:  int     = 0
        self._spike_sensor:     Optional[str] = None
        self._spike_magnitude:  float   = 0.0

        # ── RNG ────────────────────────────────────────────────────────────────
        self._rng = random.Random(seed)

        ai_logger.debug(
            f"SensorSimulator created: [{machine_id}] "
            f"type={machine_type}"
        )

    # ══════════════════════════════════════════════════════════════════════════
    # MAIN GENERATION
    # ══════════════════════════════════════════════════════════════════════════

    def generate(self) -> Dict[str, float]:
        """
        Generate one set of realistic sensor readings.

        Applies (in order):
          1. Advance simulation step
          2. Update slow-moving internal states
          3. Compute baseline from profile
          4. Apply process cycle (sinusoidal load variation)
          5. Apply degradation drift
          6. Apply failure mode impacts
          7. Apply inter-sensor correlations
          8. Add Gaussian noise
          9. Handle spike events
          10. Clamp to physical bounds

        Returns:
            Dict[str, float] — one reading per sensor key
        """
        self._step += 1

        # ── Update internal physics state ──────────────────────────────────────
        self._update_internal_state()

        # ── Build reading ──────────────────────────────────────────────────────
        reading: Dict[str, float] = {}

        for key, profile in self._profile.items():
            value = self._generate_sensor(key, profile)
            reading[key] = value

        # ── Apply inter-sensor correlations ────────────────────────────────────
        reading = self._apply_correlations(reading)

        # ── Inject spike event ─────────────────────────────────────────────────
        reading = self._apply_spike(reading)

        # ── Clamp to physical bounds ───────────────────────────────────────────
        reading = self._clamp(reading)

        # ── Round for realism ─────────────────────────────────────────────────
        reading = {k: round(v, 3) for k, v in reading.items()}

        return reading

    # ══════════════════════════════════════════════════════════════════════════
    # INTERNAL PHYSICS
    # ══════════════════════════════════════════════════════════════════════════

    def _update_internal_state(self) -> None:
        """Advance slow-moving internal simulation physics."""
        # ── Load cycle (sinusoidal, period ~200 steps = ~400 seconds) ─────────
        cycle_period        = 200 + self._rng.gauss(0, 10)
        phase               = (self._step / cycle_period) * 2 * math.pi
        load_cycle          = 0.5 + 0.4 * math.sin(phase)

        # Smooth load transitions
        self._load_state    += 0.05 * (load_cycle - self._load_state)
        self._load_state    = max(0.1, min(1.0, self._load_state))

        # ── Thermal state (machine heats up under load) ────────────────────────
        target_thermal  = self._load_state * (0.7 + self._degradation * 0.3)
        self._thermal_state += 0.02 * (target_thermal - self._thermal_state)
        self._thermal_state = max(0.0, min(1.0, self._thermal_state))

        # ── Oil consumption (slow drain) ───────────────────────────────────────
        drain_rate          = 0.001 * self._load_state
        if self._failure_mode == "hydraulic_leak":
            drain_rate      *= 5.0
        elif self._failure_mode == "lubrication_loss":
            drain_rate      *= 8.0

        self._oil_consumed  = min(30.0, self._oil_consumed + drain_rate)

        # ── Spike countdown ────────────────────────────────────────────────────
        self._spike_countdown -= 1

        if self._spike_countdown <= 0 and not self._in_spike:
            self._trigger_spike()

        if self._in_spike:
            self._spike_remaining -= 1
            if self._spike_remaining <= 0:
                self._in_spike      = False
                self._spike_sensor  = None
                # Schedule next spike
                self._spike_countdown = self._rng.randint(200, 600)

    def _generate_sensor(self, key: str, profile: Dict) -> float:
        """
        Generate a single sensor value.

        Args:
            key:     Sensor name
            profile: Baseline profile for this sensor

        Returns:
            Raw sensor value (before correlations and clamping)
        """
        base    = profile["base"]
        rng     = profile["range"]
        noise   = profile["noise"]

        # ── Load-driven variation ──────────────────────────────────────────────
        load_factor     = self._load_state

        if key == "temperature":
            value       = base + rng * load_factor * self._thermal_state
        elif key == "current":
            value       = base + rng * load_factor
        elif key == "rpm":
            value       = base + rng * (0.3 + 0.7 * load_factor)
        elif key == "load_percentage":
            value       = base + rng * load_factor
        elif key == "oil_level":
            value       = base - self._oil_consumed
        elif key == "pressure":
            value       = base + rng * load_factor
        else:
            value       = base + rng * 0.5 * load_factor

        # ── Degradation drift ──────────────────────────────────────────────────
        deg = self._degradation
        if key == "temperature":
            value   += deg * 25.0
        elif key == "vibration":
            value   += deg * 8.0
        elif key == "current":
            value   += deg * 10.0
        elif key == "oil_level":
            value   -= deg * 15.0
        elif key == "load_percentage":
            value   += deg * 10.0

        # ── Failure mode impact ────────────────────────────────────────────────
        if self._failure_mode and self._failure_mode in FAILURE_IMPACTS:
            impacts     = FAILURE_IMPACTS[self._failure_mode]
            multiplier  = impacts.get(key, 1.0)
            severity    = self._failure_severity

            if multiplier > 1.0:
                value   = value + (value * (multiplier - 1.0) * severity)
            elif multiplier < 1.0:
                value   = value * (1.0 - (1.0 - multiplier) * severity)

        # ── Add Gaussian noise ─────────────────────────────────────────────────
        value += self._rng.gauss(0.0, noise)

        return value

    def _apply_correlations(self, reading: Dict[str, float]) -> Dict[str, float]:
        """
        Apply physical inter-sensor correlations.

        Real correlations:
          - High temperature → higher current draw
          - High load → higher temperature + current + vibration
          - Low oil → higher vibration + temperature
          - High RPM → higher vibration + current

        Args:
            reading: Sensor dict before correlations

        Returns:
            Updated sensor dict
        """
        r = dict(reading)

        temp    = r.get("temperature", 0)
        load    = r.get("load_percentage", 0) / 100.0
        oil     = r.get("oil_level", 100)
        rpm     = r.get("rpm", 0)

        # Temperature → current draw coupling
        r["current"] = r.get("current", 0) + (temp - 50) * 0.05 * load

        # High load → slight temperature rise
        r["temperature"] = temp + load * 3.0

        # Low oil → increased vibration
        oil_factor = max(0.0, (50 - oil) / 50.0)
        r["vibration"] = r.get("vibration", 0) + oil_factor * 2.0

        # High RPM → more vibration
        rpm_factor = max(0.0, (rpm - 2000) / 3000.0)
        r["vibration"] = r.get("vibration", 0) + rpm_factor * 1.5

        return r

    def _apply_spike(self, reading: Dict[str, float]) -> Dict[str, float]:
        """
        Apply spike event to a sensor if one is active.

        Args:
            reading: Current reading dict

        Returns:
            Reading with spike applied
        """
        if self._in_spike and self._spike_sensor in reading:
            reading[self._spike_sensor] += self._spike_magnitude
        return reading

    def _trigger_spike(self) -> None:
        """Randomly select a sensor for a spike event."""
        # Vibration and temperature spikes are most common
        spike_weights = {
            "temperature":  0.30,
            "vibration":    0.35,
            "current":      0.15,
            "pressure":     0.10,
            "voltage":      0.10
        }

        sensors     = list(spike_weights.keys())
        weights     = list(spike_weights.values())
        sensor      = self._rng.choices(sensors, weights=weights, k=1)[0]

        profile     = self._profile[sensor]
        magnitude   = self._rng.uniform(
            profile["range"] * 0.8,
            profile["range"] * 2.5
        )

        self._spike_sensor      = sensor
        self._spike_magnitude   = magnitude
        self._spike_remaining   = self._rng.randint(2, 8)
        self._in_spike          = True

        ai_logger.debug(
            f"[{self.machine_id}] Spike: "
            f"{sensor} +{magnitude:.2f} "
            f"duration={self._spike_remaining}"
        )

    def _clamp(self, reading: Dict[str, float]) -> Dict[str, float]:
        """Clamp all sensors to physical bounds."""
        bounds = {
            "temperature":      (-20.0,  200.0),
            "vibration":        (0.0,    50.0),
            "pressure":         (0.0,    500.0),
            "current":          (0.0,    200.0),
            "rpm":              (0.0,    10000.0),
            "oil_level":        (0.0,    100.0),
            "voltage":          (0.0,    600.0),
            "load_percentage":  (0.0,    100.0),
        }
        return {
            k: max(lo, min(hi, v))
            for k, (lo, hi) in bounds.items()
            if (v := reading.get(k)) is not None
        }

    # ══════════════════════════════════════════════════════════════════════════
    # CONTROLS
    # ══════════════════════════════════════════════════════════════════════════

    def inject_failure(self, failure_mode: str, severity: float = 0.8) -> None:
        """
        Inject a simulated failure mode.

        Args:
            failure_mode: One of FAILURE_IMPACTS keys
            severity:     0.0 – 1.0
        """
        if failure_mode not in FAILURE_IMPACTS:
            ai_logger.warning(
                f"Unknown failure mode: {failure_mode}. "
                f"Valid: {list(FAILURE_IMPACTS.keys())}"
            )
            return

        self._failure_mode      = failure_mode
        self._failure_severity  = max(0.0, min(1.0, severity))
        self._degradation       = max(self._degradation, severity * 0.6)

        ai_logger.warning(
            f"🔴 [{self.machine_id}] Failure injected: "
            f"{failure_mode} severity={severity:.2f}"
        )

    def clear_failure(self) -> None:
        """Clear failure and begin gradual recovery."""
        self._failure_mode      = None
        self._failure_severity  = 0.0
        self._healing           = True

        ai_logger.info(f"✅ [{self.machine_id}] Failure cleared — recovering")

    def set_degradation(self, factor: float) -> None:
        """Manually set degradation factor (0.0 – 1.0)."""
        self._degradation = max(0.0, min(1.0, factor))

    def set_load(self, load: float) -> None:
        """Manually set machine load level (0.0 – 1.0)."""
        self._load_state = max(0.0, min(1.0, load))

    # ══════════════════════════════════════════════════════════════════════════
    # STATUS
    # ══════════════════════════════════════════════════════════════════════════

    def get_status(self) -> Dict:
        """Return current simulator status."""
        return {
            "machine_id":       self.machine_id,
            "step":             self._step,
            "degradation":      round(self._degradation, 4),
            "failure_mode":     self._failure_mode,
            "failure_severity": round(self._failure_severity, 4),
            "load_state":       round(self._load_state, 4),
            "thermal_state":    round(self._thermal_state, 4),
            "oil_consumed":     round(self._oil_consumed, 4),
            "in_spike":         self._in_spike,
            "spike_sensor":     self._spike_sensor
        }

    def __repr__(self) -> str:
        return (
            f"SensorSimulator("
            f"id={self.machine_id!r}, "
            f"step={self._step}, "
            f"deg={self._degradation:.2f})"
        )