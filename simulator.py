import asyncio
import random
import math
import uuid
from datetime import datetime
from typing import AsyncGenerator, Dict, Optional
from dataclasses import dataclass, field
from backend.utils.logger import data_logger
from backend.utils.config import config


@dataclass
class SensorReading:
    """
    Represents a single sensor reading from a machine.
    
    This is the raw data packet received from physical sensors
    or the simulator before AI processing.
    """
    reading_id: str
    machine_id: str
    timestamp: datetime
    
    # Core sensor values
    temperature: float          # °C — bearing/motor temperature
    vibration: float            # mm/s — vibration amplitude
    pressure: float             # bar — hydraulic/pneumatic pressure
    current: float              # Amps — motor current draw
    rpm: float                  # Rotations per minute
    oil_level: float            # % — lubrication level
    voltage: float              # Volts — supply voltage
    
    # Derived / environmental
    humidity: float             # % — ambient humidity
    ambient_temp: float         # °C — environment temperature
    load_percentage: float      # % — current operational load
    
    # Metadata
    sequence_number: int = 0
    sensor_health: Dict = field(default_factory=dict)


class MachineProfile:
    """
    Defines the normal operating parameters for a specific machine type.
    Each machine has unique baseline characteristics and failure modes.
    """
    
    def __init__(self, machine_id: str, machine_type: str):
        self.machine_id = machine_id
        self.machine_type = machine_type
        
        # Set baseline profiles per machine type
        self._profiles = {
            "CNC_MILL_01": {
                "temperature":      {"base": 85,  "variance": 8,  "max": 95},
                "vibration":        {"base": 6.0, "variance": 0.8, "max": 8},
                "pressure":         {"base": 6.2, "variance": 0.5, "max": 10},
                "current":          {"base": 30,  "variance": 3,  "max": 35},
                "rpm":              {"base": 3200, "variance": 100, "max": 4500},
                "oil_level":        {"base": 85,  "variance": 2,  "max": 100},
                "voltage":          {"base": 400, "variance": 5,  "max": 440},
                "humidity":         {"base": 45,  "variance": 5,  "max": 80},
                "ambient_temp":     {"base": 22,  "variance": 3,  "max": 40},
                "load_percentage":  {"base": 70,  "variance": 15, "max": 100},
            },
            "CONVEYOR_02": {
                "temperature":      {"base": 45,  "variance": 5,  "max": 75},
                "vibration":        {"base": 1.8, "variance": 0.5, "max": 6},
                "pressure":         {"base": 4.5, "variance": 0.3, "max": 8},
                "current":          {"base": 12,  "variance": 2,  "max": 25},
                "rpm":              {"base": 1200, "variance": 50, "max": 1800},
                "oil_level":        {"base": 90,  "variance": 1,  "max": 100},
                "voltage":          {"base": 380, "variance": 5,  "max": 415},
                "humidity":         {"base": 50,  "variance": 8,  "max": 85},
                "ambient_temp":     {"base": 20,  "variance": 2,  "max": 35},
                "load_percentage":  {"base": 60,  "variance": 20, "max": 100},
            },
            "HYDRAULIC_PRESS_03": {
                "temperature":      {"base": 45,  "variance": 10, "max": 90},
                "vibration":        {"base": 2.0, "variance": 1.0, "max": 10},
                "pressure":         {"base": 150, "variance": 15, "max": 250},
                "current":          {"base": 20,  "variance": 5,  "max": 50},
                "rpm":              {"base": 1400, "variance": 30, "max": 1600},
                "oil_level":        {"base": 85,  "variance": 3,  "max": 100},
                "voltage":          {"base": 410, "variance": 8,  "max": 440},
                "humidity":         {"base": 40,  "variance": 5,  "max": 75},
                "ambient_temp":     {"base": 25,  "variance": 3,  "max": 42},
                "load_percentage":  {"base": 70,  "variance": 20, "max": 100},
            },
            "ROBOT_ARM_04": {
                "temperature":      {"base": 50,  "variance": 7,  "max": 80},
                "vibration":        {"base": 1.2, "variance": 0.4, "max": 5},
                "pressure":         {"base": 5.5, "variance": 0.4, "max": 9},
                "current":          {"base": 8,   "variance": 2,  "max": 20},
                "rpm":              {"base": 800,  "variance": 50, "max": 1200},
                "oil_level":        {"base": 95,  "variance": 1,  "max": 100},
                "voltage":          {"base": 24,  "variance": 1,  "max": 28},
                "humidity":         {"base": 40,  "variance": 5,  "max": 70},
                "ambient_temp":     {"base": 22,  "variance": 2,  "max": 38},
                "load_percentage":  {"base": 50,  "variance": 25, "max": 100},
            },
            "COMPRESSOR_05": {
                "temperature":      {"base": 75,  "variance": 12, "max": 105},
                "vibration":        {"base": 4.0, "variance": 1.2, "max": 12},
                "pressure":         {"base": 8.5, "variance": 0.8, "max": 12},
                "current":          {"base": 30,  "variance": 5,  "max": 55},
                "rpm":              {"base": 2900, "variance": 100, "max": 3600},
                "oil_level":        {"base": 82,  "variance": 3,  "max": 100},
                "voltage":          {"base": 400, "variance": 6,  "max": 440},
                "humidity":         {"base": 55,  "variance": 10, "max": 90},
                "ambient_temp":     {"base": 28,  "variance": 4,  "max": 45},
                "load_percentage":  {"base": 80,  "variance": 15, "max": 100},
            }
        }
        
        # Default profile if machine not found
        self.profile = self._profiles.get(machine_id, self._profiles["CNC_MILL_01"])
        
        # Failure simulation state
        self.degradation_factor = 0.0      # 0.0 = healthy, 1.0 = critical
        self.failure_mode = None            # Active failure mode
        self.sequence_counter = 0
        
        # Drift simulation (gradual degradation over time)
        self.drift = {sensor: 0.0 for sensor in self.profile}
    
    def get_reading(self) -> Dict:
        """
        Generate a sensor reading based on machine profile.
        Incorporates:
        - Normal variance (Gaussian noise)
        - Gradual drift (wear and aging)
        - Failure modes (bearing wear, overheating, etc.)
        - Temporal patterns (load cycles, shift patterns)
        """
        self.sequence_counter += 1
        readings = {}
        
        # Time-based load cycle (sinusoidal production pattern)
        time_factor = math.sin(self.sequence_counter * 0.05) * 0.15
        
        for sensor, params in self.profile.items():
            base = params["base"]
            variance = params["variance"]
            
            # Gaussian noise
            noise = random.gauss(0, variance * 0.3)
            
            # Drift accumulation
            self.drift[sensor] += random.gauss(0, 0.01)
            self.drift[sensor] = max(-variance, min(variance, self.drift[sensor]))
            
            # Degradation effect
            degradation_impact = self._apply_degradation(sensor, params)
            
            # Load cycle impact
            load_impact = base * time_factor * 0.1
            
            # Final value
            value = base + noise + self.drift[sensor] + degradation_impact + load_impact
            
            # Clamp to realistic bounds (allow slight overshoot for anomalies)
            max_val = params["max"] * 1.2
            value = max(0, min(max_val, value))
            
            readings[sensor] = round(value, 3)
        
        return readings
    
    def _apply_degradation(self, sensor: str, params: Dict) -> float:
        """
        Apply failure mode effects to sensor readings.
        
        Failure modes:
        - bearing_wear: vibration↑, temperature↑, rpm↓
        - overheating: temperature↑, pressure↑, current↑
        - hydraulic_leak: pressure↓, oil_level↓
        - electrical_fault: voltage↓, current spike, vibration↑
        - lubrication_loss: temperature↑, vibration↑, oil_level↓
        """
        if self.degradation_factor == 0 or not self.failure_mode:
            return 0.0
        
        impact_map = {
            "bearing_wear": {
                "vibration": +4.0,
                "temperature": +15.0,
                "rpm": -100.0,
                "current": +5.0
            },
            "overheating": {
                "temperature": +25.0,
                "pressure": +2.0,
                "current": +8.0,
                "oil_level": -5.0
            },
            "hydraulic_leak": {
                "pressure": -30.0,
                "oil_level": -20.0,
                "temperature": +5.0,
                "vibration": +1.5
            },
            "electrical_fault": {
                "voltage": -25.0,
                "current": +15.0,
                "vibration": +2.0,
                "temperature": +8.0
            },
            "lubrication_loss": {
                "temperature": +20.0,
                "vibration": +3.0,
                "oil_level": -30.0,
                "current": +6.0
            }
        }
        
        mode_impacts = impact_map.get(self.failure_mode, {})
        base_impact = mode_impacts.get(sensor, 0.0)
        
        return base_impact * self.degradation_factor
    
    def set_degradation(self, factor: float, mode: str):
        """
        Set degradation state for failure simulation.
        
        Args:
            factor: 0.0 (healthy) to 1.0 (critical failure)
            mode: Failure mode name
        """
        self.degradation_factor = max(0.0, min(1.0, factor))
        self.failure_mode = mode if factor > 0 else None


class DataSimulator:
    """
    Industrial IoT Data Simulator.
    
    Simulates real-time sensor streams from multiple machines
    with configurable failure injection and realistic patterns.
    
    Features:
    - Per-machine sensor profiles
    - Realistic Gaussian noise
    - Gradual degradation simulation
    - Random failure injection
    - Configurable streaming rate
    """
    
    def __init__(self):
        self.machine_profiles: Dict[str, MachineProfile] = {}
        self.running = False
        self._init_profiles()
        
        # Failure injection schedule
        self._failure_schedule: Dict[str, Dict] = {}
        
        data_logger.info("📡 DataSimulator initialized")
    
    def _init_profiles(self):
        """Initialize machine profiles from config"""
        machine_types = getattr(config, "MACHINE_TYPES", {})
        
        for machine_id in config.MACHINE_IDS:
            machine_type = machine_types.get(machine_id, "generic")
            self.machine_profiles[machine_id] = MachineProfile(machine_id, machine_type)
            data_logger.info(f"  📊 Profile created: {machine_id} ({machine_type})")
    
    async def stream(self, machine_id: str) -> AsyncGenerator[SensorReading, None]:
        """
        Async generator that continuously yields sensor readings.
        
        Args:
            machine_id: Machine to stream data for
            
        Yields:
            SensorReading objects at configured interval
        """
        profile = self.machine_profiles.get(machine_id)
        if not profile:
            data_logger.error(f"No profile for machine: {machine_id}")
            return
        
        data_logger.info(f"📡 Stream started: {machine_id}")
        
        while True:
            try:
                # Check for scheduled failure injection
                self._apply_scheduled_failures(machine_id)
                
                # Generate sensor values
                sensor_values = profile.get_reading()
                
                # Create reading object
                reading = SensorReading(
                    reading_id=str(uuid.uuid4()),
                    machine_id=machine_id,
                    timestamp=datetime.utcnow(),
                    temperature=sensor_values["temperature"],
                    vibration=sensor_values["vibration"],
                    pressure=sensor_values["pressure"],
                    current=sensor_values["current"],
                    rpm=sensor_values["rpm"],
                    oil_level=sensor_values["oil_level"],
                    voltage=sensor_values["voltage"],
                    humidity=sensor_values["humidity"],
                    ambient_temp=sensor_values["ambient_temp"],
                    load_percentage=sensor_values["load_percentage"],
                    sequence_number=profile.sequence_counter,
                    sensor_health=self._get_sensor_health(sensor_values, profile)
                )
                
                yield reading
                
                # Configurable stream rate
                await asyncio.sleep(config.STREAM_INTERVAL)
                
            except asyncio.CancelledError:
                data_logger.info(f"Stream cancelled: {machine_id}")
                break
            except Exception as e:
                data_logger.error(f"Stream error [{machine_id}]: {e}", exc_info=True)
                await asyncio.sleep(1)
    
    def _get_sensor_health(self, values: Dict, profile: MachineProfile) -> Dict:
        """
        Assess health status of each sensor.
        
        Returns:
            Dict mapping sensor names to health percentages
        """
        health = {}
        
        for sensor, params in profile.profile.items():
            value = values.get(sensor, 0)
            max_val = params["max"]
            base = params["base"]
            
            # Calculate how far from baseline (as percentage)
            deviation = abs(value - base) / max(base, 1)
            health_score = max(0, 100 - (deviation * 100))
            
            health[sensor] = round(health_score, 1)
        
        return health
    
    def _apply_scheduled_failures(self, machine_id: str):
        """Apply any scheduled failure injections"""
        if machine_id not in self._failure_schedule:
            return
        
        schedule = self._failure_schedule[machine_id]
        profile = self.machine_profiles[machine_id]
        
        # Gradually increase degradation
        current_factor = profile.degradation_factor
        target_factor = schedule.get("target_factor", 0.8)
        ramp_speed = schedule.get("ramp_speed", 0.01)
        
        if current_factor < target_factor:
            new_factor = min(target_factor, current_factor + ramp_speed)
            profile.set_degradation(new_factor, schedule["mode"])
    
    def inject_failure(
        self,
        machine_id: str,
        failure_mode: str,
        severity: float = 0.8,
        ramp_speed: float = 0.02
    ):
        """
        Inject a failure into a specific machine.
        
        Args:
            machine_id: Target machine
            failure_mode: Type of failure (bearing_wear, overheating, etc.)
            severity: Final degradation level (0.0-1.0)
            ramp_speed: Speed of degradation increase per reading
        """
        self._failure_schedule[machine_id] = {
            "mode": failure_mode,
            "target_factor": severity,
            "ramp_speed": ramp_speed
        }
        
        data_logger.warning(
            f"💉 Failure injected: {machine_id} | "
            f"Mode: {failure_mode} | Severity: {severity}"
        )
    
    def clear_failure(self, machine_id: str):
        """
        Clear failure injection and restore healthy state.
        
        Args:
            machine_id: Machine to restore
        """
        if machine_id in self._failure_schedule:
            del self._failure_schedule[machine_id]
        
        profile = self.machine_profiles.get(machine_id)
        if profile:
            profile.set_degradation(0.0, None)
            # Reset drift
            profile.drift = {sensor: 0.0 for sensor in profile.profile}
        
        data_logger.info(f"✅ Failure cleared: {machine_id}")
    
    def get_machine_status(self, machine_id: str) -> Dict:
        """
        Get current simulation status for a machine.
        
        Returns:
            Dict with degradation state and sensor profile info
        """
        profile = self.machine_profiles.get(machine_id)
        if not profile:
            return {"error": "Machine not found"}
        
        return {
            "machine_id": machine_id,
            "machine_type": profile.machine_type,
            "degradation_factor": profile.degradation_factor,
            "failure_mode": profile.failure_mode,
            "sequence_number": profile.sequence_counter,
            "drift": profile.drift,
            "has_scheduled_failure": machine_id in self._failure_schedule
        }
    
    def get_all_status(self) -> Dict:
        """Return status for all simulated machines"""
        return {
            machine_id: self.get_machine_status(machine_id)
            for machine_id in self.machine_profiles
        }


# Global singleton instance
simulator = DataSimulator()