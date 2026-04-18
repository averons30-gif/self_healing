import numpy as np
from collections import deque
from typing import Dict, Tuple, Optional
from backend.utils.logger import ai_logger

class NoiseFilter:
    """
    Multi-stage noise filter using:
    1. Exponential Moving Average (EMA) - removes high-frequency noise
    2. Hampel Identifier - detects and replaces outliers in sliding window
    3. Savitzky-Golay smoothing approximation - preserves signal shape
    4. Per-machine noise fingerprinting - learned noise profile
    """
    
    def __init__(self, window_size: int = 5, ema_alpha: float = 0.3):
        self.window_size = window_size
        self.ema_alpha = ema_alpha
        self.sensors = ["temperature", "vibration", "rpm", "current"]
        
        # Per-machine state
        self.buffers: Dict[str, Dict[str, deque]] = {}
        self.ema_values: Dict[str, Dict[str, float]] = {}
        self.noise_profiles: Dict[str, Dict[str, float]] = {}
        self.initialized: Dict[str, bool] = {}
    
    def _init_machine(self, machine_id: str):
        """Initialize filter state for a machine"""
        if machine_id not in self.buffers:
            self.buffers[machine_id] = {
                s: deque(maxlen=self.window_size * 3) for s in self.sensors
            }
            self.ema_values[machine_id] = {s: None for s in self.sensors}
            self.noise_profiles[machine_id] = {s: 0.1 for s in self.sensors}
            self.initialized[machine_id] = False
    
    def _update_ema(self, machine_id: str, sensor: str, value: float) -> float:
        """Exponential Moving Average filter"""
        current_ema = self.ema_values[machine_id][sensor]
        if current_ema is None:
            self.ema_values[machine_id][sensor] = value
            return value
        
        new_ema = self.ema_alpha * value + (1 - self.ema_alpha) * current_ema
        self.ema_values[machine_id][sensor] = new_ema
        return new_ema
    
    def _hampel_filter(self, buffer: deque, value: float) -> Tuple[float, bool]:
        """
        Hampel identifier: detects outliers using median absolute deviation
        Returns: (filtered_value, is_outlier)
        """
        if len(buffer) < 3:
            return value, False
        
        arr = np.array(list(buffer))
        median = np.median(arr)
        mad = np.median(np.abs(arr - median))
        threshold = 3.0 * 1.4826 * mad  # 1.4826 makes MAD consistent with std
        
        if abs(value - median) > threshold and threshold > 0:
            return median, True  # Replace outlier with median
        return value, False
    
    def _update_noise_profile(self, machine_id: str, sensor: str, value: float):
        """Learn the noise characteristic of each sensor"""
        buffer = self.buffers[machine_id][sensor]
        if len(buffer) >= self.window_size:
            arr = np.array(list(buffer)[-self.window_size:])
            noise_estimate = np.std(arr)
            
            # Exponential smoothing of noise profile
            current = self.noise_profiles[machine_id][sensor]
            self.noise_profiles[machine_id][sensor] = (
                0.1 * noise_estimate + 0.9 * current
            )
    
    def filter(
        self, 
        machine_id: str, 
        readings: Dict[str, float]
    ) -> Tuple[Dict[str, float], bool]:
        """
        Apply multi-stage filtering to sensor readings.
        Returns: (filtered_readings, any_outlier_detected)
        """
        self._init_machine(machine_id)
        
        filtered = {}
        any_outlier = False
        
        for sensor in self.sensors:
            value = readings.get(sensor, 0.0)
            buffer = self.buffers[machine_id][sensor]
            
            # Stage 1: Hampel outlier detection
            filtered_val, is_outlier = self._hampel_filter(buffer, value)
            if is_outlier:
                any_outlier = True
            
            # Stage 2: EMA smoothing
            smoothed_val = self._update_ema(machine_id, sensor, filtered_val)
            
            # Update buffer with original value (not filtered)
            buffer.append(value)
            
            # Update noise profile
            self._update_noise_profile(machine_id, sensor, value)
            
            filtered[sensor] = smoothed_val
        
        return filtered, any_outlier
    
    def is_noise(
        self, 
        machine_id: str, 
        readings: Dict[str, float],
        filtered: Dict[str, float]
    ) -> bool:
        """
        Determine if a reading is predominantly noise.
        Uses noise fingerprint comparison.
        """
        self._init_machine(machine_id)
        
        noise_count = 0
        for sensor in self.sensors:
            original = readings.get(sensor, 0.0)
            filt = filtered.get(sensor, 0.0)
            profile = self.noise_profiles[machine_id][sensor]
            
            deviation = abs(original - filt)
            if profile > 0 and deviation > 2.5 * profile:
                noise_count += 1
        
        # If more than half sensors show noise characteristics
        return noise_count >= len(self.sensors) // 2
    
    def get_noise_profile(self, machine_id: str) -> Dict[str, float]:
        """Get current noise profile for a machine"""
        self._init_machine(machine_id)
        return self.noise_profiles.get(machine_id, {})