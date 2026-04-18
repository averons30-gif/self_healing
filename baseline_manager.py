import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from backend.utils.logger import data_logger
from backend.utils.config import config

logger = data_logger

class BaselineManager:
    """
    Manages per-machine sensor baselines using statistical methods.
    Supports adaptive learning - baselines update as new data comes in.
    """
    
    SENSORS = ["temperature", "vibration", "rpm", "current", "pressure", "oil_level"]
    
    def __init__(self):
        self.baselines: Dict[str, Dict] = {}
        self.window_data: Dict[str, Dict[str, List[float]]] = {}
        self.baseline_ready: Dict[str, bool] = {}
        self.last_update: Dict[str, datetime] = {}
        
    def compute_baseline(
        self, 
        machine_id: str, 
        historical_readings: List[Dict]
    ) -> Dict:
        """
        Compute statistical baseline from historical data.
        Returns per-sensor statistics.
        """
        if not historical_readings:
            logger.warning(f"No historical data for {machine_id}, using defaults")
            return self._get_default_baseline(machine_id)
        
        sensor_data: Dict[str, List[float]] = {s: [] for s in self.SENSORS}
        
        for reading in historical_readings:
            for sensor in self.SENSORS:
                val = reading.get(sensor)
                if val is not None and isinstance(val, (int, float)):
                    sensor_data[sensor].append(float(val))
        
        baseline = {}
        for sensor, values in sensor_data.items():
            if len(values) < 10:
                logger.warning(f"Insufficient data for {machine_id}.{sensor}: {len(values)} points")
                continue
                
            arr = np.array(values)
            
            # Remove obvious outliers for clean baseline (IQR method)
            q25, q75 = np.percentile(arr, [25, 75])
            iqr = q75 - q25
            clean_arr = arr[(arr >= q25 - 2.5*iqr) & (arr <= q75 + 2.5*iqr)]
            
            if len(clean_arr) < 10:
                clean_arr = arr  # Fall back to all data
            
            baseline[sensor] = {
                "mean": float(np.mean(clean_arr)),
                "std": float(np.std(clean_arr)),
                "median": float(np.median(clean_arr)),
                "min_normal": float(np.percentile(clean_arr, 2)),
                "max_normal": float(np.percentile(clean_arr, 98)),
                "percentile_5": float(np.percentile(clean_arr, 5)),
                "percentile_25": float(np.percentile(clean_arr, 25)),
                "percentile_75": float(np.percentile(clean_arr, 75)),
                "percentile_95": float(np.percentile(clean_arr, 95)),
                "count": len(clean_arr),
                "computed_at": datetime.utcnow().isoformat()
            }
        
        self.baselines[machine_id] = baseline
        self.baseline_ready[machine_id] = True
        self.last_update[machine_id] = datetime.utcnow()
        
        logger.info(f"Baseline computed for {machine_id}: "
                   f"{len(baseline)} sensors, {len(historical_readings)} readings")
        return baseline
    
    def update_baseline_incremental(
        self, 
        machine_id: str, 
        new_reading: Dict,
        weight: float = 0.02
    ):
        """
        Incrementally update baseline using exponential moving average.
        Allows baseline to adapt to slow machine changes.
        """
        if machine_id not in self.baselines:
            return
        
        for sensor in self.SENSORS:
            val = new_reading.get(sensor)
            if val is None or machine_id not in self.baselines:
                continue
            
            # Only update if reading is "normal" (not flagged as anomaly)
            if new_reading.get("is_anomaly", False):
                continue
            
            if sensor in self.baselines[machine_id]:
                old_mean = self.baselines[machine_id][sensor]["mean"]
                old_std = self.baselines[machine_id][sensor]["std"]
                
                # EMA update
                new_mean = (1 - weight) * old_mean + weight * val
                new_std = np.sqrt(
                    (1 - weight) * old_std**2 + 
                    weight * (val - new_mean)**2
                )
                
                self.baselines[machine_id][sensor]["mean"] = float(new_mean)
                self.baselines[machine_id][sensor]["std"] = float(max(new_std, 0.01))
    
    def get_baseline(self, machine_id: str) -> Optional[Dict]:
        """Get computed baseline for a machine."""
        return self.baselines.get(machine_id)
    
    def is_ready(self, machine_id: str) -> bool:
        """Check if baseline is ready for a machine."""
        return self.baseline_ready.get(machine_id, False)
    
    def compute_z_scores(
        self, 
        machine_id: str, 
        reading: Dict
    ) -> Dict[str, float]:
        """Compute Z-scores for each sensor value against baseline."""
        if machine_id not in self.baselines:
            return {}
        
        z_scores = {}
        baseline = self.baselines[machine_id]
        
        for sensor in self.SENSORS:
            val = reading.get(sensor)
            if val is None or sensor not in baseline:
                continue
            
            mean = baseline[sensor]["mean"]
            std = baseline[sensor]["std"]
            
            if std < 0.001:
                std = 0.001
            
            z_scores[sensor] = abs((val - mean) / std)
        
        return z_scores
    
    def compute_deviation_percentages(
        self, 
        machine_id: str, 
        reading: Dict
    ) -> Dict[str, float]:
        """Compute percentage deviation from baseline mean."""
        if machine_id not in self.baselines:
            return {}
        
        deviations = {}
        baseline = self.baselines[machine_id]
        
        for sensor in self.SENSORS:
            val = reading.get(sensor)
            if val is None or sensor not in baseline:
                continue
            
            mean = baseline[sensor]["mean"]
            if mean != 0:
                deviations[sensor] = ((val - mean) / mean) * 100
            else:
                deviations[sensor] = 0.0
        
        return deviations
    
    def _get_default_baseline(self, machine_id: str) -> Dict:
        """Get default baseline when no historical data available."""
        from backend.data.simulator import MACHINE_CONFIGS
        
        machine_config = MACHINE_CONFIGS.get(machine_id, {})
        sim_baselines = machine_config.get("baselines", {})
        
        baseline = {}
        for sensor, params in sim_baselines.items():
            mean = params["mean"]
            std = params["std"]
            baseline[sensor] = {
                "mean": mean,
                "std": std,
                "median": mean,
                "min_normal": params["min"],
                "max_normal": params["max"],
                "percentile_5": mean - 2 * std,
                "percentile_25": mean - std,
                "percentile_75": mean + std,
                "percentile_95": mean + 2 * std,
                "count": 0,
                "computed_at": datetime.utcnow().isoformat()
            }
        
        self.baselines[machine_id] = baseline
        self.baseline_ready[machine_id] = True
        return baseline

# Global baseline manager
baseline_manager = BaselineManager()