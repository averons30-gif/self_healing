import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
import numpy as np
from backend.database.db_manager import db
from backend.utils.logger import ai_logger
from backend.utils.config import config

class BaselineManager:
    """
    Manages 7-day rolling baseline for each machine.
    Uses a combination of:
    - Historical database records
    - Synthetic initialization data (for cold start)
    - Incremental updates
    """
    
    # Typical sensor baselines for each machine type
    TYPICAL_BASELINES = {
        "MACHINE_001": {  # Compressor
            "temperature": {"mean": 65.0, "std": 5.0},
            "vibration": {"mean": 3.5, "std": 0.8},
            "rpm": {"mean": 2400.0, "std": 50.0},
            "current": {"mean": 28.0, "std": 3.0}
        },
        "MACHINE_002": {  # Pump
            "temperature": {"mean": 72.0, "std": 6.0},
            "vibration": {"mean": 4.2, "std": 1.0},
            "rpm": {"mean": 2800.0, "std": 75.0},
            "current": {"mean": 35.0, "std": 4.0}
        },
        "MACHINE_003": {  # Motor
            "temperature": {"mean": 58.0, "std": 4.0},
            "vibration": {"mean": 2.8, "std": 0.6},
            "rpm": {"mean": 1750.0, "std": 30.0},
            "current": {"mean": 22.0, "std": 2.5}
        },
        "MACHINE_004": {  # Turbine
            "temperature": {"mean": 78.0, "std": 7.0},
            "vibration": {"mean": 5.5, "std": 1.2},
            "rpm": {"mean": 2950.0, "std": 60.0},
            "current": {"mean": 38.0, "std": 5.0}
        }
    }
    
    async def get_baseline(self, machine_id: str) -> List[Dict]:
        """Get historical baseline data, generating synthetic if insufficient"""
        
        # Try loading from database
        db_data = await db.get_baseline_data(machine_id, days=7)
        
        if len(db_data) >= 100:
            ai_logger.info(f"Loaded {len(db_data)} baseline records for {machine_id}")
            return db_data
        
        # Cold start: generate synthetic baseline
        ai_logger.info(f"Generating synthetic baseline for {machine_id}")
        synthetic = self._generate_synthetic_baseline(machine_id, n_samples=1000)
        
        # Combine real + synthetic
        combined = db_data + synthetic
        return combined
    
    def _generate_synthetic_baseline(
        self, 
        machine_id: str, 
        n_samples: int = 1000
    ) -> List[Dict]:
        """
        Generate realistic synthetic baseline data.
        Includes:
        - Normal operating patterns
        - Circadian variations (machine runs harder during day)
        - Weekend vs weekday patterns
        - Gradual drift over time
        """
        baselines = self.TYPICAL_BASELINES.get(
            machine_id, 
            self.TYPICAL_BASELINES["MACHINE_001"]
        )
        
        samples = []
        now = datetime.utcnow()
        
        for i in range(n_samples):
            # Time offset (sample every ~10 minutes over 7 days)
            offset_minutes = (7 * 24 * 60 * i) / n_samples
            sample_time = now - timedelta(minutes=7 * 24 * 60 - offset_minutes)
            
            # Circadian variation (machines load varies with time of day)
            hour = sample_time.hour
            load_factor = self._circadian_factor(hour)
            
            sample = {"timestamp": sample_time.isoformat()}
            
            for sensor, stats in baselines.items():
                mean = stats["mean"] * load_factor
                std = stats["std"]
                
                # Add realistic correlated noise
                value = np.random.normal(mean, std * 0.5)
                
                # Occasional mild transients (NOT anomalies, just normal variation)
                if np.random.random() < 0.02:
                    value += np.random.normal(0, std * 1.5)
                
                sample[sensor] = float(value)
            
            samples.append(sample)
        
        return samples
    
    def _circadian_factor(self, hour: int) -> float:
        """Model machine load throughout the day"""
        # Peak load 8am-6pm, reduced nights/weekends
        if 8 <= hour <= 18:
            return 1.0 + 0.1 * np.sin((hour - 8) / 10 * np.pi)
        elif 6 <= hour < 8 or 18 < hour <= 20:
            return 0.85
        else:
            return 0.70
    
    async def update_baseline_incrementally(
        self, 
        machine_id: str,
        new_reading: Dict
    ):
        """Add new reading to rolling baseline"""
        await db.save_reading({
            "reading_id": new_reading.get("reading_id", ""),
            "machine_id": machine_id,
            "timestamp": new_reading.get("timestamp", datetime.utcnow().isoformat()),
            "sensors": new_reading.get("sensors", {}),
            "analysis": new_reading.get("analysis", {})
        })