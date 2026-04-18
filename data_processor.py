"""
backend/ai_engine/data_processor.py
═══════════════════════════════════════════════════════════════════
DataProcessor — Pre/post processing pipeline for sensor data.

Responsibilities:
  - Input validation and type coercion
  - Physical bounds clamping
  - Missing value imputation
  - Data quality scoring
  - Delta (rate-of-change) computation
  - Feature engineering for ML models
  - Post-processing output formatting
═══════════════════════════════════════════════════════════════════
"""

from __future__     import annotations

import math
from typing         import Dict, List, Optional, Tuple

from backend.utils.logger   import ai_logger


class DataProcessor:
    """
    Sensor data pre/post processor.

    Validates, cleanses and enriches raw sensor data
    before it enters the AI pipeline.

    Usage:
        processor = DataProcessor(sensor_keys, sensor_bounds)
        validated, quality = processor.preprocess(raw)
        deltas = processor.compute_deltas(current, previous)
        features = processor.extract_features(filtered_history)
    """

    def __init__(
        self,
        sensor_keys:    List[str],
        sensor_bounds:  Dict[str, Tuple[float, float]]
    ):
        """
        Args:
            sensor_keys:    Expected sensor channel names
            sensor_bounds:  Dict of key → (min, max)
        """
        self._keys          = sensor_keys
        self._bounds        = sensor_bounds
        self._impute_cache: Dict[str, float] = {}  # last known good value
        self._process_count = 0

        ai_logger.debug(
            f"DataProcessor created: {len(sensor_keys)} channels"
        )

    # ══════════════════════════════════════════════════════════════════════════
    # PRE-PROCESSING
    # ══════════════════════════════════════════════════════════════════════════

    def preprocess(
        self,
        raw: Dict
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Validate, clamp and quality-score a raw sensor reading.

        Steps:
          1. Type coercion (str → float)
          2. NaN / Inf replacement
          3. Missing key imputation (last known good)
          4. Physical bounds clamping
          5. Quality score computation

        Args:
            raw: Untyped sensor dict from SensorSimulator

        Returns:
            Tuple of:
              - validated: Dict[str, float] — clean sensor values
              - quality:   Dict[str, float] — completeness + validity scores
        """
        self._process_count += 1

        validated:      Dict[str, float] = {}
        missing:        List[str]       = []
        out_of_bounds:  List[str]       = []

        for key in self._keys:
            raw_val = raw.get(key)

            # ── Missing value imputation ───────────────────────────────────────
            if raw_val is None:
                missing.append(key)
                # Use last known good or default to midpoint of bounds
                lo, hi  = self._bounds.get(key, (0.0, 100.0))
                fallback = self._impute_cache.get(key, (lo + hi) / 2.0)
                validated[key] = fallback
                continue

            # ── Type coercion ──────────────────────────────────────────────────
            try:
                value = float(raw_val)
            except (TypeError, ValueError):
                missing.append(key)
                lo, hi  = self._bounds.get(key, (0.0, 100.0))
                validated[key] = self._impute_cache.get(key, (lo + hi) / 2.0)
                continue

            # ── NaN / Inf handling ─────────────────────────────────────────────
            if math.isnan(value) or math.isinf(value):
                missing.append(key)
                lo, hi  = self._bounds.get(key, (0.0, 100.0))
                validated[key] = self._impute_cache.get(key, (lo + hi) / 2.0)
                continue

            # ── Bounds clamping ────────────────────────────────────────────────
            if key in self._bounds:
                lo, hi = self._bounds[key]

                if value < lo or value > hi:
                    out_of_bounds.append(key)

                value = max(lo, min(hi, value))

            validated[key]          = round(value, 4)
            self._impute_cache[key] = value     # update last known good

        # ── Quality metrics ────────────────────────────────────────────────────
        n               = len(self._keys)
        completeness    = 1.0 - len(missing) / max(n, 1)
        validity        = 1.0 - len(out_of_bounds) / max(n, 1)

        quality = {
            "completeness":     round(completeness, 4),
            "validity":         round(validity, 4),
            "missing_sensors":  missing,
            "oob_sensors":      out_of_bounds
        }

        if missing:
            ai_logger.debug(
                f"DataProcessor: {len(missing)} missing "
                f"(imputed): {missing}"
            )
        if out_of_bounds:
            ai_logger.debug(
                f"DataProcessor: {len(out_of_bounds)} out-of-bounds "
                f"(clamped): {out_of_bounds}"
            )

        return validated, quality

    # ══════════════════════════════════════════════════════════════════════════
    # DELTAS
    # ══════════════════════════════════════════════════════════════════════════

    def compute_deltas(
        self,
        current:    Dict[str, float],
        previous:   Optional[Dict[str, float]]
    ) -> Dict[str, float]:
        """
        Compute per-sensor rate of change vs previous reading.

        Args:
            current:    Current filtered sensor values
            previous:   Previous filtered sensor values (or None on first)

        Returns:
            Dict of sensor_key → delta (positive = rising, negative = falling)
        """
        if not previous:
            return {k: 0.0 for k in current}

        deltas = {}
        for key, value in current.items():
            prev_val        = previous.get(key, value)
            deltas[key]     = round(value - prev_val, 4)

        return deltas

    # ══════════════════════════════════════════════════════════════════════════
    # FEATURE ENGINEERING
    # ══════════════════════════════════════════════════════════════════════════

    def extract_features(
        self,
        history: List[Dict[str, float]]
    ) -> Dict[str, float]:
        """
        Extract statistical features from a rolling history window.
        Used for ML model inputs.

        Features per sensor:
          - mean, std, min, max
          - trend (linear slope over window)
          - volatility (std / mean)

        Args:
            history: List of filtered sensor dicts (oldest first)

        Returns:
            Flat feature dict: e.g. {"temperature_mean": 72.3, ...}
        """
        if not history:
            return {}

        features: Dict[str, float] = {}

        for key in self._keys:
            values = [h.get(key, 0.0) for h in history if key in h]

            if not values:
                continue

            n   = len(values)
            mean = sum(values) / n
            std  = (
                math.sqrt(sum((v - mean) ** 2 for v in values) / n)
                if n > 1 else 0.0
            )
            trend = self._linear_slope(values)
            vol   = std / mean if mean > 1e-9 else 0.0

            prefix = key
            features[f"{prefix}_mean"]          = round(mean, 4)
            features[f"{prefix}_std"]           = round(std, 4)
            features[f"{prefix}_min"]           = round(min(values), 4)
            features[f"{prefix}_max"]           = round(max(values), 4)
            features[f"{prefix}_trend"]         = round(trend, 6)
            features[f"{prefix}_volatility"]    = round(vol, 4)

        return features

    def _linear_slope(self, values: List[float]) -> float:
        """
        Compute linear regression slope of a value sequence.

        Args:
            values: Ordered sequence of floats

        Returns:
            Slope (positive = rising trend)
        """
        n = len(values)
        if n < 2:
            return 0.0

        x_mean  = (n - 1) / 2.0
        y_mean  = sum(values) / n

        num     = sum((i - x_mean) * (values[i] - y_mean) for i in range(n))
        den     = sum((i - x_mean) ** 2 for i in range(n))

        return num / den if den > 1e-9 else 0.0

    # ══════════════════════════════════════════════════════════════════════════
    # POST-PROCESSING
    # ══════════════════════════════════════════════════════════════════════════

    def normalize(
        self,
        sensors:    Dict[str, float],
        method:     str = "minmax"
    ) -> Dict[str, float]:
        """
        Normalize sensor values to [0, 1] or [-1, 1] range.

        Args:
            sensors: Validated sensor dict
            method:  "minmax" (0-1) or "zscore"

        Returns:
            Normalized sensor dict
        """
        normalized = {}

        for key, value in sensors.items():
            if key not in self._bounds:
                normalized[key] = value
                continue

            lo, hi  = self._bounds[key]
            rng     = hi - lo

            if method == "minmax":
                normalized[key] = round(
                    (value - lo) / rng if rng > 1e-9 else 0.0,
                    4
                )
            else:
                # Pass through for now — z-score needs mean/std from history
                normalized[key] = value

        return normalized

    # ══════════════════════════════════════════════════════════════════════════
    # STATS
    # ══════════════════════════════════════════════════════════════════════════

    def get_stats(self) -> Dict:
        """Return processor statistics."""
        return {
            "channels":         len(self._keys),
            "processed_count":  self._process_count,
            "impute_cache_size": len(self._impute_cache)
        }

    def __repr__(self) -> str:
        return (
            f"DataProcessor("
            f"channels={len(self._keys)}, "
            f"processed={self._process_count})"
        )