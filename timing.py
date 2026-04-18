# middleware/timing.py
"""
Performance Timing Middleware.

Tracks and records request processing times.
Emits warnings for slow requests that exceed thresholds.

Metrics tracked:
- Per-endpoint response times
- Slow request detection (>500ms warning, >2000ms critical)
- Rolling average calculation
"""

import time
from collections import defaultdict, deque
from typing import Callable, Dict, Deque
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
from backend.utils.logger import api_logger


# Timing thresholds
SLOW_REQUEST_WARNING_MS = 500      # Log warning if request > 500ms
SLOW_REQUEST_CRITICAL_MS = 2000    # Log critical if request > 2000ms

# Rolling window size for averages
ROLLING_WINDOW_SIZE = 100


class TimingMiddleware(BaseHTTPMiddleware):
    """
    ASGI middleware for performance timing and slow request detection.

    Maintains a rolling window of response times per endpoint
    for real-time performance monitoring.

    Usage:
        app.add_middleware(TimingMiddleware)
    """

    def __init__(self, app: ASGIApp):
        super().__init__(app)

        # Rolling window of response times per route
        # route_key -> deque of millisecond timings
        self._timings: Dict[str, Deque[float]] = defaultdict(
            lambda: deque(maxlen=ROLLING_WINDOW_SIZE)
        )

        # Total request counters
        self._request_counts: Dict[str, int] = defaultdict(int)
        self._slow_request_counts: Dict[str, int] = defaultdict(int)

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Measure request processing time and detect slow requests.

        Args:
            request:   Incoming HTTP request
            call_next: Next middleware or route handler

        Returns:
            HTTP response
        """
        start_time = time.perf_counter()

        # Process request
        response = await call_next(request)

        # Calculate elapsed time
        elapsed_ms = (time.perf_counter() - start_time) * 1000

        # Build route key for tracking
        route_key = f"{request.method}:{request.url.path}"

        # Record timing
        self._timings[route_key].append(elapsed_ms)
        self._request_counts[route_key] += 1

        # ── Slow request detection ─────────────────────────────────────────
        if elapsed_ms > SLOW_REQUEST_CRITICAL_MS:
            self._slow_request_counts[route_key] += 1
            api_logger.critical(
                f"🔴 CRITICAL SLOW REQUEST: {route_key}"
                f" | {elapsed_ms:.0f}ms"
                f" (threshold: {SLOW_REQUEST_CRITICAL_MS}ms)"
            )

        elif elapsed_ms > SLOW_REQUEST_WARNING_MS:
            self._slow_request_counts[route_key] += 1
            api_logger.warning(
                f"🟡 Slow request: {route_key}"
                f" | {elapsed_ms:.0f}ms"
                f" (threshold: {SLOW_REQUEST_WARNING_MS}ms)"
            )

        # Add timing header
        response.headers["X-Response-Time-Ms"] = f"{elapsed_ms:.2f}"

        # Record performance metrics
        try:
            from backend.utils.monitoring import performance_monitor
            performance_monitor.record_request(route_key, elapsed_ms / 1000, response.status_code)
        except ImportError:
            # Monitoring not available yet during startup
            pass

        return response

    def get_stats(self) -> Dict:
        """
        Return timing statistics for all tracked endpoints.

        Returns:
            Dict mapping route keys to timing statistics
        """
        stats = {}

        for route_key, timings in self._timings.items():
            if not timings:
                continue

            timing_list = list(timings)

            stats[route_key] = {
                "count":         self._request_counts[route_key],
                "slow_count":    self._slow_request_counts[route_key],
                "avg_ms":        round(sum(timing_list) / len(timing_list), 2),
                "min_ms":        round(min(timing_list), 2),
                "max_ms":        round(max(timing_list), 2),
                "last_ms":       round(timing_list[-1], 2),
                "p95_ms":        round(self._percentile(timing_list, 95), 2),
                "p99_ms":        round(self._percentile(timing_list, 99), 2),
            }

        return stats

    def _percentile(self, data: list, percentile: int) -> float:
        """
        Calculate the Nth percentile of a list of values.

        Args:
            data:       List of numeric values
            percentile: Percentile to calculate (0-100)

        Returns:
            Percentile value
        """
        if not data:
            return 0.0

        sorted_data = sorted(data)
        index = int(len(sorted_data) * percentile / 100)
        index = min(index, len(sorted_data) - 1)

        return sorted_data[index]


# Global timing middleware instance (for stats access from API)
timing_middleware_instance: TimingMiddleware = None


def get_timing_stats() -> Dict:
    """
    Access timing stats from anywhere in the application.

    Returns:
        Timing stats dict or empty dict if middleware not initialized
    """
    if timing_middleware_instance:
        return timing_middleware_instance.get_stats()
    return {}