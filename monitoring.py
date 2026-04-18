# Digital Twin AI System — Monitoring & Metrics

"""
Comprehensive monitoring and observability for the Digital Twin AI System.
Provides metrics, health checks, and performance monitoring.
"""

import time
import psutil
import asyncio
from typing import Dict, Any, List
from datetime import datetime, timedelta
from collections import defaultdict, deque

from backend.utils.logger import app_logger


class MetricsCollector:
    """Collects and exposes system and application metrics."""

    def __init__(self):
        self.metrics = defaultdict(dict)
        self.counters = defaultdict(int)
        self.histograms = defaultdict(lambda: deque(maxlen=1000))
        self.gauges = {}
        self.start_time = time.time()

    def increment_counter(self, name: str, value: int = 1, tags: Dict[str, str] = None):
        """Increment a counter metric."""
        key = f"{name}_{str(tags) if tags else ''}"
        self.counters[key] += value

    def set_gauge(self, name: str, value: float, tags: Dict[str, str] = None):
        """Set a gauge metric."""
        key = f"{name}_{str(tags) if tags else ''}"
        self.gauges[key] = value

    def record_histogram(self, name: str, value: float, tags: Dict[str, str] = None):
        """Record a histogram value."""
        key = f"{name}_{str(tags) if tags else ''}"
        self.histograms[key].append(value)

    def get_system_metrics(self) -> Dict[str, Any]:
        """Get system-level metrics."""
        return {
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory_percent": psutil.virtual_memory().percent,
            "memory_used_mb": psutil.virtual_memory().used / 1024 / 1024,
            "disk_usage_percent": psutil.disk_usage('/').percent,
            "uptime_seconds": time.time() - self.start_time,
            "process_count": len(psutil.pids())
        }

    def get_application_metrics(self) -> Dict[str, Any]:
        """Get application-level metrics."""
        return {
            "counters": dict(self.counters),
            "gauges": dict(self.gauges),
            "histograms": {
                name: {
                    "count": len(values),
                    "avg": sum(values) / len(values) if values else 0,
                    "min": min(values) if values else 0,
                    "max": max(values) if values else 0
                }
                for name, values in self.histograms.items()
            }
        }

    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all metrics."""
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "system": self.get_system_metrics(),
            "application": self.get_application_metrics()
        }


class HealthChecker:
    """Comprehensive health checking for all system components."""

    def __init__(self):
        self.checks = {}
        self.last_results = {}

    def register_check(self, name: str, check_func, interval_seconds: int = 60):
        """Register a health check function."""
        self.checks[name] = {
            "func": check_func,
            "interval": interval_seconds,
            "last_run": 0
        }

    async def run_check(self, name: str) -> Dict[str, Any]:
        """Run a specific health check."""
        if name not in self.checks:
            return {"status": "error", "message": f"Check {name} not found"}

        check_info = self.checks[name]

        # Check if we should run this check
        now = time.time()
        if now - check_info["last_run"] < check_info["interval"]:
            return self.last_results.get(name, {"status": "unknown"})

        try:
            result = await check_info["func"]()
            check_info["last_run"] = now
            self.last_results[name] = result
            return result
        except Exception as e:
            result = {
                "status": "error",
                "message": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
            check_info["last_run"] = now
            self.last_results[name] = result
            return result

    async def run_all_checks(self) -> Dict[str, Any]:
        """Run all registered health checks."""
        results = {}
        for name in self.checks:
            results[name] = await self.run_check(name)

        # Overall health status
        statuses = [result.get("status", "unknown") for result in results.values()]
        overall_status = "healthy" if all(s == "healthy" for s in statuses) else "unhealthy"

        return {
            "overall_status": overall_status,
            "timestamp": datetime.utcnow().isoformat(),
            "checks": results
        }


class PerformanceMonitor:
    """Monitor application performance and bottlenecks."""

    def __init__(self):
        self.request_times = deque(maxlen=1000)
        self.endpoint_stats = defaultdict(lambda: {"count": 0, "total_time": 0, "errors": 0})
        self.slow_requests = deque(maxlen=100)

    def record_request(self, endpoint: str, duration: float, status_code: int):
        """Record an API request."""
        self.request_times.append(duration)
        self.endpoint_stats[endpoint]["count"] += 1
        self.endpoint_stats[endpoint]["total_time"] += duration

        if status_code >= 400:
            self.endpoint_stats[endpoint]["errors"] += 1

        # Track slow requests (>1 second)
        if duration > 1.0:
            self.slow_requests.append({
                "endpoint": endpoint,
                "duration": duration,
                "timestamp": datetime.utcnow().isoformat()
            })

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        if not self.request_times:
            return {"message": "No requests recorded yet"}

        return {
            "total_requests": len(self.request_times),
            "avg_response_time": sum(self.request_times) / len(self.request_times),
            "min_response_time": min(self.request_times),
            "max_response_time": max(self.request_times),
            "p95_response_time": sorted(self.request_times)[int(len(self.request_times) * 0.95)],
            "p99_response_time": sorted(self.request_times)[int(len(self.request_times) * 0.99)],
            "endpoint_stats": dict(self.endpoint_stats),
            "slow_requests": list(self.slow_requests)[-10:]  # Last 10 slow requests
        }


# Global instances
metrics = MetricsCollector()
health_checker = HealthChecker()
performance_monitor = PerformanceMonitor()


async def database_health_check():
    """Check database connectivity and performance."""
    from backend.database.db_manager import db

    try:
        start_time = time.time()
        # Simple query to test connectivity
        result = await db.execute_query("SELECT 1 as test")
        query_time = time.time() - start_time

        return {
            "status": "healthy",
            "response_time": query_time,
            "message": "Database connection successful"
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "message": f"Database error: {str(e)}"
        }


async def websocket_health_check():
    """Check WebSocket connections."""
    from backend.websocket.connection_manager import manager

    try:
        connection_count = manager.get_connection_count()
        return {
            "status": "healthy",
            "active_connections": connection_count,
            "message": f"WebSocket healthy with {connection_count} connections"
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "message": f"WebSocket error: {str(e)}"
        }


async def ai_engine_health_check():
    """Check AI engine status."""
    try:
        from backend.ai_engine.digital_twin import digital_twins

        healthy_engines = 0
        total_engines = len(digital_twins)

        for twin in digital_twins.values():
            if hasattr(twin, 'is_healthy') and twin.is_healthy():
                healthy_engines += 1

        status = "healthy" if healthy_engines == total_engines else "degraded"

        return {
            "status": status,
            "healthy_engines": healthy_engines,
            "total_engines": total_engines,
            "message": f"AI engines: {healthy_engines}/{total_engines} healthy"
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "message": f"AI engine error: {str(e)}"
        }


def init_monitoring():
    """Initialize all monitoring components."""
    # Register health checks
    health_checker.register_check("database", database_health_check, 30)
    health_checker.register_check("websocket", websocket_health_check, 60)
    health_checker.register_check("ai_engine", ai_engine_health_check, 60)

    app_logger.info("Monitoring system initialized")