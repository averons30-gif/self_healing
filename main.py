# main.py  ── shows how all middleware connects
"""
Digital Twin AI System — Application Entry Point.

Startup sequence:
1. Create FastAPI app
2. Attach middleware (CORS → Logging → Timing → Errors)
3. Initialize database
4. Boot Digital Twin AI engines
5. Start WebSocket sensor streams
6. Mount API routers
"""

import asyncio
import uuid
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

# ── Middleware ─────────────────────────────────────────────────────────────────
from backend.api.middleware.cors import setup_cors
from backend.api.middleware.logging_middleware import LoggingMiddleware
from backend.api.middleware.timing import TimingMiddleware, timing_middleware_instance
from backend.api.middleware.error_handler import setup_error_handlers

# ── Core modules ───────────────────────────────────────────────────────────────
from backend.database.db_manager import db
from backend.ai_engine.digital_twin import DigitalTwinEngine
from backend.websocket.stream_handler import StreamHandler
from backend.websocket.connection_manager import manager
from backend.utils.config import config
from backend.utils.logger import app_logger

# ── API Routers ────────────────────────────────────────────────────────────────
from backend.api.routes import machines, alerts, analytics, maintenance, predictions
from backend.api.routes import alerts_management, anomalies, health_trends, performance


# ── Global state ──────────────────────────────────────────────────────────────
digital_twins = {}
stream_handler: StreamHandler = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    Handles startup and graceful shutdown.
    """
    global digital_twins, stream_handler

    # ────────────────────────── STARTUP ───────────────────────────────────
    app_logger.info("=" * 60)
    app_logger.info("🤖 Digital Twin AI System — Starting Up")
    app_logger.info("=" * 60)

    # 1. Initialize database
    app_logger.info("📦 Initializing database...")
    await db.initialize()
    await db.log_system_event("startup", "System starting up", {"version": config.API_VERSION})

    # 2. Boot Digital Twin AI engines for each machine
    app_logger.info("🧠 Booting Digital Twin AI engines...")
    for machine_id in config.MACHINE_IDS:
        twin = DigitalTwinEngine(machine_id)
        await twin.initialize()
        digital_twins[machine_id] = twin
        app_logger.info(f"  ✅ Twin ready: {machine_id}")

    # 3. Start sensor streams
    app_logger.info("📡 Starting sensor streams...")
    stream_handler = StreamHandler(digital_twins)
    
    # Set stream_handler in route modules
    machines.set_stream_handler(stream_handler)
    alerts.set_stream_handler(stream_handler)
    if hasattr(analytics, 'set_stream_handler'):
        analytics.set_stream_handler(stream_handler)
    if hasattr(maintenance, 'set_stream_handler'):
        maintenance.set_stream_handler(stream_handler)
    
    await stream_handler.start_all_streams()

    app_logger.info("=" * 60)
    app_logger.info(f"🚀 System running on http://{config.HOST}:{config.PORT}")
    app_logger.info(f"📊 Dashboard: http://{config.HOST}:{config.PORT}/docs")
    app_logger.info("=" * 60)

    yield  # ← Application runs here

    # ────────────────────────── SHUTDOWN ──────────────────────────────────
    app_logger.info("⏹ Shutting down Digital Twin AI System...")

    if stream_handler:
        await stream_handler.stop_all_streams()

    await db.log_system_event("shutdown", "System shutting down gracefully")
    app_logger.info("✅ Shutdown complete")


def create_app() -> FastAPI:
    """
    Application factory — creates and configures the FastAPI app.

    Returns:
        Fully configured FastAPI application
    """
    app = FastAPI(
        title=config.API_TITLE,
        version=config.API_VERSION,
        description=config.API_DESCRIPTION,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json"
    )

    # ── 1. Authentication & Security ──────────────────────────────────────
    from backend.api.middleware.auth import AuthMiddleware
    app.add_middleware(AuthMiddleware)

    # ── 2. CORS (must be after auth middleware) ────────────────────────────────
    setup_cors(app)

    # ── 3. Request/Response Logging ───────────────────────────────────────
    app.add_middleware(LoggingMiddleware)

    # ── 4. Performance Timing ─────────────────────────────────────────────
    timing_mw = TimingMiddleware(app)
    app.add_middleware(TimingMiddleware)

    # ── 8. Security & Compliance ─────────────────────────────────────
    from backend.utils.security import init_security_compliance
    init_security_compliance()

    # ── 6. API Routers ────────────────────────────────────────────────────
    from backend.api.middleware.auth import auth_router
    app.include_router(auth_router, prefix="/api/v1")  # Authentication endpoints
    app.include_router(machines.router,         prefix="/api/v1")
    app.include_router(alerts.router,           prefix="/api/v1")
    app.include_router(alerts_management.router, prefix="/api/v1")
    app.include_router(analytics.router,        prefix="/api/v1")
    app.include_router(maintenance.router,      prefix="/api/v1")
    app.include_router(predictions.router,      prefix="/api/v1")
    app.include_router(anomalies.router,        prefix="/api/v1")
    app.include_router(health_trends.router,    prefix="/api/v1")
    app.include_router(performance.router,      prefix="/api/v1")

    # ── 6. Root endpoint ──────────────────────────────────────────────────
    @app.get("/", tags=["Root"])
    async def root():
        return {
            "system":   "Digital Twin AI System",
            "version":  config.API_VERSION,
            "status":   "operational",
            "machines": len(digital_twins),
            "docs":     "/docs"
        }

    @app.get("/health", tags=["Health"])
    async def health_check():
        return {
            "status":       "healthy",
            "machines":     len(digital_twins),
            "streams":      stream_handler.get_stats() if stream_handler else {},
            "connections":  manager.get_connection_count(),
        }

    @app.get("/health/detailed", tags=["Health"])
    async def detailed_health_check():
        """Comprehensive health check of all system components."""
        from backend.utils.monitoring import health_checker
        return await health_checker.run_all_checks()

    @app.get("/metrics", tags=["Metrics"])
    async def get_metrics():
        """Get system and application metrics."""
        from backend.utils.monitoring import metrics
        return metrics.get_all_metrics()

    @app.get("/performance", tags=["Metrics"])
    async def get_performance_stats():
        """Get application performance statistics."""
        from backend.utils.monitoring import performance_monitor
        return performance_monitor.get_performance_stats()

    @app.get("/security/status", tags=["Security"])
    async def get_security_status():
        """Get current security status."""
        from backend.utils.security import security_manager
        return security_manager.get_security_status()

    @app.get("/audit/events", tags=["Security"])
    async def get_audit_events(limit: int = 100, event_type: str = None):
        """Get audit events with optional filtering."""
        from backend.utils.security import audit_logger
        filters = {}
        if event_type:
            filters["event_type"] = event_type
        return audit_logger.get_events(filters, limit)

    @app.get("/audit/summary", tags=["Security"])
    async def get_audit_summary():
        """Get audit events summary."""
        from backend.utils.security import audit_logger
        return audit_logger.get_security_summary()

    @app.get("/compliance/report", tags=["Compliance"])
    async def get_compliance_report():
        """Get compliance report."""
        from backend.utils.security import compliance_manager
        return compliance_manager.generate_compliance_report()

    @app.post("/compliance/check", tags=["Compliance"])
    async def run_compliance_check():
        """Run all compliance checks."""
        from backend.utils.security import compliance_manager
        return await compliance_manager.run_all_checks()

    @app.websocket("/ws/stream")
    async def websocket_endpoint(websocket: WebSocket, machine_id: str = None):
        """
        WebSocket endpoint for real-time sensor stream subscription.
        Clients connect here to receive live machine data updates.

        Query parameters:
        - machine_id: Optional machine ID to subscribe to specific machine
        """
        client_id = str(uuid.uuid4())

        try:
            # Pass the websocket connection to the stream handler
            await stream_handler.handle_websocket(websocket, client_id, machine_id)
            app_logger.info(f"📡 WebSocket client connected: {client_id}")

        except WebSocketDisconnect:
            manager.disconnect(client_id)
            app_logger.info(f"📡 WebSocket client disconnected: {client_id}")
        except Exception as e:
            app_logger.error(f"❌ WebSocket error [{client_id}]: {e}")
            manager.disconnect(client_id)

    return app


# ── Application instance ───────────────────────────────────────────────────────
app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=config.HOST,
        port=config.PORT,
        reload=config.DEBUG,
        log_level=config.LOG_LEVEL.lower(),
        ws_ping_interval=20,
        ws_ping_timeout=10
    )