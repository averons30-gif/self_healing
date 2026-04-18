# middleware/cors.py
"""
CORS (Cross-Origin Resource Sharing) Configuration.

Allows the Digital Twin frontend (React/Vue/etc.) to connect
to the FastAPI backend from different origins during development
and production.

Supports:
- WebSocket upgrades
- REST API calls
- Preflight OPTIONS requests
- Credential-based requests
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.utils.config import config
from backend.utils.logger import app_logger


def setup_cors(app: FastAPI) -> None:
    """
    Attach CORS middleware to the FastAPI application.

    Args:
        app: The FastAPI application instance

    Configuration is driven by config.CORS_ORIGINS which
    can be set via the CORS_ORIGINS environment variable.

    Example .env:
        CORS_ORIGINS=http://localhost:3000,https://mydashboard.com
    """

    origins = config.CORS_ORIGINS

    app_logger.info("🌐 Setting up CORS middleware...")
    app_logger.info(f"   Allowed origins: {origins}")

    app.add_middleware(
        CORSMiddleware,

        # ── Origins ────────────────────────────────────────────────────────
        # List of allowed origins.
        # Use ["*"] only in development — never in production with credentials.
        allow_origins=origins,

        # ── Credentials ────────────────────────────────────────────────────
        # Allow cookies and Authorization headers.
        # Must be False if allow_origins=["*"]
        allow_credentials=True,

        # ── Methods ────────────────────────────────────────────────────────
        # Allow all HTTP methods (GET, POST, PUT, DELETE, OPTIONS, PATCH)
        allow_methods=["*"],

        # ── Headers ────────────────────────────────────────────────────────
        # Allow all request headers including custom ones.
        # WebSocket needs: Upgrade, Connection, Sec-WebSocket-*
        allow_headers=["*"],

        # ── Expose Headers ─────────────────────────────────────────────────
        # Headers the browser JS can access from the response.
        expose_headers=[
            "X-Request-ID",
            "X-Process-Time",
            "X-Total-Count",
            "Content-Range"
        ],

        # ── Preflight Cache ────────────────────────────────────────────────
        # Cache preflight OPTIONS response for 1 hour (3600 seconds).
        # Reduces preflight overhead for frequent API calls.
        max_age=3600,
    )

    app_logger.info("✅ CORS middleware configured")