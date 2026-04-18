# middleware/logging_middleware.py
"""
HTTP Request/Response Logging Middleware.

Logs every incoming request and outgoing response with:
- Method, path, query params
- Status code
- Response time
- Client IP
- Request ID (UUID for tracing)

Skips health check endpoints to reduce noise.
"""

import time
import uuid
from typing import Callable
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
from backend.utils.logger import api_logger


# Paths to skip logging (too noisy / frequent)
SKIP_LOGGING_PATHS = {
    "/health",
    "/ping",
    "/favicon.ico",
    "/docs",
    "/openapi.json",
    "/redoc"
}


class LoggingMiddleware(BaseHTTPMiddleware):
    """
    ASGI middleware that logs all HTTP requests and responses.

    Adds X-Request-ID header to every response for distributed tracing.
    Measures and logs processing time in milliseconds.

    Usage:
        app.add_middleware(LoggingMiddleware)
    """

    def __init__(self, app: ASGIApp):
        super().__init__(app)

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Intercept request → process → log → return response.

        Args:
            request:   Incoming HTTP request
            call_next: Next middleware or route handler

        Returns:
            HTTP response with added X-Request-ID header
        """

        # Generate unique request ID for tracing
        request_id = str(uuid.uuid4())[:8]

        # Skip noisy health-check endpoints
        if request.url.path in SKIP_LOGGING_PATHS:
            response = await call_next(request)
            return response

        # ── Pre-request logging ────────────────────────────────────────────
        client_ip = self._get_client_ip(request)
        start_time = time.perf_counter()

        api_logger.info(
            f"→ [{request_id}] {request.method} {request.url.path}"
            f"{self._format_query(request)}"
            f" | IP: {client_ip}"
        )

        # ── Process request ────────────────────────────────────────────────
        try:
            response = await call_next(request)
            status_code = response.status_code

        except Exception as exc:
            elapsed = (time.perf_counter() - start_time) * 1000
            api_logger.error(
                f"✗ [{request_id}] {request.method} {request.url.path}"
                f" | ERROR: {exc}"
                f" | {elapsed:.1f}ms"
            )
            raise

        # ── Post-response logging ──────────────────────────────────────────
        elapsed_ms = (time.perf_counter() - start_time) * 1000

        log_fn = (
            api_logger.warning if status_code >= 400
            else api_logger.info
        )

        log_fn(
            f"← [{request_id}] {request.method} {request.url.path}"
            f" | {status_code}"
            f" | {elapsed_ms:.1f}ms"
        )

        # ── Add tracing headers ────────────────────────────────────────────
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time"] = f"{elapsed_ms:.2f}ms"

        return response

    def _get_client_ip(self, request: Request) -> str:
        """
        Extract real client IP, respecting X-Forwarded-For proxy header.

        Args:
            request: Incoming request

        Returns:
            IP address string
        """
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            # X-Forwarded-For: client, proxy1, proxy2
            return forwarded_for.split(",")[0].strip()

        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip

        if request.client:
            return request.client.host

        return "unknown"

    def _format_query(self, request: Request) -> str:
        """
        Format query string for logging.

        Args:
            request: Incoming request

        Returns:
            Formatted query string or empty string
        """
        query = str(request.url.query)
        if query:
            return f"?{query}"
        return ""