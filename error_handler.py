# middleware/error_handler.py
"""
Global Error Handler Middleware.

Catches all unhandled exceptions and returns consistent
JSON error responses instead of raw stack traces.

Error Response Format:
{
    "error": true,
    "code": 500,
    "message": "Internal server error",
    "detail": "...",
    "request_id": "abc123",
    "timestamp": "2024-01-01T00:00:00"
}

Handles:
- HTTP exceptions (404, 422, etc.)
- Validation errors (Pydantic)
- Unhandled exceptions (500)
- WebSocket errors
"""

import traceback
import uuid
from datetime import datetime
from typing import Union
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from backend.utils.logger import app_logger


def setup_error_handlers(app: FastAPI) -> None:
    """
    Register all global error handlers on the FastAPI app.

    Args:
        app: The FastAPI application instance
    """

    # ── HTTP Exception Handler ─────────────────────────────────────────────
    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(
        request: Request,
        exc: StarletteHTTPException
    ) -> JSONResponse:
        """
        Handle standard HTTP errors (404, 403, 401, etc.)
        """
        request_id = str(uuid.uuid4())[:8]

        app_logger.warning(
            f"HTTP {exc.status_code} | "
            f"[{request_id}] {request.method} {request.url.path} | "
            f"{exc.detail}"
        )

        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error":        True,
                "code":         exc.status_code,
                "message":      _get_status_message(exc.status_code),
                "detail":       str(exc.detail),
                "path":         str(request.url.path),
                "method":       request.method,
                "request_id":   request_id,
                "timestamp":    datetime.utcnow().isoformat()
            }
        )

    # ── Validation Error Handler ───────────────────────────────────────────
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request,
        exc: RequestValidationError
    ) -> JSONResponse:
        """
        Handle Pydantic validation errors (422 Unprocessable Entity).
        Returns detailed field-level error information.
        """
        request_id = str(uuid.uuid4())[:8]

        # Extract clean error messages
        errors = []
        for error in exc.errors():
            errors.append({
                "field":    " → ".join(str(loc) for loc in error["loc"]),
                "message":  error["msg"],
                "type":     error["type"]
            })

        app_logger.warning(
            f"Validation error | "
            f"[{request_id}] {request.method} {request.url.path} | "
            f"{len(errors)} field error(s)"
        )

        return JSONResponse(
            status_code=422,
            content={
                "error":        True,
                "code":         422,
                "message":      "Request validation failed",
                "detail":       "One or more fields failed validation",
                "errors":       errors,
                "path":         str(request.url.path),
                "request_id":   request_id,
                "timestamp":    datetime.utcnow().isoformat()
            }
        )

    # ── Generic Exception Handler ──────────────────────────────────────────
    @app.exception_handler(Exception)
    async def generic_exception_handler(
        request: Request,
        exc: Exception
    ) -> JSONResponse:
        """
        Catch-all handler for unhandled exceptions.
        Returns 500 with sanitized error info (no stack traces to client).
        """
        request_id = str(uuid.uuid4())[:8]

        # Log full traceback server-side
        app_logger.error(
            f"🔴 Unhandled exception | "
            f"[{request_id}] {request.method} {request.url.path}\n"
            f"{traceback.format_exc()}"
        )

        return JSONResponse(
            status_code=500,
            content={
                "error":        True,
                "code":         500,
                "message":      "Internal server error",
                "detail":       (
                    str(exc)
                    if _is_safe_to_expose(exc)
                    else "An unexpected error occurred"
                ),
                "path":         str(request.url.path),
                "request_id":   request_id,
                "timestamp":    datetime.utcnow().isoformat()
            }
        )

    app_logger.info("✅ Global error handlers registered")


def _get_status_message(status_code: int) -> str:
    """
    Map HTTP status codes to human-readable messages.

    Args:
        status_code: HTTP status code integer

    Returns:
        Human-readable status message
    """
    messages = {
        400: "Bad request",
        401: "Authentication required",
        403: "Access forbidden",
        404: "Resource not found",
        405: "Method not allowed",
        408: "Request timeout",
        409: "Resource conflict",
        410: "Resource gone",
        422: "Unprocessable entity",
        429: "Too many requests",
        500: "Internal server error",
        502: "Bad gateway",
        503: "Service unavailable",
        504: "Gateway timeout"
    }
    return messages.get(status_code, "Unknown error")


def _is_safe_to_expose(exc: Exception) -> bool:
    """
    Determine if exception message is safe to expose to clients.
    Avoids leaking internal implementation details.

    Args:
        exc: The exception instance

    Returns:
        True if safe to include in response
    """
    # These exception types have user-friendly messages
    safe_types = (
        ValueError,
        KeyError,
        TypeError,
        LookupError,
    )
    return isinstance(exc, safe_types)