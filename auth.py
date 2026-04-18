"""
backend/api/middleware/auth.py
══════════════════════════════════════════════════════════════════
JWT-based authentication middleware for API security.

Features:
- JWT token generation and validation
- Role-based access control (admin, operator, viewer)
- API key authentication for system integrations
- Request rate limiting
- Security headers and CORS protection
══════════════════════════════════════════════════════════════════
"""

import os
import time
import jwt
import hashlib
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from fastapi import Request, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

# Security configuration
JWT_SECRET = os.getenv("JWT_SECRET", "your-secret-key-change-in-production")
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_HOURS = int(os.getenv("JWT_EXPIRATION_HOURS", "24"))

# API Key configuration
API_KEY_HEADER = "X-API-Key"
API_KEYS = {
    # Hash of actual API keys for security
    hashlib.sha256("admin-key-123".encode()).hexdigest(): {"role": "admin", "name": "Admin User"},
    hashlib.sha256("operator-key-456".encode()).hexdigest(): {"role": "operator", "name": "Operator"},
    hashlib.sha256("viewer-key-789".encode()).hexdigest(): {"role": "viewer", "name": "Viewer"},
}

# Rate limiting (simple in-memory implementation)
RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", "100"))
RATE_LIMIT_WINDOW = int(os.getenv("RATE_LIMIT_WINDOW", "60"))  # seconds
rate_limit_store: Dict[str, list] = {}

security = HTTPBearer(auto_error=False)

class AuthMiddleware(BaseHTTPMiddleware):
    """Authentication and rate limiting middleware"""

    async def dispatch(self, request: Request, call_next):
        from backend.utils.security import security_manager, audit_logger

        client_ip = request.client.host if request.client else "unknown"
        user_agent = request.headers.get("User-Agent", "unknown")

        # Enhanced rate limiting with security manager
        if not security_manager.check_rate_limit(f"ip:{client_ip}", client_ip):
            audit_logger.log_event(
                "security", "system", "api", "rate_limit_exceeded",
                {"path": request.url.path}, client_ip, user_agent,
                success=False, risk_level="MEDIUM"
            )
            return JSONResponse(
                status_code=429,
                content={"detail": "Rate limit exceeded. Try again later."}
            )

        # Input validation for security
        if request.method in ["POST", "PUT", "PATCH"]:
            try:
                body = await request.body()
                if body:
                    is_valid, error_msg = security_manager.validate_input(body.decode())
                    if not is_valid:
                        audit_logger.log_event(
                            "security", "system", "api", "invalid_input",
                            {"error": error_msg, "path": request.url.path},
                            client_ip, user_agent, success=False, risk_level="HIGH"
                        )
                        return JSONResponse(
                            status_code=400,
                            content={"detail": f"Invalid input: {error_msg}"}
                        )
            except Exception:
                pass  # Continue if body parsing fails

        # Add security headers
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Content-Security-Policy"] = "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'"

        # Audit successful requests
        if response.status_code < 400:
            audit_logger.log_event(
                "access", "system", request.url.path, request.method,
                {"status_code": response.status_code}, client_ip, user_agent,
                success=True, risk_level="LOW"
            )

        return response

def create_access_token(data: dict) -> str:
    """Create JWT access token"""
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(hours=JWT_EXPIRATION_HOURS)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, JWT_SECRET, algorithm=JWT_ALGORITHM)
    return encoded_jwt

def verify_token(token: str) -> Optional[Dict[str, Any]]:
    """Verify and decode JWT token"""
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token has expired")
    except jwt.JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

def verify_api_key(api_key: str) -> Optional[Dict[str, Any]]:
    """Verify API key"""
    key_hash = hashlib.sha256(api_key.encode()).hexdigest()
    return API_KEYS.get(key_hash)

async def get_current_user(
    request: Request,
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> Dict[str, Any]:
    """Dependency to get current authenticated user"""

    # Check JWT token first
    if credentials:
        token_data = verify_token(credentials.credentials)
        if token_data:
            return {
                "user_id": token_data.get("sub"),
                "role": token_data.get("role", "viewer"),
                "auth_type": "jwt"
            }

    # Check API key
    api_key = request.headers.get(API_KEY_HEADER)
    if api_key:
        key_data = verify_api_key(api_key)
        if key_data:
            return {
                "user_id": key_data["name"],
                "role": key_data["role"],
                "auth_type": "api_key"
            }

    raise HTTPException(status_code=401, detail="Authentication required")

def require_role(required_role: str):
    """Dependency factory for role-based access control"""

    def role_checker(current_user: Dict = Depends(get_current_user)):
        role_hierarchy = {
            "admin": 3,
            "operator": 2,
            "viewer": 1
        }

        user_level = role_hierarchy.get(current_user["role"], 0)
        required_level = role_hierarchy.get(required_role, 999)

        if user_level < required_level:
            raise HTTPException(
                status_code=403,
                detail=f"Insufficient permissions. Required role: {required_role}"
            )

        return current_user

    return role_checker

# Predefined role dependencies
require_admin = require_role("admin")
require_operator = require_role("operator")
require_viewer = require_role("viewer")

# Auth router for login endpoints
from fastapi import APIRouter

auth_router = APIRouter(prefix="/auth", tags=["authentication"])

@auth_router.post("/login")
async def login(request: Request, username: str, password: str):
    """Simple login endpoint (replace with proper user management)"""
    from backend.utils.security import security_manager, audit_logger

    client_ip = request.client.host if request.client else "unknown"
    user_agent = request.headers.get("User-Agent", "unknown")

    # Check if login is blocked
    if not security_manager.check_failed_login(username, client_ip):
        audit_logger.log_event(
            "auth", username, "login", "blocked",
            {"reason": "too_many_failed_attempts"}, client_ip, user_agent,
            success=False, risk_level="HIGH"
        )
        raise HTTPException(status_code=429, detail="Account temporarily locked due to failed attempts")

    # Mock user validation (replace with database lookup)
    users = {
        "admin": {"password": "admin123", "role": "admin"},
        "operator": {"password": "op123", "role": "operator"},
        "viewer": {"password": "view123", "role": "viewer"}
    }

    user = users.get(username)
    if not user or user["password"] != password:
        security_manager.record_failed_login(username, client_ip)
        audit_logger.log_event(
            "auth", username, "login", "failed",
            {"reason": "invalid_credentials"}, client_ip, user_agent,
            success=False, risk_level="MEDIUM"
        )
        raise HTTPException(status_code=401, detail="Invalid credentials")

    # Successful login
    security_manager.record_successful_login(username, client_ip)
    audit_logger.log_event(
        "auth", username, "login", "success",
        {"role": user["role"]}, client_ip, user_agent,
        success=True, risk_level="LOW"
    )

    access_token = create_access_token({
        "sub": username,
        "role": user["role"]
    })

    return {
        "access_token": access_token,
        "token_type": "bearer",
        "expires_in": JWT_EXPIRATION_HOURS * 3600,
        "user": {
            "username": username,
            "role": user["role"]
        }
    }

@auth_router.get("/me")
async def get_current_user_info(current_user: Dict = Depends(get_current_user)):
    """Get current user information"""
    return current_user

@auth_router.post("/refresh")
async def refresh_token(current_user: Dict = Depends(get_current_user)):
    """Refresh access token"""
    access_token = create_access_token({
        "sub": current_user["user_id"],
        "role": current_user["role"]
    })

    return {
        "access_token": access_token,
        "token_type": "bearer"
    }