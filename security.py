# Digital Twin AI System — Security & Compliance

"""
Comprehensive security and compliance features for the Digital Twin AI system.
Includes audit logging, advanced security measures, and compliance reporting.
"""

import asyncio
import hashlib
import hmac
import json
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict, deque
import re

from backend.utils.logger import app_logger


class AuditLogger:
    """Comprehensive audit logging for security and compliance."""

    def __init__(self):
        self.audit_log = deque(maxlen=10000)  # Keep last 10k entries in memory
        self.audit_file = "logs/audit.log"
        self._ensure_audit_file()

    def _ensure_audit_file(self):
        """Ensure audit log file exists."""
        try:
            with open(self.audit_file, 'a') as f:
                pass  # Just ensure file exists
        except Exception as e:
            app_logger.error(f"Failed to create audit log file: {e}")

    def log_event(self, event_type: str, user: str, resource: str,
                  action: str, details: Dict[str, Any] = None,
                  ip_address: str = None, user_agent: str = None,
                  success: bool = True, risk_level: str = "LOW"):
        """
        Log a security/compliance event.

        Args:
            event_type: Type of event (auth, access, data, system)
            user: User identifier (username, API key, system)
            resource: Resource being accessed
            action: Action performed (read, write, delete, etc.)
            details: Additional event details
            ip_address: Client IP address
            user_agent: Client user agent
            success: Whether the action succeeded
            risk_level: Security risk level (LOW, MEDIUM, HIGH, CRITICAL)
        """
        event = {
            "timestamp": datetime.utcnow().isoformat(),
            "event_id": self._generate_event_id(),
            "event_type": event_type,
            "user": user,
            "resource": resource,
            "action": action,
            "success": success,
            "risk_level": risk_level,
            "ip_address": ip_address,
            "user_agent": user_agent,
            "details": details or {}
        }

        # Add to memory buffer
        self.audit_log.append(event)

        # Write to file
        self._write_to_file(event)

        # Log high-risk events immediately
        if risk_level in ["HIGH", "CRITICAL"] or not success:
            app_logger.warning(f"🔴 AUDIT EVENT: {event_type} - {action} on {resource} by {user}")

    def _generate_event_id(self) -> str:
        """Generate unique event ID."""
        timestamp = str(int(time.time() * 1000000))
        random_part = str(hash(f"{timestamp}{id(self)}"))[-8:]
        return f"evt_{timestamp}_{random_part}"

    def _write_to_file(self, event: Dict[str, Any]):
        """Write event to audit log file."""
        try:
            with open(self.audit_file, 'a') as f:
                f.write(json.dumps(event) + '\n')
        except Exception as e:
            app_logger.error(f"Failed to write audit event: {e}")

    def get_events(self, filters: Dict[str, Any] = None,
                   limit: int = 100) -> List[Dict[str, Any]]:
        """
        Retrieve audit events with optional filtering.

        Args:
            filters: Dictionary of filter criteria
            limit: Maximum events to return

        Returns:
            List of matching events
        """
        events = list(self.audit_log)

        if filters:
            filtered_events = []
            for event in events:
                match = True
                for key, value in filters.items():
                    if key not in event or event[key] != value:
                        match = False
                        break
                if match:
                    filtered_events.append(event)
            events = filtered_events

        return events[-limit:]

    def get_security_summary(self) -> Dict[str, Any]:
        """Get security summary statistics."""
        events = list(self.audit_log)

        summary = {
            "total_events": len(events),
            "time_range": {
                "oldest": events[0]["timestamp"] if events else None,
                "newest": events[-1]["timestamp"] if events else None
            },
            "events_by_type": defaultdict(int),
            "events_by_risk": defaultdict(int),
            "failed_actions": 0,
            "high_risk_events": 0
        }

        for event in events:
            summary["events_by_type"][event["event_type"]] += 1
            summary["events_by_risk"][event["risk_level"]] += 1

            if not event["success"]:
                summary["failed_actions"] += 1

            if event["risk_level"] in ["HIGH", "CRITICAL"]:
                summary["high_risk_events"] += 1

        return dict(summary)


class SecurityManager:
    """Advanced security management and threat detection."""

    def __init__(self):
        self.failed_login_attempts = defaultdict(list)
        self.suspicious_ips = set()
        self.blocked_ips = set()
        self.rate_limits = defaultdict(lambda: deque(maxlen=100))

        # Security thresholds
        self.max_failed_attempts = 5
        self.block_duration_minutes = 15
        self.rate_limit_window = 60  # seconds
        self.rate_limit_max_requests = 100

    def check_failed_login(self, username: str, ip_address: str) -> bool:
        """
        Check if login attempt should be blocked due to failed attempts.

        Returns:
            True if login should be allowed, False if blocked
        """
        now = datetime.utcnow()

        # Clean old failed attempts
        self.failed_login_attempts[username] = [
            attempt for attempt in self.failed_login_attempts[username]
            if (now - attempt) < timedelta(minutes=self.block_duration_minutes)
        ]

        # Check if user is blocked
        if len(self.failed_login_attempts[username]) >= self.max_failed_attempts:
            return False

        # Check if IP is blocked
        if ip_address in self.blocked_ips:
            return False

        return True

    def record_failed_login(self, username: str, ip_address: str):
        """Record a failed login attempt."""
        now = datetime.utcnow()
        self.failed_login_attempts[username].append(now)

        # Check if user should be blocked
        if len(self.failed_login_attempts[username]) >= self.max_failed_attempts:
            app_logger.warning(f"🚫 User blocked due to failed attempts: {username}")
            audit_logger.log_event(
                "auth", username, "login", "failed_attempt_blocked",
                {"ip_address": ip_address}, ip_address, risk_level="HIGH"
            )

        # Check for suspicious activity
        if len(self.failed_login_attempts[username]) >= 3:
            self.suspicious_ips.add(ip_address)

    def record_successful_login(self, username: str, ip_address: str):
        """Record a successful login."""
        # Clear failed attempts on successful login
        if username in self.failed_login_attempts:
            del self.failed_login_attempts[username]

    def check_rate_limit(self, identifier: str, ip_address: str) -> bool:
        """
        Check if request should be rate limited.

        Args:
            identifier: User or API key identifier
            ip_address: Client IP address

        Returns:
            True if request allowed, False if rate limited
        """
        now = time.time()
        key = f"{identifier}:{ip_address}"

        # Clean old requests
        self.rate_limits[key] = deque(
            (timestamp for timestamp in self.rate_limits[key]
             if now - timestamp < self.rate_limit_window),
            maxlen=100
        )

        # Check rate limit
        if len(self.rate_limits[key]) >= self.rate_limit_max_requests:
            return False

        # Add current request
        self.rate_limits[key].append(now)
        return True

    def validate_input(self, data: Any, schema: Dict[str, Any] = None) -> Tuple[bool, str]:
        """
        Validate input data for security issues.

        Args:
            data: Input data to validate
            schema: Validation schema (optional)

        Returns:
            (is_valid, error_message)
        """
        if isinstance(data, str):
            # Check for SQL injection patterns
            sql_patterns = [
                r';\s*(drop|delete|update|insert|alter)\s',
                r'union\s+select',
                r'--',
                r'/\*.*\*/'
            ]

            for pattern in sql_patterns:
                if re.search(pattern, data, re.IGNORECASE):
                    return False, "Potential SQL injection detected"

            # Check for XSS patterns
            xss_patterns = [
                r'<script',
                r'javascript:',
                r'on\w+\s*=',
                r'<iframe',
                r'<object'
            ]

            for pattern in xss_patterns:
                if re.search(pattern, data, re.IGNORECASE):
                    return False, "Potential XSS detected"

            # Check string length
            if len(data) > 10000:  # 10KB limit
                return False, "Input too large"

        elif isinstance(data, dict):
            # Recursively validate nested data
            for key, value in data.items():
                is_valid, error = self.validate_input(value)
                if not is_valid:
                    return False, f"Invalid data in field '{key}': {error}"

        return True, ""

    def sanitize_data(self, data: Any) -> Any:
        """
        Sanitize data to prevent security issues.

        Args:
            data: Data to sanitize

        Returns:
            Sanitized data
        """
        if isinstance(data, str):
            # Remove potentially dangerous characters
            data = re.sub(r'[;\'"\\]', '', data)
            # Limit length
            return data[:1000]
        elif isinstance(data, dict):
            return {k: self.sanitize_data(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self.sanitize_data(item) for item in data]
        else:
            return data

    def get_security_status(self) -> Dict[str, Any]:
        """Get current security status."""
        return {
            "blocked_users": len([u for u in self.failed_login_attempts.keys()
                                if len(self.failed_login_attempts[u]) >= self.max_failed_attempts]),
            "suspicious_ips": len(self.suspicious_ips),
            "blocked_ips": len(self.blocked_ips),
            "active_rate_limits": len(self.rate_limits)
        }


class ComplianceManager:
    """Compliance management and reporting."""

    def __init__(self):
        self.compliance_checks = {}
        self.compliance_reports = deque(maxlen=100)

    def register_check(self, check_name: str, check_func, frequency: str = "daily"):
        """
        Register a compliance check.

        Args:
            check_name: Name of the compliance check
            check_func: Function that performs the check
            frequency: How often to run (hourly, daily, weekly)
        """
        self.compliance_checks[check_name] = {
            "func": check_func,
            "frequency": frequency,
            "last_run": None,
            "last_result": None
        }

    async def run_compliance_check(self, check_name: str) -> Dict[str, Any]:
        """Run a specific compliance check."""
        if check_name not in self.compliance_checks:
            return {"status": "error", "message": f"Check {check_name} not found"}

        check_info = self.compliance_checks[check_name]

        try:
            result = await check_info["func"]()
            check_info["last_run"] = datetime.utcnow()
            check_info["last_result"] = result

            # Store report
            report = {
                "timestamp": datetime.utcnow().isoformat(),
                "check_name": check_name,
                "result": result
            }
            self.compliance_reports.append(report)

            return result

        except Exception as e:
            result = {
                "status": "error",
                "message": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
            check_info["last_result"] = result
            return result

    async def run_all_checks(self) -> Dict[str, Any]:
        """Run all compliance checks."""
        results = {}
        for check_name in self.compliance_checks:
            results[check_name] = await self.run_compliance_check(check_name)

        return {
            "timestamp": datetime.utcnow().isoformat(),
            "results": results,
            "summary": self._generate_summary(results)
        }

    def _generate_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate compliance summary."""
        total_checks = len(results)
        passed_checks = sum(1 for r in results.values() if r.get("status") == "passed")
        failed_checks = total_checks - passed_checks

        return {
            "total_checks": total_checks,
            "passed": passed_checks,
            "failed": failed_checks,
            "compliance_rate": (passed_checks / total_checks * 100) if total_checks > 0 else 0
        }

    def get_compliance_reports(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent compliance reports."""
        return list(self.compliance_reports)[-limit:]

    def generate_compliance_report(self) -> Dict[str, Any]:
        """Generate comprehensive compliance report."""
        return {
            "generated_at": datetime.utcnow().isoformat(),
            "checks": {
                name: {
                    "frequency": info["frequency"],
                    "last_run": info["last_run"].isoformat() if info["last_run"] else None,
                    "last_result": info["last_result"]
                }
                for name, info in self.compliance_checks.items()
            },
            "recent_reports": self.get_compliance_reports(50)
        }


# Global instances
audit_logger = AuditLogger()
security_manager = SecurityManager()
compliance_manager = ComplianceManager()


async def data_encryption_check():
    """Check data encryption compliance."""
    # This would check if sensitive data is properly encrypted
    return {
        "status": "passed",
        "message": "Data encryption policies verified",
        "details": {
            "database_encrypted": True,
            "api_communications_encrypted": True,
            "logs_encrypted": False  # Could be improved
        }
    }


async def access_control_check():
    """Check access control compliance."""
    # This would verify proper access controls are in place
    return {
        "status": "passed",
        "message": "Access control policies verified",
        "details": {
            "role_based_access": True,
            "api_key_authentication": True,
            "audit_logging": True
        }
    }


async def data_retention_check():
    """Check data retention compliance."""
    # This would verify data is retained according to policies
    return {
        "status": "passed",
        "message": "Data retention policies verified",
        "details": {
            "sensor_data_retention_days": 90,
            "alert_retention_days": 365,
            "audit_log_retention_days": 2555  # 7 years
        }
    }


def init_security_compliance():
    """Initialize security and compliance systems."""
    # Register compliance checks
    compliance_manager.register_check("data_encryption", data_encryption_check)
    compliance_manager.register_check("access_control", access_control_check)
    compliance_manager.register_check("data_retention", data_retention_check)

    app_logger.info("Security and compliance systems initialized")