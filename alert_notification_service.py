"""
Real-time Alert & Notification Management Service
Handles severity routing, alert deduplication, and smart notifications
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from typing import List, Dict, Optional
import asyncio
from backend.utils.logger import ai_logger as logger
from backend.database.db_manager import DatabaseManager


class AlertSeverity(str, Enum):
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    INFO = "INFO"


class AlertStatus(str, Enum):
    ACTIVE = "ACTIVE"
    ACKNOWLEDGED = "ACKNOWLEDGED"
    DISMISSED = "DISMISSED"
    RESOLVED = "RESOLVED"


@dataclass
class Alert:
    machine_id: str
    severity: AlertSeverity
    title: str
    description: str
    source: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    status: AlertStatus = AlertStatus.ACTIVE
    alert_id: str = field(default="")
    acknowledged_at: Optional[datetime] = None
    dismissed_at: Optional[datetime] = None
    acknowledged_by: Optional[str] = None
    
    def to_dict(self):
        return {
            "alert_id": self.alert_id,
            "machine_id": self.machine_id,
            "severity": self.severity.value,
            "title": self.title,
            "description": self.description,
            "source": self.source,
            "timestamp": self.timestamp.isoformat(),
            "status": self.status.value,
            "acknowledged_at": self.acknowledged_at.isoformat() if self.acknowledged_at else None,
            "dismissed_at": self.dismissed_at.isoformat() if self.dismissed_at else None,
            "acknowledged_by": self.acknowledged_by
        }


class AlertNotificationService:
    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
        self.alerts: Dict[str, Alert] = {}
        self.subscribers = []
        self.dedup_window = timedelta(minutes=5)
        self.severity_routing = {
            AlertSeverity.CRITICAL: ["email", "slack", "sms"],
            AlertSeverity.HIGH: ["email", "slack"],
            AlertSeverity.MEDIUM: ["slack"],
            AlertSeverity.LOW: ["dashboard"],
            AlertSeverity.INFO: ["dashboard"]
        }
    
    async def create_alert(self, machine_id: str, severity: AlertSeverity, 
                           title: str, description: str, source: str) -> Alert:
        """Create and register new alert"""
        # Check for duplicate alerts within dedup window
        if self._is_duplicate(machine_id, title):
            logger.info(f"Alert deduplicated: {machine_id} - {title}")
            return None
        
        import uuid
        alert_id = str(uuid.uuid4())
        alert = Alert(
            alert_id=alert_id,
            machine_id=machine_id,
            severity=severity,
            title=title,
            description=description,
            source=source
        )
        
        self.alerts[alert_id] = alert
        
        # Route notification
        await self._route_notification(alert)
        
        # Notify subscribers
        await self._notify_subscribers(alert)
        
        logger.info(f"Alert created: {alert_id} - {severity.value} - {title}")
        return alert
    
    async def acknowledge_alert(self, alert_id: str, user_id: str = None) -> bool:
        """Mark alert as acknowledged"""
        if alert_id in self.alerts:
            alert = self.alerts[alert_id]
            alert.status = AlertStatus.ACKNOWLEDGED
            alert.acknowledged_at = datetime.utcnow()
            alert.acknowledged_by = user_id
            logger.info(f"Alert acknowledged: {alert_id}")
            await self._notify_subscribers(alert)
            return True
        return False
    
    async def dismiss_alert(self, alert_id: str, duration_minutes: int = 60) -> bool:
        """Dismiss alert for specified duration"""
        if alert_id in self.alerts:
            alert = self.alerts[alert_id]
            alert.status = AlertStatus.DISMISSED
            alert.dismissed_at = datetime.utcnow()
            
            # Auto-resolve after duration
            asyncio.create_task(self._auto_resolve_alert(alert_id, duration_minutes))
            logger.info(f"Alert dismissed for {duration_minutes}min: {alert_id}")
            return True
        return False
    
    async def resolve_alert(self, alert_id: str) -> bool:
        """Mark alert as resolved"""
        if alert_id in self.alerts:
            alert = self.alerts[alert_id]
            alert.status = AlertStatus.RESOLVED
            logger.info(f"Alert resolved: {alert_id}")
            await self._notify_subscribers(alert)
            return True
        return False
    
    async def get_active_alerts(self, machine_id: Optional[str] = None, 
                                severity: Optional[AlertSeverity] = None) -> List[Alert]:
        """Get active alerts, optionally filtered by machine or severity"""
        active = [a for a in self.alerts.values() 
                 if a.status in [AlertStatus.ACTIVE, AlertStatus.ACKNOWLEDGED]]
        
        if machine_id:
            active = [a for a in active if a.machine_id == machine_id]
        
        if severity:
            active = [a for a in active if a.severity == severity]
        
        return sorted(active, key=lambda x: x.timestamp, reverse=True)
    
    async def get_alert_stats(self) -> Dict:
        """Get alert statistics"""
        all_alerts = list(self.alerts.values())
        active = [a for a in all_alerts if a.status in [AlertStatus.ACTIVE, AlertStatus.ACKNOWLEDGED]]
        
        severity_counts = {
            AlertSeverity.CRITICAL: len([a for a in active if a.severity == AlertSeverity.CRITICAL]),
            AlertSeverity.HIGH: len([a for a in active if a.severity == AlertSeverity.HIGH]),
            AlertSeverity.MEDIUM: len([a for a in active if a.severity == AlertSeverity.MEDIUM]),
            AlertSeverity.LOW: len([a for a in active if a.severity == AlertSeverity.LOW]),
        }
        
        return {
            "total_active": len(active),
            "total_all_time": len(all_alerts),
            "by_severity": severity_counts,
            "critical_count": severity_counts[AlertSeverity.CRITICAL],
            "requires_immediate_action": severity_counts[AlertSeverity.CRITICAL] + severity_counts[AlertSeverity.HIGH]
        }
    
    def _is_duplicate(self, machine_id: str, title: str) -> bool:
        """Check if similar alert exists within dedup window"""
        cutoff_time = datetime.utcnow() - self.dedup_window
        
        for alert in self.alerts.values():
            if (alert.machine_id == machine_id and 
                alert.title == title and
                alert.timestamp > cutoff_time and
                alert.status in [AlertStatus.ACTIVE, AlertStatus.ACKNOWLEDGED]):
                return True
        return False
    
    async def _route_notification(self, alert: Alert):
        """Route alert to appropriate notification channels"""
        channels = self.severity_routing.get(alert.severity, ["dashboard"])
        
        for channel in channels:
            if channel == "email":
                await self._send_email_notification(alert)
            elif channel == "slack":
                await self._send_slack_notification(alert)
            elif channel == "sms":
                await self._send_sms_notification(alert)
            elif channel == "dashboard":
                pass  # Handled by subscribers
    
    async def _send_email_notification(self, alert: Alert):
        """Send email notification (stub)"""
        logger.info(f"Email notification (stub): {alert.title} to {alert.machine_id} owner")
    
    async def _send_slack_notification(self, alert: Alert):
        """Send Slack notification (stub)"""
        emoji = "🔴" if alert.severity == AlertSeverity.CRITICAL else "🟠" if alert.severity == AlertSeverity.HIGH else "🟡"
        logger.info(f"Slack notification (stub) {emoji}: {alert.title}")
    
    async def _send_sms_notification(self, alert: Alert):
        """Send SMS notification (stub)"""
        logger.info(f"SMS notification (stub): {alert.title} for {alert.machine_id}")
    
    async def _notify_subscribers(self, alert: Alert):
        """Notify all subscribed clients"""
        for subscriber in self.subscribers:
            try:
                await subscriber(alert)
            except Exception as e:
                logger.error(f"Error notifying subscriber: {e}")
    
    async def _auto_resolve_alert(self, alert_id: str, duration_minutes: int):
        """Auto-resolve dismissed alert after duration"""
        await asyncio.sleep(duration_minutes * 60)
        if alert_id in self.alerts:
            await self.resolve_alert(alert_id)
    
    def subscribe(self, callback):
        """Subscribe to alert notifications"""
        self.subscribers.append(callback)
