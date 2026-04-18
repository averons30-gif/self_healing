"""
backend/ai_engine/alert_manager.py
═══════════════════════════════════════════════════════════════════
AlertManager — Priority queue alert management system.

Features:
  - Priority queue ordered by risk level + timestamp
  - Deduplication window (suppress duplicate alerts)
  - Per-machine per-level cooldown
  - Alert statistics and queue inspection
  - WebSocket broadcast integration
  - Database persistence
  - Maximum queue size with LRU eviction
═══════════════════════════════════════════════════════════════════
"""

from __future__     import annotations

import asyncio
import heapq
import time
import uuid
from dataclasses    import dataclass, field
from datetime       import datetime, timezone
from typing         import Any, Callable, Dict, List, Optional, Set

from backend.database.db_manager    import db
from backend.utils.logger           import ai_logger


# ── Priority levels (lower number = higher priority) ──────────────────────────
class AlertPriority:
    CRITICAL    = 1
    HIGH        = 2
    MEDIUM      = 3
    LOW         = 4

RISK_TO_PRIORITY: Dict[str, int] = {
    "CRITICAL": AlertPriority.CRITICAL,
    "HIGH":     AlertPriority.HIGH,
    "MEDIUM":   AlertPriority.MEDIUM,
    "LOW":      AlertPriority.LOW
}


@dataclass(order=True)
class AlertEntry:
    """
    Heap-ordered alert entry.
    Ordered by: priority ASC, then timestamp ASC (older = higher priority).
    """
    priority:       int
    timestamp_ms:   float

    # ── Payload (not included in ordering) ────────────────────────────────────
    alert_id:       str     = field(compare=False)
    machine_id:     str     = field(compare=False)
    risk_level:     str     = field(compare=False)
    alert_type:     str     = field(compare=False)
    message:        str     = field(compare=False)
    sensors:        Dict    = field(compare=False, default_factory=dict)
    risk_score:     float   = field(compare=False, default=0.0)
    timestamp:      str     = field(compare=False, default="")
    resolved:       bool    = field(compare=False, default=False)


class AlertManager:
    """
    Centralised alert management with priority queue.

    Collects alerts from all DigitalTwinEngines, deduplicates them,
    and makes them available for WebSocket broadcast and database storage.

    Usage:
        manager = AlertManager(max_queue_size=500)
        await manager.submit(alert_data)
        alerts = await manager.get_top_alerts(10)
    """

    def __init__(
        self,
        max_queue_size:     int     = 500,
        dedup_window_s:     float   = 30.0,
        cooldown_s:         float   = 30.0,
        on_alert_callback:  Optional[Callable] = None
    ):
        """
        Args:
            max_queue_size:     Maximum alerts in priority queue
            dedup_window_s:     Seconds to suppress duplicate alerts
            cooldown_s:         Per machine+level cooldown
            on_alert_callback:  Async callback (alert_dict) → None
                                (used for WebSocket broadcast)
        """
        self._max_size          = max_queue_size
        self._dedup_window_s    = dedup_window_s
        self._cooldown_s        = cooldown_s
        self._on_alert          = on_alert_callback

        # ── Priority heap ──────────────────────────────────────────────────────
        self._heap:         List[AlertEntry]    = []

        # ── Deduplication tracking ─────────────────────────────────────────────
        self._dedup_keys:   Dict[str, float]    = {}    # fingerprint → ts
        self._cooldowns:    Dict[str, float]    = {}    # machine+level → ts

        # ── Resolved alert IDs (for fast lookup) ──────────────────────────────
        self._resolved_ids: Set[str]            = set()

        # ── Statistics ────────────────────────────────────────────────────────
        self._submitted         : int = 0
        self._deduped           : int = 0
        self._throttled         : int = 0
        self._resolved_count    : int = 0

        # ── Async lock ────────────────────────────────────────────────────────
        self._lock = asyncio.Lock()

        ai_logger.info(
            f"AlertManager created: "
            f"max_size={max_queue_size} "
            f"dedup={dedup_window_s}s"
        )

    # ══════════════════════════════════════════════════════════════════════════
    # SUBMIT
    # ══════════════════════════════════════════════════════════════════════════

    async def submit(self, alert_data: Dict[str, Any]) -> Optional[str]:
        """
        Submit an alert for processing.

        Applies deduplication, cooldown, priority assignment
        and inserts into the priority queue.

        Args:
            alert_data: {
                machine_id, alert_type, risk_level,
                message, sensors, risk_score, timestamp
            }

        Returns:
            alert_id if accepted, None if rejected
        """
        async with self._lock:
            self._submitted += 1

            machine_id  = alert_data.get("machine_id", "unknown")
            risk_level  = alert_data.get("risk_level", "MEDIUM")
            alert_type  = alert_data.get("alert_type", "unknown")

            # ── Deduplication ──────────────────────────────────────────────────
            fingerprint = f"{machine_id}:{alert_type}:{risk_level}"
            now         = time.time()

            self._clean_dedup_cache(now)

            if fingerprint in self._dedup_keys:
                self._deduped += 1
                ai_logger.debug(
                    f"AlertManager: deduped {fingerprint}"
                )
                return None

            # ── Cooldown check ─────────────────────────────────────────────────
            cooldown_key = f"{machine_id}:{risk_level}"
            last_ts      = self._cooldowns.get(cooldown_key, 0.0)

            if now - last_ts < self._cooldown_s:
                self._throttled += 1
                return None

            # ── Build alert entry ──────────────────────────────────────────────
            alert_id    = str(uuid.uuid4())
            priority    = RISK_TO_PRIORITY.get(risk_level, AlertPriority.MEDIUM)
            timestamp   = alert_data.get(
                "timestamp",
                datetime.now(timezone.utc).isoformat()
            )

            entry = AlertEntry(
                priority        = priority,
                timestamp_ms    = now,
                alert_id        = alert_id,
                machine_id      = machine_id,
                risk_level      = risk_level,
                alert_type      = alert_type,
                message         = alert_data.get("message", ""),
                sensors         = alert_data.get("sensors", {}),
                risk_score      = alert_data.get("risk_score", 0.0),
                timestamp       = timestamp
            )

            # ── Evict lowest-priority if at capacity ───────────────────────────
            if len(self._heap) >= self._max_size:
                self._evict_lowest_priority()

            # ── Push to heap ───────────────────────────────────────────────────
            heapq.heappush(self._heap, entry)

            # ── Record dedup + cooldown ────────────────────────────────────────
            self._dedup_keys[fingerprint]   = now
            self._cooldowns[cooldown_key]   = now

            ai_logger.info(
                f"📥 AlertManager: [{machine_id}] {risk_level} "
                f"id={alert_id[:8]}... "
                f"queue_size={len(self._heap)}"
            )

            # ── Fire callback (WebSocket broadcast) ────────────────────────────
            if self._on_alert:
                alert_dict = self._entry_to_dict(entry)
                asyncio.create_task(
                    self._fire_callback(alert_dict)
                )

            return alert_id

    async def _fire_callback(self, alert_dict: Dict) -> None:
        """Fire the on_alert callback safely."""
        try:
            await self._on_alert(alert_dict)
        except Exception as exc:
            ai_logger.error(f"AlertManager callback error: {exc}")

    # ══════════════════════════════════════════════════════════════════════════
    # QUERY
    # ══════════════════════════════════════════════════════════════════════════

    async def get_top_alerts(self, n: int = 10) -> List[Dict]:
        """
        Get top N alerts by priority (non-destructive peek).

        Args:
            n: Max alerts to return

        Returns:
            List of alert dicts, highest priority first
        """
        async with self._lock:
            # Sort copy of heap for display
            sorted_entries = sorted(self._heap)
            active  = [
                e for e in sorted_entries
                if e.alert_id not in self._resolved_ids
            ]
            return [self._entry_to_dict(e) for e in active[:n]]

    async def get_queue_snapshot(self) -> Dict:
        """
        Get a summary snapshot of the current alert queue.

        Returns:
            {
                size, by_priority, top_alerts,
                submitted, deduped, throttled, resolved
            }
        """
        async with self._lock:
            active  = [
                e for e in self._heap
                if e.alert_id not in self._resolved_ids
            ]

            by_priority: Dict[str, int] = {
                "CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0
            }
            for e in active:
                by_priority[e.risk_level] = (
                    by_priority.get(e.risk_level, 0) + 1
                )

            top = sorted(active)[:5]

            return {
                "queue_size":   len(active),
                "total_size":   len(self._heap),
                "by_priority":  by_priority,
                "top_alerts":   [self._entry_to_dict(e) for e in top],
                "submitted":    self._submitted,
                "deduped":      self._deduped,
                "throttled":    self._throttled,
                "resolved":     self._resolved_count
            }

    async def get_machine_alerts(
        self,
        machine_id: str,
        limit:      int = 20
    ) -> List[Dict]:
        """
        Get alerts for a specific machine.

        Args:
            machine_id: Machine to filter for
            limit:      Max results

        Returns:
            List of alert dicts
        """
        async with self._lock:
            entries = [
                e for e in self._heap
                if e.machine_id == machine_id
                and e.alert_id not in self._resolved_ids
            ]
            entries.sort()
            return [self._entry_to_dict(e) for e in entries[:limit]]

    # ══════════════════════════════════════════════════════════════════════════
    # RESOLVE
    # ══════════════════════════════════════════════════════════════════════════

    async def resolve(self, alert_id: str) -> bool:
        """
        Mark an alert as resolved.

        Args:
            alert_id: Alert UUID

        Returns:
            True if found and resolved
        """
        async with self._lock:
            for entry in self._heap:
                if entry.alert_id == alert_id:
                    entry.resolved = True
                    self._resolved_ids.add(alert_id)
                    self._resolved_count += 1
                    ai_logger.info(f"✅ Alert resolved: {alert_id[:8]}...")
                    return True
            return False

    async def resolve_all_machine(self, machine_id: str) -> int:
        """
        Resolve all alerts for a machine.

        Args:
            machine_id: Target machine

        Returns:
            Number of alerts resolved
        """
        async with self._lock:
            resolved = 0
            for entry in self._heap:
                if (
                    entry.machine_id == machine_id
                    and entry.alert_id not in self._resolved_ids
                ):
                    entry.resolved = True
                    self._resolved_ids.add(entry.alert_id)
                    resolved += 1

            self._resolved_count += resolved
            ai_logger.info(
                f"✅ Resolved {resolved} alerts for [{machine_id}]"
            )
            return resolved

    async def clear_resolved(self) -> int:
        """
        Remove resolved alerts from the heap (cleanup).

        Returns:
            Number of entries removed
        """
        async with self._lock:
            before          = len(self._heap)
            self._heap      = [
                e for e in self._heap
                if e.alert_id not in self._resolved_ids
            ]
            heapq.heapify(self._heap)
            removed         = before - len(self._heap)
            ai_logger.debug(f"AlertManager: cleared {removed} resolved")
            return removed

    # ══════════════════════════════════════════════════════════════════════════
    # HELPERS
    # ══════════════════════════════════════════════════════════════════════════

    def _evict_lowest_priority(self) -> None:
        """
        Remove the lowest-priority (highest priority number) alert.
        Needed when queue is at max capacity.
        """
        if not self._heap:
            return

        # Highest priority number = lowest importance
        worst_idx   = max(range(len(self._heap)), key=lambda i: (
            self._heap[i].priority,
            -self._heap[i].timestamp_ms     # newer = higher to evict
        ))

        self._heap.pop(worst_idx)
        heapq.heapify(self._heap)

    def _clean_dedup_cache(self, now: float) -> None:
        """Remove expired dedup cache entries."""
        expired = [
            k for k, ts in self._dedup_keys.items()
            if now - ts > self._dedup_window_s
        ]
        for k in expired:
            del self._dedup_keys[k]

    def _entry_to_dict(self, entry: AlertEntry) -> Dict:
        """Convert AlertEntry to serialisable dict."""
        return {
            "alert_id":     entry.alert_id,
            "machine_id":   entry.machine_id,
            "risk_level":   entry.risk_level,
            "alert_type":   entry.alert_type,
            "message":      entry.message,
            "sensors":      entry.sensors,
            "risk_score":   entry.risk_score,
            "timestamp":    entry.timestamp,
            "resolved":     entry.resolved,
            "type":         "alert"
        }

    # ══════════════════════════════════════════════════════════════════════════
    # STATS
    # ══════════════════════════════════════════════════════════════════════════

    def get_stats(self) -> Dict:
        """Return AlertManager statistics."""
        active = len([e for e in self._heap if e.alert_id not in self._resolved_ids])
        return {
            "queue_size":       active,
            "total_in_heap":    len(self._heap),
            "submitted":        self._submitted,
            "deduped":          self._deduped,
            "throttled":        self._throttled,
            "resolved":         self._resolved_count,
            "dedup_cache_size": len(self._dedup_keys),
            "cooldowns_active": len(self._cooldowns)
        }

    def set_callback(self, callback: Callable) -> None:
        """Update the on_alert broadcast callback."""
        self._on_alert = callback

    def __repr__(self) -> str:
        return (
            f"AlertManager("
            f"size={len(self._heap)}, "
            f"submitted={self._submitted})"
        )