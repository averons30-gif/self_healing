import heapq
import threading
from dataclasses import dataclass, field
from typing import Any, Optional
from datetime import datetime
from enum import IntEnum

class Priority(IntEnum):
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4
    INFO = 5

@dataclass(order=True)
class PriorityItem:
    priority: int
    timestamp: float
    item: Any = field(compare=False)
    machine_id: str = field(compare=False, default="")
    
    def to_dict(self):
        return {
            "priority": Priority(self.priority).name,
            "timestamp": self.timestamp,
            "machine_id": self.machine_id,
            "item": self.item
        }

class AlertPriorityQueue:
    """Thread-safe priority queue for machine alerts"""
    
    def __init__(self, maxsize: int = 1000):
        self._queue = []
        self._lock = threading.Lock()
        self._maxsize = maxsize
        self._counter = 0
        self._stats = {
            "total_pushed": 0,
            "total_popped": 0,
            "dropped": 0
        }
    
    def push(self, item: Any, priority: Priority, machine_id: str = "") -> bool:
        with self._lock:
            if len(self._queue) >= self._maxsize:
                # Drop lowest priority if full
                if self._queue and self._queue[-1].priority > priority.value:
                    self._queue.pop()
                    heapq.heapify(self._queue)
                    self._stats["dropped"] += 1
                else:
                    self._stats["dropped"] += 1
                    return False
            
            timestamp = datetime.now().timestamp()
            entry = PriorityItem(
                priority=priority.value,
                timestamp=timestamp,
                item=item,
                machine_id=machine_id
            )
            heapq.heappush(self._queue, entry)
            self._stats["total_pushed"] += 1
            return True
    
    def pop(self) -> Optional[PriorityItem]:
        with self._lock:
            if self._queue:
                item = heapq.heappop(self._queue)
                self._stats["total_popped"] += 1
                return item
            return None
    
    def peek(self) -> Optional[PriorityItem]:
        with self._lock:
            return self._queue[0] if self._queue else None
    
    def size(self) -> int:
        with self._lock:
            return len(self._queue)
    
    def get_all_by_machine(self, machine_id: str) -> list:
        with self._lock:
            return [item for item in self._queue if item.machine_id == machine_id]
    
    def get_stats(self) -> dict:
        with self._lock:
            return {**self._stats, "current_size": len(self._queue)}
    
    def clear(self):
        with self._lock:
            self._queue.clear()

# Global alert queue
alert_queue = AlertPriorityQueue()