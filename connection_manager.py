import asyncio
import json
from datetime import datetime
from typing import Dict, List, Optional, Set
from fastapi import WebSocket
from backend.utils.logger import ws_logger
from backend.utils.config import config
from backend.database.db_manager import convert_numpy_types


class ConnectionManager:
    """
    WebSocket Connection Manager.
    
    Manages:
    - Active client connections
    - Machine-specific subscriptions
    - Global broadcast channels
    - Alert priority broadcasting
    
    Architecture:
    ┌─────────────────────────────────────┐
    │         ConnectionManager           │
    │                                     │
    │  global_clients: all connections    │
    │  machine_clients: per-machine subs  │
    │  alert_clients: alert-only subs     │
    └─────────────────────────────────────┘
    """
    
    def __init__(self):
        # All active connections: client_id -> WebSocket
        self.active_connections: Dict[str, WebSocket] = {}
        
        # Machine subscriptions: machine_id -> Set[client_id]
        self.machine_subscriptions: Dict[str, Set[str]] = {}
        
        # Global subscribers (receive all machine updates)
        self.global_subscribers: Set[str] = set()
        
        # Alert-only subscribers
        self.alert_subscribers: Set[str] = set()
        
        # Client metadata: client_id -> metadata dict
        self.client_metadata: Dict[str, Dict] = {}
        
        # Lock for thread safety
        self._lock = asyncio.Lock()
    
    async def connect(
        self,
        websocket: WebSocket,
        client_id: str,
        machine_id: Optional[str] = None
    ):
        """
        Register a new WebSocket connection.
        
        Args:
            websocket: The WebSocket connection object
            client_id: Unique client identifier
            machine_id: Optional machine to subscribe to (None = all machines)
        """
        await websocket.accept()
        
        async with self._lock:
            self.active_connections[client_id] = websocket
            
            # Store metadata
            self.client_metadata[client_id] = {
                "connected_at": datetime.utcnow().isoformat(),
                "machine_id": machine_id,
                "messages_sent": 0,
                "last_ping": datetime.utcnow().isoformat()
            }
            
            if machine_id:
                # Subscribe to specific machine
                if machine_id not in self.machine_subscriptions:
                    self.machine_subscriptions[machine_id] = set()
                self.machine_subscriptions[machine_id].add(client_id)
            else:
                # Subscribe to all machines
                self.global_subscribers.add(client_id)
        
        ws_logger.info(
            f"✅ Client connected: {client_id} | "
            f"Machine: {machine_id or 'ALL'} | "
            f"Total: {len(self.active_connections)}"
        )
    
    def disconnect(self, client_id: str):
        """
        Remove a client connection and clean up subscriptions.
        
        Args:
            client_id: The client to disconnect
        """
        if client_id in self.active_connections:
            del self.active_connections[client_id]
        
        # Remove from global subscribers
        self.global_subscribers.discard(client_id)
        self.alert_subscribers.discard(client_id)
        
        # Remove from machine subscriptions
        for machine_id in self.machine_subscriptions:
            self.machine_subscriptions[machine_id].discard(client_id)
        
        # Remove metadata
        if client_id in self.client_metadata:
            del self.client_metadata[client_id]
        
        ws_logger.info(
            f"❌ Client disconnected: {client_id} | "
            f"Remaining: {len(self.active_connections)}"
        )
    
    def update_subscription(self, client_id: str, new_machine_id: Optional[str]):
        """
        Update a client's machine subscription.
        
        Args:
            client_id: The client to update
            new_machine_id: New machine to subscribe to (None = all)
        """
        # Remove from all current subscriptions
        self.global_subscribers.discard(client_id)
        for machine_id in self.machine_subscriptions:
            self.machine_subscriptions[machine_id].discard(client_id)
        
        # Add to new subscription
        if new_machine_id:
            if new_machine_id not in self.machine_subscriptions:
                self.machine_subscriptions[new_machine_id] = set()
            self.machine_subscriptions[new_machine_id].add(client_id)
        else:
            self.global_subscribers.add(client_id)
        
        # Update metadata
        if client_id in self.client_metadata:
            self.client_metadata[client_id]["machine_id"] = new_machine_id
        
        ws_logger.info(
            f"🔄 Subscription updated: {client_id} -> "
            f"{new_machine_id or 'ALL'}"
        )
    
    async def broadcast_to_machine(self, machine_id: str, data: Dict):
        """
        Send data to all clients subscribed to a specific machine.
        
        Args:
            machine_id: Target machine ID
            data: Data payload to broadcast
        """
        subscribers = self.machine_subscriptions.get(machine_id, set()).copy()
        
        if not subscribers:
            return
        
        disconnected = []
        
        for client_id in subscribers:
            websocket = self.active_connections.get(client_id)
            if websocket:
                success = await self._send_safe(client_id, websocket, data)
                if not success:
                    disconnected.append(client_id)
        
        # Clean up dead connections
        for client_id in disconnected:
            self.disconnect(client_id)
    
    async def broadcast_global(self, data: Dict):
        """
        Send data to all global subscribers (subscribed to all machines).
        
        Args:
            data: Data payload to broadcast
        """
        subscribers = self.global_subscribers.copy()
        
        if not subscribers:
            return
        
        disconnected = []
        
        for client_id in subscribers:
            websocket = self.active_connections.get(client_id)
            if websocket:
                success = await self._send_safe(client_id, websocket, data)
                if not success:
                    disconnected.append(client_id)
        
        # Clean up dead connections
        for client_id in disconnected:
            self.disconnect(client_id)
    
    async def broadcast_alert(self, alert_data: Dict):
        """
        Broadcast critical alerts to ALL connected clients immediately.
        Alerts bypass subscription filters — everyone needs to know.
        
        Args:
            alert_data: Alert payload
        """
        all_clients = list(self.active_connections.keys())
        disconnected = []
        
        for client_id in all_clients:
            websocket = self.active_connections.get(client_id)
            if websocket:
                success = await self._send_safe(client_id, websocket, alert_data)
                if not success:
                    disconnected.append(client_id)
        
        # Clean up dead connections
        for client_id in disconnected:
            self.disconnect(client_id)
        
        ws_logger.warning(
            f"🚨 Alert broadcast | "
            f"Clients notified: {len(all_clients)} | "
            f"Machine: {alert_data.get('machine_id', 'N/A')}"
        )
    
    async def send_to_client(self, client_id: str, data: Dict) -> bool:
        """
        Send data to a specific client.
        
        Args:
            client_id: Target client ID
            data: Data payload
            
        Returns:
            True if successful, False if client not found or error
        """
        websocket = self.active_connections.get(client_id)
        if not websocket:
            return False
        
        return await self._send_safe(client_id, websocket, data)
    
    async def _send_safe(
        self,
        client_id: str,
        websocket: WebSocket,
        data: Dict
    ) -> bool:
        """
        Safely send data to a WebSocket, handling errors gracefully.
        
        Args:
            client_id: Client ID for logging
            websocket: The WebSocket connection
            data: Data to send
            
        Returns:
            True if successful, False on error
        """
        try:
            # Convert NumPy types to JSON-serializable types
            clean_data = convert_numpy_types(data)
            
            await websocket.send_json(clean_data)
            
            # Update message count
            if client_id in self.client_metadata:
                self.client_metadata[client_id]["messages_sent"] += 1
                self.client_metadata[client_id]["last_ping"] = (
                    datetime.utcnow().isoformat()
                )
            
            return True
            
        except Exception as e:
            ws_logger.warning(f"Send failed [{client_id}]: {e}")
            return False
    
    def get_connection_count(self) -> int:
        """Return total number of active connections"""
        return len(self.active_connections)
    
    def get_connection_info(self) -> Dict:
        """
        Return detailed connection statistics.
        
        Returns:
            Dict with connection counts and metadata
        """
        machine_counts = {
            machine_id: len(clients)
            for machine_id, clients in self.machine_subscriptions.items()
            if clients
        }
        
        return {
            "total_connections": len(self.active_connections),
            "global_subscribers": len(self.global_subscribers),
            "machine_subscribers": machine_counts,
            "alert_subscribers": len(self.alert_subscribers),
            "clients": [
                {
                    "client_id": cid,
                    **meta
                }
                for cid, meta in self.client_metadata.items()
            ]
        }
    
    def is_connected(self, client_id: str) -> bool:
        """Check if a client is currently connected"""
        return client_id in self.active_connections


# Global singleton instance
manager = ConnectionManager()