import asyncio
import json
from dataclasses import asdict
from datetime import datetime
from typing import Dict, Optional
from fastapi import WebSocket, WebSocketDisconnect
from backend.websocket.connection_manager import manager
from backend.data.simulator import simulator
from backend.data.priority_queue import alert_queue
from backend.database.db_manager import db
from backend.ai_engine.baseline_learner import BaselineManager
from backend.ai_engine.failure_predictor import FailurePredictor
from backend.utils.logger import ws_logger
from backend.utils.config import config
import uuid

class StreamHandler:
    """
    Real-time sensor stream handler.
    This is the 'Sensory Layer' — the nervous system entry point.
    
    Processes:
    1. Receives sensor streams from simulator/real sensors
    2. Routes to Digital Twin AI engines
    3. Broadcasts processed results via WebSocket
    4. Queues critical alerts with priority
    """
    
    def __init__(self, digital_twins: Dict):
        self.digital_twins = digital_twins
        self.stream_tasks: Dict[str, asyncio.Task] = {}
        self.running = False
        self.alert_queue = alert_queue  # Add reference to global alert_queue
        self.simulator = simulator
        self.baseline_manager = BaselineManager()
        self.failure_predictor = FailurePredictor()
        self.machine_states: Dict[str, Dict] = {}  # Store current state for each machine
        self.stats: Dict[str, int] = {
            "readings_processed": 0,
            "anomalies_detected": 0,
            "alerts_generated": 0,
            "healing_actions": 0
        }
    
    async def start_all_streams(self):
        """Start streaming for all machines simultaneously"""
        self.running = True
        ws_logger.info("🚀 Starting all machine streams...")
        
        # Inject high-risk failure for CNC_MILL_01
        simulator.inject_failure("CNC_MILL_01", "bearing_wear", severity=1.0, ramp_speed=0.2)
        ws_logger.info("💉 Injected bearing wear failure on CNC_MILL_01 for high risk demonstration")
        
        for machine_id in config.MACHINE_IDS:
            task = asyncio.create_task(
                self._stream_machine(machine_id),
                name=f"stream_{machine_id}"
            )
            self.stream_tasks[machine_id] = task
            ws_logger.info(f"  ▶ Stream started: {machine_id}")
        
        ws_logger.info(f"✅ {len(self.stream_tasks)} streams active")
    
    async def stop_all_streams(self):
        """Gracefully stop all streams"""
        self.running = False
        for machine_id, task in self.stream_tasks.items():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        self.stream_tasks.clear()
        ws_logger.info("⏹ All streams stopped")
    
    async def _stream_machine(self, machine_id: str):
        """
        Continuous stream processing for a single machine.
        Target: <200ms processing per reading.
        """
        twin = self.digital_twins.get(machine_id)
        if not twin:
            ws_logger.error(f"No digital twin for {machine_id}")
            return
        
        ws_logger.info(f"🔄 Streaming {machine_id}")
        
        async for reading in simulator.stream(machine_id):
            if not self.running:
                break
            
            try:
                start_time = asyncio.get_event_loop().time()
                
                # Run AI pipeline
                sensor_dict = {
                    "temperature": reading.temperature,
                    "vibration": reading.vibration,
                    "pressure": reading.pressure,
                    "current": reading.current,
                    "rpm": reading.rpm,
                    "oil_level": reading.oil_level,
                    "voltage": reading.voltage,
                    "humidity": reading.humidity,
                    "ambient_temp": reading.ambient_temp,
                    "load_percentage": reading.load_percentage,
                }
                result = await twin.process(sensor_dict)
                
                # Convert SensorReading dataclass to dict
                result_dict = asdict(result)
                
                processing_time = (asyncio.get_event_loop().time() - start_time) * 1000
                
                # Update stats
                self.stats["readings_processed"] += 1
                
                if result_dict.get("analysis", {}).get("is_anomaly"):
                    self.stats["anomalies_detected"] += 1
                
                # Alerts are persisted by digital_twin._persist() for HIGH/CRITICAL risks
                # The digital twin also triggers alerts via _submit_alert during processing
                if result_dict["risk"].get("level") in ("HIGH", "CRITICAL"):
                    self.stats["alerts_generated"] += 1
                
                if result_dict.get("healing_performed"):
                    self.stats["healing_actions"] += 1
                
                # Save the current machine state for API access
                self.machine_states[machine_id] = result

                # Save reading to DB (async, don't await)
                asyncio.create_task(db.save_reading({
                    "reading_id": reading.reading_id,
                    "machine_id": machine_id,
                    "timestamp": reading.timestamp.isoformat(),
                    "sensors": result_dict.get("sensors", {}).get("filtered", {}),
                    "analysis": result_dict.get("analysis", {})
                }))
                
                # Broadcast sensor update to all connected clients
                broadcast_data = {
                    "type": "sensor_update",
                    "machine_id": machine_id,
                    "timestamp": result_dict["timestamp"],
                    "state": result_dict["state"],
                    "sensors": result_dict["sensors"],
                    "risk": result_dict["risk"],
                    "analysis": result_dict["analysis"],
                    "machine": {
                        "id": machine_id,
                        "name": config.MACHINE_NAMES.get(machine_id, machine_id),
                        "type": config.MACHINE_TYPES.get(machine_id, "unknown")
                    },
                    "processing_time_ms": round(processing_time, 2),
                    "stats": self.stats.copy()
                }
                
                await manager.broadcast_to_machine(machine_id, broadcast_data)
                
                # Also broadcast to global subscribers
                await manager.broadcast_global(broadcast_data)

                # Feed failure predictor with latest filtered reading
                try:
                    self.failure_predictor.add_reading(
                        machine_id,
                        filtered,
                        reading.timestamp
                    )
                except Exception:
                    pass
                
                ws_logger.debug(
                    f"[{machine_id}] Processed in {processing_time:.1f}ms | "
                    f"Risk: {result_dict['risk'].get('level', 'N/A')} | "
                    f"Anomaly: {result_dict.get('analysis', {}).get('is_anomaly', False)}"
                )
                
            except asyncio.CancelledError:
                ws_logger.info(f"Stream cancelled: {machine_id}")
                break
            except Exception as e:
                ws_logger.error(f"Stream error [{machine_id}]: {e}", exc_info=True)
                # Don't break — keep streaming despite errors
                await asyncio.sleep(1)

    def inject_fault(self, machine_id: str, fault_type: str, intensity: float = 1.0):
        """
        Inject a test fault into the data simulator.
        """
        self.simulator.inject_failure(
            machine_id,
            fault_type,
            severity=min(max(intensity, 0.0), 1.0),
            ramp_speed=0.1
        )

    def clear_fault(self, machine_id: str):
        """
        Clear any injected fault from the simulator.
        """
        self.simulator.clear_failure(machine_id)

    def simulate_scenarios(self, machine_id: str, risk_score: float, baseline_dict: Dict):
        """
        Simulate failure scenarios for the machine based on current state.
        """
        current_state = self.machine_states.get(machine_id)
        current_readings = {}
        if current_state:
            current_readings = current_state.sensors.get("filtered", {})

        return self.failure_predictor.simulate_failure_scenarios(
            machine_id,
            baseline_dict,
            current_readings,
            failure_signature="bearing_fault"
        )

    async def handle_websocket(self, websocket: WebSocket, client_id: str, machine_id: Optional[str] = None):
        """
        Handle individual WebSocket client connections.
        Supports both global listeners and machine-specific subscriptions.
        """
        await manager.connect(websocket, client_id, machine_id)
        
        try:
            # Send welcome handshake
            await websocket.send_json({
                "type": "connected",
                "client_id": client_id,
                "machine_id": machine_id,
                "subscribed_to": machine_id if machine_id else "all_machines",
                "timestamp": datetime.utcnow().isoformat(),
                "server_stats": self.stats.copy(),
                "active_machines": list(self.stream_tasks.keys()),
                "message": "🤖 Digital Twin AI System — Connected"
            })
            
            ws_logger.info(
                f"Client connected: {client_id} | "
                f"Machine: {machine_id or 'ALL'}"
            )
            
            # Keep connection alive, listen for client messages
            while True:
                try:
                    # Wait for client messages (ping/pong, commands)
                    raw = await asyncio.wait_for(
                        websocket.receive_text(),
                        timeout=config.WS_HEARTBEAT_INTERVAL
                    )
                    
                    try:
                        message = json.loads(raw)
                        await self._handle_client_message(
                            websocket, client_id, machine_id, message
                        )
                    except json.JSONDecodeError as e:
                        ws_logger.warning(f"Invalid JSON from {client_id}: {e}")
                        await websocket.send_json({
                            "type": "error",
                            "message": "Invalid message format",
                            "timestamp": datetime.utcnow().isoformat()
                        })
                    
                except asyncio.TimeoutError:
                    # Send heartbeat ping
                    try:
                        await websocket.send_json({
                            "type": "ping",
                            "timestamp": datetime.utcnow().isoformat(),
                            "stats": self.stats.copy()
                        })
                    except Exception as e:
                        ws_logger.debug(f"Heartbeat send failed for {client_id}: {e}")
                        break
                    
        except WebSocketDisconnect:
            ws_logger.info(f"Client disconnected: {client_id}")
        except Exception as e:
            ws_logger.error(f"WebSocket error [{client_id}]: {e}", exc_info=True)
        finally:
            manager.disconnect(client_id)
    
    async def _handle_client_message(
        self,
        websocket: WebSocket,
        client_id: str,
        machine_id: Optional[str],
        message: Dict
    ):
        """
        Process incoming client WebSocket messages.
        
        Supported commands:
        - pong: heartbeat response
        - get_stats: request current system stats
        - get_alerts: request pending alerts from queue
        - subscribe: change machine subscription
        - get_twin_state: request current twin state
        """
        msg_type = message.get("type", "unknown")
        
        if msg_type == "pong":
            # Heartbeat acknowledged
            pass
        
        elif msg_type == "get_stats":
            await websocket.send_json({
                "type": "stats",
                "data": {
                    "stream_stats": self.stats.copy(),
                    "active_streams": len(self.stream_tasks),
                    "connected_clients": manager.get_connection_count(),
                    "queue_size": alert_queue.size(),
                    "timestamp": datetime.utcnow().isoformat()
                }
            })
        
        elif msg_type == "get_alerts":
            # Return top N alerts from priority queue
            limit = message.get("limit", 10)
            alerts = alert_queue.peek_top(limit)
            await websocket.send_json({
                "type": "alert_queue",
                "data": alerts,
                "count": len(alerts),
                "timestamp": datetime.utcnow().isoformat()
            })
        
        elif msg_type == "subscribe":
            # Change machine subscription
            new_machine_id = message.get("machine_id")
            manager.update_subscription(client_id, new_machine_id)
            await websocket.send_json({
                "type": "subscribed",
                "machine_id": new_machine_id,
                "message": f"Now subscribed to: {new_machine_id or 'all machines'}",
                "timestamp": datetime.utcnow().isoformat()
            })
        
        elif msg_type == "get_twin_state":
            # Return current state of a specific digital twin
            target_machine = message.get("machine_id", machine_id)
            twin = self.digital_twins.get(target_machine)
            
            if twin:
                await websocket.send_json({
                    "type": "twin_state",
                    "machine_id": target_machine,
                    "data": twin.get_state_summary(),
                    "timestamp": datetime.utcnow().isoformat()
                })
            else:
                await websocket.send_json({
                    "type": "error",
                    "message": f"No twin found for machine: {target_machine}",
                    "timestamp": datetime.utcnow().isoformat()
                })
        
        elif msg_type == "trigger_healing":
            # Manually trigger self-healing for a machine
            target_machine = message.get("machine_id", machine_id)
            twin = self.digital_twins.get(target_machine)
            
            if twin:
                healing_result = await twin.force_healing()
                await websocket.send_json({
                    "type": "healing_triggered",
                    "machine_id": target_machine,
                    "result": healing_result,
                    "timestamp": datetime.utcnow().isoformat()
                })
            else:
                await websocket.send_json({
                    "type": "error",
                    "message": f"No twin found: {target_machine}",
                    "timestamp": datetime.utcnow().isoformat()
                })
        
        else:
            await websocket.send_json({
                "type": "error",
                "message": f"Unknown command: {msg_type}",
                "supported": [
                    "pong", "get_stats", "get_alerts",
                    "subscribe", "get_twin_state", "trigger_healing"
                ],
                "timestamp": datetime.utcnow().isoformat()
            })
    
    def get_stats(self) -> Dict:
        """Return current streaming statistics"""
        return {
            **self.stats,
            "active_streams": len(self.stream_tasks),
            "running": self.running,
            "stream_names": list(self.stream_tasks.keys())
        }