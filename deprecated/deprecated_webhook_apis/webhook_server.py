#!/usr/bin/env python3
"""Example webhook server to receive FaceCV notifications"""

from fastapi import FastAPI, Request, BackgroundTasks
from datetime import datetime
import json
import logging
import uvicorn
from typing import Dict, List
import asyncio
import aiofiles

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="FaceCV Webhook Receiver Example")

# In-memory storage for demo (use database in production)
events_storage = []
alerts_storage = []


@app.post("/webhook/face-recognition")
async def receive_face_recognition(request: Request, background_tasks: BackgroundTasks):
    """
    Receive face recognition events from FaceCV
    """
    try:
        data = await request.json()
        
        # Log the webhook
        logger.info(f"Received webhook from {data.get('webhook_id', 'unknown')}")
        
        # Process events
        for event in data.get("events", []):
            event_type = event.get("event_type")
            camera_id = event.get("camera_id")
            timestamp = event.get("timestamp")
            
            if event_type == "face_recognized":
                # Handle recognized faces
                for face in event["data"].get("recognized_faces", []):
                    logger.info(f"Camera {camera_id}: Recognized {face['name']} with confidence {face['confidence']}")
                    
                    # Store event
                    events_storage.append({
                        "timestamp": timestamp,
                        "camera_id": camera_id,
                        "person": face["name"],
                        "confidence": face["confidence"]
                    })
                    
                    # Trigger additional actions in background
                    background_tasks.add_task(
                        process_recognition,
                        camera_id,
                        face
                    )
            
            elif event_type == "stranger_alert":
                # Handle stranger alerts
                alert_data = event["data"]
                logger.warning(f"Camera {camera_id}: Stranger alert - Level {alert_data.get('alert_level')}")
                
                alerts_storage.append({
                    "timestamp": timestamp,
                    "camera_id": camera_id,
                    "alert": alert_data
                })
                
                # Trigger security protocol
                background_tasks.add_task(
                    handle_security_alert,
                    camera_id,
                    alert_data
                )
        
        return {"status": "received", "events_count": len(data.get("events", []))}
        
    except Exception as e:
        logger.error(f"Error processing webhook: {e}")
        return {"status": "error", "message": str(e)}, 500


@app.post("/webhook/attendance")
async def receive_attendance(request: Request):
    """
    Receive attendance events from FaceCV
    """
    try:
        data = await request.json()
        
        for event in data.get("events", []):
            if event["event_type"] == "attendance_recorded":
                attendance_data = event["data"]
                logger.info(
                    f"Attendance: {attendance_data['name']} - "
                    f"{attendance_data['action']} at {event['timestamp']}"
                )
                
                # Here you would update your attendance system
                # For example: update database, send notifications, etc.
        
        return {"status": "received"}
        
    except Exception as e:
        logger.error(f"Error processing attendance webhook: {e}")
        return {"status": "error", "message": str(e)}, 500


@app.get("/events")
async def get_recent_events(limit: int = 100):
    """Get recent recognition events"""
    return {
        "events": events_storage[-limit:],
        "total": len(events_storage)
    }


@app.get("/alerts")
async def get_alerts(limit: int = 50):
    """Get recent security alerts"""
    return {
        "alerts": alerts_storage[-limit:],
        "total": len(alerts_storage)
    }


async def process_recognition(camera_id: str, face_data: Dict):
    """Process recognized face (example background task)"""
    # Simulate processing
    await asyncio.sleep(0.1)
    
    # Example actions:
    # - Update access control system
    # - Log to time tracking system
    # - Send notification to relevant parties
    # - Update dashboard
    
    logger.info(f"Processed recognition for {face_data['name']} from camera {camera_id}")


async def handle_security_alert(camera_id: str, alert_data: Dict):
    """Handle security alert (example background task)"""
    # Simulate security response
    await asyncio.sleep(0.1)
    
    # Example actions:
    # - Send SMS/Email to security team
    # - Trigger alarm system
    # - Lock down certain areas
    # - Save alert video clip
    
    logger.warning(f"Security protocol activated for camera {camera_id}")


# WebSocket endpoint for real-time updates
from fastapi import WebSocket, WebSocketDisconnect
from typing import Set

class ConnectionManager:
    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.add(websocket)
    
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
    
    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except:
                pass

manager = ConnectionManager()


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket for real-time updates to connected clients"""
    await manager.connect(websocket)
    try:
        while True:
            # Keep connection alive
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)


@app.post("/webhook/realtime")
async def receive_realtime_webhook(request: Request):
    """
    Receive real-time events and broadcast via WebSocket
    """
    try:
        data = await request.json()
        
        # Broadcast to all connected WebSocket clients
        await manager.broadcast(data)
        
        return {"status": "broadcasted"}
        
    except Exception as e:
        logger.error(f"Error broadcasting webhook: {e}")
        return {"status": "error", "message": str(e)}, 500


if __name__ == "__main__":
    # Run the webhook receiver
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8080,
        log_level="info"
    )