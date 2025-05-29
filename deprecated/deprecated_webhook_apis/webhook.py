"""Webhook manager for real-time result forwarding"""

import asyncio
import aiohttp
import json
import logging
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
from dataclasses import dataclass, asdict
import threading
from queue import Queue, Empty
import time

logger = logging.getLogger(__name__)


@dataclass
class WebhookConfig:
    """Webhook configuration"""
    url: str
    headers: Optional[Dict[str, str]] = None
    timeout: int = 30
    retry_count: int = 3
    retry_delay: int = 1
    batch_size: int = 10
    batch_timeout: float = 1.0
    enabled: bool = True
    
    
@dataclass
class WebhookEvent:
    """Webhook event data"""
    event_type: str  # face_detected, stranger_alert, attendance_recorded
    timestamp: str
    camera_id: str
    data: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class WebhookManager:
    """Manages webhook deliveries for real-time notifications"""
    
    def __init__(self):
        self.webhooks: Dict[str, WebhookConfig] = {}
        self.event_queue = Queue(maxsize=10000)
        self.running = False
        self.worker_thread = None
        self._session: Optional[aiohttp.ClientSession] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        
    def add_webhook(self, webhook_id: str, config: WebhookConfig):
        """Add a webhook configuration"""
        self.webhooks[webhook_id] = config
        logger.info(f"Added webhook {webhook_id}: {config.url}")
        
    def remove_webhook(self, webhook_id: str):
        """Remove a webhook configuration"""
        if webhook_id in self.webhooks:
            del self.webhooks[webhook_id]
            logger.info(f"Removed webhook {webhook_id}")
            
    def update_webhook(self, webhook_id: str, config: WebhookConfig):
        """Update webhook configuration"""
        self.webhooks[webhook_id] = config
        logger.info(f"Updated webhook {webhook_id}")
        
    def start(self):
        """Start the webhook delivery worker"""
        if not self.running:
            self.running = True
            self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
            self.worker_thread.start()
            logger.info("Webhook manager started")
            
    def stop(self):
        """Stop the webhook delivery worker"""
        self.running = False
        if self.worker_thread:
            self.worker_thread.join(timeout=5)
        logger.info("Webhook manager stopped")
        
    def send_event(self, event: WebhookEvent):
        """Queue an event for delivery"""
        if not self.running:
            logger.warning("Webhook manager not running, event dropped")
            return
            
        try:
            self.event_queue.put_nowait(event)
        except:
            logger.error(f"Event queue full, dropping event: {event.event_type}")
            
    def _worker_loop(self):
        """Worker loop for processing webhook deliveries"""
        # Create new event loop for this thread
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        
        try:
            self._loop.run_until_complete(self._async_worker())
        except Exception as e:
            logger.error(f"Webhook worker error: {e}")
        finally:
            self._loop.close()
            
    async def _async_worker(self):
        """Async worker for webhook delivery"""
        self._session = aiohttp.ClientSession()
        batch = []
        last_send_time = time.time()
        
        try:
            while self.running:
                try:
                    # Get event with timeout
                    event = self.event_queue.get(timeout=0.1)
                    batch.append(event)
                    
                    # Send batch if size reached or timeout
                    current_time = time.time()
                    should_send = (
                        len(batch) >= max(config.batch_size for config in self.webhooks.values()) or
                        (current_time - last_send_time) >= min(config.batch_timeout for config in self.webhooks.values())
                    )
                    
                    if should_send and batch:
                        await self._send_batch(batch)
                        batch = []
                        last_send_time = current_time
                        
                except Empty:
                    # Send any pending events on timeout
                    if batch:
                        await self._send_batch(batch)
                        batch = []
                        last_send_time = time.time()
                        
                except Exception as e:
                    logger.error(f"Error processing webhook event: {e}")
                    
        finally:
            if self._session:
                await self._session.close()
                
    async def _send_batch(self, events: List[WebhookEvent]):
        """Send a batch of events to all configured webhooks"""
        tasks = []
        
        for webhook_id, config in self.webhooks.items():
            if config.enabled:
                task = self._deliver_to_webhook(webhook_id, config, events)
                tasks.append(task)
                
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
            
    async def _deliver_to_webhook(self, webhook_id: str, config: WebhookConfig, events: List[WebhookEvent]):
        """Deliver events to a specific webhook"""
        payload = {
            "webhook_id": webhook_id,
            "timestamp": datetime.now().isoformat(),
            "events": [event.to_dict() for event in events]
        }
        
        headers = config.headers or {}
        headers["Content-Type"] = "application/json"
        
        for attempt in range(config.retry_count):
            try:
                async with self._session.post(
                    config.url,
                    json=payload,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=config.timeout)
                ) as response:
                    if response.status < 300:
                        logger.debug(f"Successfully delivered {len(events)} events to {webhook_id}")
                        return
                    else:
                        logger.warning(f"Webhook {webhook_id} returned {response.status}")
                        
            except asyncio.TimeoutError:
                logger.error(f"Webhook {webhook_id} timeout (attempt {attempt + 1})")
            except Exception as e:
                logger.error(f"Webhook {webhook_id} delivery error: {e}")
                
            if attempt < config.retry_count - 1:
                await asyncio.sleep(config.retry_delay * (attempt + 1))
                
        logger.error(f"Failed to deliver events to {webhook_id} after {config.retry_count} attempts")


class CallbackManager:
    """Manages real-time callbacks for different event types"""
    
    def __init__(self):
        self.callbacks: Dict[str, List[Callable]] = {
            "face_detected": [],
            "face_recognized": [],
            "stranger_alert": [],
            "attendance_recorded": [],
            "stream_started": [],
            "stream_stopped": []
        }
        
    def register_callback(self, event_type: str, callback: Callable):
        """Register a callback for an event type"""
        if event_type in self.callbacks:
            self.callbacks[event_type].append(callback)
            logger.info(f"Registered callback for {event_type}")
        else:
            logger.warning(f"Unknown event type: {event_type}")
            
    def unregister_callback(self, event_type: str, callback: Callable):
        """Unregister a callback"""
        if event_type in self.callbacks and callback in self.callbacks[event_type]:
            self.callbacks[event_type].remove(callback)
            logger.info(f"Unregistered callback for {event_type}")
            
    async def trigger_event(self, event_type: str, data: Dict[str, Any]):
        """Trigger callbacks for an event"""
        if event_type not in self.callbacks:
            return
            
        # Run callbacks concurrently
        tasks = []
        for callback in self.callbacks[event_type]:
            if asyncio.iscoroutinefunction(callback):
                tasks.append(callback(data))
            else:
                # Run sync callbacks in executor
                loop = asyncio.get_event_loop()
                tasks.append(loop.run_in_executor(None, callback, data))
                
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)


# Global instances
webhook_manager = WebhookManager()
callback_manager = CallbackManager()


def send_face_detection_event(camera_id: str, faces: List[Dict[str, Any]], frame_metadata: Optional[Dict] = None):
    """Send face detection event to webhooks"""
    event = WebhookEvent(
        event_type="face_detected",
        timestamp=datetime.now().isoformat(),
        camera_id=camera_id,
        data={
            "faces": faces,
            "face_count": len(faces)
        },
        metadata=frame_metadata
    )
    webhook_manager.send_event(event)
    
    
def send_recognition_event(camera_id: str, recognized_faces: List[Dict[str, Any]], metadata: Optional[Dict] = None):
    """Send face recognition event to webhooks"""
    event = WebhookEvent(
        event_type="face_recognized",
        timestamp=datetime.now().isoformat(),
        camera_id=camera_id,
        data={
            "recognized_faces": recognized_faces,
            "recognized_count": len(recognized_faces)
        },
        metadata=metadata
    )
    webhook_manager.send_event(event)
    

def send_stranger_alert(camera_id: str, alert_data: Dict[str, Any]):
    """Send stranger alert to webhooks"""
    event = WebhookEvent(
        event_type="stranger_alert",
        timestamp=datetime.now().isoformat(),
        camera_id=camera_id,
        data=alert_data
    )
    webhook_manager.send_event(event)
    

def send_attendance_event(camera_id: str, attendance_data: Dict[str, Any]):
    """Send attendance event to webhooks"""
    event = WebhookEvent(
        event_type="attendance_recorded",
        timestamp=datetime.now().isoformat(),
        camera_id=camera_id,
        data=attendance_data
    )
    webhook_manager.send_event(event)