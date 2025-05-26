"""Webhook management API routes"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from typing import List, Dict, Optional
from pydantic import BaseModel, HttpUrl
import logging
from datetime import datetime

from facecv.core.webhook import webhook_manager, WebhookConfig, WebhookEvent
from facecv.config import get_settings

router = APIRouter()
logger = logging.getLogger(__name__)


class WebhookConfigRequest(BaseModel):
    """Webhook configuration request model"""
    webhook_id: str
    url: HttpUrl
    headers: Optional[Dict[str, str]] = None
    timeout: int = 30
    retry_count: int = 3
    retry_delay: int = 1
    batch_size: int = 10
    batch_timeout: float = 1.0
    enabled: bool = True
    event_types: Optional[List[str]] = None  # Filter specific event types


class WebhookConfigResponse(BaseModel):
    """Webhook configuration response model"""
    webhook_id: str
    url: str
    enabled: bool
    created_at: datetime
    event_types: Optional[List[str]]
    statistics: Optional[Dict[str, int]] = None


class WebhookTestRequest(BaseModel):
    """Webhook test request model"""
    url: HttpUrl
    test_data: Optional[Dict] = None


@router.post("/webhooks", response_model=WebhookConfigResponse)
async def create_webhook(config: WebhookConfigRequest):
    """
    Create a new webhook configuration
    """
    try:
        webhook_config = WebhookConfig(
            url=str(config.url),
            headers=config.headers,
            timeout=config.timeout,
            retry_count=config.retry_count,
            retry_delay=config.retry_delay,
            batch_size=config.batch_size,
            batch_timeout=config.batch_timeout,
            enabled=config.enabled
        )
        
        webhook_manager.add_webhook(config.webhook_id, webhook_config)
        
        # Start webhook manager if not running
        if not webhook_manager.running:
            webhook_manager.start()
        
        logger.info(f"Created webhook {config.webhook_id}")
        
        return WebhookConfigResponse(
            webhook_id=config.webhook_id,
            url=str(config.url),
            enabled=config.enabled,
            created_at=datetime.now(),
            event_types=config.event_types
        )
        
    except Exception as e:
        logger.error(f"Error creating webhook: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create webhook: {str(e)}")


@router.get("/webhooks", response_model=List[WebhookConfigResponse])
async def list_webhooks():
    """
    List all configured webhooks
    """
    webhooks = []
    
    for webhook_id, config in webhook_manager.webhooks.items():
        webhooks.append(WebhookConfigResponse(
            webhook_id=webhook_id,
            url=config.url,
            enabled=config.enabled,
            created_at=datetime.now(),  # Would need to track this properly
            event_types=None
        ))
    
    return webhooks


@router.get("/webhooks/{webhook_id}", response_model=WebhookConfigResponse)
async def get_webhook(webhook_id: str):
    """
    Get specific webhook configuration
    """
    if webhook_id not in webhook_manager.webhooks:
        raise HTTPException(status_code=404, detail="Webhook not found")
    
    config = webhook_manager.webhooks[webhook_id]
    
    return WebhookConfigResponse(
        webhook_id=webhook_id,
        url=config.url,
        enabled=config.enabled,
        created_at=datetime.now(),
        event_types=None
    )


@router.put("/webhooks/{webhook_id}", response_model=WebhookConfigResponse)
async def update_webhook(webhook_id: str, config: WebhookConfigRequest):
    """
    Update webhook configuration
    """
    if webhook_id not in webhook_manager.webhooks:
        raise HTTPException(status_code=404, detail="Webhook not found")
    
    webhook_config = WebhookConfig(
        url=str(config.url),
        headers=config.headers,
        timeout=config.timeout,
        retry_count=config.retry_count,
        retry_delay=config.retry_delay,
        batch_size=config.batch_size,
        batch_timeout=config.batch_timeout,
        enabled=config.enabled
    )
    
    webhook_manager.update_webhook(webhook_id, webhook_config)
    
    return WebhookConfigResponse(
        webhook_id=webhook_id,
        url=str(config.url),
        enabled=config.enabled,
        created_at=datetime.now(),
        event_types=config.event_types
    )


@router.delete("/webhooks/{webhook_id}")
async def delete_webhook(webhook_id: str):
    """
    Delete webhook configuration
    """
    if webhook_id not in webhook_manager.webhooks:
        raise HTTPException(status_code=404, detail="Webhook not found")
    
    webhook_manager.remove_webhook(webhook_id)
    
    return {"message": f"Webhook {webhook_id} deleted successfully"}


@router.post("/webhooks/{webhook_id}/enable")
async def enable_webhook(webhook_id: str):
    """
    Enable a webhook
    """
    if webhook_id not in webhook_manager.webhooks:
        raise HTTPException(status_code=404, detail="Webhook not found")
    
    config = webhook_manager.webhooks[webhook_id]
    config.enabled = True
    webhook_manager.update_webhook(webhook_id, config)
    
    return {"message": f"Webhook {webhook_id} enabled"}


@router.post("/webhooks/{webhook_id}/disable")
async def disable_webhook(webhook_id: str):
    """
    Disable a webhook
    """
    if webhook_id not in webhook_manager.webhooks:
        raise HTTPException(status_code=404, detail="Webhook not found")
    
    config = webhook_manager.webhooks[webhook_id]
    config.enabled = False
    webhook_manager.update_webhook(webhook_id, config)
    
    return {"message": f"Webhook {webhook_id} disabled"}


@router.post("/webhooks/test")
async def test_webhook(test_request: WebhookTestRequest, background_tasks: BackgroundTasks):
    """
    Test a webhook URL with sample data
    """
    test_event = WebhookEvent(
        event_type="test",
        timestamp=datetime.now().isoformat(),
        camera_id="test_camera",
        data=test_request.test_data or {
            "message": "This is a test webhook event",
            "faces": [
                {
                    "name": "Test Person",
                    "confidence": 0.95,
                    "bbox": [100, 100, 200, 200]
                }
            ]
        },
        metadata={"test": True}
    )
    
    # Create temporary webhook config
    test_config = WebhookConfig(
        url=str(test_request.url),
        timeout=10,
        retry_count=1
    )
    
    # Test the webhook in background
    async def send_test():
        try:
            if not webhook_manager._session:
                import aiohttp
                async with aiohttp.ClientSession() as session:
                    await webhook_manager._deliver_to_webhook(
                        "test",
                        test_config,
                        [test_event]
                    )
            else:
                await webhook_manager._deliver_to_webhook(
                    "test",
                    test_config,
                    [test_event]
                )
            logger.info(f"Test webhook sent to {test_request.url}")
        except Exception as e:
            logger.error(f"Test webhook failed: {e}")
    
    background_tasks.add_task(send_test)
    
    return {
        "message": "Test webhook sent",
        "url": str(test_request.url),
        "test_event": test_event.to_dict()
    }


@router.get("/webhooks/stats")
async def get_webhook_statistics():
    """
    Get webhook delivery statistics
    """
    stats = {
        "total_webhooks": len(webhook_manager.webhooks),
        "enabled_webhooks": sum(1 for w in webhook_manager.webhooks.values() if w.enabled),
        "queue_size": webhook_manager.event_queue.qsize(),
        "manager_running": webhook_manager.running
    }
    
    return stats


@router.on_event("startup")
async def startup_webhook_manager():
    """Start webhook manager on API startup"""
    webhook_manager.start()
    logger.info("Webhook manager started")


@router.on_event("shutdown")
async def shutdown_webhook_manager():
    """Stop webhook manager on API shutdown"""
    webhook_manager.stop()
    logger.info("Webhook manager stopped")