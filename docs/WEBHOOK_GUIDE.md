# FaceCV Webhook Integration Guide

## Overview

FaceCV supports real-time webhook notifications for face recognition events. This allows you to receive instant notifications when faces are detected, recognized, or when security alerts are triggered.

## Webhook Events

### Event Types

1. **face_detected** - Triggered when faces are detected in a frame
2. **face_recognized** - Triggered when faces are successfully recognized
3. **stranger_alert** - Triggered when an unknown person is detected
4. **attendance_recorded** - Triggered when attendance action is recorded
5. **stream_started** - Triggered when a camera stream starts
6. **stream_stopped** - Triggered when a camera stream stops

### Event Structure

All webhook events follow this structure:

```json
{
  "webhook_id": "main_recognition",
  "timestamp": "2025-05-26T10:30:45.123Z",
  "events": [
    {
      "event_type": "face_recognized",
      "timestamp": "2025-05-26T10:30:45.100Z",
      "camera_id": "entrance_camera",
      "data": {
        "recognized_faces": [
          {
            "name": "John Doe",
            "confidence": 0.95,
            "bbox": [100, 100, 200, 200],
            "metadata": {
              "department": "Engineering"
            }
          }
        ]
      },
      "metadata": {
        "source": "insightface_stream",
        "threshold": 0.6
      }
    }
  ]
}
```

## Integration Methods

### 1. Stream-Level Webhooks (Quick Setup)

Add webhook URLs directly to streaming endpoints:

```bash
# InsightFace Stream with Webhook
GET /api/v1/face_recognition_insightface/recognize/webcam/stream?
    camera_id=entrance&
    webhook_urls=http://localhost:8080/webhook,http://backup-server/webhook&
    webhook_timeout=30&
    webhook_retry_count=3

# DeepFace Stream with Webhook
GET /api/v1/face_recognition_deepface/recognize/webcam/stream?
    camera_id=parking&
    webhook_urls=http://notification-server/events&
    webhook_timeout=15
```

### 2. Webhook Management API (Advanced)

#### Create Webhook Configuration

```bash
POST /api/v1/webhooks
Content-Type: application/json

{
  "webhook_id": "main_recognition",
  "url": "http://localhost:8080/webhook/face-recognition",
  "headers": {
    "Authorization": "Bearer your-token"
  },
  "timeout": 30,
  "retry_count": 3,
  "batch_size": 10,
  "batch_timeout": 1.0,
  "enabled": true,
  "event_types": ["face_recognized", "stranger_alert"]
}
```

#### List Webhooks

```bash
GET /api/v1/webhooks
```

#### Update Webhook

```bash
PUT /api/v1/webhooks/{webhook_id}
```

#### Delete Webhook

```bash
DELETE /api/v1/webhooks/{webhook_id}
```

#### Test Webhook

```bash
POST /api/v1/webhooks/test
Content-Type: application/json

{
  "url": "http://localhost:8080/webhook/test",
  "test_data": {
    "message": "Test webhook"
  }
}
```

## Example Integration

### 1. Python Webhook Server

```python
from fastapi import FastAPI, Request
import logging

app = FastAPI()
logger = logging.getLogger(__name__)

@app.post("/webhook/face-recognition")
async def receive_webhook(request: Request):
    data = await request.json()
    
    for event in data.get("events", []):
        if event["event_type"] == "face_recognized":
            for face in event["data"]["recognized_faces"]:
                logger.info(f"Recognized: {face['name']} from {event['camera_id']}")
                # Process recognition...
                
    return {"status": "received"}
```

### 2. Node.js Webhook Server

```javascript
const express = require('express');
const app = express();

app.use(express.json());

app.post('/webhook/face-recognition', (req, res) => {
    const { events } = req.body;
    
    events.forEach(event => {
        if (event.event_type === 'face_recognized') {
            event.data.recognized_faces.forEach(face => {
                console.log(`Recognized: ${face.name} from ${event.camera_id}`);
                // Process recognition...
            });
        }
    });
    
    res.json({ status: 'received' });
});

app.listen(8080);
```

### 3. Multiple Camera Setup

```python
import requests

# Configure multiple cameras with webhooks
cameras = [
    {
        "camera_id": "entrance",
        "source": "0",
        "webhook_url": "http://localhost:8080/webhook/entrance"
    },
    {
        "camera_id": "parking",
        "source": "rtsp://192.168.1.100/stream",
        "webhook_url": "http://localhost:8080/webhook/parking"
    }
]

for camera in cameras:
    params = {
        "camera_id": camera["camera_id"],
        "source": camera["source"],
        "webhook_urls": camera["webhook_url"],
        "threshold": 0.6
    }
    
    # Start streaming with webhooks
    response = requests.get(
        "http://localhost:7000/api/v1/face_recognition_insightface/recognize/webcam/stream",
        params=params,
        stream=True
    )
```

## Best Practices

### 1. Webhook Security

- Use HTTPS for webhook URLs in production
- Implement authentication (Bearer tokens, API keys)
- Validate webhook signatures
- Use IP whitelisting if possible

### 2. Reliability

- Implement idempotency to handle duplicate events
- Store webhook events temporarily in case of processing failures
- Use appropriate timeout values (15-30 seconds)
- Monitor webhook delivery status

### 3. Performance

- Process webhooks asynchronously
- Use batch processing when possible
- Implement rate limiting on your webhook receiver
- Consider using a message queue for high volume

### 4. Error Handling

```python
@app.post("/webhook/face-recognition")
async def receive_webhook(request: Request):
    try:
        data = await request.json()
        # Process webhook...
        return {"status": "received"}
    except json.JSONDecodeError:
        return {"status": "error", "message": "Invalid JSON"}, 400
    except Exception as e:
        logger.error(f"Webhook processing error: {e}")
        return {"status": "error", "message": "Internal error"}, 500
```

## Monitoring

### Check Webhook Statistics

```bash
GET /api/v1/webhooks/stats

Response:
{
  "total_webhooks": 3,
  "enabled_webhooks": 3,
  "queue_size": 0,
  "manager_running": true
}
```

### View Webhook Logs

Monitor FaceCV logs for webhook delivery status:

```
INFO: Successfully delivered 5 events to main_recognition
WARNING: Webhook attendance_system returned 503
ERROR: Failed to deliver events to backup_webhook after 3 attempts
```

## Troubleshooting

### Common Issues

1. **Webhooks not being called**
   - Check if webhook manager is running
   - Verify webhook URL is accessible from FaceCV server
   - Check webhook configuration is enabled

2. **Webhook timeouts**
   - Increase timeout value in webhook configuration
   - Ensure webhook endpoint processes quickly
   - Consider async processing on receiver side

3. **Missing events**
   - Check batch settings (some events may be batched)
   - Verify event types match configuration
   - Check queue size isn't full

### Debug Mode

Enable detailed webhook logging:

```python
# In your FaceCV configuration
LOG_LEVEL=DEBUG
```

## Complete Example

See `/examples/webhook_integration.py` for a complete working example that demonstrates:

- Setting up webhooks
- Starting camera streams with webhook notifications
- Receiving and processing webhook events
- Real-time WebSocket broadcasting
- Monitoring webhook statistics

```bash
# Run the example webhook server
python examples/webhook_server.py

# In another terminal, run the integration
python examples/webhook_integration.py
```

---

**Note**: Webhook delivery is best-effort. For critical notifications, implement additional verification mechanisms.