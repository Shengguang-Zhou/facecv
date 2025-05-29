#!/usr/bin/env python3
"""Example integration script showing how to use FaceCV with webhooks"""

import requests
import json
import time
import threading
from datetime import datetime

# FaceCV API configuration
FACECV_API = "http://localhost:7000"
WEBHOOK_SERVER = "http://localhost:8080"

def setup_webhooks():
    """Configure webhooks in FaceCV"""
    
    webhooks = [
        {
            "webhook_id": "main_recognition",
            "url": f"{WEBHOOK_SERVER}/webhook/face-recognition",
            "event_types": ["face_recognized", "stranger_alert"],
            "batch_size": 5,
            "batch_timeout": 0.5
        },
        {
            "webhook_id": "attendance_system",
            "url": f"{WEBHOOK_SERVER}/webhook/attendance",
            "event_types": ["attendance_recorded"],
            "batch_size": 1,
            "batch_timeout": 0.1
        },
        {
            "webhook_id": "realtime_broadcast",
            "url": f"{WEBHOOK_SERVER}/webhook/realtime",
            "event_types": ["face_recognized"],
            "batch_size": 1,
            "batch_timeout": 0.1
        }
    ]
    
    for webhook in webhooks:
        try:
            response = requests.post(
                f"{FACECV_API}/api/v1/webhooks",
                json=webhook
            )
            if response.status_code == 200:
                print(f"✓ Configured webhook: {webhook['webhook_id']}")
            else:
                print(f"✗ Failed to configure webhook: {webhook['webhook_id']}")
        except Exception as e:
            print(f"Error setting up webhook: {e}")


def start_camera_stream_with_webhook(camera_id: str, source: str = "0"):
    """Start a camera stream with webhook notifications"""
    
    # Configure stream with webhooks
    params = {
        "camera_id": camera_id,
        "source": source,
        "threshold": 0.6,
        "fps": 10,
        "webhook_urls": f"{WEBHOOK_SERVER}/webhook/face-recognition",
        "webhook_timeout": 30,
        "webhook_retry_count": 3
    }
    
    print(f"Starting camera stream {camera_id} with webhooks...")
    
    try:
        # Start InsightFace stream
        response = requests.get(
            f"{FACECV_API}/api/v1/face_recognition_insightface/recognize/webcam/stream",
            params=params,
            stream=True
        )
        
        # Process SSE stream
        for line in response.iter_lines():
            if line:
                line = line.decode('utf-8')
                if line.startswith('data: '):
                    data = json.loads(line[6:])
                    print(f"Stream {camera_id}: {len(data.get('faces', []))} faces detected")
                    
    except KeyboardInterrupt:
        print(f"Stopping camera stream {camera_id}")
    except Exception as e:
        print(f"Stream error: {e}")


def test_webhook_delivery():
    """Test webhook delivery with sample data"""
    
    test_data = {
        "url": f"{WEBHOOK_SERVER}/webhook/face-recognition",
        "test_data": {
            "message": "Testing webhook delivery",
            "faces": [
                {
                    "name": "John Doe",
                    "confidence": 0.92,
                    "bbox": [100, 100, 200, 200]
                }
            ]
        }
    }
    
    try:
        response = requests.post(
            f"{FACECV_API}/api/v1/webhooks/test",
            json=test_data
        )
        if response.status_code == 200:
            print("✓ Test webhook sent successfully")
        else:
            print("✗ Test webhook failed")
    except Exception as e:
        print(f"Error testing webhook: {e}")


def monitor_webhook_stats():
    """Monitor webhook delivery statistics"""
    
    while True:
        try:
            response = requests.get(f"{FACECV_API}/api/v1/webhooks/stats")
            if response.status_code == 200:
                stats = response.json()
                print(f"\nWebhook Stats - {datetime.now().strftime('%H:%M:%S')}")
                print(f"  Total webhooks: {stats['total_webhooks']}")
                print(f"  Enabled: {stats['enabled_webhooks']}")
                print(f"  Queue size: {stats['queue_size']}")
                print(f"  Manager running: {stats['manager_running']}")
            
            time.sleep(5)
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error getting stats: {e}")
            time.sleep(5)


def main():
    """Main integration example"""
    
    print("FaceCV Webhook Integration Example")
    print("==================================\n")
    
    # Step 1: Setup webhooks
    print("1. Setting up webhooks...")
    setup_webhooks()
    
    # Step 2: Test webhook delivery
    print("\n2. Testing webhook delivery...")
    test_webhook_delivery()
    
    # Step 3: Start monitoring in background
    print("\n3. Starting webhook statistics monitor...")
    monitor_thread = threading.Thread(target=monitor_webhook_stats, daemon=True)
    monitor_thread.start()
    
    # Step 4: Start camera streams
    print("\n4. Starting camera streams with webhooks...")
    print("Press Ctrl+C to stop\n")
    
    # You can start multiple camera streams in threads
    cameras = [
        {"camera_id": "entrance", "source": "0"},
        # {"camera_id": "parking", "source": "rtsp://192.168.1.100/stream"},
        # {"camera_id": "lobby", "source": "1"},
    ]
    
    threads = []
    for camera in cameras:
        thread = threading.Thread(
            target=start_camera_stream_with_webhook,
            args=(camera["camera_id"], camera["source"]),
            daemon=True
        )
        thread.start()
        threads.append(thread)
    
    # Wait for threads
    try:
        for thread in threads:
            thread.join()
    except KeyboardInterrupt:
        print("\nShutting down...")


if __name__ == "__main__":
    main()