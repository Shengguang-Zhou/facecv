#!/usr/bin/env python3
"""
Test script to check stream status and verify it's processing
"""

import requests
import time
import sys

BASE_URL = "http://localhost:7003/api/v1/insightface"

def start_stream(camera_id=0):
    """Start a recognition stream"""
    params = {
        "camera_id": camera_id,
        "webhook_url": "",  # No webhook
        "skip_frames": 1,
        "model": "buffalo_l",
        "use_scrfd": True,
        "return_frame": False,
        "draw_bbox": True,
        "threshold": 0.35
    }
    
    response = requests.get(f"{BASE_URL}/stream/process_recognition", params=params)
    if response.status_code == 200:
        result = response.json()
        print(f"✓ Stream started successfully")
        print(f"  Stream ID: {result['stream_id']}")
        print(f"  Status: {result['status']}")
        print(f"  Camera: {result['camera_id']}")
        return result['stream_id']
    else:
        print(f"✗ Failed to start stream: {response.status_code}")
        print(f"  Error: {response.text}")
        return None

def get_active_streams():
    """Get list of active streams"""
    response = requests.get(f"{BASE_URL}/stream/active")
    if response.status_code == 200:
        return response.json()
    return None

def get_stream_status(stream_id):
    """Get status of a specific stream"""
    response = requests.get(f"{BASE_URL}/stream/status/{stream_id}")
    if response.status_code == 200:
        return response.json()
    return None

def stop_stream(stream_id):
    """Stop a stream"""
    response = requests.post(f"{BASE_URL}/stream/stop/{stream_id}")
    if response.status_code == 200:
        print(f"✓ Stream {stream_id} stopped successfully")
        return True
    else:
        print(f"✗ Failed to stop stream: {response.text}")
        return False

def test_with_simple_webhook():
    """Test with a simple webhook to see results"""
    print("\n" + "="*60)
    print("Testing with simple webhook server...")
    
    # Start a simple HTTP server to receive webhooks
    print("\nTo see actual results, run this in another terminal:")
    print("python -m http.server 8888")
    print("\nThen run this command:")
    print('curl -X GET "http://localhost:7003/api/v1/insightface/stream/process_recognition?camera_id=0&webhook_url=http://localhost:8888&return_frame=true"')
    print("\nYou'll see POST requests in the HTTP server logs when faces are detected.")

if __name__ == "__main__":
    print("Stream Status Test")
    print("="*60)
    
    # Start a stream
    print("\n1. Starting stream...")
    stream_id = start_stream()
    
    if stream_id:
        print("\n2. Waiting 3 seconds for processing...")
        time.sleep(3)
        
        print("\n3. Checking active streams...")
        active = get_active_streams()
        if active:
            print(f"✓ Found {active['count']} active streams")
            for stream in active['streams']:
                print(f"  - Stream {stream['stream_id'][:8]}... on camera {stream['camera_id']}: {stream['status']}")
        
        print("\n4. Checking specific stream status...")
        status = get_stream_status(stream_id)
        if status:
            print(f"✓ Stream status: {status['status']}")
            print(f"  Active: {status['active']}")
            if 'error' in status:
                print(f"  Error: {status['error']}")
        
        print("\n5. The stream is running in background!")
        print("   - It's processing frames from camera 0")
        print("   - But without webhook, results aren't sent anywhere")
        print("   - No visible window opens (it's a background task)")
        
        print("\n6. Press Enter to stop the stream...")
        input()
        
        print("\n7. Stopping stream...")
        stop_stream(stream_id)
    
    # Show how to test with webhook
    test_with_simple_webhook()
    
    print("\n" + "="*60)
    print("Summary:")
    print("- Streams run in background (no visible window)")
    print("- Without webhook URL, results aren't sent anywhere")
    print("- Use simple_frame_viewer.py to see frames visually")
    print("- Or use any HTTP server as webhook to see results")