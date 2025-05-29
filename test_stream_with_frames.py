#!/usr/bin/env python3
"""
Test script for stream processing with frame viewing

This script demonstrates how to:
1. Start stream processing without webhook
2. View frames with bounding boxes drawn
"""

import requests
import base64
import cv2
import numpy as np
import time
import sys

# API base URL
BASE_URL = "http://localhost:7003/api/v1/insightface"

def start_recognition_stream(camera_id=0, return_frame=True):
    """Start face recognition stream"""
    params = {
        "camera_id": camera_id,
        "webhook_url": "",  # Empty webhook - no webhook sending
        "return_frame": return_frame,  # Get frames with bbox drawn
        "draw_bbox": True,
        "skip_frames": 5  # Process every 5th frame for better performance
    }
    
    response = requests.get(f"{BASE_URL}/stream/process_recognition", params=params)
    if response.status_code == 200:
        result = response.json()
        print(f"Stream started: {result['stream_id']}")
        return result['stream_id']
    else:
        print(f"Failed to start stream: {response.text}")
        return None

def view_frames_with_simple_webhook():
    """
    Since frames are sent via webhook, we need a simple webhook server
    to receive and display them. Without webhook, frames won't be sent.
    """
    print("\nTo view frames in real-time, you have two options:")
    print("\n1. Use a simple webhook server (recommended):")
    print("   Run this command in another terminal:")
    print("   python -m http.server 8888")
    print("   Then use webhook_url='http://localhost:8888' when starting stream")
    
    print("\n2. Use the test webhook server:")
    print("   python tests/api/stream/webhook_server.py")
    print("   This will display frames in a window")
    
    print("\n3. For testing without viewing frames:")
    print("   Leave webhook_url empty - processing will happen but no results sent")

def stop_stream(stream_id):
    """Stop a running stream"""
    response = requests.post(f"{BASE_URL}/stream/stop/{stream_id}")
    if response.status_code == 200:
        print(f"Stream {stream_id} stopped")
    else:
        print(f"Failed to stop stream: {response.text}")

if __name__ == "__main__":
    print("Stream Processing Test (Without Webhook)")
    print("=" * 50)
    
    # Show options for viewing frames
    view_frames_with_simple_webhook()
    
    print("\n" + "=" * 50)
    print("Starting stream without webhook (no frame viewing)...")
    
    # Start recognition stream without webhook
    stream_id = start_recognition_stream(camera_id=0, return_frame=False)
    
    if stream_id:
        print(f"\nStream {stream_id} is running in background")
        print("Processing frames but not sending results anywhere")
        print("Press Enter to stop the stream...")
        input()
        
        # Stop the stream
        stop_stream(stream_id)
    
    print("\nTo view frames, you need to provide a webhook URL!")
    print("Example with webhook:")
    print('curl -X GET "http://localhost:7003/api/v1/insightface/stream/process_recognition?camera_id=0&webhook_url=http://localhost:8888&return_frame=true&draw_bbox=true"')