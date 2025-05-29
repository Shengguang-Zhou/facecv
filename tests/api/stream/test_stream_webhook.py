#!/usr/bin/env python3
"""Test stream processing with webhook"""

import requests
import json
import time
import threading
from flask import Flask, request
import sys

# Webhook server setup
app = Flask(__name__)
received_events = []

@app.route('/webhook', methods=['POST'])
def webhook():
    """Receive webhook events"""
    data = request.json
    print(f"\n[WEBHOOK] Received event at {data.get('timestamp')}:")
    print(f"  Source: {data.get('source')}")
    print(f"  Faces detected: {len(data.get('faces', []))}")
    
    for face in data.get('faces', []):
        print(f"    - {face['name']}: {face['confidence']:.2f} at {face['bbox']}")
    
    received_events.append(data)
    return {'status': 'ok'}, 200

def run_webhook_server():
    """Run webhook server in background"""
    app.run(port=8080, debug=False)

def test_stream_processing():
    """Test stream processing API"""
    base_url = "http://localhost:7003/api/v1"
    
    # Start webhook server in background
    webhook_thread = threading.Thread(target=run_webhook_server, daemon=True)
    webhook_thread.start()
    time.sleep(2)  # Give server time to start
    
    print("Testing stream processing with webhook...")
    
    # Test parameters
    params = {
        "source": "0",  # Default camera
        "webhook_url": "http://localhost:8080/webhook",
        "skip_frames": "2",  # Process every 2nd frame
        "show_preview": "false"
    }
    
    print(f"\nStarting stream processing:")
    print(f"  Source: {params['source']}")
    print(f"  Webhook URL: {params['webhook_url']}")
    print(f"  Skip frames: {params['skip_frames']}")
    
    try:
        # Start stream processing
        response = requests.post(
            f"{base_url}/stream/process",
            params=params,
            timeout=10  # Short timeout as it runs continuously
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"\nStream processing completed:")
            print(f"  Status: {result['status']}")
            print(f"  Total detections: {result['total_detections']}")
            print(f"  Unique persons: {result['unique_persons']}")
            print(f"  Persons: {result['persons']}")
        else:
            print(f"\nError: {response.status_code}")
            print(response.text)
            
    except requests.exceptions.Timeout:
        print("\nStream is processing (this is expected for continuous processing)")
        print("Press Ctrl+C to stop...")
        
        # Wait a bit to collect some webhook events
        time.sleep(5)
        
        print(f"\nReceived {len(received_events)} webhook events")
        
    except KeyboardInterrupt:
        print("\nStopping stream processing...")
        
    except Exception as e:
        print(f"\nError: {e}")

if __name__ == "__main__":
    test_stream_processing()