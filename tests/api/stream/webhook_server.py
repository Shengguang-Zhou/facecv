#!/usr/bin/env python3
"""Simple webhook server for testing"""

from flask import Flask, request, jsonify
import json
from datetime import datetime

app = Flask(__name__)

# Store received events
events = []

@app.route('/webhook', methods=['POST'])
def webhook():
    """Receive webhook events"""
    try:
        data = request.json
        event_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        print(f"\n[{event_time}] Webhook received:")
        print(f"  Event type: {data.get('webhook_id', 'unknown')}")
        print(f"  Timestamp: {data.get('timestamp')}")
        print(f"  Events count: {len(data.get('events', []))}")
        
        for event in data.get('events', []):
            print(f"\n  Event: {event.get('event_type')}")
            print(f"  Camera: {event.get('camera_id')}")
            print(f"  Data: {json.dumps(event.get('data', {}), indent=2)}")
        
        events.append({
            'received_at': event_time,
            'data': data
        })
        
        return jsonify({'status': 'ok', 'message': 'Event received'}), 200
    except Exception as e:
        print(f"Error processing webhook: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/events', methods=['GET'])
def get_events():
    """Get all received events"""
    return jsonify({
        'total': len(events),
        'events': events[-10:]  # Last 10 events
    })

@app.route('/health', methods=['GET'])
def health():
    """Health check"""
    return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()})

if __name__ == '__main__':
    print("Starting webhook server on http://localhost:8080")
    print("Webhook endpoint: http://localhost:8080/webhook")
    print("Events endpoint: http://localhost:8080/events")
    print("Health endpoint: http://localhost:8080/health")
    app.run(host='0.0.0.0', port=8080, debug=True)