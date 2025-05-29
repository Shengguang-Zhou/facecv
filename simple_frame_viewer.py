#!/usr/bin/env python3
"""
Simple webhook server to view frames from stream processing

Run this server, then use its URL as webhook_url when starting streams.
Frames will be displayed in an OpenCV window.
"""

from flask import Flask, request, jsonify
import base64
import cv2
import numpy as np
import threading
import queue
import time

app = Flask(__name__)
frame_queue = queue.Queue(maxsize=10)

@app.route('/', methods=['POST'])
def webhook():
    """Receive webhook events with frames"""
    try:
        data = request.json
        
        # Extract frame if present
        if data and 'frame_base64' in data:
            # Decode base64 frame
            frame_data = base64.b64decode(data['frame_base64'])
            np_arr = np.frombuffer(frame_data, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
            # Add frame info
            info_text = f"Stream: {data.get('stream_id', 'unknown')}"
            if 'faces' in data:
                info_text += f" | Faces: {len(data['faces'])}"
                for i, face in enumerate(data['faces']):
                    name = face.get('name', 'unknown')
                    confidence = face.get('confidence', 0)
                    info_text += f" | {name}: {confidence:.2f}"
            
            cv2.putText(frame, info_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Put frame in queue (drop old frames if full)
            try:
                frame_queue.put_nowait((frame, data))
            except queue.Full:
                try:
                    frame_queue.get_nowait()  # Remove oldest
                    frame_queue.put_nowait((frame, data))
                except:
                    pass
        
        # Print event info
        event_type = data.get('event_type', 'unknown')
        print(f"\nReceived {event_type} event:")
        print(f"  Stream ID: {data.get('stream_id')}")
        print(f"  Camera: {data.get('camera_id')}")
        if 'faces' in data:
            print(f"  Faces detected: {len(data['faces'])}")
            for face in data['faces']:
                print(f"    - {face.get('name')}: {face.get('confidence', 0):.3f}")
        
        return jsonify({"status": "received"}), 200
    
    except Exception as e:
        print(f"Error processing webhook: {e}")
        return jsonify({"error": str(e)}), 400

def display_frames():
    """Display frames in OpenCV window"""
    print("Starting frame display thread...")
    cv2.namedWindow('Stream Viewer', cv2.WINDOW_NORMAL)
    
    last_frame = None
    while True:
        try:
            # Get frame from queue (timeout to show last frame)
            frame, data = frame_queue.get(timeout=0.1)
            last_frame = frame
        except queue.Empty:
            frame = last_frame
        
        if frame is not None:
            cv2.imshow('Stream Viewer', frame)
        
        # Check for 'q' key to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Quit requested")
            cv2.destroyAllWindows()
            break
        
        time.sleep(0.01)  # Small delay

if __name__ == '__main__':
    print("Simple Frame Viewer Webhook Server")
    print("=" * 50)
    print("This server receives webhook events and displays frames")
    print("Press 'q' in the frame window to quit")
    print("\nWebhook URL: http://localhost:8889")
    print("\nExample usage:")
    print('curl -X GET "http://localhost:7003/api/v1/insightface/stream/process_recognition?camera_id=0&webhook_url=http://localhost:8889&return_frame=true&draw_bbox=true"')
    print("\n" + "=" * 50)
    
    # Start frame display thread
    display_thread = threading.Thread(target=display_frames, daemon=True)
    display_thread.start()
    
    # Start Flask server
    app.run(host='0.0.0.0', port=8889, debug=False)