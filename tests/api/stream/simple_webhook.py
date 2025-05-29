#!/usr/bin/env python3
"""Simple webhook server using built-in http.server"""

from http.server import HTTPServer, BaseHTTPRequestHandler
import json
from datetime import datetime

class WebhookHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path == '/webhook':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            
            try:
                data = json.loads(post_data.decode('utf-8'))
                event_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                print(f"\n[{event_time}] Webhook received:")
                print(f"  Timestamp: {data.get('timestamp')}")
                
                # Handle different webhook formats
                if 'events' in data:
                    # Batch format
                    print(f"  Events count: {len(data.get('events', []))}")
                    for event in data.get('events', []):
                        print(f"\n  Event: {event.get('event_type')}")
                        print(f"  Camera: {event.get('camera_id')}")
                        faces = event.get('data', {}).get('recognized_faces', [])
                        print(f"  Faces: {len(faces)}")
                        for face in faces:
                            print(f"    - {face.get('name')}: {face.get('confidence', 0):.2f}")
                else:
                    # Direct format
                    faces = data.get('faces', [])
                    print(f"  Source: {data.get('source')}")
                    print(f"  Faces: {len(faces)}")
                    for face in faces:
                        print(f"    - {face.get('name')}: {face.get('confidence', 0):.2f}")
                
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({'status': 'ok'}).encode())
                
            except Exception as e:
                print(f"Error processing webhook: {e}")
                self.send_response(500)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({'status': 'error', 'message': str(e)}).encode())
        else:
            self.send_response(404)
            self.end_headers()
    
    def do_GET(self):
        if self.path == '/health':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'status': 'healthy'}).encode())
        else:
            self.send_response(404)
            self.end_headers()
    
    def log_message(self, format, *args):
        # Suppress default logging
        pass

def run_server(port=8080):
    server_address = ('', port)
    httpd = HTTPServer(server_address, WebhookHandler)
    print(f"Webhook server running on http://localhost:{port}")
    print(f"Webhook endpoint: http://localhost:{port}/webhook")
    print(f"Health endpoint: http://localhost:{port}/health")
    print("Press Ctrl+C to stop...")
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping webhook server...")
        httpd.shutdown()

if __name__ == '__main__':
    run_server()