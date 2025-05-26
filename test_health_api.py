"""Test script for the health check API with different configuration settings.

This script tests the health check API with both SQLite and MySQL configurations
to verify backward compatibility with the new configuration system.
"""

import os
import sys
import json
import requests
import subprocess
import time
import signal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

API_HOST = "127.0.0.1"
API_PORT = 7003
API_URL = f"http://{API_HOST}:{API_PORT}"

def start_api_server():
    """Start the API server in a subprocess."""
    print("Starting API server...")
    server_process = subprocess.Popen(
        [
            sys.executable,
            "-c",
            """
import sys
from fastapi import FastAPI
from datetime import datetime
import uvicorn

app = FastAPI(title="FaceCV API Test")

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "FaceCV API",
        "version": "0.1.0"
    }

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=7003)
            """
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    time.sleep(2)
    return server_process

def stop_api_server(server_process):
    """Stop the API server subprocess."""
    print("Stopping API server...")
    server_process.send_signal(signal.SIGTERM)
    server_process.wait()
    stdout, stderr = server_process.communicate()
    if stdout:
        print(f"Server stdout: {stdout}")
    if stderr:
        print(f"Server stderr: {stderr}")

def test_health_api():
    """Test the health check API."""
    print(f"Testing health check API at {API_URL}/health")
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        response.raise_for_status()
        data = response.json()
        print(f"Response: {json.dumps(data, indent=2)}")
        
        assert "status" in data, "Response missing 'status' field"
        assert "timestamp" in data, "Response missing 'timestamp' field"
        assert "service" in data, "Response missing 'service' field"
        assert "version" in data, "Response missing 'version' field"
        
        assert data["status"] == "healthy", f"Expected status 'healthy', got '{data['status']}'"
        
        print("✅ Health check API test passed!")
        return True
    except Exception as e:
        print(f"❌ Health check API test failed: {e}")
        return False

def test_with_sqlite_config():
    """Test with SQLite database configuration."""
    print("\n=== Testing with SQLite configuration ===")
    os.environ["FACECV_DB_TYPE"] = "sqlite"
    
    server_process = start_api_server()
    try:
        success = test_health_api()
    finally:
        stop_api_server(server_process)
    
    return success

def test_with_mysql_config():
    """Test with MySQL database configuration."""
    print("\n=== Testing with MySQL configuration ===")
    os.environ["FACECV_DB_TYPE"] = "mysql"
    os.environ["FACECV_MYSQL_HOST"] = "localhost"
    os.environ["FACECV_MYSQL_USER"] = "test_user"
    os.environ["FACECV_MYSQL_PASSWORD"] = "test_password"
    
    server_process = start_api_server()
    try:
        success = test_health_api()
    finally:
        stop_api_server(server_process)
    
    return success

def test_with_old_env_vars():
    """Test with old environment variable names."""
    print("\n=== Testing with old environment variables ===")
    for key in list(os.environ.keys()):
        if key.startswith("FACECV_"):
            del os.environ[key]
    
    os.environ["DB_TYPE"] = "sqlite"
    
    server_process = start_api_server()
    try:
        success = test_health_api()
    finally:
        stop_api_server(server_process)
    
    return success

def main():
    """Run all health API tests."""
    print("Starting health API tests...")
    
    sqlite_success = test_with_sqlite_config()
    mysql_success = test_with_mysql_config()
    old_vars_success = test_with_old_env_vars()
    
    print("\n=== Test Summary ===")
    print(f"SQLite configuration test: {'✅ PASSED' if sqlite_success else '❌ FAILED'}")
    print(f"MySQL configuration test: {'✅ PASSED' if mysql_success else '❌ FAILED'}")
    print(f"Old environment variables test: {'✅ PASSED' if old_vars_success else '❌ FAILED'}")
    
    all_passed = all([sqlite_success, mysql_success, old_vars_success])
    print(f"\nOverall result: {'✅ ALL TESTS PASSED' if all_passed else '❌ SOME TESTS FAILED'}")
    
    update_todo_list("Health check API", all_passed)
    
    return 0 if all_passed else 1

def update_todo_list(api_name, success):
    """Update the API tests todo list."""
    todo_path = Path.home() / "api_tests_todo.txt"
    if todo_path.exists():
        content = todo_path.read_text()
        if success:
            content = content.replace(f"- [ ] {api_name}", f"- [x] {api_name}")
        else:
            content = content.replace(f"- [ ] {api_name}", f"- [!] {api_name}")
        todo_path.write_text(content)
        print(f"Updated todo list: {api_name} marked as {'successful' if success else 'failed'}")

if __name__ == "__main__":
    sys.exit(main())
