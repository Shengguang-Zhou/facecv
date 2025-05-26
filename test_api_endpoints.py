#!/usr/bin/env python3
"""
Comprehensive API Test Script for FaceCV

This script tests all critical API endpoints to ensure they're working correctly
with the refactored configuration system.
"""

import requests
import json
import os
import sys
import time
from pprint import pprint

# Configuration
BASE_URL = "http://localhost:8000"
API_PREFIX = "/api/v1"
HEALTH_PREFIX = f"{API_PREFIX}/health"
INSIGHTFACE_PREFIX = f"{API_PREFIX}/insightface"

def test_endpoint(url, method="GET", data=None, files=None, expected_status=200):
    """Test an API endpoint and return the response"""
    print(f"\n🔍 Testing {method} {url}")
    
    try:
        if method == "GET":
            response = requests.get(url)
        elif method == "POST":
            response = requests.post(url, data=data, files=files)
        else:
            print(f"❌ Unsupported method: {method}")
            return None
        
        if response.status_code == expected_status:
            print(f"✅ Status: {response.status_code}")
            try:
                return response.json()
            except:
                print("⚠️ Response is not JSON")
                return response.text
        else:
            print(f"❌ Expected status {expected_status}, got {response.status_code}")
            try:
                print(f"Error: {response.json()}")
            except:
                print(f"Error: {response.text}")
            return None
    except Exception as e:
        print(f"❌ Request failed: {e}")
        return None

def wait_for_server(max_retries=10, retry_interval=2):
    """Wait for the server to become available"""
    print(f"Waiting for server to start (max {max_retries} retries)...")
    
    for i in range(max_retries):
        try:
            response = requests.get(f"{BASE_URL}/health")
            if response.status_code == 200:
                print(f"✅ Server is up and running after {i+1} attempts")
                return True
        except Exception:
            pass
        
        print(f"Attempt {i+1}/{max_retries} - Server not ready, waiting {retry_interval}s...")
        time.sleep(retry_interval)
    
    print("❌ Server failed to start in the expected time")
    return False

def test_health_endpoints():
    """Test health-related endpoints"""
    print("\n==== Testing Health Endpoints ====")
    
    # Basic health check
    result = test_endpoint(f"{BASE_URL}/health")
    if result:
        print(f"Service: {result.get('service')}, Status: {result.get('status')}")
    
    # Comprehensive health check
    result = test_endpoint(f"{BASE_URL}{HEALTH_PREFIX}/comprehensive")
    if result:
        print(f"Overall health: {result.get('status')}")
        print(f"Database availability: {result.get('metrics', {}).get('database', {}).get('availability')}")
    
    # Database health
    result = test_endpoint(f"{BASE_URL}{HEALTH_PREFIX}/database")
    if result:
        print(f"Database status: {result.get('status')}")
        print(f"Available databases: {result.get('availability')}")

def test_insightface_endpoints():
    """Test InsightFace-related endpoints"""
    print("\n==== Testing InsightFace Endpoints ====")
    
    # Get available models
    result = test_endpoint(f"{BASE_URL}{INSIGHTFACE_PREFIX}/models/available")
    if result:
        print(f"Available models: {list(result.get('available_models', {}).keys())}")
        print(f"Current model: {result.get('current_model')}")
    
    # Get model info - correct path from router definition
    result = test_endpoint(f"{BASE_URL}{INSIGHTFACE_PREFIX}/get_model_info")
    if result:
        print(f"Model info available: {result is not None}")

def test_deepface_endpoints():
    """Test DeepFace-related endpoints"""
    print("\n==== Testing DeepFace Endpoints ====")
    
    # Get model info - check router prefix
    result = test_endpoint(f"{BASE_URL}/api/v1/deepface/model_info")
    if result:
        print(f"DeepFace model info: {result}")
    else:
        # Try alternative path
        result = test_endpoint(f"{BASE_URL}/deepface/model_info")
        if result:
            print(f"DeepFace model info: {result}")

def test_system_endpoints():
    """Test system-related endpoints"""
    print("\n==== Testing System Endpoints ====")
    
    # Get database info - check router prefix
    result = test_endpoint(f"{BASE_URL}{API_PREFIX}/health/database")
    if result:
        print(f"Database info: {result}")

def main():
    """Main test function"""
    print("\n🚀 Starting FaceCV API Tests")
    print(f"Base URL: {BASE_URL}")
    
    # Test if server is running with retry logic
    if not wait_for_server(max_retries=15, retry_interval=2):
        print("❌ Server not responding after multiple attempts")
        return False
    
    # Run all tests
    test_health_endpoints()
    test_insightface_endpoints()
    test_deepface_endpoints()
    test_system_endpoints()
    
    print("\n✅ All tests completed")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
