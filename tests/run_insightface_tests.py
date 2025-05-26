#!/usr/bin/env python3
"""
InsightFace API Test Runner
===========================

Runs comprehensive tests for InsightFace APIs.
Requires server to be running on port 7003.
"""
import sys
import subprocess
import requests
import time


def check_server():
    """Check if API server is running"""
    try:
        response = requests.get("http://localhost:7003/api/v1/insightface/health", timeout=5)
        return response.status_code == 200
    except:
        return False


def main():
    print("🧪 InsightFace API Test Runner")
    print("=" * 50)
    
    # Check server
    print("🔍 Checking API server...")
    if not check_server():
        print("❌ API server not running on port 7003")
        print("💡 Please start server with: python start_api_server.py")
        sys.exit(1)
    
    print("✅ API server is running")
    
    # Run tests
    print("\n🚀 Running InsightFace API tests...")
    
    cmd = [
        sys.executable, "-m", "pytest", 
        "tests/api/insightface/test_insightface_comprehensive.py",
        "-v",
        "--tb=short",
        "--durations=10"
    ]
    
    try:
        result = subprocess.run(cmd, check=False)
        
        if result.returncode == 0:
            print("\n✅ All tests passed!")
        else:
            print(f"\n❌ Tests failed with return code: {result.returncode}")
            
        return result.returncode
        
    except Exception as e:
        print(f"\n💥 Error running tests: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())