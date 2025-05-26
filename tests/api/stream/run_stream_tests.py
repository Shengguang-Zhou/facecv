#!/usr/bin/env python3
"""
Stream API Test Runner
Runs all stream-related tests with proper organization and reporting
"""

import sys
import os
import time
import requests
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from tests.api.stream.test_stream_sources import run_standalone_test as run_sources_tests
from tests.api.stream.test_stream_process import run_standalone_test as run_process_tests
from tests.api.stream.test_stream_comprehensive import run_comprehensive_tests


def check_server_running(base_url="http://localhost:7003"):
    """Check if the API server is running"""
    try:
        response = requests.get(f"{base_url}/api/v1/insightface/health", timeout=5)
        return response.status_code == 200
    except:
        return False


def wait_for_server(base_url="http://localhost:7003", max_attempts=10, delay=2):
    """Wait for server to be available"""
    print(f"🔍 Checking if server is running at {base_url}...")
    
    for attempt in range(max_attempts):
        if check_server_running(base_url):
            print(f"✅ Server is running at {base_url}")
            return True
        
        if attempt < max_attempts - 1:
            print(f"⏳ Server not ready, waiting {delay}s... (attempt {attempt + 1}/{max_attempts})")
            time.sleep(delay)
    
    return False


def run_all_stream_tests():
    """Run all stream API tests"""
    print("🚀 FaceCV Stream API Test Suite")
    print("=" * 80)
    print(f"Running tests against: http://localhost:7003")
    print("=" * 80)
    
    # Check server availability
    if not wait_for_server():
        print("❌ API server is not running at http://localhost:7003")
        print("   Please start the server with: python main.py")
        return False
    
    print()
    
    # Track overall results
    all_tests_passed = True
    test_suites = []
    
    # Test Suite 1: Stream Sources API
    print("📋 Test Suite 1: Stream Sources API")
    print("-" * 50)
    try:
        run_sources_tests()
        test_suites.append(("Stream Sources", True))
        print("✅ Stream Sources tests completed")
    except Exception as e:
        print(f"❌ Stream Sources tests failed: {str(e)}")
        test_suites.append(("Stream Sources", False))
        all_tests_passed = False
    
    print("\n")
    
    # Test Suite 2: Stream Processing API
    print("🎥 Test Suite 2: Stream Processing API")
    print("-" * 50)
    try:
        run_process_tests()
        test_suites.append(("Stream Processing", True))
        print("✅ Stream Processing tests completed")
    except Exception as e:
        print(f"❌ Stream Processing tests failed: {str(e)}")
        test_suites.append(("Stream Processing", False))
        all_tests_passed = False
    
    print("\n")
    
    # Test Suite 3: Comprehensive Integration Tests
    print("🔄 Test Suite 3: Comprehensive Integration Tests")
    print("-" * 50)
    try:
        comprehensive_passed = run_comprehensive_tests()
        test_suites.append(("Comprehensive", comprehensive_passed))
        if comprehensive_passed:
            print("✅ Comprehensive tests completed")
        else:
            print("❌ Some comprehensive tests failed")
            all_tests_passed = False
    except Exception as e:
        print(f"❌ Comprehensive tests failed: {str(e)}")
        test_suites.append(("Comprehensive", False))
        all_tests_passed = False
    
    # Final Summary
    print("\n" + "=" * 80)
    print("📊 FINAL TEST SUMMARY")
    print("=" * 80)
    
    for suite_name, passed in test_suites:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{status} {suite_name}")
    
    print("-" * 80)
    passed_count = sum(1 for _, passed in test_suites if passed)
    total_count = len(test_suites)
    
    print(f"Total Test Suites: {total_count}")
    print(f"Passed: {passed_count}")
    print(f"Failed: {total_count - passed_count}")
    print(f"Success Rate: {(passed_count/total_count*100):.1f}%")
    
    if all_tests_passed:
        print("\n🎉 ALL STREAM API TESTS PASSED!")
        print("The stream APIs are working correctly.")
    else:
        print("\n⚠️  SOME TESTS FAILED")
        print("Please check the detailed output above for specific failures.")
        print("Note: Camera/RTSP failures may be expected in CI environments.")
    
    print("=" * 80)
    
    return all_tests_passed


def run_quick_smoke_test():
    """Run a quick smoke test of the stream APIs"""
    print("💨 Quick Smoke Test for Stream APIs")
    print("-" * 40)
    
    if not check_server_running():
        print("❌ Server not running")
        return False
    
    try:
        # Test sources endpoint
        response = requests.get("http://localhost:7003/api/v1/stream/sources", timeout=10)
        if response.status_code == 200:
            print("✅ GET /stream/sources - OK")
        else:
            print(f"❌ GET /stream/sources - Status: {response.status_code}")
            return False
        
        # Test process endpoint with minimal params (quick test)
        params = {"source": "0", "duration": 1, "skip_frames": 30, "show_preview": False}
        response = requests.post("http://localhost:7003/api/v1/stream/process", params=params, timeout=15)
        
        if response.status_code in [200, 400, 404]:  # 400/404 expected if no camera
            print("✅ POST /stream/process - OK (or expected failure)")
        else:
            print(f"❌ POST /stream/process - Status: {response.status_code}")
            return False
        
        print("✅ Smoke test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Smoke test failed: {str(e)}")
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Stream API tests")
    parser.add_argument("--smoke", action="store_true", help="Run quick smoke test only")
    parser.add_argument("--server-port", type=int, default=7003, help="Server port (default: 7003)")
    
    args = parser.parse_args()
    
    if args.smoke:
        success = run_quick_smoke_test()
    else:
        success = run_all_stream_tests()
    
    sys.exit(0 if success else 1)