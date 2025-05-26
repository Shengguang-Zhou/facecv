#!/usr/bin/env python3
"""
Test runner for implemented APIs only (Camera and Batch)
"""
import sys
import os
from pathlib import Path

# Add parent directories to path
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent / "utils"))

from utils.test_base import APITestBase, TestResults
from api.camera.test_camera_streaming import TestCameraStreaming
from api.batch.test_batch_processing import TestBatchProcessing


def main():
    """Test only the implemented APIs"""
    print("🎯 Testing Implemented APIs (Camera + Batch)")
    print("=" * 60)
    
    # Check service availability
    api_test = APITestBase("http://localhost:7003/api/v1")
    print("Checking service availability...")
    if not api_test.wait_for_service():
        print("❌ Service not available at http://localhost:7003")
        return 1
    
    print("✅ Service is available\n")
    
    # Run camera tests
    print("📹 Running Camera API Tests...")
    camera_tester = TestCameraStreaming()
    camera_results = camera_tester.run_all_tests()
    
    print("\n" + "="*60)
    
    # Run batch tests
    print("📦 Running Batch API Tests...")
    batch_tester = TestBatchProcessing()
    batch_results = batch_tester.run_all_tests()
    
    # Combined summary
    total_tests = camera_results.total + batch_results.total
    total_passed = camera_results.passed + batch_results.passed
    total_failed = camera_results.failed + batch_results.failed
    
    print(f"\n{'='*60}")
    print(f"🎯 COMBINED RESULTS")
    print(f"{'='*60}")
    print(f"Camera Tests: {camera_results.passed}/{camera_results.total} passed")
    print(f"Batch Tests:  {batch_results.passed}/{batch_results.total} passed")
    print(f"TOTAL:        {total_passed}/{total_tests} passed ({total_passed/total_tests*100:.1f}%)")
    
    if total_failed == 0:
        print("🎉 All implemented APIs working correctly!")
        return 0
    else:
        print(f"⚠️  {total_failed} tests failed - see details above")
        return 1


if __name__ == "__main__":
    exit(main())