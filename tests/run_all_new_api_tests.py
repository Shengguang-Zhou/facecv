#!/usr/bin/env python3
"""
Comprehensive test runner for all new FaceCV APIs
"""
import sys
import os
import time
import argparse
from pathlib import Path

# Add parent directories to path
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent / "utils"))

from utils.test_base import APITestBase, TestResults
from api.models.test_model_management import TestModelManagement
from api.health.test_system_health import TestSystemHealth
from api.camera.test_camera_streaming import TestCameraStreaming
from api.batch.test_batch_processing import TestBatchProcessing


class ComprehensiveTestRunner:
    """Run all API tests comprehensively"""
    
    def __init__(self, base_url: str = "http://localhost:7003/api/v1"):
        self.base_url = base_url
        self.overall_results = TestResults()
        self.test_suites = []
    
    def wait_for_service(self, max_attempts: int = 30) -> bool:
        """Wait for the FaceCV service to be available"""
        print(f"🔍 Checking service availability at {self.base_url.replace('/api/v1', '')}")
        
        api_test = APITestBase(self.base_url)
        for attempt in range(max_attempts):
            try:
                response = api_test.get("/health")
                if response.status_code == 200:
                    print(f"✅ Service is available (attempt {attempt + 1})")
                    return True
                elif response.status_code == 404:
                    # Try basic root endpoint
                    response = api_test.get("")
                    if response.status_code in [200, 404]:
                        print(f"✅ Service is available (attempt {attempt + 1})")
                        return True
            except Exception as e:
                if attempt < 5:  # Only show errors for first few attempts
                    print(f"⏳ Waiting for service... (attempt {attempt + 1}: {str(e)[:50]})")
            time.sleep(1)
        
        print(f"❌ Service not available after {max_attempts} attempts")
        return False
    
    def run_test_suite(self, suite_name: str, test_class, run_individually: bool = False):
        """Run a test suite and collect results"""
        print(f"\n{'='*60}")
        print(f"🧪 Running {suite_name} Tests")
        print(f"{'='*60}")
        
        try:
            if run_individually:
                # For troubleshooting individual tests
                tester = test_class()
                results = tester.run_all_tests()
            else:
                # Standard run
                tester = test_class()
                results = tester.run_all_tests()
            
            # Merge results
            self.overall_results.total += results.total
            self.overall_results.passed += results.passed
            self.overall_results.failed += results.failed
            self.overall_results.failed_tests.extend([f"{suite_name}: {test}" for test in results.failed_tests])
            
            self.test_suites.append({
                'name': suite_name,
                'total': results.total,
                'passed': results.passed,
                'failed': results.failed,
                'success_rate': (results.passed / results.total * 100) if results.total > 0 else 0
            })
            
            return results.failed == 0
            
        except Exception as e:
            print(f"❌ Failed to run {suite_name} tests: {str(e)}")
            self.overall_results.total += 1
            self.overall_results.failed += 1
            self.overall_results.failed_tests.append(f"{suite_name}: Suite execution failed")
            return False
    
    def print_comprehensive_summary(self):
        """Print comprehensive test summary"""
        print(f"\n{'='*80}")
        print(f"🎯 COMPREHENSIVE TEST RESULTS")
        print(f"{'='*80}")
        
        # Suite-by-suite summary
        print(f"📊 Test Suite Summary:")
        print(f"{'-'*80}")
        print(f"{'Suite Name':<25} {'Total':<8} {'Passed':<8} {'Failed':<8} {'Success Rate':<12}")
        print(f"{'-'*80}")
        
        for suite in self.test_suites:
            print(f"{suite['name']:<25} {suite['total']:<8} {suite['passed']:<8} {suite['failed']:<8} {suite['success_rate']:>9.1f}%")
        
        print(f"{'-'*80}")
        
        # Overall summary
        total_success_rate = (self.overall_results.passed / self.overall_results.total * 100) if self.overall_results.total > 0 else 0
        print(f"{'OVERALL TOTAL':<25} {self.overall_results.total:<8} {self.overall_results.passed:<8} {self.overall_results.failed:<8} {total_success_rate:>9.1f}%")
        
        # Status indication
        if self.overall_results.failed == 0:
            print(f"\n🎉 ALL TESTS PASSED! Service is working correctly.")
        elif total_success_rate >= 80:
            print(f"\n⚠️  MOSTLY WORKING: {total_success_rate:.1f}% success rate (some features may not be implemented)")
        elif total_success_rate >= 50:
            print(f"\n⚠️  PARTIALLY WORKING: {total_success_rate:.1f}% success rate (significant issues found)")
        else:
            print(f"\n❌ MAJOR ISSUES: {total_success_rate:.1f}% success rate (service may not be properly running)")
        
        # Failed tests details
        if self.overall_results.failed_tests:
            print(f"\n❌ Failed Tests ({len(self.overall_results.failed_tests)}):")
            for i, test in enumerate(self.overall_results.failed_tests, 1):
                print(f"   {i:2d}. {test}")
        
        # Recommendations
        print(f"\n💡 Recommendations:")
        if self.overall_results.failed == 0:
            print("   • All APIs are working correctly")
            print("   • Ready for production use")
        elif any("not implemented" in test.lower() for test in self.overall_results.failed_tests):
            print("   • Some APIs may not be implemented yet - this is expected")
            print("   • Focus on implementing missing endpoints")
        elif any("not available" in test.lower() for test in self.overall_results.failed_tests):
            print("   • Some hardware/external resources not available - this is normal in test environments")
            print("   • Camera and GPU tests may fail without hardware")
        else:
            print("   • Check service logs for detailed error information")
            print("   • Verify all dependencies are installed and configured")
            print("   • Ensure models are properly loaded")
        
        print(f"{'='*80}")
    
    def run_all_tests(self, include_hardware: bool = True, quick_test: bool = False):
        """Run all test suites"""
        print(f"🚀 Starting Comprehensive API Test Suite")
        print(f"Base URL: {self.base_url}")
        print(f"Include Hardware Tests: {include_hardware}")
        print(f"Quick Test Mode: {quick_test}")
        
        # Check service availability first
        if not self.wait_for_service():
            print("\n❌ Cannot proceed with tests - service is not available")
            return False
        
        # Core API tests (always run)
        core_suites = [
            ("System Health", TestSystemHealth),
            ("Model Management", TestModelManagement),
        ]
        
        # Hardware-dependent tests
        hardware_suites = [
            ("Camera Streaming", TestCameraStreaming),
            ("Batch Processing", TestBatchProcessing),
        ]
        
        # Run core tests
        all_passed = True
        for suite_name, test_class in core_suites:
            success = self.run_test_suite(suite_name, test_class)
            all_passed = all_passed and success
            
            if quick_test and not success:
                print(f"⚠️  Quick test mode: Stopping after first failure in {suite_name}")
                break
        
        # Run hardware tests if requested
        if include_hardware and (not quick_test or all_passed):
            for suite_name, test_class in hardware_suites:
                success = self.run_test_suite(suite_name, test_class)
                all_passed = all_passed and success
                
                if quick_test and not success:
                    print(f"⚠️  Quick test mode: Stopping after first failure in {suite_name}")
                    break
        
        # Print comprehensive summary
        self.print_comprehensive_summary()
        
        return all_passed


def main():
    """Main test runner"""
    parser = argparse.ArgumentParser(description="FaceCV API Comprehensive Test Suite")
    parser.add_argument("--url", default="http://localhost:7003/api/v1", 
                       help="Base URL for the API (default: http://localhost:7003/api/v1)")
    parser.add_argument("--no-hardware", action="store_true", 
                       help="Skip hardware-dependent tests (camera, GPU)")
    parser.add_argument("--quick", action="store_true", 
                       help="Quick test mode - stop on first failure")
    parser.add_argument("--suite", choices=["health", "models", "camera", "batch"], 
                       help="Run only specific test suite")
    
    args = parser.parse_args()
    
    runner = ComprehensiveTestRunner(args.url)
    
    if args.suite:
        # Run specific suite
        suite_mapping = {
            "health": ("System Health", TestSystemHealth),
            "models": ("Model Management", TestModelManagement),
            "camera": ("Camera Streaming", TestCameraStreaming),
            "batch": ("Batch Processing", TestBatchProcessing),
        }
        
        if args.suite in suite_mapping:
            suite_name, test_class = suite_mapping[args.suite]
            
            if not runner.wait_for_service():
                print("❌ Service not available")
                return 1
            
            success = runner.run_test_suite(suite_name, test_class)
            runner.print_comprehensive_summary()
            return 0 if success else 1
        else:
            print(f"❌ Unknown test suite: {args.suite}")
            return 1
    else:
        # Run all tests
        success = runner.run_all_tests(
            include_hardware=not args.no_hardware,
            quick_test=args.quick
        )
        return 0 if success else 1


if __name__ == "__main__":
    exit(main())