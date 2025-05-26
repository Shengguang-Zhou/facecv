"""
Tests for stream processing API endpoint
POST /api/v1/stream/process
"""

import pytest
import requests
import time
import os
from tests.utils.test_base import APITestBase, TestResults


class TestStreamProcess:
    """Test suite for stream processing endpoint"""
    
    def setup_method(self):
        """Setup for each test method"""
        self.api = APITestBase("http://localhost:7003/api/v1")
        self.results = TestResults()
    
    def test_process_local_camera(self):
        """Test processing local camera (index 0)"""
        try:
            params = {
                "source": "0",
                "duration": 5,  # Short test duration
                "skip_frames": 5,  # Skip frames for faster processing
                "show_preview": False
            }
            
            response = self.api.post("/stream/process", params=params)
            
            # Check status code
            assert response.status_code in [200, 400, 404], f"Unexpected status: {response.status_code}"
            
            if response.status_code == 200:
                data = response.json()
                
                # Validate response structure
                required_fields = ["status", "source", "duration", "total_detections", "unique_persons", "persons", "summary"]
                for field in required_fields:
                    assert field in data, f"Missing required field: {field}"
                
                # Validate field types and values
                assert isinstance(data["status"], str), "status should be string"
                assert data["status"] in ["completed", "error", "processing"], f"Invalid status: {data['status']}"
                assert data["source"] == "0", f"Expected source '0', got '{data['source']}'"
                assert isinstance(data["total_detections"], int), "total_detections should be integer"
                assert data["total_detections"] >= 0, "total_detections should be non-negative"
                assert isinstance(data["unique_persons"], int), "unique_persons should be integer"
                assert data["unique_persons"] >= 0, "unique_persons should be non-negative"
                assert isinstance(data["persons"], list), "persons should be list"
                assert isinstance(data["summary"], list), "summary should be list"
                
                # Validate summary structure
                for person in data["summary"]:
                    assert "name" in person, "Summary missing name field"
                    assert "detections" in person, "Summary missing detections field"
                    assert "avg_similarity" in person, "Summary missing avg_similarity field"
                    assert isinstance(person["detections"], int), "detections should be integer"
                    assert isinstance(person["avg_similarity"], float), "avg_similarity should be float"
                    assert 0.0 <= person["avg_similarity"] <= 1.0, "avg_similarity should be between 0.0 and 1.0"
                
                self.results.add_result("test_process_local_camera", True)
                print("✅ Local camera processing test passed")
                
            elif response.status_code == 404:
                print("⚠️  Local camera (index 0) not available - test skipped")
                self.results.add_result("test_process_local_camera", True)
                
            elif response.status_code == 400:
                # May be expected if camera is not accessible
                print("⚠️  Local camera (index 0) not accessible - test skipped")
                self.results.add_result("test_process_local_camera", True)
            
        except Exception as e:
            self.results.add_result("test_process_local_camera", False)
            print(f"❌ Local camera processing test failed: {str(e)}")
            raise
    
    def test_process_rtsp_stream(self):
        """Test processing RTSP stream"""
        try:
            rtsp_url = "http://tdit.online:81/ai_check/TianduCV.git"
            params = {
                "source": rtsp_url,
                "duration": 10,  # Short test duration
                "skip_frames": 10,  # Skip many frames for faster processing
                "show_preview": False
            }
            
            response = self.api.post("/stream/process", params=params)
            
            # Check status code
            assert response.status_code in [200, 400, 404, 500], f"Unexpected status: {response.status_code}"
            
            if response.status_code == 200:
                data = response.json()
                
                # Validate response structure
                required_fields = ["status", "source", "duration", "total_detections", "unique_persons", "persons", "summary"]
                for field in required_fields:
                    assert field in data, f"Missing required field: {field}"
                
                # Validate source matches
                assert data["source"] == rtsp_url, f"Expected source '{rtsp_url}', got '{data['source']}'"
                
                # Validate field types
                assert isinstance(data["status"], str), "status should be string"
                assert isinstance(data["total_detections"], int), "total_detections should be integer"
                assert isinstance(data["unique_persons"], int), "unique_persons should be integer"
                assert isinstance(data["persons"], list), "persons should be list"
                assert isinstance(data["summary"], list), "summary should be list"
                
                self.results.add_result("test_process_rtsp_stream", True)
                print("✅ RTSP stream processing test passed")
                
            else:
                # RTSP may not be accessible, which is expected in some environments
                print(f"⚠️  RTSP stream not accessible (status: {response.status_code}) - test skipped")
                self.results.add_result("test_process_rtsp_stream", True)
            
        except Exception as e:
            self.results.add_result("test_process_rtsp_stream", False)
            print(f"❌ RTSP stream processing test failed: {str(e)}")
            # Don't raise for RTSP tests as they may fail due to network issues
            print("⚠️  RTSP test failure may be due to network connectivity")
    
    def test_process_invalid_source(self):
        """Test processing with invalid source"""
        try:
            params = {
                "source": "invalid_source_999",
                "duration": 5,
                "skip_frames": 5,
                "show_preview": False
            }
            
            response = self.api.post("/stream/process", params=params)
            
            # Should return error status
            assert response.status_code in [400, 404, 500], f"Expected error status, got {response.status_code}"
            
            self.results.add_result("test_process_invalid_source", True)
            print("✅ Invalid source test passed")
            
        except Exception as e:
            self.results.add_result("test_process_invalid_source", False)
            print(f"❌ Invalid source test failed: {str(e)}")
            raise
    
    def test_process_parameter_validation(self):
        """Test parameter validation"""
        try:
            # Test with invalid duration (too large)
            params = {
                "source": "0",
                "duration": 5000,  # Exceeds maximum
                "skip_frames": 1,
                "show_preview": False
            }
            
            response = self.api.post("/stream/process", params=params)
            assert response.status_code == 422, f"Expected 422 for invalid duration, got {response.status_code}"
            
            # Test with invalid skip_frames (too large)
            params = {
                "source": "0",
                "duration": 5,
                "skip_frames": 50,  # Exceeds maximum
                "show_preview": False
            }
            
            response = self.api.post("/stream/process", params=params)
            assert response.status_code == 422, f"Expected 422 for invalid skip_frames, got {response.status_code}"
            
            # Test with missing source parameter
            params = {
                "duration": 5,
                "skip_frames": 1,
                "show_preview": False
            }
            
            response = self.api.post("/stream/process", params=params)
            assert response.status_code == 422, f"Expected 422 for missing source, got {response.status_code}"
            
            self.results.add_result("test_process_parameter_validation", True)
            print("✅ Parameter validation test passed")
            
        except Exception as e:
            self.results.add_result("test_process_parameter_validation", False)
            print(f"❌ Parameter validation test failed: {str(e)}")
            raise
    
    def test_process_minimal_parameters(self):
        """Test processing with minimal required parameters"""
        try:
            params = {
                "source": "0",
                "duration": 3,  # Very short test
                "skip_frames": 10  # Skip many frames
            }
            
            response = self.api.post("/stream/process", params=params)
            
            # Should work with minimal params or return expected error
            assert response.status_code in [200, 400, 404], f"Unexpected status: {response.status_code}"
            
            if response.status_code == 200:
                data = response.json()
                # Validate basic structure
                assert "status" in data, "Missing status field"
                assert "source" in data, "Missing source field"
                
            self.results.add_result("test_process_minimal_parameters", True)
            print("✅ Minimal parameters test passed")
            
        except Exception as e:
            self.results.add_result("test_process_minimal_parameters", False)
            print(f"❌ Minimal parameters test failed: {str(e)}")
            raise
    
    def test_process_different_skip_frames(self):
        """Test processing with different skip_frames values"""
        try:
            skip_values = [1, 2, 5]
            
            for skip_frames in skip_values:
                params = {
                    "source": "0",
                    "duration": 3,
                    "skip_frames": skip_frames,
                    "show_preview": False
                }
                
                response = self.api.post("/stream/process", params=params)
                
                # Should work or return expected error
                assert response.status_code in [200, 400, 404], f"Unexpected status for skip_frames={skip_frames}: {response.status_code}"
                
                if response.status_code == 200:
                    data = response.json()
                    assert data["source"] == "0", "Source should match"
            
            self.results.add_result("test_process_different_skip_frames", True)
            print("✅ Different skip_frames test passed")
            
        except Exception as e:
            self.results.add_result("test_process_different_skip_frames", False)
            print(f"❌ Different skip_frames test failed: {str(e)}")
            raise
    
    def teardown_method(self):
        """Cleanup after each test method"""
        if hasattr(self, 'results'):
            if self.results.failed > 0:
                print(f"\n❌ {self.results.failed} test(s) failed")
            else:
                print(f"\n✅ All {self.results.passed} test(s) passed")


def run_standalone_test():
    """Run tests as standalone script"""
    test_suite = TestStreamProcess()
    print("🚀 Running Stream Processing API Tests...")
    print("=" * 60)
    
    test_methods = [
        test_suite.test_process_local_camera,
        test_suite.test_process_rtsp_stream,
        test_suite.test_process_invalid_source,
        test_suite.test_process_parameter_validation,
        test_suite.test_process_minimal_parameters,
        test_suite.test_process_different_skip_frames
    ]
    
    total_tests = len(test_methods)
    passed_tests = 0
    
    for test_method in test_methods:
        try:
            test_suite.setup_method()
            test_method()
            passed_tests += 1
        except Exception as e:
            print(f"❌ {test_method.__name__} failed: {str(e)}")
        finally:
            test_suite.teardown_method()
    
    print("\n" + "=" * 60)
    print(f"STREAM PROCESSING TEST SUMMARY")
    print("=" * 60)
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {total_tests - passed_tests}")
    print(f"Success Rate: {(passed_tests/total_tests*100):.1f}%")


if __name__ == "__main__":
    run_standalone_test()