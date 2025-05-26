"""
Tests for stream sources API endpoint
GET /api/v1/stream/sources
"""

import pytest
import requests
from tests.utils.test_base import APITestBase, TestResults


class TestStreamSources:
    """Test suite for stream sources endpoint"""
    
    def setup_method(self):
        """Setup for each test method"""
        self.api = APITestBase("http://localhost:7003/api/v1")
        self.results = TestResults()
    
    def test_get_sources_success(self):
        """Test successful retrieval of video sources"""
        try:
            response = self.api.get("/stream/sources")
            
            # Check status code
            assert response.status_code == 200, f"Expected 200, got {response.status_code}"
            
            # Parse JSON response
            data = response.json()
            
            # Validate response structure
            required_fields = ["camera_sources", "rtsp_examples", "file_support", "notes"]
            for field in required_fields:
                assert field in data, f"Missing required field: {field}"
            
            # Validate camera_sources structure
            assert isinstance(data["camera_sources"], list), "camera_sources should be a list"
            for camera in data["camera_sources"]:
                assert "index" in camera, "Camera missing index field"
                assert "description" in camera, "Camera missing description field"
                assert isinstance(camera["index"], int), "Camera index should be integer"
                assert isinstance(camera["description"], str), "Camera description should be string"
            
            # Validate rtsp_examples structure
            assert isinstance(data["rtsp_examples"], list), "rtsp_examples should be a list"
            assert len(data["rtsp_examples"]) > 0, "Should have at least one RTSP example"
            for example in data["rtsp_examples"]:
                assert isinstance(example, str), "RTSP example should be string"
                assert "rtsp://" in example, "RTSP example should contain rtsp://"
            
            # Validate file_support structure
            file_support = data["file_support"]
            assert isinstance(file_support, dict), "file_support should be a dict"
            assert "formats" in file_support, "file_support missing formats field"
            assert "example" in file_support, "file_support missing example field"
            assert isinstance(file_support["formats"], list), "formats should be a list"
            assert len(file_support["formats"]) > 0, "Should support at least one format"
            
            # Validate notes structure
            assert isinstance(data["notes"], list), "notes should be a list"
            assert len(data["notes"]) > 0, "Should have at least one note"
            
            self.results.add_result("test_get_sources_success", True)
            print("✅ Stream sources API test passed")
            
        except Exception as e:
            self.results.add_result("test_get_sources_success", False)
            print(f"❌ Stream sources API test failed: {str(e)}")
            raise
    
    def test_get_sources_response_format(self):
        """Test response format and content validation"""
        try:
            response = self.api.get("/stream/sources")
            assert response.status_code == 200
            
            data = response.json()
            
            # Check that camera sources contain expected fields
            if data["camera_sources"]:
                camera = data["camera_sources"][0]
                assert camera["index"] >= 0, "Camera index should be non-negative"
            
            # Check RTSP examples format
            for rtsp in data["rtsp_examples"]:
                assert rtsp.startswith("rtsp://"), "RTSP URL should start with rtsp://"
            
            # Check file formats are valid
            valid_formats = ["mp4", "avi", "mov", "mkv", "wmv", "flv", "webm"]
            for format_type in data["file_support"]["formats"]:
                assert format_type.lower() in valid_formats, f"Unexpected format: {format_type}"
            
            # Check example path format
            example_path = data["file_support"]["example"]
            assert "/" in example_path, "Example path should contain path separator"
            assert "." in example_path, "Example path should contain file extension"
            
            self.results.add_result("test_get_sources_response_format", True)
            print("✅ Response format validation passed")
            
        except Exception as e:
            self.results.add_result("test_get_sources_response_format", False)
            print(f"❌ Response format validation failed: {str(e)}")
            raise
    
    def test_get_sources_no_params_required(self):
        """Test that no parameters are required"""
        try:
            # Test with no parameters
            response = self.api.get("/stream/sources")
            assert response.status_code == 200
            
            # Test with empty parameters (should still work)
            response_with_params = self.api.get("/stream/sources", params={})
            assert response_with_params.status_code == 200
            
            # Both responses should be identical
            data1 = response.json()
            data2 = response_with_params.json()
            assert data1 == data2, "Responses should be identical regardless of empty params"
            
            self.results.add_result("test_get_sources_no_params_required", True)
            print("✅ No parameters required test passed")
            
        except Exception as e:
            self.results.add_result("test_get_sources_no_params_required", False)
            print(f"❌ No parameters required test failed: {str(e)}")
            raise
    
    def test_get_sources_content_type(self):
        """Test response content type"""
        try:
            response = self.api.get("/stream/sources")
            assert response.status_code == 200
            
            # Check content type
            content_type = response.headers.get('content-type', '')
            assert 'application/json' in content_type, f"Expected JSON response, got: {content_type}"
            
            self.results.add_result("test_get_sources_content_type", True)
            print("✅ Content type test passed")
            
        except Exception as e:
            self.results.add_result("test_get_sources_content_type", False)
            print(f"❌ Content type test failed: {str(e)}")
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
    test_suite = TestStreamSources()
    print("🚀 Running Stream Sources API Tests...")
    print("=" * 60)
    
    test_methods = [
        test_suite.test_get_sources_success,
        test_suite.test_get_sources_response_format,
        test_suite.test_get_sources_no_params_required,
        test_suite.test_get_sources_content_type
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
    print(f"STREAM SOURCES TEST SUMMARY")
    print("=" * 60)
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {total_tests - passed_tests}")
    print(f"Success Rate: {(passed_tests/total_tests*100):.1f}%")


if __name__ == "__main__":
    run_standalone_test()