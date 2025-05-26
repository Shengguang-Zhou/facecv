"""
Base test utilities for FaceCV API tests
"""
import requests
import json
import time
import os
from typing import Dict, Any, Optional, List
from io import BytesIO
from PIL import Image
import numpy as np


class APITestBase:
    """Base class for API testing with common utilities"""
    
    def __init__(self, base_url: str = "http://localhost:7003/api/v1"):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'FaceCV-API-Test/1.0'
        })
    
    def get(self, endpoint: str, params: Optional[Dict] = None, **kwargs) -> requests.Response:
        """Make GET request"""
        url = f"{self.base_url}{endpoint}"
        return self.session.get(url, params=params, **kwargs)
    
    def post(self, endpoint: str, data: Optional[Dict] = None, json_data: Optional[Dict] = None, 
             files: Optional[Dict] = None, params: Optional[Dict] = None, **kwargs) -> requests.Response:
        """Make POST request"""
        url = f"{self.base_url}{endpoint}"
        return self.session.post(url, data=data, json=json_data, files=files, params=params, **kwargs)
    
    def put(self, endpoint: str, data: Optional[Dict] = None, json_data: Optional[Dict] = None, **kwargs) -> requests.Response:
        """Make PUT request"""
        url = f"{self.base_url}{endpoint}"
        return self.session.put(url, data=data, json=json_data, **kwargs)
    
    def delete(self, endpoint: str, **kwargs) -> requests.Response:
        """Make DELETE request"""
        url = f"{self.base_url}{endpoint}"
        return self.session.delete(url, **kwargs)
    
    def assert_success_response(self, response: requests.Response, expected_status: int = 200):
        """Assert successful response"""
        assert response.status_code == expected_status, f"Expected {expected_status}, got {response.status_code}: {response.text}"
        if response.headers.get('content-type', '').startswith('application/json'):
            data = response.json()
            assert 'success' in data, f"Response missing 'success' field: {data}"
            assert data['success'] is True, f"API returned success=False: {data}"
        return response
    
    def assert_error_response(self, response: requests.Response, expected_status: int):
        """Assert error response"""
        assert response.status_code == expected_status, f"Expected {expected_status}, got {response.status_code}: {response.text}"
        return response
    
    def create_test_image(self, width: int = 300, height: int = 300, format: str = 'JPEG') -> BytesIO:
        """Create a test image"""
        # Create a simple colored image
        image_array = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        image = Image.fromarray(image_array)
        
        buffer = BytesIO()
        image.save(buffer, format=format)
        buffer.seek(0)
        return buffer
    
    def get_test_face_image(self) -> BytesIO:
        """Get a test face image (if available from test dataset)"""
        test_image_path = "/home/a/PycharmProjects/EurekCV/dataset/faces"
        if os.path.exists(test_image_path):
            # Try to find a test image
            for file in os.listdir(test_image_path):
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    with open(os.path.join(test_image_path, file), 'rb') as f:
                        return BytesIO(f.read())
        
        # Fallback to generated image
        return self.create_test_image()
    
    def wait_for_service(self, max_attempts: int = 30, delay: float = 1.0) -> bool:
        """Wait for service to be available"""
        for attempt in range(max_attempts):
            try:
                response = self.get("/insightface/health")
                if response.status_code == 200:
                    return True
            except requests.exceptions.ConnectionError:
                pass
            time.sleep(delay)
        return False
    
    def print_test_result(self, test_name: str, success: bool, message: str = ""):
        """Print formatted test result"""
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}")
        if message:
            print(f"   {message}")
    
    def run_endpoint_test(self, test_name: str, endpoint: str, method: str = "GET", 
                         data: Optional[Dict] = None, files: Optional[Dict] = None,
                         expected_status: int = 200, should_succeed: bool = True) -> bool:
        """Run a single endpoint test"""
        try:
            if method.upper() == "GET":
                response = self.get(endpoint, params=data)
            elif method.upper() == "POST":
                response = self.post(endpoint, data=data, files=files)
            elif method.upper() == "PUT":
                response = self.put(endpoint, data=data)
            elif method.upper() == "DELETE":
                response = self.delete(endpoint)
            else:
                raise ValueError(f"Unsupported method: {method}")
            
            if should_succeed:
                self.assert_success_response(response, expected_status)
                self.print_test_result(test_name, True, f"{method} {endpoint} - Status: {response.status_code}")
            else:
                self.assert_error_response(response, expected_status)
                self.print_test_result(test_name, True, f"{method} {endpoint} - Expected error: {response.status_code}")
            
            return True
            
        except Exception as e:
            self.print_test_result(test_name, False, f"{method} {endpoint} - Error: {str(e)}")
            return False


class TestResults:
    """Track test results"""
    
    def __init__(self):
        self.total = 0
        self.passed = 0
        self.failed = 0
        self.failed_tests = []
    
    def add_result(self, test_name: str, success: bool):
        """Add test result"""
        self.total += 1
        if success:
            self.passed += 1
        else:
            self.failed += 1
            self.failed_tests.append(test_name)
    
    def print_summary(self):
        """Print test summary"""
        print(f"\n{'='*60}")
        print(f"TEST SUMMARY")
        print(f"{'='*60}")
        print(f"Total Tests: {self.total}")
        print(f"Passed: {self.passed}")
        print(f"Failed: {self.failed}")
        print(f"Success Rate: {(self.passed/self.total*100):.1f}%" if self.total > 0 else "No tests run")
        
        if self.failed_tests:
            print(f"\nFailed Tests:")
            for test in self.failed_tests:
                print(f"  - {test}")
        
        print(f"{'='*60}")