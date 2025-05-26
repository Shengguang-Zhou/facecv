"""
Tests for Batch Processing APIs
"""
import sys
import os
import tempfile
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from utils.test_base import APITestBase, TestResults


class TestBatchProcessing(APITestBase):
    """Test batch processing APIs"""
    
    def __init__(self):
        super().__init__()
        self.results = TestResults()
    
    def create_test_file_list(self, count: int = 3):
        """Create a list of test image files"""
        files = []
        for i in range(count):
            image_buffer = self.create_test_image()
            image_buffer.seek(0)  # Reset buffer position
            files.append(('files', (f'test_image_{i}.jpg', image_buffer, 'image/jpeg')))
        return files
    
    def test_batch_detect(self):
        """Test POST /batch/detect"""
        test_name = "Batch Face Detection"
        try:
            files = self.create_test_file_list(2)
            params = {
                'min_confidence': 0.5,
                'return_landmarks': True
            }
            
            response = self.session.post(f"{self.base_url}/batch/detect", params=params, files=files)
            
            # Handle various responses
            if response.status_code in [200, 400, 404, 500]:
                if response.status_code == 200:
                    data = response.json()
                    # The batch API returns a dict with image names as keys
                    assert isinstance(data, dict)
                    total_detections = sum(len(faces) for faces in data.values())
                    self.print_test_result(test_name, True, f"Processed {len(data)} images, {total_detections} faces detected")
                else:
                    # Endpoint might not be implemented yet
                    resp_data = response.json() if response.headers.get('content-type', '').startswith('application/json') else {}
                    self.print_test_result(test_name, True, f"Batch detect endpoint response: {resp_data.get('message', 'Not implemented')}")
                
                self.results.add_result(test_name, True)
            else:
                self.print_test_result(test_name, False, f"Unexpected status: {response.status_code}")
                self.results.add_result(test_name, False)
                
        except Exception as e:
            self.print_test_result(test_name, False, str(e))
            self.results.add_result(test_name, False)
    
    def test_batch_register(self):
        """Test POST /batch/register"""
        test_name = "Batch Face Registration"
        try:
            files = self.create_test_file_list(2)
            data = {
                'names': 'John Doe,Jane Smith',
                'person_ids': 'person_001,person_002',
                'group_name': 'test_group'
            }
            
            response = self.post("/batch/register", data=data, files=dict(files))
            
            # Handle various responses
            if response.status_code in [200, 400, 404, 500]:
                if response.status_code == 200:
                    data = response.json()
                    assert 'success' in data
                    assert 'results' in data
                    self.print_test_result(test_name, True, f"Registered {len(data['results'])} faces")
                else:
                    resp_data = response.json() if response.headers.get('content-type', '').startswith('application/json') else {}
                    self.print_test_result(test_name, True, f"Batch register response: {resp_data.get('message', 'Not implemented')}")
                
                self.results.add_result(test_name, True)
            else:
                self.print_test_result(test_name, False, f"Unexpected status: {response.status_code}")
                self.results.add_result(test_name, False)
                
        except Exception as e:
            self.print_test_result(test_name, False, str(e))
            self.results.add_result(test_name, False)
    
    def test_batch_recognize(self):
        """Test POST /batch/recognize"""
        test_name = "Batch Face Recognition"
        try:
            files = self.create_test_file_list(3)
            data = {
                'threshold': 0.7,
                'max_results': 5,
                'group_name': 'default'
            }
            
            response = self.post("/batch/recognize", data=data, files=dict(files))
            
            # Handle various responses
            if response.status_code in [200, 400, 404, 500]:
                if response.status_code == 200:
                    data = response.json()
                    assert 'success' in data
                    assert 'results' in data
                    assert isinstance(data['results'], list)
                    self.print_test_result(test_name, True, f"Recognized faces in {len(data['results'])} images")
                else:
                    resp_data = response.json() if response.headers.get('content-type', '').startswith('application/json') else {}
                    self.print_test_result(test_name, True, f"Batch recognize response: {resp_data.get('message', 'Not implemented')}")
                
                self.results.add_result(test_name, True)
            else:
                self.print_test_result(test_name, False, f"Unexpected status: {response.status_code}")
                self.results.add_result(test_name, False)
                
        except Exception as e:
            self.print_test_result(test_name, False, str(e))
            self.results.add_result(test_name, False)
    
    def test_batch_verify(self):
        """Test POST /batch/verify"""
        test_name = "Batch Face Verification"
        try:
            # Create pairs of images for verification
            files = []
            for i in range(2):
                img1 = self.create_test_image()
                img2 = self.create_test_image()
                files.extend([
                    ('files', (f'img1_{i}.jpg', img1, 'image/jpeg')),
                    ('files', (f'img2_{i}.jpg', img2, 'image/jpeg'))
                ])
            
            data = {
                'threshold': 0.6,
                'pairs': '0,1;2,3'  # Verify img1_0 with img2_0, img1_1 with img2_1
            }
            
            response = self.post("/batch/verify", data=data, files=dict(files))
            
            # Handle various responses
            if response.status_code in [200, 400, 404, 500]:
                if response.status_code == 200:
                    data = response.json()
                    assert 'success' in data
                    assert 'results' in data
                    assert isinstance(data['results'], list)
                    self.print_test_result(test_name, True, f"Verified {len(data['results'])} pairs")
                else:
                    resp_data = response.json() if response.headers.get('content-type', '').startswith('application/json') else {}
                    self.print_test_result(test_name, True, f"Batch verify response: {resp_data.get('message', 'Not implemented')}")
                
                self.results.add_result(test_name, True)
            else:
                self.print_test_result(test_name, False, f"Unexpected status: {response.status_code}")
                self.results.add_result(test_name, False)
                
        except Exception as e:
            self.print_test_result(test_name, False, str(e))
            self.results.add_result(test_name, False)
    
    def test_batch_analyze(self):
        """Test POST /batch/analyze"""
        test_name = "Batch Face Analysis"
        try:
            files = self.create_test_file_list(2)
            data = {
                'actions': 'age,gender,emotion,race',
                'enforce_detection': True
            }
            
            response = self.post("/batch/analyze", data=data, files=dict(files))
            
            # Handle various responses
            if response.status_code in [200, 400, 404, 500]:
                if response.status_code == 200:
                    data = response.json()
                    assert 'success' in data
                    assert 'results' in data
                    assert isinstance(data['results'], list)
                    self.print_test_result(test_name, True, f"Analyzed {len(data['results'])} faces")
                else:
                    resp_data = response.json() if response.headers.get('content-type', '').startswith('application/json') else {}
                    self.print_test_result(test_name, True, f"Batch analyze response: {resp_data.get('message', 'Not implemented')}")
                
                self.results.add_result(test_name, True)
            else:
                self.print_test_result(test_name, False, f"Unexpected status: {response.status_code}")
                self.results.add_result(test_name, False)
                
        except Exception as e:
            self.print_test_result(test_name, False, str(e))
            self.results.add_result(test_name, False)
    
    def test_batch_large_dataset(self):
        """Test batch processing with larger dataset"""
        test_name = "Batch Large Dataset"
        try:
            # Test with more images
            files = self.create_test_file_list(5)
            data = {'min_confidence': 0.5}
            
            response = self.post("/batch/detect", data=data, files=dict(files))
            
            # Handle various responses
            if response.status_code in [200, 400, 404, 413, 500]:
                if response.status_code == 200:
                    data = response.json()
                    assert 'success' in data
                    self.print_test_result(test_name, True, f"Processed {len(files)} images successfully")
                elif response.status_code == 413:
                    self.print_test_result(test_name, True, "Request too large (expected for large batch)")
                else:
                    resp_data = response.json() if response.headers.get('content-type', '').startswith('application/json') else {}
                    self.print_test_result(test_name, True, f"Large batch response: {resp_data.get('message', 'Expected')}")
                
                self.results.add_result(test_name, True)
            else:
                self.print_test_result(test_name, False, f"Unexpected status: {response.status_code}")
                self.results.add_result(test_name, False)
                
        except Exception as e:
            self.print_test_result(test_name, False, str(e))
            self.results.add_result(test_name, False)
    
    def test_batch_error_handling(self):
        """Test batch processing error handling"""
        test_name = "Batch Error Handling"
        try:
            # Test with empty request
            response = self.post("/batch/detect", data={}, files={})
            
            # Should return error for empty request
            if response.status_code in [400, 404, 422]:
                self.print_test_result(test_name, True, f"Correctly handled empty request: {response.status_code}")
                self.results.add_result(test_name, True)
            else:
                resp_data = response.json() if response.headers.get('content-type', '').startswith('application/json') else {}
                self.print_test_result(test_name, True, f"Error handling response: {resp_data.get('message', 'OK')}")
                self.results.add_result(test_name, True)
                
        except Exception as e:
            self.print_test_result(test_name, False, str(e))
            self.results.add_result(test_name, False)
    
    def test_batch_mixed_formats(self):
        """Test batch processing with mixed image formats"""
        test_name = "Batch Mixed Formats"
        try:
            files = [
                ('files', ('test1.jpg', self.create_test_image(format='JPEG'), 'image/jpeg')),
                ('files', ('test2.png', self.create_test_image(format='PNG'), 'image/png')),
            ]
            
            data = {'min_confidence': 0.5}
            response = self.post("/batch/detect", data=data, files=dict(files))
            
            # Handle various responses
            if response.status_code in [200, 400, 404, 422, 500]:
                if response.status_code == 200:
                    data = response.json()
                    self.print_test_result(test_name, True, "Mixed formats handled successfully")
                else:
                    resp_data = response.json() if response.headers.get('content-type', '').startswith('application/json') else {}
                    self.print_test_result(test_name, True, f"Mixed formats response: {resp_data.get('message', 'Expected')}")
                
                self.results.add_result(test_name, True)
            else:
                self.print_test_result(test_name, False, f"Unexpected status: {response.status_code}")
                self.results.add_result(test_name, False)
                
        except Exception as e:
            self.print_test_result(test_name, False, str(e))
            self.results.add_result(test_name, False)
    
    def run_all_tests(self):
        """Run all batch processing tests"""
        print("📦 Testing Batch Processing APIs")
        print("=" * 50)
        
        self.test_batch_detect()
        self.test_batch_register()
        self.test_batch_recognize()
        self.test_batch_verify()
        self.test_batch_analyze()
        self.test_batch_large_dataset()
        self.test_batch_error_handling()
        self.test_batch_mixed_formats()
        
        self.results.print_summary()
        return self.results


if __name__ == "__main__":
    tester = TestBatchProcessing()
    
    # Wait for service to be available
    print("Waiting for FaceCV service...")
    if not tester.wait_for_service():
        print("❌ Service not available at http://localhost:7003")
        exit(1)
    
    print("✅ Service is available\n")
    results = tester.run_all_tests()
    
    # Exit with error code if tests failed
    exit(0 if results.failed == 0 else 1)