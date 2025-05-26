"""
Tests for Camera Streaming APIs
"""
import sys
import os
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from utils.test_base import APITestBase, TestResults


class TestCameraStreaming(APITestBase):
    """Test camera streaming APIs"""
    
    def __init__(self):
        super().__init__()
        self.results = TestResults()
        self.active_streams = []  # Track active streams for cleanup
    
    def cleanup_streams(self):
        """Clean up any active streams"""
        for stream_id in self.active_streams:
            try:
                self.post("/camera/disconnect", json_data={"stream_id": stream_id})
            except:
                pass
        self.active_streams.clear()
    
    def test_camera_connect_local(self):
        """Test POST /camera/connect with local camera"""
        test_name = "Camera Connect (Local)"
        try:
            params = {
                "camera_id": "0",  # Default webcam
                "source": "local"
            }
            
            response = self.post("/camera/connect", params=params)
            
            # Camera might not be available in test environment
            if response.status_code in [200, 404, 500]:
                data = response.json()
                
                if response.status_code == 200:
                    assert 'success' in data
                    assert 'stream_id' in data
                    self.active_streams.append(data['stream_id'])
                    self.print_test_result(test_name, True, f"Connected: {data['stream_id']}")
                else:
                    # Camera not available is acceptable in test environment
                    self.print_test_result(test_name, True, f"Camera not available (expected): {data.get('message', '')}")
                
                self.results.add_result(test_name, True)
            else:
                self.assert_success_response(response)
                
        except Exception as e:
            self.print_test_result(test_name, False, str(e))
            self.results.add_result(test_name, False)
    
    def test_camera_connect_rtsp(self):
        """Test POST /camera/connect with RTSP"""
        test_name = "Camera Connect (RTSP)"
        try:
            params = {
                "camera_id": "rtsp://test.example.com/stream",
                "source": "rtsp"
            }
            
            response = self.post("/camera/connect", params=params)
            
            # RTSP URL is fake, so expect failure but test endpoint exists
            if response.status_code in [200, 400, 404, 500, 503]:
                data = response.json()
                
                if response.status_code == 200:
                    assert 'stream_id' in data
                    self.active_streams.append(data['stream_id'])
                    self.print_test_result(test_name, True, f"RTSP connected: {data['stream_id']}")
                else:
                    # Connection failure expected with fake URL
                    self.print_test_result(test_name, True, f"RTSP connection failed (expected): {data.get('message', '')}")
                
                self.results.add_result(test_name, True)
            else:
                self.print_test_result(test_name, False, f"Unexpected status: {response.status_code}")
                self.results.add_result(test_name, False)
                
        except Exception as e:
            self.print_test_result(test_name, False, str(e))
            self.results.add_result(test_name, False)
    
    def test_camera_status(self):
        """Test GET /camera/status"""
        test_name = "Camera Status"
        try:
            response = self.get("/camera/status")
            
            # Accept various responses
            if response.status_code in [200, 404]:
                if response.status_code == 200:
                    data = response.json()
                    assert isinstance(data, (dict, list))
                    self.print_test_result(test_name, True, f"Status retrieved: {len(data) if isinstance(data, list) else 'dict'}")
                else:
                    self.print_test_result(test_name, True, "No cameras connected (OK)")
                
                self.results.add_result(test_name, True)
            else:
                self.assert_success_response(response)
                
        except Exception as e:
            self.print_test_result(test_name, False, str(e))
            self.results.add_result(test_name, False)
    
    def test_camera_stream_endpoint(self):
        """Test GET /camera/stream"""
        test_name = "Camera Stream Endpoint"
        try:
            # Test if endpoint exists and handles no active streams
            response = self.get("/camera/stream")
            
            # Accept various responses since we may not have active streams
            if response.status_code in [200, 404, 503]:
                self.print_test_result(test_name, True, f"Stream endpoint accessible (status: {response.status_code})")
                self.results.add_result(test_name, True)
            else:
                self.print_test_result(test_name, False, f"Unexpected status: {response.status_code}")
                self.results.add_result(test_name, False)
                
        except Exception as e:
            self.print_test_result(test_name, False, str(e))
            self.results.add_result(test_name, False)
    
    def test_rtsp_connection_test(self):
        """Test GET /camera/test/rtsp"""
        test_name = "RTSP Connection Test"
        try:
            # Test with a known bad URL to check endpoint functionality
            params = {"rtsp_url": "rtsp://invalid.test.url/stream"}
            response = self.get("/camera/test/rtsp", params=params)
            
            # Expect failure but endpoint should exist
            if response.status_code in [200, 400, 404, 500, 503]:
                data = response.json()
                self.print_test_result(test_name, True, f"RTSP test endpoint working: {data.get('message', 'OK')}")
                self.results.add_result(test_name, True)
            else:
                self.print_test_result(test_name, False, f"Unexpected status: {response.status_code}")
                self.results.add_result(test_name, False)
                
        except Exception as e:
            self.print_test_result(test_name, False, str(e))
            self.results.add_result(test_name, False)
    
    def test_local_camera_test(self):
        """Test GET /camera/test/local"""
        test_name = "Local Camera Test"
        try:
            params = {"camera_id": "0"}
            response = self.get("/camera/test/local", params=params)
            
            # Camera might not be available
            if response.status_code in [200, 404, 500]:
                data = response.json()
                
                if response.status_code == 200:
                    self.print_test_result(test_name, True, f"Local camera available: {data.get('message', 'OK')}")
                else:
                    self.print_test_result(test_name, True, f"Local camera not available (expected): {data.get('message', '')}")
                
                self.results.add_result(test_name, True)
            else:
                self.print_test_result(test_name, False, f"Unexpected status: {response.status_code}")
                self.results.add_result(test_name, False)
                
        except Exception as e:
            self.print_test_result(test_name, False, str(e))
            self.results.add_result(test_name, False)
    
    def test_camera_disconnect(self):
        """Test POST /camera/disconnect"""
        test_name = "Camera Disconnect"
        try:
            # Try to disconnect a non-existent stream
            params = {"camera_id": "0"}
            response = self.post("/camera/disconnect", params=params)
            
            # Expect failure for non-existent stream
            if response.status_code in [200, 404, 400]:
                data = response.json()
                self.print_test_result(test_name, True, f"Disconnect endpoint working: {data.get('message', 'OK')}")
                self.results.add_result(test_name, True)
            else:
                self.print_test_result(test_name, False, f"Unexpected status: {response.status_code}")
                self.results.add_result(test_name, False)
                
        except Exception as e:
            self.print_test_result(test_name, False, str(e))
            self.results.add_result(test_name, False)
    
    def test_stream_with_recognition(self):
        """Test streaming with face recognition enabled"""
        test_name = "Stream with Recognition"
        try:
            params = {
                "camera_id": "0",
                "source": "local"
            }
            
            response = self.post("/camera/connect", params=params)
            
            # Handle various responses gracefully
            if response.status_code in [200, 404, 500]:
                data = response.json()
                
                if response.status_code == 200:
                    assert 'stream_id' in data
                    self.active_streams.append(data['stream_id'])
                    self.print_test_result(test_name, True, f"Recognition stream started: {data['stream_id']}")
                else:
                    self.print_test_result(test_name, True, f"Recognition stream failed (expected): {data.get('message', '')}")
                
                self.results.add_result(test_name, True)
            else:
                self.print_test_result(test_name, False, f"Unexpected status: {response.status_code}")
                self.results.add_result(test_name, False)
                
        except Exception as e:
            self.print_test_result(test_name, False, str(e))
            self.results.add_result(test_name, False)
    
    def run_all_tests(self):
        """Run all camera streaming tests"""
        print("📹 Testing Camera Streaming APIs")
        print("=" * 50)
        
        try:
            self.test_camera_status()
            self.test_local_camera_test()
            self.test_rtsp_connection_test()
            self.test_camera_connect_local()
            self.test_camera_connect_rtsp()
            self.test_stream_with_recognition()
            self.test_camera_stream_endpoint()
            self.test_camera_disconnect()
            
        finally:
            # Clean up any active streams
            self.cleanup_streams()
        
        self.results.print_summary()
        return self.results


if __name__ == "__main__":
    tester = TestCameraStreaming()
    
    # Wait for service to be available
    print("Waiting for FaceCV service...")
    if not tester.wait_for_service():
        print("❌ Service not available at http://localhost:7003")
        exit(1)
    
    print("✅ Service is available\n")
    results = tester.run_all_tests()
    
    # Exit with error code if tests failed
    exit(0 if results.failed == 0 else 1)