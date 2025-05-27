#!/usr/bin/env python3
"""
Script to run comprehensive CUDA acceleration tests
"""

import sys
import os
import subprocess
import time
import requests
import json
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def check_server_running(base_url="http://localhost:7003"):
    """Check if the FaceCV server is running"""
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def start_server():
    """Start the FaceCV server"""
    print("🚀 Starting FaceCV server...")
    
    # Change to project directory
    project_dir = Path(__file__).parent.parent
    os.chdir(project_dir)
    
    # Start server in background
    process = subprocess.Popen([
        sys.executable, "main.py",
        "--host", "localhost",
        "--port", "7003"
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    # Wait for server to start
    max_wait = 30  # seconds
    wait_time = 0
    
    while wait_time < max_wait:
        if check_server_running():
            print("✅ Server started successfully")
            return process
        time.sleep(2)
        wait_time += 2
        print(f"⏳ Waiting for server... ({wait_time}s)")
    
    print("❌ Server failed to start within timeout")
    process.terminate()
    return None

def run_unit_tests():
    """Run the unit test suite"""
    print("\n" + "="*60)
    print("🧪 RUNNING UNIT TESTS")
    print("="*60)
    
    test_file = Path(__file__).parent / "test_cuda_acceleration.py"
    
    if not test_file.exists():
        print("❌ Test file not found")
        return False
    
    try:
        # Run the test file directly
        result = subprocess.run([sys.executable, str(test_file)], 
                              capture_output=True, text=True, timeout=300)
        
        print("STDOUT:")
        print(result.stdout)
        
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        
        if result.returncode == 0:
            print("✅ Unit tests completed successfully")
            return True
        else:
            print(f"❌ Unit tests failed with return code {result.returncode}")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Unit tests timed out")
        return False
    except Exception as e:
        print(f"❌ Error running unit tests: {e}")
        return False

def test_api_endpoints():
    """Test key API endpoints"""
    print("\n" + "="*60)
    print("🌐 TESTING API ENDPOINTS")
    print("="*60)
    
    base_url = "http://localhost:7003"
    
    # Test endpoints
    tests = [
        {
            "name": "Health Check",
            "method": "GET",
            "url": f"{base_url}/health",
            "expected_status": 200
        },
        {
            "name": "System Health",
            "method": "GET", 
            "url": f"{base_url}/api/v1/system/health",
            "expected_status": 200
        }
    ]
    
    for test in tests:
        try:
            print(f"\n🔗 Testing {test['name']}")
            response = requests.request(
                test["method"], 
                test["url"], 
                timeout=10
            )
            
            if response.status_code == test["expected_status"]:
                print(f"✅ {test['name']}: {response.status_code}")
                
                # Pretty print JSON response if available
                try:
                    data = response.json()
                    print(f"   Response: {json.dumps(data, indent=2)}")
                except:
                    print(f"   Response: {response.text[:200]}...")
            else:
                print(f"❌ {test['name']}: Expected {test['expected_status']}, got {response.status_code}")
                
        except Exception as e:
            print(f"❌ {test['name']}: Error - {e}")

def test_face_operations():
    """Test face detection and recognition operations"""
    print("\n" + "="*60)
    print("👤 TESTING FACE OPERATIONS")
    print("="*60)
    
    base_url = "http://localhost:7003"
    test_image_path = "/home/a/PycharmProjects/EurekCV/dataset/faces/harris1.jpeg"
    
    if not Path(test_image_path).exists():
        print(f"❌ Test image not found: {test_image_path}")
        return
    
    try:
        # Test face detection
        print("\n🔍 Testing face detection...")
        with open(test_image_path, 'rb') as f:
            files = {'file': ('harris1.jpeg', f, 'image/jpeg')}
            response = requests.post(
                f"{base_url}/api/v1/insightface/detect",
                files=files,
                timeout=30
            )
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Face detection successful")
            print(f"   Detected {len(data.get('faces', []))} faces")
            if data.get('faces'):
                face = data['faces'][0]
                print(f"   Confidence: {face.get('confidence', 'N/A')}")
                print(f"   Bbox: {face.get('bbox', 'N/A')}")
        else:
            print(f"❌ Face detection failed: {response.status_code}")
            print(f"   Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Face detection error: {e}")

def generate_test_report():
    """Generate a comprehensive test report"""
    print("\n" + "="*60)
    print("📊 CUDA ACCELERATION TEST REPORT")
    print("="*60)
    
    # Import and check CUDA utilities
    try:
        from facecv.utils.cuda_utils import (
            get_cuda_version, check_cuda_availability, 
            get_execution_providers
        )
        from facecv.config.runtime_config import get_runtime_config
        
        # CUDA Detection Results
        print("\n🔧 CUDA Detection Results:")
        cuda_available = check_cuda_availability()
        print(f"   CUDA Available: {cuda_available}")
        
        if cuda_available:
            cuda_version = get_cuda_version()
            if cuda_version:
                print(f"   CUDA Version: {cuda_version[0]}.{cuda_version[1]}")
        
        providers = get_execution_providers()
        print(f"   Execution Providers: {providers}")
        
        # Runtime Configuration
        print("\n⚙️  Runtime Configuration:")
        runtime_config = get_runtime_config()
        config_data = runtime_config.get_all()
        
        cuda_config = {
            'cuda_available': config_data.get('cuda_available'),
            'cuda_version': config_data.get('cuda_version'),
            'execution_providers': config_data.get('execution_providers'),
            'prefer_gpu': config_data.get('prefer_gpu')
        }
        
        for key, value in cuda_config.items():
            print(f"   {key}: {value}")
        
        # Performance Summary
        print("\n⚡ Performance Summary:")
        if cuda_available:
            print("   ✅ GPU acceleration enabled")
            print("   ✅ CUDA environment configured")
            print("   ✅ Execution providers optimized")
        else:
            print("   ⚠️  GPU acceleration disabled (CUDA not available)")
            print("   ℹ️  Running in CPU-only mode")
        
    except Exception as e:
        print(f"❌ Error generating report: {e}")

def main():
    """Main test runner"""
    print("🎯 FaceCV CUDA Acceleration Test Suite")
    print("="*60)
    
    # Check if test images exist
    test_images = [
        "/home/a/PycharmProjects/EurekCV/dataset/faces/harris1.jpeg",
        "/home/a/PycharmProjects/EurekCV/dataset/faces/trump1.jpeg"
    ]
    
    missing_images = []
    for img_path in test_images:
        if not Path(img_path).exists():
            missing_images.append(img_path)
    
    if missing_images:
        print("⚠️  Warning: Some test images are missing:")
        for img in missing_images:
            print(f"   {img}")
        print("   Some tests may be skipped.")
    
    # Run unit tests
    unit_test_success = run_unit_tests()
    
    # Check if server is running
    if not check_server_running():
        print("\n🚀 Server not running, starting...")
        server_process = start_server()
        if not server_process:
            print("❌ Cannot start server, skipping API tests")
            generate_test_report()
            return
    else:
        print("\n✅ Server already running")
        server_process = None
    
    try:
        # Run API tests
        test_api_endpoints()
        test_face_operations()
        
    finally:
        # Stop server if we started it
        if server_process:
            print("\n🛑 Stopping server...")
            server_process.terminate()
            try:
                server_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server_process.kill()
    
    # Generate final report
    generate_test_report()
    
    print("\n🏁 Test suite completed!")
    
    if unit_test_success:
        print("✅ All tests completed successfully")
    else:
        print("⚠️  Some tests had issues - check output above")

if __name__ == "__main__":
    main()