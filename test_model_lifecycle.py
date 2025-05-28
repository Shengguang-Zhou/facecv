"""
Test script for model lifecycle management in FaceCV.
This script tests model loading, offloading, and switching functionality.
"""

import os
import sys
import time
import json
import requests
import psutil
import numpy as np
from pprint import pprint

API_BASE_URL = "http://localhost:7003/api/v1/insightface"
TEST_IMAGE = "test_images/test_face.jpg"

def get_memory_usage():
    """Get current memory usage of the process."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # MB

def test_model_loading(model_name):
    """Test loading a specific InsightFace model."""
    print(f"Testing loading model: {model_name}")
    
    url = f"{API_BASE_URL}/detect"
    
    try:
        with open(TEST_IMAGE, "rb") as f:
            files = {"file": (os.path.basename(TEST_IMAGE), f, "image/jpeg")}
            data = {"model": model_name}
            
            print(f"Sending request with model: {model_name}")
            start_time = time.time()
            response = requests.post(url, files=files, data=data, timeout=30)
            elapsed = time.time() - start_time
            
            print(f"Request completed in {elapsed:.2f} seconds with status code: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                print(f"Detection results: {len(result)} faces")
                if len(result) > 0:
                    print(f"First face confidence: {result[0].get('confidence')}")
                return True, elapsed
            else:
                print(f"Error: {response.status_code}")
                print(response.text)
                return False, elapsed
    
    except Exception as e:
        print(f"Error: {e}")
        return False, 0

def test_model_switching():
    """Test switching between different InsightFace models."""
    print("\nTesting model switching...")
    
    models = ["buffalo_l", "buffalo_s", "buffalo_m"]
    results = {}
    
    for model in models:
        print(f"\nSwitching to model: {model}")
        success, elapsed = test_model_loading(model)
        results[model] = {
            "success": success,
            "elapsed": elapsed
        }
    
    print("\nModel switching results:")
    for model, result in results.items():
        status = "✓ Success" if result["success"] else "✗ Failed"
        print(f"- {model}: {status} (elapsed: {result['elapsed']:.2f}s)")
    
    return all(r["success"] for r in results.values())

def test_model_offloading():
    """Test model offloading after timeout period."""
    print("\nTesting model offloading...")
    
    print("1. Loading initial model")
    success, _ = test_model_loading("buffalo_l")
    if not success:
        print("Failed to load initial model")
        return False
    
    memory_after_load = get_memory_usage()
    print(f"Memory usage after model load: {memory_after_load:.2f} MB")
    
    offload_timeout = 60  # 1 minute for testing instead of 5 minutes
    print(f"2. Waiting for {offload_timeout} seconds to test offloading...")
    time.sleep(offload_timeout)
    
    memory_after_wait = get_memory_usage()
    print(f"Memory usage after waiting: {memory_after_wait:.2f} MB")
    
    print("3. Loading model again")
    success, elapsed_reload = test_model_loading("buffalo_l")
    if not success:
        print("Failed to reload model")
        return False
    
    print(f"4. Checking reload time: {elapsed_reload:.2f}s")
    
    memory_after_reload = get_memory_usage()
    print(f"Memory usage after reload: {memory_after_reload:.2f} MB")
    
    return True

def test_model_cache():
    """Test model caching system."""
    print("\nTesting model cache system...")
    
    model = "buffalo_s"
    times = []
    
    print(f"Testing cache with model: {model}")
    
    print("1. First request (cold start)")
    success, elapsed1 = test_model_loading(model)
    times.append(elapsed1)
    
    print("2. Second request (should be cached)")
    success, elapsed2 = test_model_loading(model)
    times.append(elapsed2)
    
    print("3. Third request (should be cached)")
    success, elapsed3 = test_model_loading(model)
    times.append(elapsed3)
    
    print("\nCache test results:")
    print(f"- First load: {elapsed1:.2f}s")
    print(f"- Second load: {elapsed2:.2f}s")
    print(f"- Third load: {elapsed3:.2f}s")
    
    cache_working = elapsed2 < elapsed1 or elapsed3 < elapsed1
    print(f"Cache system working: {'✓ Yes' if cache_working else '✗ No'}")
    
    return cache_working

def test_direct_model_loading():
    """Test direct model loading using InsightFace library."""
    print("\nTesting direct model loading with InsightFace library...")
    
    try:
        import insightface
        from insightface.app import FaceAnalysis
        
        model_dir = insightface.utils.get_model_dir()
        print(f"InsightFace model directory: {model_dir}")
        
        models = ["buffalo_l", "buffalo_s", "buffalo_m"]
        results = {}
        
        for model_name in models:
            try:
                print(f"\nLoading model: {model_name}")
                start_time = time.time()
                app = FaceAnalysis(name=model_name)
                app.prepare(ctx_id=-1)  # CPU mode
                elapsed = time.time() - start_time
                
                print(f"Model {model_name} loaded successfully in {elapsed:.2f}s")
                results[model_name] = {
                    "success": True,
                    "elapsed": elapsed
                }
            except Exception as e:
                print(f"Failed to load model {model_name}: {e}")
                results[model_name] = {
                    "success": False,
                    "error": str(e)
                }
        
        print("\nDirect model loading results:")
        for model, result in results.items():
            status = "✓ Success" if result.get("success") else "✗ Failed"
            if result.get("success"):
                print(f"- {model}: {status} (elapsed: {result['elapsed']:.2f}s)")
            else:
                print(f"- {model}: {status} (error: {result.get('error')})")
        
        return all(r.get("success", False) for r in results.values())
        
    except ImportError as e:
        print(f"Error importing InsightFace: {e}")
        return False
    except Exception as e:
        print(f"Error during direct model loading: {e}")
        return False

if __name__ == "__main__":
    os.environ["FACECV_INSIGHTFACE_PREFER_GPU"] = "false"
    os.environ["FACECV_INSIGHTFACE_DET_SIZE"] = "[320,320]"
    os.environ["FACECV_DB_TYPE"] = "hybrid"
    
    print("\n=== Testing Model Lifecycle Management ===\n")
    
    direct_loading_success = test_direct_model_loading()
    
    switching_success = test_model_switching()
    
    cache_success = test_model_cache()
    
    offloading_success = test_model_offloading()
    
    print("\n=== Model Lifecycle Management Test Summary ===")
    print(f"- Direct Model Loading: {'✓ Success' if direct_loading_success else '✗ Failed'}")
    print(f"- Model Switching: {'✓ Success' if switching_success else '✗ Failed'}")
    print(f"- Model Cache: {'✓ Success' if cache_success else '✗ Failed'}")
    print(f"- Model Offloading: {'✓ Success' if offloading_success else '✗ Failed'}")
    
    if direct_loading_success and switching_success and cache_success and offloading_success:
        print("\nAll model lifecycle management tests passed!")
        sys.exit(0)
    else:
        print("\nSome model lifecycle management tests failed!")
        sys.exit(1)
