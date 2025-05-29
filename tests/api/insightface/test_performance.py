#!/usr/bin/env python3
"""Test InsightFace API Performance - All endpoints should respond < 1s"""

import time
import requests
import json
from pathlib import Path

BASE_URL = "http://localhost:7003/api/v1/insightface"
TEST_IMAGE_PATH = "/home/a/PycharmProjects/facecv/test_images/test_face.jpg"
TEST_IMAGE_PATH2 = "/home/a/PycharmProjects/facecv/test_images/test_face.jpg"

def measure_time(func):
    """Decorator to measure function execution time"""
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        return result, elapsed
    return wrapper

@measure_time
def test_health():
    """Test health check endpoint"""
    response = requests.get(f"{BASE_URL}/health")
    return response.status_code == 200, response.json()

@measure_time
def test_list_faces():
    """Test list faces endpoint"""
    response = requests.get(f"{BASE_URL}/faces")
    return response.status_code == 200, response.json()

@measure_time
def test_model_info():
    """Test model info endpoint"""
    response = requests.get(f"{BASE_URL}/models/info")
    return response.status_code == 200, response.json()

@measure_time
def test_detect_faces():
    """Test face detection endpoint"""
    with open(TEST_IMAGE_PATH, 'rb') as f:
        files = {'file': ('test.jpg', f, 'image/jpeg')}
        response = requests.post(f"{BASE_URL}/detect", files=files)
    return response.status_code == 200, response.json()

@measure_time
def test_register_face(name="Test_User"):
    """Test face registration endpoint"""
    with open(TEST_IMAGE_PATH, 'rb') as f:
        files = {'file': ('test.jpg', f, 'image/jpeg')}
        data = {'name': name}
        response = requests.post(f"{BASE_URL}/register", files=files, data=data)
    return response.status_code == 200, response.json() if response.status_code == 200 else response.text

@measure_time
def test_search_face():
    """Test face search endpoint"""
    with open(TEST_IMAGE_PATH, 'rb') as f:
        files = {'file': ('test.jpg', f, 'image/jpeg')}
        data = {'threshold': '0.6'}
        response = requests.post(f"{BASE_URL}/recognize", files=files, data=data)
    return response.status_code == 200, response.json() if response.status_code == 200 else response.text

@measure_time
def test_verify_faces():
    """Test face verification endpoint"""
    with open(TEST_IMAGE_PATH, 'rb') as f1, open(TEST_IMAGE_PATH2, 'rb') as f2:
        files = {
            'file1': ('img1.jpg', f1, 'image/jpeg'),
            'file2': ('img2.jpg', f2, 'image/jpeg')
        }
        response = requests.post(f"{BASE_URL}/verify", files=files)
    return response.status_code == 200, response.json() if response.status_code == 200 else response.text

@measure_time
def test_get_face_by_id(face_id):
    """Test get face by ID endpoint"""
    response = requests.get(f"{BASE_URL}/faces/{face_id}")
    return response.status_code == 200, response.json() if response.status_code == 200 else response.text

@measure_time
def test_get_faces_by_name(name):
    """Test get faces by name endpoint"""
    response = requests.get(f"{BASE_URL}/faces/by-name/{name}")
    return response.status_code == 200, response.json() if response.status_code == 200 else response.text

@measure_time
def test_update_face(face_id, new_name):
    """Test update face endpoint"""
    response = requests.put(f"{BASE_URL}/faces/{face_id}", json={"name": new_name})
    return response.status_code == 200, response.json() if response.status_code == 200 else response.text

@measure_time
def test_delete_face_by_id(face_id):
    """Test delete face by ID endpoint"""
    response = requests.delete(f"{BASE_URL}/faces/{face_id}")
    return response.status_code == 200, response.json() if response.status_code == 200 else response.text

@measure_time
def test_delete_faces_by_name(name):
    """Test delete faces by name endpoint"""
    response = requests.delete(f"{BASE_URL}/faces/by-name/{name}")
    return response.status_code == 200, response.json() if response.status_code == 200 else response.text

@measure_time
def test_available_models():
    """Test available models endpoint"""
    response = requests.get(f"{BASE_URL}/models/available")
    return response.status_code == 200, response.json()

def main():
    """Run all performance tests"""
    print("InsightFace API Performance Test")
    print("================================")
    print(f"Target: All endpoints should respond in < 1 second\n")
    
    results = []
    face_id = None
    
    # Test 1: Health Check
    (success, data), elapsed = test_health()
    status = "✓ PASS" if elapsed < 1.0 else "✗ FAIL"
    print(f"1. Health Check: {elapsed:.3f}s {status}")
    results.append(("Health Check", elapsed, elapsed < 1.0))
    
    # Test 2: List Faces
    (success, data), elapsed = test_list_faces()
    status = "✓ PASS" if elapsed < 1.0 else "✗ FAIL"
    print(f"2. List Faces: {elapsed:.3f}s {status}")
    results.append(("List Faces", elapsed, elapsed < 1.0))
    
    # Test 3: Model Info
    (success, data), elapsed = test_model_info()
    status = "✓ PASS" if elapsed < 1.0 else "✗ FAIL"
    print(f"3. Model Info: {elapsed:.3f}s {status}")
    results.append(("Model Info", elapsed, elapsed < 1.0))
    
    # Test 4: Available Models
    (success, data), elapsed = test_available_models()
    status = "✓ PASS" if elapsed < 1.0 else "✗ FAIL"
    print(f"4. Available Models: {elapsed:.3f}s {status}")
    results.append(("Available Models", elapsed, elapsed < 1.0))
    
    # Test 5: Detect Faces
    (success, data), elapsed = test_detect_faces()
    status = "✓ PASS" if elapsed < 1.0 else "✗ FAIL"
    print(f"5. Detect Faces: {elapsed:.3f}s {status}")
    results.append(("Detect Faces", elapsed, elapsed < 1.0))
    
    # Test 6: Register Face
    (success, data), elapsed = test_register_face("Performance_Test_User")
    status = "✓ PASS" if elapsed < 1.0 else "✗ FAIL"
    print(f"6. Register Face: {elapsed:.3f}s {status}")
    if not success:
        print(f"   Error: {data}")
    results.append(("Register Face", elapsed, elapsed < 1.0))
    if success and isinstance(data, dict):
        face_id = data.get('face_id')
    
    # Test 7: Search Face
    (success, data), elapsed = test_search_face()
    status = "✓ PASS" if elapsed < 1.0 else "✗ FAIL"
    print(f"7. Search Face: {elapsed:.3f}s {status}")
    if not success:
        print(f"   Error: {data}")
    results.append(("Search Face", elapsed, elapsed < 1.0))
    
    # Test 8: Verify Faces
    (success, data), elapsed = test_verify_faces()
    status = "✓ PASS" if elapsed < 1.0 else "✗ FAIL"
    print(f"8. Verify Faces: {elapsed:.3f}s {status}")
    if not success:
        print(f"   Error: {data}")
    results.append(("Verify Faces", elapsed, elapsed < 1.0))
    
    # Test 9: Get Faces by Name
    (success, data), elapsed = test_get_faces_by_name("Performance_Test_User")
    status = "✓ PASS" if elapsed < 1.0 else "✗ FAIL"
    print(f"9. Get Faces by Name: {elapsed:.3f}s {status}")
    results.append(("Get Faces by Name", elapsed, elapsed < 1.0))
    
    if face_id:
        # Test 10: Get Face by ID
        (success, data), elapsed = test_get_face_by_id(face_id)
        status = "✓ PASS" if elapsed < 1.0 else "✗ FAIL"
        print(f"10. Get Face by ID: {elapsed:.3f}s {status}")
        results.append(("Get Face by ID", elapsed, elapsed < 1.0))
        
        # Test 11: Update Face
        (success, data), elapsed = test_update_face(face_id, "Updated_Performance_Test_User")
        status = "✓ PASS" if elapsed < 1.0 else "✗ FAIL"
        print(f"11. Update Face: {elapsed:.3f}s {status}")
        results.append(("Update Face", elapsed, elapsed < 1.0))
        
        # Test 12: Delete Face by ID
        (success, data), elapsed = test_delete_face_by_id(face_id)
        status = "✓ PASS" if elapsed < 1.0 else "✗ FAIL"
        print(f"12. Delete Face by ID: {elapsed:.3f}s {status}")
        results.append(("Delete Face by ID", elapsed, elapsed < 1.0))
    
    # Test 13: Delete Faces by Name (cleanup)
    (success, data), elapsed = test_delete_faces_by_name("Performance_Test_User")
    status = "✓ PASS" if elapsed < 1.0 else "✗ FAIL"
    print(f"13. Delete Faces by Name: {elapsed:.3f}s {status}")
    results.append(("Delete Faces by Name", elapsed, elapsed < 1.0))
    
    # Summary
    print("\n" + "="*50)
    print("PERFORMANCE SUMMARY")
    print("="*50)
    
    passed = sum(1 for _, _, success in results if success)
    total = len(results)
    
    print(f"\nTotal Tests: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {total - passed}")
    
    print(f"\nAverage Response Time: {sum(t for _, t, _ in results) / len(results):.3f}s")
    print(f"Max Response Time: {max(t for _, t, _ in results):.3f}s")
    print(f"Min Response Time: {min(t for _, t, _ in results):.3f}s")
    
    if total - passed > 0:
        print("\nFailed Tests (>1s):")
        for name, elapsed, success in results:
            if not success:
                print(f"  - {name}: {elapsed:.3f}s")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)