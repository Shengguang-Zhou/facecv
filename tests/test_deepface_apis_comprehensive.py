#!/usr/bin/env python3
"""
Comprehensive test suite for DeepFace APIs using EurekCV dataset
Tests all endpoints to ensure error-free operation with real face images
"""

import requests
import json
import os
import time
from pathlib import Path

# Configuration
BASE_URL = "http://localhost:7003"
DATASET_PATH = "/home/a/PycharmProjects/EurekCV/dataset/faces"
TEST_RESULTS = {}

def log_test(test_name, success, details=""):
    """Log test results"""
    status = "✅ PASS" if success else "❌ FAIL"
    print(f"{status} {test_name}")
    if details:
        print(f"    {details}")
    TEST_RESULTS[test_name] = {"success": success, "details": details}

def test_server_health():
    """Test if server is running"""
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        success = response.status_code == 200
        log_test("Server Health Check", success, f"Status: {response.status_code}")
        return success
    except Exception as e:
        log_test("Server Health Check", False, f"Error: {e}")
        return False

def test_deepface_health():
    """Test DeepFace health endpoint"""
    try:
        response = requests.get(f"{BASE_URL}/api/v1/deepface/health", timeout=10)
        success = response.status_code == 200
        if success:
            data = response.json()
            log_test("DeepFace Health Check", True, f"Status: {data.get('status', 'unknown')}")
        else:
            log_test("DeepFace Health Check", False, f"HTTP {response.status_code}: {response.text}")
        return success
    except Exception as e:
        log_test("DeepFace Health Check", False, f"Error: {e}")
        return False

def test_face_registration():
    """Test face registration with different models"""
    test_images = [
        ("harris1.jpeg", "Kamala_Harris", "VGG-Face"),
        ("trump1.jpeg", "Donald_Trump", "Facenet"),
        ("harris2.jpeg", "Kamala_Harris_2", "OpenFace")
    ]
    
    registered_faces = []
    
    for image_file, name, model in test_images:
        image_path = os.path.join(DATASET_PATH, image_file)
        
        if not os.path.exists(image_path):
            log_test(f"Register {name}", False, f"Image not found: {image_path}")
            continue
            
        try:
            with open(image_path, 'rb') as f:
                files = {'file': f}
                data = {
                    'name': name,
                    'model': model,
                    'detector_backend': 'opencv',  # Use opencv instead of mtcnn
                    'metadata': json.dumps({"source": "EurekCV_dataset", "test": True})
                }
                
                response = requests.post(
                    f"{BASE_URL}/api/v1/deepface/faces/",
                    files=files,
                    data=data,
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    success = result.get('success', False)
                    log_test(f"Register {name} ({model})", success, f"Message: {result.get('message', '')}")
                    if success:
                        registered_faces.append({"name": name, "model": model})
                else:
                    log_test(f"Register {name} ({model})", False, f"HTTP {response.status_code}: {response.text}")
                    
        except Exception as e:
            log_test(f"Register {name} ({model})", False, f"Error: {e}")
    
    return registered_faces

def test_list_faces():
    """Test listing faces"""
    try:
        response = requests.get(f"{BASE_URL}/api/v1/deepface/faces/", timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            faces = data.get('faces', [])
            total = data.get('total', 0)
            
            # Check if we have model information
            model_info_present = False
            if faces:
                first_face = faces[0]
                model_info_present = 'model_type' in first_face and 'model_name' in first_face
            
            details = f"Found {total} faces"
            if model_info_present:
                details += " with model tracking"
            
            log_test("List Faces", True, details)
            
            # Show some face details
            for face in faces[:3]:  # Show first 3
                name = face.get('person_name', 'Unknown')
                model_type = face.get('model_type', 'unknown')
                model_name = face.get('model_name', 'unknown')
                print(f"    - {name} ({model_type}/{model_name})")
                
            return faces
        else:
            log_test("List Faces", False, f"HTTP {response.status_code}: {response.text}")
            return []
            
    except Exception as e:
        log_test("List Faces", False, f"Error: {e}")
        return []

def test_face_recognition():
    """Test face recognition with different models"""
    test_images = [
        ("harris2.jpeg", "Should recognize Kamala_Harris"),
        ("trump2.jpeg", "Should recognize Donald_Trump"),
        ("trump3.jpeg", "Should recognize Donald_Trump")
    ]
    
    for image_file, expected in test_images:
        image_path = os.path.join(DATASET_PATH, image_file)
        
        if not os.path.exists(image_path):
            log_test(f"Recognize {image_file}", False, f"Image not found: {image_path}")
            continue
            
        try:
            with open(image_path, 'rb') as f:
                files = {'file': f}
                data = {
                    'threshold': 0.6,
                    'model': 'VGG-Face',
                    'return_all_candidates': True
                }
                
                response = requests.post(
                    f"{BASE_URL}/api/v1/deepface/recognition",
                    files=files,
                    data=data,
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    faces = result.get('faces', [])
                    
                    if faces:
                        best_match = faces[0]
                        name = best_match.get('person_name', 'Unknown')
                        confidence = best_match.get('confidence', 0)
                        details = f"Recognized: {name} (confidence: {confidence:.3f})"
                    else:
                        details = "No faces recognized"
                    
                    log_test(f"Recognize {image_file}", True, details)
                else:
                    log_test(f"Recognize {image_file}", False, f"HTTP {response.status_code}: {response.text}")
                    
        except Exception as e:
            log_test(f"Recognize {image_file}", False, f"Error: {e}")

def test_face_verification():
    """Test face verification between different images"""
    test_pairs = [
        ("harris1.jpeg", "harris2.jpeg", "Same person verification"),
        ("trump1.jpeg", "trump2.jpeg", "Same person verification"),
        ("harris1.jpeg", "trump1.jpeg", "Different person verification")
    ]
    
    for image1, image2, description in test_pairs:
        image1_path = os.path.join(DATASET_PATH, image1)
        image2_path = os.path.join(DATASET_PATH, image2)
        
        if not (os.path.exists(image1_path) and os.path.exists(image2_path)):
            log_test(f"Verify {image1} vs {image2}", False, "Image files not found")
            continue
            
        try:
            with open(image1_path, 'rb') as f1, open(image2_path, 'rb') as f2:
                files = {
                    'file1': f1,
                    'file2': f2
                }
                data = {'threshold': 0.6, 'model_name': 'VGG-Face', 'detector_backend': 'opencv'}
                
                response = requests.post(
                    f"{BASE_URL}/api/v1/deepface/verify/",
                    files=files,
                    data=data,
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    verified = result.get('verified', False)
                    distance = result.get('distance', 1.0)
                    details = f"Verified: {verified}, Distance: {distance:.3f}"
                    log_test(f"Verify {image1} vs {image2}", True, details)
                else:
                    log_test(f"Verify {image1} vs {image2}", False, f"HTTP {response.status_code}: {response.text}")
                    
        except Exception as e:
            log_test(f"Verify {image1} vs {image2}", False, f"Error: {e}")

def test_face_analysis():
    """Test face analysis (age, gender, emotion, race)"""
    test_images = ["harris1.jpeg", "trump1.jpeg"]
    
    for image_file in test_images:
        image_path = os.path.join(DATASET_PATH, image_file)
        
        if not os.path.exists(image_path):
            log_test(f"Analyze {image_file}", False, f"Image not found: {image_path}")
            continue
            
        try:
            with open(image_path, 'rb') as f:
                files = {'file': f}
                data = {
                    'actions': "age,gender,emotion,race",  # Use comma-separated string
                    'detector_backend': 'opencv'  # Use opencv instead of mtcnn
                }
                
                response = requests.post(
                    f"{BASE_URL}/api/v1/deepface/analyze/",
                    files=files,
                    data=data,
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    # Extract analysis results
                    analysis = result.get('analysis', {})
                    if analysis:
                        age = analysis.get('age', 'unknown')
                        gender = analysis.get('dominant_gender', 'unknown')
                        emotion = analysis.get('dominant_emotion', 'unknown')
                        details = f"Age: {age}, Gender: {gender}, Emotion: {emotion}"
                    else:
                        details = "Analysis completed"
                    
                    log_test(f"Analyze {image_file}", True, details)
                else:
                    log_test(f"Analyze {image_file}", False, f"HTTP {response.status_code}: {response.text}")
                    
        except Exception as e:
            log_test(f"Analyze {image_file}", False, f"Error: {e}")

def test_face_update():
    """Test updating face information"""
    # First get a face to update
    try:
        response = requests.get(f"{BASE_URL}/api/v1/deepface/faces/", timeout=10)
        if response.status_code == 200:
            data = response.json()
            faces = data.get('faces', [])
            
            if faces:
                # Update the first test face
                test_face = None
                for face in faces:
                    if face.get('person_name', '').startswith(('Kamala_Harris', 'Donald_Trump')):
                        test_face = face
                        break
                
                if test_face:
                    face_id = test_face['face_id']
                    original_name = test_face['person_name']
                    new_name = f"{original_name}_Updated"
                    
                    # Update the face
                    data = {
                        'name': new_name,
                        'metadata': json.dumps({"updated": True, "test": True})
                    }
                    
                    update_response = requests.put(
                        f"{BASE_URL}/api/v1/deepface/faces/{face_id}",
                        data=data,
                        timeout=15
                    )
                    
                    if update_response.status_code == 200:
                        log_test("Update Face Info", True, f"Updated {original_name} to {new_name}")
                    else:
                        log_test("Update Face Info", False, f"HTTP {update_response.status_code}: {update_response.text}")
                else:
                    log_test("Update Face Info", False, "No test faces found to update")
            else:
                log_test("Update Face Info", False, "No faces available to update")
        else:
            log_test("Update Face Info", False, f"Failed to get faces: HTTP {response.status_code}")
            
    except Exception as e:
        log_test("Update Face Info", False, f"Error: {e}")

def test_face_deletion():
    """Test deleting faces (cleanup test data)"""
    try:
        response = requests.get(f"{BASE_URL}/api/v1/deepface/faces/", timeout=10)
        if response.status_code == 200:
            data = response.json()
            faces = data.get('faces', [])
            
            # Delete test faces (ones created in this test)
            deleted_count = 0
            for face in faces:
                name = face.get('person_name', '')
                metadata = face.get('metadata', {})
                
                # Delete if it's a test face from this session
                if (name.startswith(('Kamala_Harris', 'Donald_Trump')) and 
                    isinstance(metadata, dict) and metadata.get('test') == True):
                    
                    face_id = face['face_id']
                    try:
                        delete_response = requests.delete(
                            f"{BASE_URL}/api/v1/deepface/faces/{face_id}",
                            timeout=10
                        )
                        
                        if delete_response.status_code == 200:
                            deleted_count += 1
                        
                    except Exception as e:
                        print(f"    Warning: Failed to delete {name}: {e}")
            
            log_test("Delete Test Faces", True, f"Deleted {deleted_count} test faces")
            
    except Exception as e:
        log_test("Delete Test Faces", False, f"Error: {e}")

def run_all_tests():
    """Run comprehensive test suite"""
    print("🧪 DeepFace API Comprehensive Test Suite")
    print("🎯 Using EurekCV Dataset for Real Face Images")
    print("=" * 60)
    
    # Check if dataset exists
    if not os.path.exists(DATASET_PATH):
        print(f"❌ Dataset not found at: {DATASET_PATH}")
        return False
    
    # List available test images
    print(f"📁 Dataset Path: {DATASET_PATH}")
    try:
        images = [f for f in os.listdir(DATASET_PATH) if f.endswith(('.jpg', '.jpeg', '.png'))]
        print(f"🖼️  Available Images: {', '.join(images)}")
    except Exception as e:
        print(f"❌ Cannot access dataset: {e}")
        return False
    
    print("\n🚀 Starting API Tests...")
    print("-" * 40)
    
    # Test sequence
    start_time = time.time()
    
    # 1. Basic health checks
    if not test_server_health():
        print("❌ Server not running. Please start with: python main.py")
        return False
    
    if not test_deepface_health():
        print("❌ DeepFace service not available")
        return False
    
    # 2. CRUD Operations
    print("\n📝 Testing CRUD Operations:")
    test_face_registration()
    test_list_faces()
    test_face_update()
    
    # 3. AI Operations  
    print("\n🤖 Testing AI Operations:")
    test_face_recognition()
    test_face_verification() 
    test_face_analysis()
    
    # 4. Cleanup
    print("\n🧹 Cleanup:")
    test_face_deletion()
    
    # Results summary
    end_time = time.time()
    duration = end_time - start_time
    
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    total_tests = len(TEST_RESULTS)
    passed_tests = sum(1 for result in TEST_RESULTS.values() if result['success'])
    failed_tests = total_tests - passed_tests
    
    print(f"Total Tests: {total_tests}")
    print(f"✅ Passed: {passed_tests}")
    print(f"❌ Failed: {failed_tests}")
    print(f"⏱️  Duration: {duration:.2f} seconds")
    print(f"📈 Success Rate: {(passed_tests/total_tests)*100:.1f}%")
    
    if failed_tests > 0:
        print("\n❌ FAILED TESTS:")
        for test_name, result in TEST_RESULTS.items():
            if not result['success']:
                print(f"   - {test_name}: {result['details']}")
    
    print("\n" + "=" * 60)
    
    return failed_tests == 0

if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)