"""
Test script to verify recognition of the original test faces.
"""

import os
import sys
import requests
import json
import time
from pprint import pprint

def test_recognize_with_image(image_path, timeout=60):
    """Test recognition with a specific image."""
    print(f"Testing recognition with image: {image_path}")
    
    url = "http://localhost:7003/api/v1/insightface/recognize"
    
    try:
        with open(image_path, "rb") as f:
            files = {"file": (os.path.basename(image_path), f, "image/jpeg")}
            data = {
                "model": "buffalo_s",
                "threshold": "0.35"
            }
            
            print("Sending request...")
            start_time = time.time()
            response = requests.post(url, files=files, data=data, timeout=timeout)
            elapsed = time.time() - start_time
            
            print(f"Request completed in {elapsed:.2f} seconds with status code: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                print(f"Recognition results: {len(result)}")
                pprint(result)
                
                recognized = [f for f in result if f.get('name') != 'Unknown' and f.get('similarity', 0) > 0.35]
                print(f"\nRecognized faces: {len(recognized)}")
                for face in recognized:
                    print(f"- Name: {face.get('name')}, ID: {face.get('id')}, Similarity: {face.get('similarity')}")
                
                return result
            else:
                print(f"Error: {response.status_code}")
                print(response.text)
                return None
    
    except requests.exceptions.Timeout:
        print(f"Request timed out after {timeout} seconds")
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None

def test_original_faces():
    """Test recognition with the original test faces."""
    test_faces = [
        {"name": "Harris", "id": "e72ed2f0-abaf-45c8-9967-6e7084003665"},
        {"name": "Donald Trump", "id": "a5b82313-e954-400a-8de8-c01c4d2cb6fe"},
        {"name": "string", "id": "444a5b1c-2d79-4ba0-97fe-dfa3d3d7149c"}
    ]
    
    print("Expected test faces in database:")
    for face in test_faces:
        print(f"- {face['name']} (ID: {face['id']})")
    
    test_image = "test_images/test_face.jpg"
    print(f"\nTesting with image: {test_image}")
    result = test_recognize_with_image(test_image)
    
    if result:
        found_original = False
        for face in result:
            face_id = face.get('id')
            for test_face in test_faces:
                if face_id == test_face['id']:
                    print(f"\n✓ Found original test face: {test_face['name']} (ID: {face_id})")
                    found_original = True
        
        if not found_original:
            print("\n✗ None of the original test faces were recognized")
    
    return result

if __name__ == "__main__":
    os.environ["FACECV_INSIGHTFACE_PREFER_GPU"] = "false"
    os.environ["FACECV_INSIGHTFACE_DET_SIZE"] = "[320,320]"
    os.environ["FACECV_DB_TYPE"] = "hybrid"
    
    print("\n=== Testing Recognition of Original Test Faces ===\n")
    result = test_original_faces()
    
    if result:
        print("\nTest completed successfully!")
        sys.exit(0)
    else:
        print("\nTest failed!")
        sys.exit(1)
