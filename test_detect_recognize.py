"""
Test script for the InsightFace detect and recognize API endpoints.
This script tests both detection and recognition functionality.
"""

import os
import sys
import requests
import json
import time
from pprint import pprint

def test_detect_endpoint(image_path, timeout=60):
    """Test the detect endpoint with optimized parameters."""
    print(f"Testing detect endpoint with image: {image_path}")
    
    url = "http://localhost:7003/api/v1/insightface/detect"
    
    try:
        with open(image_path, "rb") as f:
            files = {"file": (os.path.basename(image_path), f, "image/jpeg")}
            data = {
                "model": "buffalo_s",  # Use smaller model for better performance
                "min_confidence": "0.5"
            }
            
            print("Sending request...")
            start_time = time.time()
            response = requests.post(url, files=files, data=data, timeout=timeout)
            elapsed = time.time() - start_time
            
            print(f"Request completed in {elapsed:.2f} seconds with status code: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                print(f"Detected {len(result)} faces")
                
                if result:
                    face = result[0]
                    print("\nResponse format verification:")
                    print(f"- 'id' field present: {'id' in face}")
                    print(f"- 'face_id' field absent: {'face_id' not in face}")
                    print(f"- 'person_id' field absent: {'person_id' not in face}")
                    print(f"- Required fields present: {all(k in face for k in ['bbox', 'confidence', 'id', 'landmarks', 'quality_score', 'name', 'similarity'])}")
                    
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

def test_recognize_endpoint(image_path, timeout=60):
    """Test the recognize endpoint."""
    print(f"Testing recognize endpoint with image: {image_path}")
    
    url = "http://localhost:7003/api/v1/insightface/recognize"
    
    try:
        with open(image_path, "rb") as f:
            files = {"file": (os.path.basename(image_path), f, "image/jpeg")}
            data = {
                "model": "buffalo_s",  # Use smaller model for better performance
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
                
                if result:
                    face = result[0]
                    print("\nResponse format verification:")
                    print(f"- 'id' field present: {'id' in face}")
                    print(f"- 'face_id' field absent: {'face_id' not in face}")
                    print(f"- 'person_id' field absent: {'person_id' not in face}")
                    
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

if __name__ == "__main__":
    os.environ["FACECV_INSIGHTFACE_PREFER_GPU"] = "false"
    os.environ["FACECV_INSIGHTFACE_DET_SIZE"] = "[320,320]"
    os.environ["FACECV_DB_TYPE"] = "hybrid"
    
    test_image = "test_images/test_face.jpg"
    if len(sys.argv) > 1:
        test_image = sys.argv[1]
    
    print("\n=== Testing InsightFace Detect Endpoint ===\n")
    detect_result = test_detect_endpoint(test_image)
    
    print("\n=== Testing InsightFace Recognize Endpoint ===\n")
    recognize_result = test_recognize_endpoint(test_image)
    
    if detect_result and recognize_result:
        print("\nAll tests successful!")
        sys.exit(0)
    elif detect_result:
        print("\nOnly detect test successful. Recognize test failed.")
        sys.exit(1)
    else:
        print("\nAll tests failed!")
        sys.exit(2)
