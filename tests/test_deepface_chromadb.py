#!/usr/bin/env python3
"""Test DeepFace API with separate ChromaDB"""

import requests
import json
from pathlib import Path

BASE_URL = "http://localhost:8000/api/v1/deepface"

def test_deepface_registration():
    """Test face registration using DeepFace API"""
    
    # Test image path
    test_image = "/home/a/PycharmProjects/EurekCV/dataset/faces/1.jpg"
    
    if not Path(test_image).exists():
        print(f"Test image not found: {test_image}")
        return False
    
    # Register face
    print("Testing DeepFace registration...")
    with open(test_image, 'rb') as f:
        files = {'file': ('test.jpg', f, 'image/jpeg')}
        data = {
            'name': 'test_person_deepface',
            'metadata': json.dumps({'department': 'testing', 'source': 'deepface_chromadb_test'})
        }
        
        response = requests.post(f"{BASE_URL}/faces/", files=files, data=data)
        
    print(f"Registration response status: {response.status_code}")
    print(f"Registration response: {response.json()}")
    
    if response.status_code != 200:
        return False
    
    # List faces to verify
    print("\nListing all faces...")
    response = requests.get(f"{BASE_URL}/faces/")
    print(f"List response status: {response.status_code}")
    
    if response.status_code == 200:
        faces = response.json()
        print(f"Total faces: {faces.get('total', 0)}")
        for face in faces.get('faces', []):
            print(f"  - {face['person_name']} (ID: {face['face_id']})")
            if face.get('metadata'):
                print(f"    Metadata: {face['metadata']}")
    
    return True

def test_deepface_recognition():
    """Test face recognition using DeepFace API"""
    
    test_image = "/home/a/PycharmProjects/EurekCV/dataset/faces/1.jpg"
    
    if not Path(test_image).exists():
        print(f"Test image not found: {test_image}")
        return False
    
    print("\nTesting DeepFace recognition...")
    with open(test_image, 'rb') as f:
        files = {'file': ('test.jpg', f, 'image/jpeg')}
        data = {'threshold': 0.6}
        
        response = requests.post(f"{BASE_URL}/recognition", files=files, data=data)
        
    print(f"Recognition response status: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"Recognition result: {json.dumps(result, indent=2)}")
    else:
        print(f"Recognition failed: {response.text}")
    
    return response.status_code == 200

def test_deepface_stats():
    """Check DeepFace ChromaDB stats"""
    
    print("\nChecking DeepFace ChromaDB stats...")
    
    try:
        from facecv.database.factory import FaceDBFactory
        db = FaceDBFactory.create_database('deepface_chromadb')
        stats = db.get_stats()
        print(f"DeepFace ChromaDB stats: {json.dumps(stats, indent=2)}")
    except Exception as e:
        print(f"Failed to get stats: {e}")

if __name__ == "__main__":
    print("=== Testing DeepFace with separate ChromaDB ===\n")
    
    # Test registration
    if test_deepface_registration():
        print("\n✓ Registration test passed")
    else:
        print("\n✗ Registration test failed")
    
    # Test recognition
    if test_deepface_recognition():
        print("\n✓ Recognition test passed")
    else:
        print("\n✗ Recognition test failed")
    
    # Check stats
    test_deepface_stats()
    
    print("\n=== Test completed ===")