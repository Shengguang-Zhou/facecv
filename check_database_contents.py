"""
Script to check the contents of the hybrid database and verify pre-registered faces.
"""

import os
import sys
from pprint import pprint
import uuid

os.environ["FACECV_DB_TYPE"] = "hybrid"

def check_database_contents():
    """Check the contents of the hybrid database."""
    try:
        from facecv.database.factory import create_face_database
        
        db = create_face_database('hybrid')
        print(f"Database type: {type(db).__name__}")
        
        mysql_faces = db.relational_db.get_all_faces()
        print(f"MySQL faces count: {len(mysql_faces)}")
        
        chroma_count = db.embedding_collection.count()
        print(f"ChromaDB faces count: {chroma_count}")
        
        print("\nMySQL Faces:")
        for face in mysql_faces:
            face_id = face.get('id', 'Unknown')
            name = face.get('name', 'Unknown')
            created_at = face.get('created_at', 'Unknown')
            print(f"  - ID: {face_id}, Name: {name}, Created: {created_at}")
            
            test_faces = [
                "e72ed2f0-abaf-45c8-9967-6e7084003665",  # Harris
                "a5b82313-e954-400a-8de8-c01c4d2cb6fe",  # Donald Trump
                "444a5b1c-2d79-4ba0-97fe-dfa3d3d7149c",  # "string"
            ]
            if face_id in test_faces:
                print(f"    ✓ Found test face: {name}")
        
        print("\nChecking ChromaDB embeddings for MySQL faces:")
        for face in mysql_faces:
            face_id = face.get('id', 'Unknown')
            name = face.get('name', 'Unknown')
            
            try:
                result = db.embedding_collection.get(ids=[face_id])
                if result and result['ids'] and len(result['ids']) > 0:
                    print(f"  ✓ Found embedding for {name} (ID: {face_id})")
                else:
                    print(f"  ✗ No embedding found for {name} (ID: {face_id})")
            except Exception as e:
                print(f"  ✗ Error checking embedding for {name} (ID: {face_id}): {e}")
        
        return True
    except Exception as e:
        print(f"Error checking database contents: {e}")
        import traceback
        print(traceback.format_exc())
        return False

def test_register_endpoint():
    """Test the register endpoint with a test image."""
    import requests
    import time
    
    print("\nTesting register endpoint...")
    
    url = "http://localhost:7003/api/v1/insightface/register"
    test_image = "test_images/test_face.jpg"
    
    try:
        with open(test_image, "rb") as f:
            files = {"file": (os.path.basename(test_image), f, "image/jpeg")}
            data = {
                "name": f"Test Person {int(time.time())}",
                "model": "buffalo_s"
            }
            
            print(f"Registering new face with name: {data['name']}")
            start_time = time.time()
            response = requests.post(url, files=files, data=data, timeout=60)
            elapsed = time.time() - start_time
            
            print(f"Request completed in {elapsed:.2f} seconds with status code: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                print("Registration successful:")
                pprint(result)
                return result
            else:
                print(f"Error: {response.status_code}")
                print(response.text)
                return None
    
    except requests.exceptions.Timeout:
        print(f"Request timed out after 60 seconds")
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None

if __name__ == "__main__":
    print("=== Checking Database Contents ===")
    check_database_contents()
    
    print("\n=== Testing Register Endpoint ===")
    register_result = test_register_endpoint()
    
    if register_result:
        print("\nRegister endpoint test successful!")
        
        print("\n=== Checking Database After Registration ===")
        check_database_contents()
    else:
        print("\nRegister endpoint test failed!")
