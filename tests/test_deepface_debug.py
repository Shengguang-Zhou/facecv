#!/usr/bin/env python3
"""Debug script for DeepFace registration error"""

import os
import sys
import json
import requests
from pathlib import Path

# Set environment variables
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

# Add project to path
sys.path.insert(0, '/')

def test_deepface_registration():
    """Test DeepFace registration with detailed debugging"""
    
    base_url = "http://127.0.0.1:7003"
    
    # 1. Check health
    print("1. Checking DeepFace health...")
    try:
        resp = requests.get(f"{base_url}/api/v1/deepface/health")
        print(f"Health Status: {resp.status_code}")
        print(f"Response: {json.dumps(resp.json(), indent=2)}")
    except Exception as e:
        print(f"Health check failed: {e}")
    
    print("\n" + "="*50 + "\n")
    
    # 2. List faces before
    print("2. Listing faces before registration...")
    try:
        resp = requests.get(f"{base_url}/api/v1/deepface/faces/")
        print(f"List Status: {resp.status_code}")
        print(f"Faces: {json.dumps(resp.json(), indent=2)}")
    except Exception as e:
        print(f"List failed: {e}")
    
    print("\n" + "="*50 + "\n")
    
    # 3. Test registration with different metadata formats
    print("3. Testing registration...")
    
    # Test image
    image_path = "/home/a/PycharmProjects/EurekCV/dataset/faces/harris1.jpeg"
    
    if not Path(image_path).exists():
        print(f"Error: Test image not found at {image_path}")
        return
    
    # Test different metadata formats
    test_cases = [
        {
            "name": "Test Case 1: JSON string metadata",
            "files": {'file': ('harris1.jpeg', open(image_path, 'rb'), 'image/jpeg')},
            "data": {
                'name': 'Harris Test 1',
                'metadata': '{"department": "Testing", "employee_id": "EMP001"}'
            }
        },
        {
            "name": "Test Case 2: Dict metadata",
            "files": {'file': ('harris1.jpeg', open(image_path, 'rb'), 'image/jpeg')},
            "data": {
                'name': 'Harris Test 2',
                'metadata': json.dumps({"department": "Testing", "employee_id": "EMP002"})
            }
        },
        {
            "name": "Test Case 3: No metadata",
            "files": {'file': ('harris1.jpeg', open(image_path, 'rb'), 'image/jpeg')},
            "data": {
                'name': 'Harris Test 3'
            }
        }
    ]
    
    for test in test_cases:
        print(f"\n{test['name']}...")
        try:
            resp = requests.post(
                f"{base_url}/api/v1/deepface/faces/",
                files=test['files'],
                data=test['data']
            )
            print(f"Status: {resp.status_code}")
            print(f"Response: {json.dumps(resp.json(), indent=2)}")
        except Exception as e:
            print(f"Failed: {e}")
        finally:
            # Close file
            if 'files' in test and 'file' in test['files']:
                test['files']['file'][1].close()
    
    print("\n" + "="*50 + "\n")
    
    # 4. Debug DeepFace directly
    print("4. Testing DeepFace directly...")
    try:
        from deepface import DeepFace
        
        # Test representation
        print("Testing DeepFace.represent()...")
        result = DeepFace.represent(
            img_path=image_path,
            model_name="Facenet512",
            detector_backend="retinaface",
            enforce_detection=True
        )
        
        print(f"Result type: {type(result)}")
        print(f"Result length: {len(result) if isinstance(result, list) else 'N/A'}")
        
        if result:
            print(f"First item type: {type(result[0])}")
            if isinstance(result[0], dict):
                print(f"Keys: {list(result[0].keys())}")
                if "embedding" in result[0]:
                    print(f"Embedding length: {len(result[0]['embedding'])}")
            else:
                print(f"First item: {result[0]}")
                
    except Exception as e:
        print(f"Direct DeepFace test failed: {e}")
        import traceback
        traceback.print_exc()

def check_mysql_database():
    """Check MySQL database for faces"""
    print("\n" + "="*50 + "\n")
    print("5. Checking MySQL database...")
    
    try:
        import pymysql
        from facecv.config import get_db_config
        
        db_config = get_db_config()
        
        # Connect to MySQL
        connection = pymysql.connect(
            host=db_config.mysql_host,
            port=db_config.mysql_port,
            user=db_config.mysql_user,
            password=db_config.mysql_password,
            database=db_config.mysql_database,
            charset='utf8mb4'
        )
        
        with connection.cursor() as cursor:
            # Count faces
            cursor.execute("SELECT COUNT(*) as count FROM faces")
            result = cursor.fetchone()
            print(f"Total faces in MySQL: {result[0]}")
            
            # List recent faces
            cursor.execute("""
                SELECT id, name, created_at, updated_at 
                FROM faces 
                ORDER BY created_at DESC 
                LIMIT 5
            """)
            faces = cursor.fetchall()
            
            print("\nRecent faces:")
            for face in faces:
                print(f"  ID: {face[0]}, Name: {face[1]}, Created: {face[2]}")
                
        connection.close()
        
    except Exception as e:
        print(f"MySQL check failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("DeepFace Registration Debug Script")
    print("="*50)
    
    test_deepface_registration()
    check_mysql_database()
    
    print("\nDebug complete!")