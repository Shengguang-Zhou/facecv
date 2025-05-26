"""
Pytest configuration and shared fixtures for FaceCV tests
"""
import pytest
import requests
import base64
import os
from pathlib import Path
from typing import Dict, Any


# Test configuration
API_BASE_URL = "http://localhost:7003"

# DeepFace specific configuration
DEEPFACE_BASE_URL = f"{API_BASE_URL}/api/v1/deepface"
TIMEOUT = 30


@pytest.fixture(scope="session")
def api_client():
    """HTTP client for API testing"""
    session = requests.Session()
    # Don't set Content-Type for multipart form data
    session.headers.update({
        "Accept": "application/json"
    })
    return session


@pytest.fixture(scope="session") 
def test_image_file():
    """Create test image file for multipart upload"""
    # Try to find a test image
    test_faces_dir = Path("/home/a/PycharmProjects/EurekCV/dataset/faces")
    
    # Look for any image file
    image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
    test_image = None
    
    if test_faces_dir.exists():
        for ext in image_extensions:
            images = list(test_faces_dir.glob(f"**/{ext}"))
            if images:
                test_image = images[0]
                break
    
    if not test_image:
        # Create a simple test image if none found
        import numpy as np
        from PIL import Image
        import tempfile
        
        # Create a simple RGB image with face-like pattern
        img_array = np.random.randint(100, 200, (224, 224, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        
        # Save to temporary file
        temp_file = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
        img.save(temp_file.name, format='JPEG')
        temp_file.close()
        return temp_file.name
    
    return str(test_image)


@pytest.fixture(scope="session")
def sample_face_data():
    """Sample face registration data"""
    return {
        "name": "test_user",
        "metadata": {"department": "engineering", "role": "developer"}
    }


@pytest.fixture(scope="function")
def cleanup_test_faces(api_client):
    """Cleanup test faces after each test"""
    yield
    
    # Cleanup any faces with test names
    try:
        response = api_client.get(f"{API_BASE_URL}/api/v1/insightface/faces")
        if response.status_code == 200:
            faces = response.json().get("faces", [])
            for face in faces:
                if face.get("name", "").startswith("test_"):
                    face_id = face.get("id")
                    if face_id:
                        api_client.delete(f"{API_BASE_URL}/api/v1/insightface/faces/{face_id}")
    except:
        pass  # Ignore cleanup errors


def check_api_health():
    """Check if API server is running"""
    try:
        response = requests.get(f"{API_BASE_URL}/api/v1/insightface/health", timeout=5)
        return response.status_code == 200
    except:
        return False


@pytest.fixture(scope="session", autouse=True)
def verify_api_server():
    """Verify API server is running before tests"""
    if not check_api_health():
        pytest.skip(f"API server not running at {API_BASE_URL}")