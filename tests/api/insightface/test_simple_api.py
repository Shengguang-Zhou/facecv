"""
Simple InsightFace API Test without autouse fixtures
"""
import pytest
import requests
from pathlib import Path
import tempfile
from PIL import Image


def test_health_check():
    """Test health check endpoint"""
    response = requests.get("http://localhost:7003/api/v1/insightface/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"


def test_model_info():
    """Test model info endpoint"""
    response = requests.get("http://localhost:7003/api/v1/insightface/models/info")
    assert response.status_code == 200
    data = response.json()
    assert "model_pack" in data
    assert data["initialized"] == True


def test_face_count():
    """Test face count endpoint"""
    response = requests.get("http://localhost:7003/api/v1/insightface/faces/count")
    assert response.status_code == 200
    data = response.json()
    assert "total_faces" in data
    assert isinstance(data["total_faces"], int)


def test_detection_with_test_image():
    """Test face detection with a test image"""
    # Create a simple test image
    img = Image.new('RGB', (224, 224), color=(128, 128, 128))
    temp_file = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
    img.save(temp_file.name, format='JPEG')
    temp_file.close()
    
    try:
        with open(temp_file.name, 'rb') as f:
            files = {'file': ('test.jpg', f, 'image/jpeg')}
            params = {'min_confidence': 0.5}
            
            response = requests.post(
                "http://localhost:7003/api/v1/insightface/detect",
                files=files,
                params=params
            )
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        
    finally:
        # Cleanup
        Path(temp_file.name).unlink(missing_ok=True)


def test_registration_with_test_image():
    """Test face registration"""
    # Create a test image
    img = Image.new('RGB', (224, 224), color=(200, 150, 100))
    temp_file = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
    img.save(temp_file.name, format='JPEG')
    temp_file.close()
    
    try:
        with open(temp_file.name, 'rb') as f:
            files = {'file': ('test.jpg', f, 'image/jpeg')}
            data = {
                'name': 'test_simple_user',
                'department': 'testing'
            }
            
            response = requests.post(
                "http://localhost:7003/api/v1/insightface/register",
                files=files,
                data=data
            )
        
        assert response.status_code == 200
        result = response.json()
        assert "success" in result
        assert "message" in result
        
        # Cleanup - delete the registered face
        if result.get("success"):
            requests.delete("http://localhost:7003/api/v1/insightface/faces/by-name/test_simple_user")
            
    finally:
        # Cleanup file
        Path(temp_file.name).unlink(missing_ok=True)