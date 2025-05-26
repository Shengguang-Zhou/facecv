"""
Standalone InsightFace API Tests - No conftest dependencies
"""
import requests
import tempfile
from PIL import Image
from pathlib import Path


def test_health_direct():
    """Direct health check test"""
    response = requests.get("http://localhost:7003/api/v1/insightface/health")
    print(f"Health Status: {response.status_code}")
    if response.status_code == 200:
        print(f"Response: {response.json()}")
    else:
        print(f"Error: {response.text}")
    assert response.status_code == 200


def test_detection_with_synthetic_image():
    """Test detection with synthetic image"""
    # Create test image
    img = Image.new('RGB', (300, 300), color=(150, 150, 150))
    temp_file = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
    img.save(temp_file.name, format='JPEG')
    temp_file.close()
    
    try:
        with open(temp_file.name, 'rb') as f:
            files = {'file': ('test.jpg', f, 'image/jpeg')}
            response = requests.post(
                "http://localhost:7003/api/v1/insightface/detect",
                files=files
            )
        
        print(f"Detection Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"Detected faces: {len(data)}")
        else:
            print(f"Detection failed: {response.text}")
            
        assert response.status_code == 200
        
    finally:
        Path(temp_file.name).unlink(missing_ok=True)


def test_registration_expected_failure():
    """Test registration - expected to fail with synthetic image (no faces)"""
    img = Image.new('RGB', (200, 200), color=(100, 100, 100))
    temp_file = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
    img.save(temp_file.name, format='JPEG')
    temp_file.close()
    
    try:
        with open(temp_file.name, 'rb') as f:
            files = {'file': ('test.jpg', f, 'image/jpeg')}
            data = {'name': 'test_user', 'department': 'test'}
            response = requests.post(
                "http://localhost:7003/api/v1/insightface/register",
                files=files,
                data=data
            )
        
        print(f"Registration Status: {response.status_code}")
        print(f"Response: {response.text}")
        
        # Should fail with 400 - no faces detected (expected behavior)
        assert response.status_code == 400
        assert "No faces detected" in response.text
        
    finally:
        Path(temp_file.name).unlink(missing_ok=True)


def test_verification_with_same_image():
    """Test verification with same synthetic image"""
    img = Image.new('RGB', (250, 250), color=(200, 200, 200))
    temp_file = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
    img.save(temp_file.name, format='JPEG')
    temp_file.close()
    
    try:
        with open(temp_file.name, 'rb') as f1, open(temp_file.name, 'rb') as f2:
            files = {
                'file1': ('test1.jpg', f1, 'image/jpeg'),
                'file2': ('test2.jpg', f2, 'image/jpeg')
            }
            response = requests.post(
                "http://localhost:7003/api/v1/insightface/verify",
                files=files
            )
        
        print(f"Verification Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"Verification result: {data}")
        else:
            print(f"Verification failed: {response.text}")
            
        assert response.status_code == 200
        
    finally:
        Path(temp_file.name).unlink(missing_ok=True)


if __name__ == "__main__":
    print("Running standalone InsightFace API tests...")
    
    try:
        test_health_direct()
        print("✅ Health check passed")
    except Exception as e:
        print(f"❌ Health check failed: {e}")
    
    try:
        test_detection_with_synthetic_image()
        print("✅ Detection test passed")
    except Exception as e:
        print(f"❌ Detection test failed: {e}")
    
    try:
        test_registration_expected_failure()
        print("✅ Registration test passed (expected failure)")
    except Exception as e:
        print(f"❌ Registration test failed: {e}")
    
    try:
        test_verification_with_same_image()
        print("✅ Verification test passed")
    except Exception as e:
        print(f"❌ Verification test failed: {e}")
    
    print("Standalone tests completed!")