"""
Test core computer vision functionality
Following Ultralytics approach - test actual CV features
"""
import pytest
import numpy as np
import cv2
from PIL import Image
import tempfile
import os


def test_opencv_basic():
    """Test OpenCV basic functionality"""
    # Create a test image
    img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    
    # Basic OpenCV operations
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    assert gray.shape == (100, 100)
    assert blurred.shape == (100, 100)
    print("✓ OpenCV basic operations work")


def test_pillow_basic():
    """Test PIL/Pillow basic functionality"""
    # Create test image
    img = Image.new('RGB', (100, 100), color='red')
    
    # Basic PIL operations
    resized = img.resize((50, 50))
    rotated = img.rotate(45)
    
    assert resized.size == (50, 50)
    assert rotated.size == (100, 100)
    print("✓ PIL basic operations work")


def test_numpy_arrays():
    """Test numpy array operations for CV"""
    # Test image-like arrays
    img_array = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Common CV operations
    mean_val = np.mean(img_array)
    normalized = img_array / 255.0
    
    assert img_array.shape == (480, 640, 3)
    assert 0 <= mean_val <= 255
    assert 0 <= np.max(normalized) <= 1
    print("✓ NumPy array operations work")


def test_image_io():
    """Test image loading and saving"""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test image
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        # Save with OpenCV
        cv_path = os.path.join(tmpdir, 'test_cv.jpg')
        cv2.imwrite(cv_path, img)
        
        # Load with OpenCV
        loaded_cv = cv2.imread(cv_path)
        
        # Save with PIL
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        pil_path = os.path.join(tmpdir, 'test_pil.jpg')
        pil_img.save(pil_path)
        
        # Load with PIL
        loaded_pil = Image.open(pil_path)
        
        assert loaded_cv is not None
        assert loaded_pil is not None
        assert loaded_cv.shape == img.shape
        print("✓ Image I/O operations work")


def test_face_detection_prerequisites():
    """Test that we have the basic components for face detection"""
    try:
        import onnxruntime as ort
        print(f"✓ ONNX Runtime available: {ort.__version__}")
        
        # Test ONNX providers
        providers = ort.get_available_providers()
        assert 'CPUExecutionProvider' in providers
        print("✓ CPU execution provider available")
        
    except ImportError:
        pytest.skip("ONNX Runtime not available")


if __name__ == "__main__":
    test_opencv_basic()
    test_pillow_basic()
    test_numpy_arrays()
    test_image_io()
    test_face_detection_prerequisites()
    print("🎉 All CV functionality tests passed!")