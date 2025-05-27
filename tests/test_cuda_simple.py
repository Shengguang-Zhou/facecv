#!/usr/bin/env python3
"""
Simple CUDA acceleration tests without external dependencies
"""

import sys
import os
import time
import traceback
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Fix protobuf compatibility issue
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

# Test data paths
TEST_DATA_PATH = Path("/home/a/PycharmProjects/EurekCV/dataset/faces")

def test_cuda_detection():
    """Test CUDA detection utilities"""
    print("\n🔧 Testing CUDA Detection...")
    
    try:
        from facecv.utils.cuda_utils import (
            get_cuda_version,
            get_cudnn_version,
            check_cuda_availability,
            get_execution_providers,
            setup_cuda_environment
        )
        
        # Test CUDA version detection
        cuda_version = get_cuda_version()
        if cuda_version:
            print(f"✅ CUDA version detected: {cuda_version[0]}.{cuda_version[1]}")
        else:
            print("ℹ️  No CUDA detected")
        
        # Test CUDA availability
        cuda_available = check_cuda_availability()
        print(f"✅ CUDA available: {cuda_available}")
        
        # Test execution providers
        providers = get_execution_providers()
        print(f"✅ Execution providers: {providers}")
        
        # Test environment setup
        env_result = setup_cuda_environment()
        print(f"✅ Environment setup result: {env_result}")
        
        return True
        
    except Exception as e:
        print(f"❌ CUDA detection test failed: {e}")
        traceback.print_exc()
        return False

def test_runtime_config():
    """Test runtime configuration"""
    print("\n⚙️  Testing Runtime Configuration...")
    
    try:
        from facecv.config.runtime_config import get_runtime_config
        
        runtime_config = get_runtime_config()
        config_data = runtime_config.get_all()
        
        # Check required keys
        required_keys = ['cuda_available', 'execution_providers', 'prefer_gpu']
        missing_keys = [key for key in required_keys if key not in config_data]
        
        if missing_keys:
            print(f"❌ Missing config keys: {missing_keys}")
            return False
        
        print(f"✅ CUDA available in config: {config_data['cuda_available']}")
        print(f"✅ Execution providers: {config_data['execution_providers']}")
        print(f"✅ Prefer GPU: {config_data['prefer_gpu']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Runtime config test failed: {e}")
        traceback.print_exc()
        return False

def test_insightface_initialization():
    """Test InsightFace initialization with GPU settings"""
    print("\n👁️  Testing InsightFace Initialization...")
    
    try:
        from facecv.models.insightface.real_recognizer import RealInsightFaceRecognizer
        
        # Create recognizer with GPU preference
        recognizer = RealInsightFaceRecognizer(prefer_gpu=True)
        
        if not recognizer.initialized:
            print("❌ InsightFace failed to initialize")
            return False
        
        print("✅ InsightFace initialized successfully")
        print(f"✅ GPU preference: {recognizer.prefer_gpu}")
        
        # Test model info
        model_info = recognizer.get_model_info()
        print(f"✅ Model pack: {model_info['model_pack']}")
        print(f"✅ InsightFace available: {model_info['insightface_available']}")
        
        return True
        
    except Exception as e:
        print(f"❌ InsightFace initialization test failed: {e}")
        traceback.print_exc()
        return False

def test_face_detection():
    """Test face detection with test images"""
    print("\n👤 Testing Face Detection...")
    
    harris_image = TEST_DATA_PATH / "harris1.jpeg"
    trump_image = TEST_DATA_PATH / "trump1.jpeg"
    
    if not harris_image.exists():
        print(f"⚠️  Test image not found: {harris_image}")
        return False
    
    try:
        import cv2
        from facecv.models.insightface.real_recognizer import RealInsightFaceRecognizer
        
        recognizer = RealInsightFaceRecognizer(prefer_gpu=True)
        
        # Test with Harris image
        image = cv2.imread(str(harris_image))
        if image is None:
            print(f"❌ Failed to load image: {harris_image}")
            return False
        
        start_time = time.time()
        faces = recognizer.detect_faces(image)
        detection_time = time.time() - start_time
        
        print(f"✅ Detected {len(faces)} faces in harris1.jpeg")
        print(f"✅ Detection time: {detection_time:.3f}s")
        
        if faces:
            face = faces[0]
            print(f"✅ Face confidence: {face.confidence:.3f}")
            print(f"✅ Face bbox: {face.bbox}")
        
        # Test with Trump image if available
        if trump_image.exists():
            image2 = cv2.imread(str(trump_image))
            if image2 is not None:
                start_time = time.time()
                faces2 = recognizer.detect_faces(image2)
                detection_time2 = time.time() - start_time
                
                print(f"✅ Detected {len(faces2)} faces in trump1.jpeg")
                print(f"✅ Detection time: {detection_time2:.3f}s")
        
        return True
        
    except Exception as e:
        print(f"❌ Face detection test failed: {e}")
        traceback.print_exc()
        return False

def test_face_verification():
    """Test face verification"""
    print("\n🔍 Testing Face Verification...")
    
    trump1_image = TEST_DATA_PATH / "trump1.jpeg"
    trump2_image = TEST_DATA_PATH / "trump2.jpeg"
    harris_image = TEST_DATA_PATH / "harris1.jpeg"
    
    if not (trump1_image.exists() and trump2_image.exists() and harris_image.exists()):
        print("⚠️  Required test images not found")
        return False
    
    try:
        import cv2
        from facecv.models.insightface.real_recognizer import RealInsightFaceRecognizer
        
        recognizer = RealInsightFaceRecognizer(prefer_gpu=True)
        
        # Load images
        trump1 = cv2.imread(str(trump1_image))
        trump2 = cv2.imread(str(trump2_image))
        harris = cv2.imread(str(harris_image))
        
        if any(img is None for img in [trump1, trump2, harris]):
            print("❌ Failed to load test images")
            return False
        
        # Test Trump1 vs Trump2 (should be same person)
        start_time = time.time()
        result1 = recognizer.verify(trump1, trump2, threshold=0.4)
        verify_time1 = time.time() - start_time
        
        print(f"✅ Trump1 vs Trump2 verification:")
        print(f"   Same person: {result1.is_same_person}")
        print(f"   Confidence: {result1.confidence:.3f}")
        print(f"   Time: {verify_time1:.3f}s")
        
        # Test Trump1 vs Harris (should be different people)
        start_time = time.time()
        result2 = recognizer.verify(trump1, harris, threshold=0.4)
        verify_time2 = time.time() - start_time
        
        print(f"✅ Trump1 vs Harris verification:")
        print(f"   Same person: {result2.is_same_person}")
        print(f"   Confidence: {result2.confidence:.3f}")
        print(f"   Time: {verify_time2:.3f}s")
        
        return True
        
    except Exception as e:
        print(f"❌ Face verification test failed: {e}")
        traceback.print_exc()
        return False

def test_deepface_gpu():
    """Test DeepFace GPU configuration"""
    print("\n🧠 Testing DeepFace GPU Configuration...")
    
    try:
        # Try to import TensorFlow and check GPU
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        print(f"✅ TensorFlow detected {len(gpus)} GPU(s)")
        
        # Test DeepFace recognizer
        from facecv.models.deepface.core.recognizer import DeepFaceRecognizer
        
        recognizer = DeepFaceRecognizer(
            model_name="VGG-Face",
            detector_backend="opencv",
            enforce_detection=False
        )
        
        print("✅ DeepFace recognizer initialized successfully")
        return True
        
    except ImportError as e:
        if "tensorflow" in str(e):
            print("ℹ️  TensorFlow not available, skipping DeepFace GPU test")
            return True  # Not a failure, just not available
        else:
            print(f"❌ DeepFace GPU test failed: {e}")
            return False
    except Exception as e:
        print(f"❌ DeepFace GPU test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("🎯 FaceCV CUDA Acceleration Test Suite")
    print("=" * 60)
    
    tests = [
        ("CUDA Detection", test_cuda_detection),
        ("Runtime Configuration", test_runtime_config),
        ("InsightFace Initialization", test_insightface_initialization),
        ("Face Detection", test_face_detection),
        ("Face Verification", test_face_verification),
        ("DeepFace GPU", test_deepface_gpu),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'=' * 20} {test_name} {'=' * 20}")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    failed = 0
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:30} {status}")
        if result:
            passed += 1
        else:
            failed += 1
    
    print(f"\nTotal: {len(results)} tests")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    
    if failed == 0:
        print("\n🎉 All tests passed!")
    else:
        print(f"\n⚠️  {failed} test(s) failed")
    
    return failed == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)