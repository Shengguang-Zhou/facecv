#!/usr/bin/env python3
"""
Comprehensive test suite for CUDA detection and GPU acceleration functionality
"""

import sys
import os
import pytest
import numpy as np
import cv2
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Fix protobuf compatibility issue
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

from facecv.utils.cuda_utils import (
    get_cuda_version,
    get_cudnn_version,
    check_cuda_availability,
    get_execution_providers,
    setup_cuda_environment,
    install_appropriate_onnxruntime,
    check_onnxruntime_cuda_compatibility
)
from facecv.config.runtime_config import get_runtime_config
from facecv.models.insightface.real_recognizer import RealInsightFaceRecognizer
from facecv.database.factory import get_default_database

# Test data paths
TEST_DATA_PATH = Path("/home/a/PycharmProjects/EurekCV/dataset/faces")
HARRIS_IMAGE_1 = TEST_DATA_PATH / "harris1.jpeg"
HARRIS_IMAGE_2 = TEST_DATA_PATH / "harris2.jpeg"
TRUMP_IMAGE_1 = TEST_DATA_PATH / "trump1.jpeg"
TRUMP_IMAGE_2 = TEST_DATA_PATH / "trump2.jpeg"
TRUMP_IMAGE_3 = TEST_DATA_PATH / "trump3.jpeg"


class TestCUDADetection:
    """Test CUDA detection utilities"""
    
    def test_cuda_version_detection(self):
        """Test CUDA version detection"""
        cuda_version = get_cuda_version()
        
        if cuda_version:
            print(f"✓ CUDA version detected: {cuda_version[0]}.{cuda_version[1]}")
            assert isinstance(cuda_version, tuple)
            assert len(cuda_version) == 2
            assert cuda_version[0] >= 10  # At least CUDA 10
        else:
            print("ℹ No CUDA detected on this system")
    
    def test_cudnn_version_detection(self):
        """Test cuDNN version detection"""
        cudnn_version = get_cudnn_version()
        
        if cudnn_version:
            print(f"✓ cuDNN version detected: {cudnn_version}")
            assert isinstance(cudnn_version, int)
            assert cudnn_version >= 7  # At least cuDNN 7
        else:
            print("ℹ No cuDNN detected on this system")
    
    def test_cuda_availability(self):
        """Test CUDA availability check"""
        cuda_available = check_cuda_availability()
        print(f"CUDA available: {cuda_available}")
        assert isinstance(cuda_available, bool)
    
    def test_execution_providers(self):
        """Test execution provider selection"""
        providers = get_execution_providers()
        print(f"Execution providers: {providers}")
        
        assert isinstance(providers, list)
        assert len(providers) > 0
        assert 'CPUExecutionProvider' in providers
        
        if check_cuda_availability():
            # Should include CUDA providers when CUDA is available
            assert any('CUDA' in provider for provider in providers)
    
    def test_cuda_environment_setup(self):
        """Test CUDA environment setup"""
        cuda_available = setup_cuda_environment()
        print(f"CUDA environment setup result: {cuda_available}")
        assert isinstance(cuda_available, bool)
    
    def test_onnxruntime_installation_command(self):
        """Test ONNX Runtime installation command generation"""
        install_cmd = install_appropriate_onnxruntime()
        print(f"Recommended installation: {install_cmd}")
        
        assert isinstance(install_cmd, str)
        assert 'pip install' in install_cmd
        assert 'onnxruntime' in install_cmd
    
    def test_onnxruntime_compatibility(self):
        """Test ONNX Runtime CUDA compatibility"""
        try:
            compatible = check_onnxruntime_cuda_compatibility()
            print(f"ONNX Runtime CUDA compatible: {compatible}")
            assert isinstance(compatible, bool)
        except ImportError:
            print("ℹ ONNX Runtime not installed, skipping compatibility test")


class TestRuntimeConfig:
    """Test runtime configuration with CUDA settings"""
    
    def test_runtime_config_initialization(self):
        """Test runtime config initializes with CUDA settings"""
        runtime_config = get_runtime_config()
        
        # Check that CUDA settings are present
        assert 'cuda_available' in runtime_config.get_all()
        assert 'cuda_version' in runtime_config.get_all()
        assert 'execution_providers' in runtime_config.get_all()
        
        cuda_available = runtime_config.get('cuda_available')
        print(f"Runtime config CUDA available: {cuda_available}")
        
        providers = runtime_config.get('execution_providers')
        print(f"Runtime config providers: {providers}")
        
        assert isinstance(cuda_available, bool)
        assert isinstance(providers, list)
    
    def test_cuda_version_in_config(self):
        """Test CUDA version is properly stored in config"""
        runtime_config = get_runtime_config()
        cuda_version = runtime_config.get('cuda_version')
        
        if cuda_version:
            print(f"Runtime config CUDA version: {cuda_version[0]}.{cuda_version[1]}")
            assert isinstance(cuda_version, tuple)
            assert len(cuda_version) == 2


class TestInsightFaceGPU:
    """Test InsightFace GPU acceleration"""
    
    @pytest.fixture
    def recognizer(self):
        """Create InsightFace recognizer instance"""
        return RealInsightFaceRecognizer(prefer_gpu=True)
    
    def test_insightface_initialization(self, recognizer):
        """Test InsightFace initializes with GPU settings"""
        assert recognizer.initialized
        assert recognizer.prefer_gpu
        
        # Check model info
        model_info = recognizer.get_model_info()
        print(f"InsightFace model info: {model_info}")
        assert model_info['initialized']
        assert model_info['insightface_available']
    
    def test_face_detection_with_gpu(self, recognizer):
        """Test face detection using GPU acceleration"""
        if not HARRIS_IMAGE_1.exists():
            pytest.skip("Test image not found")
        
        # Load test image
        image = cv2.imread(str(HARRIS_IMAGE_1))
        assert image is not None
        
        # Detect faces
        faces = recognizer.detect_faces(image)
        print(f"Detected {len(faces)} faces in harris1.jpeg")
        
        assert isinstance(faces, list)
        if len(faces) > 0:
            face = faces[0]
            assert hasattr(face, 'bbox')
            assert hasattr(face, 'confidence')
            assert hasattr(face, 'embedding')
            print(f"Face confidence: {face.confidence}")
            print(f"Face bbox: {face.bbox}")
    
    def test_face_recognition_performance(self, recognizer):
        """Test face recognition performance with multiple images"""
        test_images = [HARRIS_IMAGE_1, HARRIS_IMAGE_2, TRUMP_IMAGE_1, TRUMP_IMAGE_2]
        
        face_counts = []
        processing_times = []
        
        for img_path in test_images:
            if not img_path.exists():
                continue
            
            image = cv2.imread(str(img_path))
            if image is None:
                continue
            
            import time
            start_time = time.time()
            faces = recognizer.detect_faces(image)
            end_time = time.time()
            
            processing_time = end_time - start_time
            face_counts.append(len(faces))
            processing_times.append(processing_time)
            
            print(f"{img_path.name}: {len(faces)} faces, {processing_time:.3f}s")
        
        if processing_times:
            avg_time = sum(processing_times) / len(processing_times)
            print(f"Average processing time: {avg_time:.3f}s")
            
            # Performance should be reasonable (less than 5 seconds per image)
            assert avg_time < 5.0


class TestFaceRegistrationAndRecognition:
    """Test face registration and recognition with GPU acceleration"""
    
    @pytest.fixture
    def recognizer(self):
        """Create recognizer with database"""
        db = get_default_database()
        return RealInsightFaceRecognizer(face_db=db, prefer_gpu=True)
    
    def test_face_registration(self, recognizer):
        """Test face registration with GPU acceleration"""
        if not HARRIS_IMAGE_1.exists():
            pytest.skip("Test image not found")
        
        # Load test image
        image = cv2.imread(str(HARRIS_IMAGE_1))
        assert image is not None
        
        # Register face
        face_ids = recognizer.register(image, "harris_test", {"test": True})
        print(f"Registered face IDs: {face_ids}")
        
        assert isinstance(face_ids, list)
        if len(face_ids) > 0:
            assert all(isinstance(fid, str) for fid in face_ids)
    
    def test_face_recognition(self, recognizer):
        """Test face recognition with GPU acceleration"""
        if not HARRIS_IMAGE_2.exists():
            pytest.skip("Test image not found")
        
        # Load test image
        image = cv2.imread(str(HARRIS_IMAGE_2))
        assert image is not None
        
        # Recognize faces
        results = recognizer.recognize(image, threshold=0.4)
        print(f"Recognition results: {len(results)}")
        
        for result in results:
            print(f"  Name: {result.name}, Confidence: {result.confidence}")
        
        assert isinstance(results, list)
    
    def test_face_verification(self, recognizer):
        """Test face verification with GPU acceleration"""
        if not (TRUMP_IMAGE_1.exists() and TRUMP_IMAGE_2.exists()):
            pytest.skip("Test images not found")
        
        # Load test images
        image1 = cv2.imread(str(TRUMP_IMAGE_1))
        image2 = cv2.imread(str(TRUMP_IMAGE_2))
        assert image1 is not None and image2 is not None
        
        # Verify faces
        verification_result = recognizer.verify(image1, image2, threshold=0.4)
        print(f"Verification result: {verification_result}")
        
        assert hasattr(verification_result, 'is_same_person')
        assert hasattr(verification_result, 'confidence')
        assert hasattr(verification_result, 'distance')
        
        print(f"Same person: {verification_result.is_same_person}")
        print(f"Confidence: {verification_result.confidence}")
        print(f"Distance: {verification_result.distance}")


class TestBatchProcessing:
    """Test batch processing with GPU acceleration"""
    
    @pytest.fixture
    def recognizer(self):
        """Create recognizer instance"""
        return RealInsightFaceRecognizer(prefer_gpu=True)
    
    def test_batch_face_detection(self, recognizer):
        """Test batch face detection"""
        test_images = []
        image_names = []
        
        for img_path in [HARRIS_IMAGE_1, HARRIS_IMAGE_2, TRUMP_IMAGE_1, TRUMP_IMAGE_2]:
            if img_path.exists():
                image = cv2.imread(str(img_path))
                if image is not None:
                    test_images.append(image)
                    image_names.append(img_path.name)
        
        if not test_images:
            pytest.skip("No test images available")
        
        print(f"Testing batch detection with {len(test_images)} images")
        
        # Batch detect faces
        import time
        start_time = time.time()
        results = recognizer.batch_detect_faces(test_images)
        end_time = time.time()
        
        processing_time = end_time - start_time
        print(f"Batch processing time: {processing_time:.3f}s")
        
        assert len(results) == len(test_images)
        
        for i, (faces, img_name) in enumerate(zip(results, image_names)):
            print(f"{img_name}: {len(faces)} faces detected")
            assert isinstance(faces, list)


def run_comprehensive_test():
    """Run all tests and generate report"""
    print("=" * 60)
    print("COMPREHENSIVE CUDA ACCELERATION TEST SUITE")
    print("=" * 60)
    
    # Test classes to run
    test_classes = [
        TestCUDADetection,
        TestRuntimeConfig,
        TestInsightFaceGPU,
        TestFaceRegistrationAndRecognition,
        TestBatchProcessing
    ]
    
    for test_class in test_classes:
        print(f"\n--- Running {test_class.__name__} ---")
        
        # Create instance and run tests
        instance = test_class()
        
        # Get all test methods
        test_methods = [method for method in dir(instance) if method.startswith('test_')]
        
        for method_name in test_methods:
            try:
                print(f"\n🧪 {method_name}")
                method = getattr(instance, method_name)
                
                # Handle pytest fixtures
                if hasattr(method, '__code__') and 'recognizer' in method.__code__.co_varnames:
                    # Create recognizer for methods that need it
                    if 'face_db' in str(method):
                        db = get_default_database()
                        recognizer = RealInsightFaceRecognizer(face_db=db, prefer_gpu=True)
                    else:
                        recognizer = RealInsightFaceRecognizer(prefer_gpu=True)
                    method(recognizer)
                else:
                    method()
                
                print(f"✅ {method_name} PASSED")
                
            except Exception as e:
                print(f"❌ {method_name} FAILED: {e}")
    
    print("\n" + "=" * 60)
    print("TEST SUITE COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    run_comprehensive_test()