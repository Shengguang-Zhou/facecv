#!/usr/bin/env python3
"""
Test script to verify DeepFace dependencies
"""

def test_imports():
    """Test importing all required libraries"""
    print("Testing imports...")
    
    try:
        import cv2
        print("✅ OpenCV imported successfully")
    except Exception as e:
        print(f"❌ OpenCV failed: {e}")
    
    try:
        import dlib
        print("✅ dlib imported successfully") 
    except Exception as e:
        print(f"❌ dlib failed: {e}")
        
    try:
        import mediapipe
        print("✅ MediaPipe imported successfully")
    except Exception as e:
        print(f"❌ MediaPipe failed: {e}")
        
    try:
        import mtcnn
        print("✅ MTCNN imported successfully")
    except Exception as e:
        print(f"❌ MTCNN failed: {e}")
        
    try:
        import tensorflow as tf
        print(f"✅ TensorFlow {tf.__version__} imported successfully")
    except Exception as e:
        print(f"❌ TensorFlow failed: {e}")
        
    try:
        import deepface
        print(f"✅ DeepFace imported successfully")
    except Exception as e:
        print(f"❌ DeepFace failed: {e}")

def test_basic_functionality():
    """Test basic functionality"""
    print("\nTesting basic functionality...")
    
    # Test dlib face detection
    try:
        import dlib
        detector = dlib.get_frontal_face_detector()
        print("✅ dlib face detector initialized")
    except Exception as e:
        print(f"❌ dlib face detector failed: {e}")
    
    # Test MediaPipe
    try:
        import mediapipe as mp
        mp_face_detection = mp.solutions.face_detection
        print("✅ MediaPipe face detection available")
    except Exception as e:
        print(f"❌ MediaPipe face detection failed: {e}")
    
    # Test MTCNN
    try:
        from mtcnn import MTCNN
        detector = MTCNN()
        print("✅ MTCNN detector initialized")
    except Exception as e:
        print(f"❌ MTCNN detector failed: {e}")
    
    # Test DeepFace basic import
    try:
        from deepface import DeepFace
        print("✅ DeepFace core imported")
    except Exception as e:
        print(f"❌ DeepFace core failed: {e}")

if __name__ == "__main__":
    test_imports()
    test_basic_functionality()
    print("\nDependency test completed!")