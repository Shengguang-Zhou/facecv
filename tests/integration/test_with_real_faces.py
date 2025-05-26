#!/usr/bin/env python3
"""Integration test with real face images"""

import os
import sys
import json

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Test configuration
TEST_IMAGE_DIR = "/home/a/PycharmProjects/EurekCV/dataset/faces"
TEST_IMAGES = {
    "harris": ["harris1.jpeg", "harris2.jpeg"],
    "trump": ["trump.jpeg", "trump1.jpeg", "trump2.jpeg", "trump3.jpeg"]
}

def test_face_quality():
    """Test face quality assessment on real images"""
    print("=== Testing Face Quality Assessment ===")
    from facecv.utils.face_quality import FaceQualityAssessor
    from facecv.utils.image_utils import ImageProcessor
    
    assessor = FaceQualityAssessor()
    processor = ImageProcessor()
    
    for person, images in TEST_IMAGES.items():
        print(f"\nTesting {person}:")
        for img_name in images:
            img_path = os.path.join(TEST_IMAGE_DIR, img_name)
            image = processor.load_image(img_path)
            
            if image is not None:
                # Assume full image is face for testing
                bbox = [0, 0, image.shape[1], image.shape[0]]
                quality = assessor.assess_face(image, bbox)
                
                print(f"  {img_name}:")
                print(f"    Overall score: {quality.overall_score:.3f}")
                print(f"    Sharpness: {quality.sharpness_score:.3f}")
                print(f"    Brightness: {quality.brightness_score:.3f}")
                print(f"    Contrast: {quality.contrast_score:.3f}")
                print(f"    Pose: {quality.pose_score:.3f}")
                print(f"    Recommendation: {quality.recommendation}")

def test_image_processing():
    """Test image processing utilities"""
    print("\n=== Testing Image Processing ===")
    from facecv.utils.image_utils import ImageProcessor, ImageValidator
    
    processor = ImageProcessor()
    validator = ImageValidator()
    
    # Test loading and validation
    test_image = os.path.join(TEST_IMAGE_DIR, "trump1.jpeg")
    image = processor.load_image(test_image)
    
    if image is not None:
        print(f"Loaded image shape: {image.shape}")
        is_valid, msg = validator.validate_image(image)
        print(f"Validation: {is_valid} - {msg}")
        
        # Test resizing
        resized = processor.resize_image(image, (224, 224), keep_aspect_ratio=True)
        print(f"Resized shape: {resized.shape}")
        
        # Test normalization
        normalized = processor.normalize_image(image)
        print(f"Normalized range: [{normalized.min():.3f}, {normalized.max():.3f}]")

def test_database_operations():
    """Test database operations with face data"""
    print("\n=== Testing Database Operations ===")
    from facecv.database.factory import create_face_database
    from facecv.config.settings import Settings
    
    settings = Settings()
    # Use SQLite for testing
    db = create_face_database("sqlite", "sqlite:///:memory:")
    
    # Test adding faces
    test_faces = [
        {"name": "trump", "embedding": [0.1] * 512, "metadata": {"source": "trump1.jpeg"}},
        {"name": "trump", "embedding": [0.2] * 512, "metadata": {"source": "trump2.jpeg"}},
        {"name": "harris", "embedding": [0.3] * 512, "metadata": {"source": "harris1.jpeg"}},
    ]
    
    for face in test_faces:
        face_id = db.add_face(face["name"], face["embedding"], face["metadata"])
        print(f"Added face: {face_id} for {face['name']}")
    
    # Test queries
    print(f"\nTotal faces: {db.get_face_count()}")
    print(f"Trump faces: {len(db.get_faces_by_name('trump'))}")
    print(f"Harris faces: {len(db.get_faces_by_name('harris'))}")
    
    # Test similarity search (mock)
    similar = db.search_similar_faces([0.15] * 512, threshold=0.5)
    print(f"Similar faces found: {len(similar)}")

def test_core_processors():
    """Test core processing modules"""
    print("\n=== Testing Core Processors ===")
    from facecv.core.attendance import AttendanceSystem
    from facecv.core.stranger import StrangerDetector
    from facecv.utils.image_utils import ImageProcessor
    
    # Initialize
    attendance = AttendanceSystem()
    stranger_detector = StrangerDetector(save_dir="/tmp/strangers")
    processor = ImageProcessor()
    
    # Test with real images
    test_image = os.path.join(TEST_IMAGE_DIR, "trump1.jpeg")
    image = processor.load_image(test_image)
    
    if image is not None:
        # Mock face detection result
        mock_face = {
            "name": "trump",
            "confidence": 0.95,
            "bbox": [50, 50, 150, 150],
            "embedding": [0.1] * 512
        }
        
        # Test attendance
        attendance_result = attendance.process_face(mock_face, image)
        print(f"Attendance: {attendance_result}")
        
        # Test stranger detection
        stranger_alert = stranger_detector.process_face({"name": "Unknown", "confidence": 0.3}, image)
        if stranger_alert:
            print(f"Stranger alert: Level {stranger_alert.alert_level}, Count: {stranger_alert.appearance_count}")

def main():
    """Run all integration tests"""
    print("Starting Integration Tests with Real Images\n")
    
    try:
        test_face_quality()
        test_image_processing()
        test_database_operations()
        test_core_processors()
        
        print("\n✅ All integration tests completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()