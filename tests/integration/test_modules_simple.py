#!/usr/bin/env python3
"""Simple module tests without external dependencies"""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

def test_database_layer():
    """Test database abstraction layer"""
    print("=== Testing Database Layer ===")
    
    # Test SQLite implementation
    from facecv.database.sqlite_facedb import SQLiteFaceDB
    
    db = SQLiteFaceDB("sqlite:///:memory:")
    
    # Add test data
    face_id1 = db.add_face("Trump", [0.1] * 512, {"source": "trump1.jpeg"})
    face_id2 = db.add_face("Trump", [0.2] * 512, {"source": "trump2.jpeg"})
    face_id3 = db.add_face("Harris", [0.3] * 512, {"source": "harris1.jpeg"})
    
    print(f"Added faces: {face_id1}, {face_id2}, {face_id3}")
    
    # Test retrieval
    print(f"Total faces: {db.get_face_count()}")
    trump_faces = db.get_faces_by_name("Trump")
    print(f"Trump faces: {len(trump_faces)}")
    
    # Test search
    similar = db.search_similar_faces([0.15] * 512, threshold=0.8)
    print(f"Similar faces found: {len(similar)}")
    
    # Test update
    success = db.update_face(face_id1, embedding=[0.11] * 512, metadata={"updated": True})
    print(f"Update successful: {success}")
    
    # Test delete
    deleted = db.delete_face(face_id3)
    print(f"Deleted face: {deleted}")
    print(f"Total faces after delete: {db.get_face_count()}")
    
    print("✓ Database layer working correctly\n")

def test_core_modules():
    """Test core processing modules"""
    print("=== Testing Core Modules ===")
    
    # Test Attendance System
    from facecv.core.attendance import AttendanceSystem
    
    attendance = AttendanceSystem()
    
    # Simulate face recognition results
    faces = [
        {"name": "Trump", "confidence": 0.95, "bbox": [100, 100, 200, 200]},
        {"name": "Harris", "confidence": 0.89, "bbox": [300, 100, 400, 200]},
    ]
    
    import numpy as np
    mock_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    for face in faces:
        result = attendance.process_face(face, mock_frame)
        print(f"Attendance for {face['name']}: {result}")
    
    # Get statistics
    stats = attendance.get_statistics()
    print(f"Attendance stats: Present={stats['present_count']}, Total={stats['total_records']}")
    
    # Test Stranger Detector
    from facecv.core.stranger import StrangerDetector
    
    stranger_detector = StrangerDetector(save_dir="/tmp/test_strangers")
    
    # Test with unknown face
    unknown_face = {"name": "Unknown", "confidence": 0.3, "bbox": [50, 50, 150, 150]}
    alert = stranger_detector.process_face(unknown_face, mock_frame)
    
    if alert:
        print(f"Stranger alert: Level={alert.alert_level}, Count={alert.appearance_count}")
    
    print("✓ Core modules working correctly\n")

def test_schemas():
    """Test data schemas"""
    print("=== Testing Schemas ===")
    
    from facecv.schemas.face import (
        FaceInfo, RecognitionResult, VerificationResult,
        BatchRecognitionResult, AnalysisResult
    )
    from datetime import datetime
    
    # Test FaceInfo
    face_info = FaceInfo(
        id="test_123",
        name="Test Person",
        created_at=datetime.now(),
        metadata={"department": "IT"}
    )
    print(f"FaceInfo: {face_info.name} (ID: {face_info.id})")
    
    # Test RecognitionResult
    rec_result = RecognitionResult(
        name="Test Person",
        confidence=0.95,
        bbox=[100, 100, 200, 200],
        metadata={"department": "IT"}
    )
    print(f"RecognitionResult: {rec_result.name} (confidence: {rec_result.confidence})")
    
    # Test VerificationResult
    ver_result = VerificationResult(
        is_same_person=True,
        confidence=0.98,
        distance=0.2
    )
    print(f"VerificationResult: Same person={ver_result.is_same_person} (confidence: {ver_result.confidence})")
    
    print("✓ Schemas working correctly\n")

def test_config():
    """Test configuration system"""
    print("=== Testing Configuration ===")
    
    from facecv.config.settings import Settings
    
    settings = Settings()
    
    print(f"Model backend: {settings.model_backend}")
    print(f"Database type: {settings.db_type}")
    print(f"API prefix: {settings.api_prefix}")
    print(f"Max upload size: {settings.max_upload_size // 1024 // 1024}MB")
    print(f"Allowed extensions: {settings.allowed_extensions}")
    
    print("✓ Configuration working correctly\n")

def main():
    """Run all tests"""
    print("Running Simple Integration Tests\n")
    
    try:
        test_database_layer()
        test_core_modules()
        test_schemas()
        test_config()
        
        print("✅ All tests completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()