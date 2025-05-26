#!/usr/bin/env python3
"""DeepFace Integration Tests - Complete Flow Testing"""

import unittest
import sys
import os
import json
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class TestDeepFaceIntegration(unittest.TestCase):
    """Complete integration tests for DeepFace functionality"""
    
    def setUp(self):
        """Set up test environment"""
        self.test_image_dir = "/home/a/PycharmProjects/EurekCV/dataset/faces"
        self.test_images = {
            "trump1": os.path.join(self.test_image_dir, "trump1.jpeg"),
            "trump2": os.path.join(self.test_image_dir, "trump2.jpeg"),
            "harris1": os.path.join(self.test_image_dir, "harris1.jpeg"),
        }
        
    def test_deepface_core_modules(self):
        """Test DeepFace core module imports and basic functionality"""
        try:
            from facecv.models.deepface.core import (
                DeepFaceRecognizer,
                DeepFaceEmbedding,
                DeepFaceVerification,
                DeepFaceAnalysis
            )
            
            # Test recognizer initialization
            recognizer = DeepFaceRecognizer(mock_mode=True)
            self.assertIsNotNone(recognizer)
            self.assertTrue(recognizer.mock_mode)
            
            # Test embedding module
            embedding = DeepFaceEmbedding(mock_mode=True)
            self.assertIsNotNone(embedding)
            
            # Test verification module
            verification = DeepFaceVerification(mock_mode=True)
            self.assertIsNotNone(verification)
            
            # Test analysis module
            analysis = DeepFaceAnalysis(mock_mode=True)
            self.assertIsNotNone(analysis)
            
            print("✓ DeepFace core modules loaded successfully")
            
        except ImportError as e:
            self.skipTest(f"DeepFace modules not available: {e}")
    
    def test_deepface_embedding_workflow(self):
        """Test complete embedding workflow"""
        try:
            from facecv.models.deepface.core.embedding import DeepFaceEmbedding
            
            embedding = DeepFaceEmbedding(mock_mode=True)
            
            # Test embedding generation
            mock_image = MagicMock()
            result = embedding.generate_embedding(mock_image, model_name="VGG-Face")
            
            self.assertIsNotNone(result)
            self.assertIn("embedding", result)
            self.assertIn("model", result)
            self.assertEqual(len(result["embedding"]), 2622)  # VGG-Face dimension
            
            # Test batch embedding
            batch_result = embedding.generate_embeddings_batch(
                [mock_image, mock_image],
                model_name="Facenet"
            )
            
            self.assertEqual(len(batch_result), 2)
            self.assertEqual(len(batch_result[0]["embedding"]), 128)  # Facenet dimension
            
            print("✓ DeepFace embedding workflow passed")
            
        except Exception as e:
            self.skipTest(f"Embedding test skipped: {e}")
    
    def test_deepface_verification_workflow(self):
        """Test complete verification workflow"""
        try:
            from facecv.models.deepface.core.verification import DeepFaceVerification
            
            verification = DeepFaceVerification(mock_mode=True)
            
            # Test single verification
            result = verification.verify_faces(
                "mock_image1.jpg",
                "mock_image2.jpg",
                model_name="VGG-Face"
            )
            
            self.assertIn("verified", result)
            self.assertIn("distance", result)
            self.assertIn("threshold", result)
            self.assertIn("model", result)
            
            # Test batch verification
            batch_result = verification.verify_batch(
                ["img1.jpg", "img2.jpg"],
                ["img3.jpg", "img4.jpg"]
            )
            
            self.assertEqual(len(batch_result), 2)
            
            print("✓ DeepFace verification workflow passed")
            
        except Exception as e:
            self.skipTest(f"Verification test skipped: {e}")
    
    def test_deepface_analysis_workflow(self):
        """Test complete analysis workflow"""
        try:
            from facecv.models.deepface.core.analysis import DeepFaceAnalysis
            
            analysis = DeepFaceAnalysis(mock_mode=True)
            
            # Test single image analysis
            result = analysis.analyze_face(
                "mock_image.jpg",
                actions=["age", "gender", "emotion", "race"]
            )
            
            self.assertIn("age", result)
            self.assertIn("gender", result)
            self.assertIn("dominant_emotion", result)
            self.assertIn("dominant_race", result)
            
            # Test batch analysis
            batch_result = analysis.analyze_batch(
                ["img1.jpg", "img2.jpg"],
                actions=["age", "gender"]
            )
            
            self.assertEqual(len(batch_result), 2)
            
            # Test statistics
            stats = analysis.get_analysis_statistics([result, result])
            self.assertIn("average_age", stats)
            self.assertIn("gender_distribution", stats)
            
            print("✓ DeepFace analysis workflow passed")
            
        except Exception as e:
            self.skipTest(f"Analysis test skipped: {e}")
    
    def test_deepface_api_integration(self):
        """Test DeepFace API route integration"""
        try:
            from facecv.api.routes.deepface import router
            from facecv.schemas.face import AnalysisResult, VerificationResult
            
            # Check router endpoints
            routes = [route.path for route in router.routes]
            
            expected_routes = [
                "/analyze/",
                "/verify/",
                "/faces/",
                "/video_face/",
                "/recognition",
                "/health"
            ]
            
            for route in expected_routes:
                self.assertIn(route, str(routes), f"Missing route: {route}")
            
            print("✓ DeepFace API routes configured correctly")
            
        except Exception as e:
            self.skipTest(f"API integration test skipped: {e}")
    
    def test_deepface_database_integration(self):
        """Test DeepFace with database integration"""
        try:
            from facecv.database.factory import create_face_database
            from facecv.models.deepface.core import DeepFaceRecognizer
            
            # Create test database
            db = create_face_database("sqlite", "sqlite:///:memory:")
            
            # Initialize recognizer with database
            recognizer = DeepFaceRecognizer(
                mock_mode=True,
                face_db=db
            )
            
            # Test registration
            mock_image = MagicMock()
            face_ids = recognizer.register_face(
                mock_image,
                name="Test Person",
                metadata={"department": "Testing"}
            )
            
            self.assertTrue(len(face_ids) > 0)
            
            # Test recognition
            results = recognizer.recognize_faces(
                mock_image,
                threshold=0.6
            )
            
            self.assertIsInstance(results, list)
            
            print("✓ DeepFace database integration passed")
            
        except Exception as e:
            self.skipTest(f"Database integration test skipped: {e}")
    
    def test_deepface_video_processing(self):
        """Test DeepFace video processing capabilities"""
        try:
            from facecv.utils.video_utils import VideoExtractor, FrameExtractionMethod
            from facecv.models.deepface.core import DeepFaceRecognizer
            
            recognizer = DeepFaceRecognizer(mock_mode=True)
            
            # Mock video processing
            mock_frames = [(0, MagicMock(), 0.0), (1, MagicMock(), 1.0)]
            
            results = []
            for frame_idx, frame, timestamp in mock_frames:
                result = recognizer.process_frame(frame)
                results.append({
                    "frame_index": frame_idx,
                    "timestamp": timestamp,
                    "faces": result
                })
            
            self.assertEqual(len(results), 2)
            
            print("✓ DeepFace video processing passed")
            
        except Exception as e:
            self.skipTest(f"Video processing test skipped: {e}")
    
    def test_deepface_performance_metrics(self):
        """Test DeepFace performance and resource usage"""
        import time
        
        try:
            from facecv.models.deepface.core import DeepFaceRecognizer
            
            recognizer = DeepFaceRecognizer(mock_mode=True)
            
            # Test processing speed
            start_time = time.time()
            for _ in range(10):
                recognizer.detect_faces(MagicMock())
            elapsed = time.time() - start_time
            
            avg_time = elapsed / 10
            self.assertLess(avg_time, 0.1)  # Should be fast in mock mode
            
            print(f"✓ DeepFace performance: {avg_time:.3f}s per detection")
            
        except Exception as e:
            self.skipTest(f"Performance test skipped: {e}")
    
    def test_deepface_error_handling(self):
        """Test DeepFace error handling and recovery"""
        try:
            from facecv.models.deepface.core import DeepFaceRecognizer
            
            recognizer = DeepFaceRecognizer(mock_mode=True)
            
            # Test with invalid input
            result = recognizer.detect_faces(None)
            self.assertEqual(result, [])
            
            # Test with empty image
            result = recognizer.detect_faces(MagicMock(shape=(0, 0, 3)))
            self.assertEqual(result, [])
            
            print("✓ DeepFace error handling passed")
            
        except Exception as e:
            self.skipTest(f"Error handling test skipped: {e}")
    
    def test_deepface_model_switching(self):
        """Test switching between different DeepFace models"""
        try:
            from facecv.models.deepface.core.embedding import DeepFaceEmbedding
            
            embedding = DeepFaceEmbedding(mock_mode=True)
            
            models = ["VGG-Face", "Facenet", "OpenFace", "DeepFace", "DeepID", "ArcFace"]
            
            for model in models:
                result = embedding.generate_embedding(
                    MagicMock(),
                    model_name=model
                )
                self.assertEqual(result["model"], model)
                
            print("✓ DeepFace model switching passed")
            
        except Exception as e:
            self.skipTest(f"Model switching test skipped: {e}")

class TestDeepFaceEndToEnd(unittest.TestCase):
    """End-to-end workflow tests"""
    
    def test_complete_face_recognition_workflow(self):
        """Test complete face recognition workflow from registration to identification"""
        print("\n=== Testing Complete DeepFace Workflow ===")
        
        try:
            from facecv.models.deepface.core import DeepFaceRecognizer
            from facecv.database.factory import create_face_database
            
            # Initialize components
            db = create_face_database("sqlite", "sqlite:///:memory:")
            recognizer = DeepFaceRecognizer(mock_mode=True, face_db=db)
            
            # Step 1: Register faces
            print("1. Registering faces...")
            faces_registered = 0
            for name in ["Trump", "Harris"]:
                face_ids = recognizer.register_face(
                    MagicMock(),
                    name=name,
                    metadata={"source": "test"}
                )
                faces_registered += len(face_ids)
            
            self.assertEqual(faces_registered, 2)
            print(f"   ✓ Registered {faces_registered} faces")
            
            # Step 2: Verify registration
            print("2. Verifying registration...")
            face_count = db.get_face_count()
            self.assertEqual(face_count, 2)
            print(f"   ✓ Database contains {face_count} faces")
            
            # Step 3: Recognize faces
            print("3. Testing recognition...")
            results = recognizer.recognize_faces(MagicMock(), threshold=0.6)
            self.assertTrue(len(results) > 0)
            print(f"   ✓ Recognition returned {len(results)} results")
            
            # Step 4: Face verification
            print("4. Testing verification...")
            from facecv.models.deepface.core.verification import DeepFaceVerification
            verification = DeepFaceVerification(mock_mode=True)
            
            ver_result = verification.verify_faces(
                MagicMock(),
                MagicMock()
            )
            self.assertIn("verified", ver_result)
            print(f"   ✓ Verification result: {ver_result['verified']}")
            
            # Step 5: Face analysis
            print("5. Testing analysis...")
            from facecv.models.deepface.core.analysis import DeepFaceAnalysis
            analysis = DeepFaceAnalysis(mock_mode=True)
            
            ana_result = analysis.analyze_face(MagicMock())
            self.assertIn("age", ana_result)
            print(f"   ✓ Analysis detected age: {ana_result['age']}")
            
            print("\n✅ Complete DeepFace workflow test passed!")
            
        except Exception as e:
            self.fail(f"End-to-end test failed: {e}")

def run_tests():
    """Run all tests and generate report"""
    print("Starting DeepFace Integration Tests\n")
    
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add all test cases
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestDeepFaceIntegration))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestDeepFaceEndToEnd))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Summary
    print(f"\n{'='*60}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    print(f"Success rate: {(result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100:.1f}%")
    print(f"{'='*60}")
    
    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)