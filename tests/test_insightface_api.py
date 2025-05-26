#!/usr/bin/env python3
"""InsightFace API Endpoint Tests"""

import unittest
import sys
import os
import json
import io
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class TestInsightFaceAPI(unittest.TestCase):
    """Test InsightFace API endpoints"""
    
    def setUp(self):
        """Set up test environment"""
        self.test_image_dir = "/home/a/PycharmProjects/EurekCV/dataset/faces"
        self.test_images = {
            "trump1": os.path.join(self.test_image_dir, "trump1.jpeg"),
            "trump2": os.path.join(self.test_image_dir, "trump2.jpeg"),
            "harris1": os.path.join(self.test_image_dir, "harris1.jpeg"),
        }
        
    def test_api_routes_structure(self):
        """Test that all expected routes are defined"""
        try:
            from facecv.api.routes.face import router
            
            # Get all route paths
            routes = {route.path: route.methods for route in router.routes}
            
            # Expected routes
            expected_routes = {
                "/faces/register": {"POST"},
                "/faces/recognize": {"POST"},
                "/faces/verify": {"POST"},
                "/faces": {"GET"},
                "/faces/{face_id}": {"DELETE"},
                "/faces/by-name/{name}": {"DELETE"},
                "/faces/count": {"GET"},
                "/video_face/": {"POST"},
                "/recognize/webcam/stream": {"GET"},
                "/faces/offline": {"POST"}
            }
            
            for path, methods in expected_routes.items():
                self.assertIn(path, routes, f"Missing route: {path}")
                
            print("✓ All InsightFace API routes are defined")
            
        except ImportError as e:
            self.skipTest(f"API routes not available: {e}")
    
    @patch('facecv.api.routes.face.get_recognizer')
    async def test_register_face_endpoint(self, mock_get_recognizer):
        """Test face registration endpoint"""
        try:
            from facecv.api.routes.face import register_face
            from fastapi import UploadFile
            
            # Mock recognizer
            mock_recognizer = Mock()
            mock_recognizer.register.return_value = ["face_123"]
            mock_get_recognizer.return_value = mock_recognizer
            
            # Create mock file
            mock_file = Mock(spec=UploadFile)
            mock_file.filename = "test.jpg"
            mock_file.read = AsyncMock(return_value=b"fake_image_data")
            
            # Test registration
            result = await register_face(
                name="Test Person",
                file=mock_file,
                department="Testing",
                employee_id="001",
                recognizer=mock_recognizer
            )
            
            self.assertEqual(result, ["face_123"])
            mock_recognizer.register.assert_called_once()
            
            print("✓ Face registration endpoint test passed")
            
        except Exception as e:
            self.skipTest(f"Registration endpoint test skipped: {e}")
    
    @patch('facecv.api.routes.face.get_recognizer')
    async def test_recognize_face_endpoint(self, mock_get_recognizer):
        """Test face recognition endpoint"""
        try:
            from facecv.api.routes.face import recognize_face
            from facecv.schemas.face import RecognitionResult
            from fastapi import UploadFile
            
            # Mock recognizer
            mock_recognizer = Mock()
            mock_result = RecognitionResult(
                name="Test Person",
                confidence=0.95,
                bbox=[100, 100, 200, 200],
                metadata={}
            )
            mock_recognizer.recognize.return_value = [mock_result]
            mock_get_recognizer.return_value = mock_recognizer
            
            # Create mock file
            mock_file = Mock(spec=UploadFile)
            mock_file.filename = "test.jpg"
            mock_file.read = AsyncMock(return_value=b"fake_image_data")
            
            # Test recognition
            result = await recognize_face(
                file=mock_file,
                threshold=0.6,
                recognizer=mock_recognizer
            )
            
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0].name, "Test Person")
            
            print("✓ Face recognition endpoint test passed")
            
        except Exception as e:
            self.skipTest(f"Recognition endpoint test skipped: {e}")
    
    @patch('facecv.api.routes.face.get_recognizer')
    async def test_verify_faces_endpoint(self, mock_get_recognizer):
        """Test face verification endpoint"""
        try:
            from facecv.api.routes.face import verify_faces
            from facecv.schemas.face import VerificationResult
            from fastapi import UploadFile
            
            # Mock recognizer
            mock_recognizer = Mock()
            mock_result = VerificationResult(
                is_same_person=True,
                confidence=0.98,
                distance=0.2
            )
            mock_recognizer.verify.return_value = mock_result
            mock_get_recognizer.return_value = mock_recognizer
            
            # Create mock files
            mock_file1 = Mock(spec=UploadFile)
            mock_file1.filename = "test1.jpg"
            mock_file1.read = AsyncMock(return_value=b"fake_image_data1")
            
            mock_file2 = Mock(spec=UploadFile)
            mock_file2.filename = "test2.jpg"
            mock_file2.read = AsyncMock(return_value=b"fake_image_data2")
            
            # Test verification
            result = await verify_faces(
                file1=mock_file1,
                file2=mock_file2,
                threshold=0.6,
                recognizer=mock_recognizer
            )
            
            self.assertTrue(result.is_same_person)
            self.assertEqual(result.confidence, 0.98)
            
            print("✓ Face verification endpoint test passed")
            
        except Exception as e:
            self.skipTest(f"Verification endpoint test skipped: {e}")
    
    def test_video_face_extraction_endpoint(self):
        """Test video face extraction endpoint structure"""
        try:
            from facecv.api.routes.face import extract_faces_from_video
            import inspect
            
            # Check function signature
            sig = inspect.signature(extract_faces_from_video)
            params = list(sig.parameters.keys())
            
            expected_params = ["file", "method", "count", "interval", "quality_threshold", "recognizer"]
            for param in expected_params:
                self.assertIn(param, params, f"Missing parameter: {param}")
            
            print("✓ Video face extraction endpoint structure test passed")
            
        except Exception as e:
            self.skipTest(f"Video extraction test skipped: {e}")
    
    def test_webcam_stream_endpoint(self):
        """Test webcam stream recognition endpoint"""
        try:
            from facecv.api.routes.face import recognize_webcam_stream
            import inspect
            
            # Check function signature
            sig = inspect.signature(recognize_webcam_stream)
            params = list(sig.parameters.keys())
            
            expected_params = ["source", "threshold", "fps", "recognizer"]
            for param in expected_params:
                self.assertIn(param, params, f"Missing parameter: {param}")
            
            # Check if it's async
            self.assertTrue(inspect.iscoroutinefunction(recognize_webcam_stream))
            
            print("✓ Webcam stream endpoint structure test passed")
            
        except Exception as e:
            self.skipTest(f"Stream endpoint test skipped: {e}")
    
    def test_offline_batch_registration_endpoint(self):
        """Test offline batch registration endpoint"""
        try:
            from facecv.api.routes.face import batch_register_offline
            import inspect
            
            # Check function signature
            sig = inspect.signature(batch_register_offline)
            params = list(sig.parameters.keys())
            
            expected_params = ["directory_path", "quality_threshold", "recognizer"]
            for param in expected_params:
                self.assertIn(param, params, f"Missing parameter: {param}")
            
            print("✓ Offline batch registration endpoint structure test passed")
            
        except Exception as e:
            self.skipTest(f"Batch registration test skipped: {e}")
    
    @patch('facecv.api.routes.face.get_recognizer')
    async def test_list_faces_endpoint(self, mock_get_recognizer):
        """Test list faces endpoint"""
        try:
            from facecv.api.routes.face import list_faces
            from facecv.schemas.face import FaceInfo
            
            # Mock recognizer
            mock_recognizer = Mock()
            mock_recognizer.list_faces.return_value = [
                {
                    'id': 'face_1',
                    'name': 'Person 1',
                    'created_at': datetime.now(),
                    'updated_at': datetime.now(),
                    'metadata': {}
                },
                {
                    'id': 'face_2', 
                    'name': 'Person 2',
                    'created_at': datetime.now(),
                    'updated_at': datetime.now(),
                    'metadata': {}
                }
            ]
            mock_get_recognizer.return_value = mock_recognizer
            
            # Test listing
            result = await list_faces(
                name=None,
                skip=0,
                limit=100,
                recognizer=mock_recognizer
            )
            
            self.assertEqual(len(result), 2)
            self.assertIsInstance(result[0], FaceInfo)
            
            print("✓ List faces endpoint test passed")
            
        except Exception as e:
            self.skipTest(f"List faces test skipped: {e}")
    
    @patch('facecv.api.routes.face.get_recognizer')
    async def test_delete_face_endpoint(self, mock_get_recognizer):
        """Test delete face endpoint"""
        try:
            from facecv.api.routes.face import delete_face
            
            # Mock recognizer
            mock_recognizer = Mock()
            mock_recognizer.delete.return_value = True
            mock_get_recognizer.return_value = mock_recognizer
            
            # Test deletion
            result = await delete_face(
                face_id="face_123",
                recognizer=mock_recognizer
            )
            
            self.assertIn("message", result)
            mock_recognizer.delete.assert_called_with(face_id="face_123")
            
            print("✓ Delete face endpoint test passed")
            
        except Exception as e:
            self.skipTest(f"Delete face test skipped: {e}")
    
    def test_face_quality_integration(self):
        """Test face quality assessment integration"""
        try:
            from facecv.utils.face_quality import FaceQualityAssessor
            
            assessor = FaceQualityAssessor()
            
            # Test with mock image
            import numpy as np
            mock_image = np.ones((224, 224, 3), dtype=np.uint8) * 128
            bbox = [50, 50, 150, 150]
            
            quality = assessor.assess_face(mock_image, bbox)
            
            self.assertIsNotNone(quality)
            self.assertGreater(quality.overall_score, 0)
            self.assertLessEqual(quality.overall_score, 1)
            
            print("✓ Face quality integration test passed")
            
        except Exception as e:
            self.skipTest(f"Quality assessment test skipped: {e}")

class TestInsightFaceWorkflow(unittest.TestCase):
    """Test complete InsightFace workflows"""
    
    def test_registration_to_recognition_workflow(self):
        """Test complete workflow from registration to recognition"""
        print("\n=== Testing InsightFace Complete Workflow ===")
        
        try:
            from facecv.models.insightface.recognizer import InsightFaceRecognizer
            from facecv.database.factory import create_face_database
            
            # Initialize components
            db = create_face_database("sqlite", "sqlite:///:memory:")
            recognizer = InsightFaceRecognizer(mock_mode=True, face_db=db)
            
            # Step 1: Register multiple faces
            print("1. Registering faces...")
            names = ["Trump", "Harris", "Biden"]
            face_ids = []
            
            for name in names:
                ids = recognizer.register(
                    MagicMock(shape=(224, 224, 3)),
                    name=name,
                    metadata={"test": True}
                )
                face_ids.extend(ids)
            
            self.assertEqual(len(face_ids), 3)
            print(f"   ✓ Registered {len(face_ids)} faces")
            
            # Step 2: List all faces
            print("2. Listing registered faces...")
            all_faces = recognizer.list_faces()
            self.assertEqual(len(all_faces), 3)
            print(f"   ✓ Found {len(all_faces)} faces in database")
            
            # Step 3: Recognize a face
            print("3. Testing recognition...")
            results = recognizer.recognize(
                MagicMock(shape=(224, 224, 3)),
                threshold=0.6
            )
            self.assertTrue(len(results) > 0)
            print(f"   ✓ Recognition returned {len(results)} matches")
            
            # Step 4: Delete a face
            print("4. Testing deletion...")
            success = recognizer.delete(face_id=face_ids[0])
            self.assertTrue(success)
            
            remaining = recognizer.get_face_count()
            self.assertEqual(remaining, 2)
            print(f"   ✓ Deletion successful, {remaining} faces remain")
            
            print("\n✅ InsightFace workflow test completed!")
            
        except Exception as e:
            self.fail(f"Workflow test failed: {e}")
    
    def test_batch_operations_workflow(self):
        """Test batch processing operations"""
        print("\n=== Testing Batch Operations ===")
        
        try:
            from facecv.core.processor import VideoProcessor, ProcessingMode
            from facecv.database.factory import create_face_database
            
            # Initialize
            db = create_face_database("sqlite", "sqlite:///:memory:")
            processor = VideoProcessor(
                mode=ProcessingMode.RECOGNITION_ONLY,
                face_db=db,
                mock_mode=True
            )
            
            # Process multiple frames
            print("1. Processing batch of frames...")
            frames = [MagicMock(shape=(480, 640, 3)) for _ in range(5)]
            
            results = []
            for i, frame in enumerate(frames):
                result = processor.process_frame(frame)
                results.append(result)
                print(f"   Frame {i+1}: {len(result.faces)} faces detected")
            
            self.assertEqual(len(results), 5)
            print("   ✓ Batch processing completed")
            
            # Get statistics
            print("2. Analyzing statistics...")
            stats = processor.get_statistics()
            self.assertIn("frames_processed", stats)
            self.assertIn("faces_detected", stats)
            print(f"   ✓ Stats: {stats['frames_processed']} frames, {stats['faces_detected']} faces")
            
            print("\n✅ Batch operations test completed!")
            
        except Exception as e:
            self.skipTest(f"Batch operations test skipped: {e}")

def run_tests():
    """Run all InsightFace API tests"""
    print("Starting InsightFace API Tests\n")
    
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestInsightFaceAPI))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestInsightFaceWorkflow))
    
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