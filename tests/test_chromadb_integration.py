#!/usr/bin/env python3
"""ChromaDB Integration Tests"""

import unittest
import sys
import os
import tempfile
import shutil
import numpy as np
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class TestChromaDBIntegration(unittest.TestCase):
    """Test ChromaDB vector database integration"""
    
    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.test_embedding = np.random.rand(512).tolist()
        
    def tearDown(self):
        """Clean up test environment"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_chromadb_in_memory(self):
        """Test ChromaDB in-memory mode"""
        try:
            from facecv.database.factory import create_face_database
            
            # Create in-memory ChromaDB
            db = create_face_database('chromadb')
            
            print("\n=== Testing ChromaDB In-Memory Mode ===")
            
            # Test add face
            face_id = db.add_face(
                name="Test Person",
                embedding=self.test_embedding,
                metadata={"test": True}
            )
            self.assertIsNotNone(face_id)
            print(f"✓ Added face: {face_id}")
            
            # Test get face
            face = db.get_face(face_id)
            self.assertIsNotNone(face)
            self.assertEqual(face["name"], "Test Person")
            print("✓ Retrieved face by ID")
            
            # Test search similar
            similar = db.search_similar_faces(self.test_embedding, threshold=0.9)
            self.assertEqual(len(similar), 1)
            self.assertGreater(similar[0][1], 0.9)  # Similarity score
            print(f"✓ Found similar face with score: {similar[0][1]:.3f}")
            
            # Test statistics
            stats = db.get_statistics()
            self.assertEqual(stats["total_faces"], 1)
            self.assertEqual(stats["backend"], "ChromaDB")
            print(f"✓ Statistics: {stats}")
            
            print("\n✅ ChromaDB in-memory tests passed!")
            
        except ImportError:
            self.skipTest("ChromaDB not available")
    
    def test_chromadb_persistent(self):
        """Test ChromaDB persistent mode"""
        try:
            from facecv.database.factory import create_face_database
            
            persist_dir = os.path.join(self.temp_dir, "chromadb")
            
            print("\n=== Testing ChromaDB Persistent Mode ===")
            
            # Create persistent ChromaDB
            db = create_face_database(
                'chromadb',
                persist_directory=persist_dir
            )
            
            # Add multiple faces
            face_ids = []
            for i in range(5):
                face_id = db.add_face(
                    name=f"Person_{i}",
                    embedding=np.random.rand(512).tolist(),
                    metadata={"index": i}
                )
                face_ids.append(face_id)
            
            print(f"✓ Added {len(face_ids)} faces")
            
            # Close and reopen database
            del db
            
            # Reopen database
            db2 = create_face_database(
                'chromadb',
                persist_directory=persist_dir
            )
            
            # Verify persistence
            count = db2.get_face_count()
            self.assertEqual(count, 5)
            print(f"✓ Persistence verified: {count} faces")
            
            # Test batch operations
            all_faces = db2.get_all_faces()
            self.assertEqual(len(all_faces), 5)
            print("✓ Batch retrieval working")
            
            print("\n✅ ChromaDB persistent tests passed!")
            
        except ImportError:
            self.skipTest("ChromaDB not available")
    
    def test_chromadb_with_connection_string(self):
        """Test ChromaDB with connection string"""
        try:
            from facecv.database.factory import create_face_database
            
            print("\n=== Testing ChromaDB Connection String ===")
            
            # Test in-memory
            db1 = create_face_database(connection_string='chromadb://')
            self.assertIsNotNone(db1)
            print("✓ Created in-memory ChromaDB with connection string")
            
            # Test persistent
            persist_path = os.path.join(self.temp_dir, "chromadb2")
            db2 = create_face_database(connection_string=f'chromadb://{persist_path}')
            self.assertIsNotNone(db2)
            print("✓ Created persistent ChromaDB with connection string")
            
            print("\n✅ Connection string tests passed!")
            
        except ImportError:
            self.skipTest("ChromaDB not available")
    
    def test_chromadb_vector_search(self):
        """Test ChromaDB vector similarity search"""
        try:
            from facecv.database.factory import create_face_database
            
            print("\n=== Testing ChromaDB Vector Search ===")
            
            db = create_face_database('chromadb')
            
            # Add faces with known embeddings
            base_embedding = np.random.rand(512)
            
            # Add original face
            face_id1 = db.add_face(
                name="Original",
                embedding=base_embedding.tolist(),
                metadata={"type": "original"}
            )
            
            # Add similar face (small perturbation)
            similar_embedding = base_embedding + np.random.randn(512) * 0.01
            face_id2 = db.add_face(
                name="Similar",
                embedding=similar_embedding.tolist(),
                metadata={"type": "similar"}
            )
            
            # Add different face
            different_embedding = np.random.rand(512)
            face_id3 = db.add_face(
                name="Different",
                embedding=different_embedding.tolist(),
                metadata={"type": "different"}
            )
            
            # Search with original embedding
            results = db.search_similar_faces(
                base_embedding.tolist(),
                threshold=0.8,
                limit=10
            )
            
            # Verify results
            self.assertGreaterEqual(len(results), 2)
            
            # Check ordering (most similar first)
            similarities = [r[1] for r in results]
            self.assertEqual(similarities, sorted(similarities, reverse=True))
            
            print(f"✓ Vector search found {len(results)} results")
            for i, (face, score) in enumerate(results):
                print(f"  {i+1}. {face['name']}: {score:.3f}")
            
            print("\n✅ Vector search tests passed!")
            
        except ImportError:
            self.skipTest("ChromaDB not available")
    
    def test_chromadb_backup_restore(self):
        """Test ChromaDB backup and restore"""
        try:
            from facecv.database.factory import create_face_database
            
            print("\n=== Testing ChromaDB Backup/Restore ===")
            
            # Create persistent database
            persist_dir = os.path.join(self.temp_dir, "chromadb_original")
            backup_dir = os.path.join(self.temp_dir, "chromadb_backup")
            
            db = create_face_database(
                'chromadb',
                persist_directory=persist_dir
            )
            
            # Add test data
            for i in range(3):
                db.add_face(
                    name=f"Person_{i}",
                    embedding=np.random.rand(512).tolist(),
                    metadata={"index": i}
                )
            
            original_count = db.get_face_count()
            print(f"✓ Created database with {original_count} faces")
            
            # Backup
            success = db.backup(backup_dir)
            self.assertTrue(success)
            print("✓ Database backed up")
            
            # Clear database
            db.clear_database()
            self.assertEqual(db.get_face_count(), 0)
            print("✓ Database cleared")
            
            # Restore
            success = db.restore(backup_dir)
            self.assertTrue(success)
            restored_count = db.get_face_count()
            self.assertEqual(restored_count, original_count)
            print(f"✓ Database restored: {restored_count} faces")
            
            print("\n✅ Backup/restore tests passed!")
            
        except ImportError:
            self.skipTest("ChromaDB not available")
    
    def test_chromadb_with_deepface(self):
        """Test ChromaDB integration with DeepFace"""
        try:
            from facecv.database.factory import create_face_database
            from facecv.models.deepface.core.embedding import DeepFaceEmbedding
            
            print("\n=== Testing ChromaDB with DeepFace ===")
            
            # Create ChromaDB
            db = create_face_database('chromadb')
            
            # Create DeepFace embedding generator
            embedding_gen = DeepFaceEmbedding(mock_mode=True)
            
            # Generate embeddings for different models
            models = ["VGG-Face", "Facenet", "ArcFace"]
            
            for model in models:
                result = embedding_gen.generate_embedding(
                    "mock_image.jpg",
                    model_name=model
                )
                
                # Store in ChromaDB
                face_id = db.add_face(
                    name=f"Test_{model}",
                    embedding=result["embedding"],
                    metadata={
                        "model": model,
                        "dimension": len(result["embedding"])
                    }
                )
                
                print(f"✓ Stored {model} embedding (dim: {len(result['embedding'])})")
            
            # Verify storage
            all_faces = db.get_all_faces()
            self.assertEqual(len(all_faces), len(models))
            
            print("\n✅ ChromaDB-DeepFace integration tests passed!")
            
        except ImportError as e:
            self.skipTest(f"Integration test skipped: {e}")

def run_tests():
    """Run all ChromaDB integration tests"""
    print("Starting ChromaDB Integration Tests\n")
    
    # Create test suite
    suite = unittest.TestSuite()
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestChromaDBIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Summary
    print(f"\n{'='*60}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    if result.testsRun > 0:
        success_count = result.testsRun - len(result.failures) - len(result.errors) - len(result.skipped)
        print(f"Success rate: {success_count / result.testsRun * 100:.1f}%")
    print(f"{'='*60}")
    
    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)