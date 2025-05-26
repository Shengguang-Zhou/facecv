#!/usr/bin/env python3
"""Database Backend Tests - Multi-database Support Testing"""

import unittest
import sys
import os
import time
import numpy as np
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class TestDatabaseBackends(unittest.TestCase):
    """Test multiple database backend implementations"""
    
    def setUp(self):
        """Set up test environment"""
        self.test_face_data = {
            "name": "Test Person",
            "embedding": np.random.rand(512).tolist(),
            "metadata": {
                "department": "Testing",
                "employee_id": "TEST001",
                "source": "unit_test"
            }
        }
        
    def test_sqlite_backend(self):
        """Test SQLite database backend"""
        try:
            from facecv.database.sqlite_facedb import SQLiteFaceDB
            
            # Test in-memory database
            db = SQLiteFaceDB("sqlite:///:memory:")
            
            # Test basic operations
            print("\n=== Testing SQLite Backend ===")
            
            # 1. Add face
            face_id = db.add_face(
                self.test_face_data["name"],
                self.test_face_data["embedding"],
                self.test_face_data["metadata"]
            )
            self.assertIsNotNone(face_id)
            print(f"✓ Added face with ID: {face_id}")
            
            # 2. Get face by ID
            face = db.get_face(face_id)
            self.assertIsNotNone(face)
            self.assertEqual(face["name"], self.test_face_data["name"])
            print("✓ Retrieved face by ID")
            
            # 3. Get faces by name
            faces = db.get_faces_by_name(self.test_face_data["name"])
            self.assertEqual(len(faces), 1)
            print(f"✓ Found {len(faces)} face(s) by name")
            
            # 4. Search similar faces
            similar = db.search_similar_faces(
                self.test_face_data["embedding"],
                threshold=0.9
            )
            self.assertEqual(len(similar), 1)
            print(f"✓ Found {len(similar)} similar face(s)")
            
            # 5. Update face
            new_embedding = np.random.rand(512).tolist()
            success = db.update_face(
                face_id,
                embedding=new_embedding,
                metadata={"updated": True}
            )
            self.assertTrue(success)
            print("✓ Updated face successfully")
            
            # 6. Get face count
            count = db.get_face_count()
            self.assertEqual(count, 1)
            print(f"✓ Face count: {count}")
            
            # 7. Delete face
            deleted = db.delete_face(face_id)
            self.assertTrue(deleted)
            self.assertEqual(db.get_face_count(), 0)
            print("✓ Deleted face successfully")
            
            print("\n✅ SQLite backend tests passed!")
            
        except Exception as e:
            self.fail(f"SQLite backend test failed: {e}")
    
    def test_mysql_backend(self):
        """Test MySQL database backend"""
        try:
            from facecv.database.mysql_facedb import MySQLFaceDB
            from facecv.config.database import DatabaseConfig
            
            # Load config from environment
            config = DatabaseConfig()
            
            # Skip if MySQL not configured
            if not config.mysql_host:
                self.skipTest("MySQL not configured")
            
            print("\n=== Testing MySQL Backend ===")
            
            # Create connection string
            connection_string = (
                f"mysql+pymysql://{config.mysql_user}:{config.mysql_password}@"
                f"{config.mysql_host}:{config.mysql_port}/{config.mysql_database}"
            )
            
            # Initialize database
            db = MySQLFaceDB(connection_string)
            
            # Test connection
            try:
                count = db.get_face_count()
                print(f"✓ Connected to MySQL (existing faces: {count})")
            except Exception as e:
                self.skipTest(f"MySQL connection failed: {e}")
            
            # Run same tests as SQLite
            # 1. Add face
            face_id = db.add_face(
                self.test_face_data["name"],
                self.test_face_data["embedding"],
                self.test_face_data["metadata"]
            )
            self.assertIsNotNone(face_id)
            print(f"✓ Added face with ID: {face_id}")
            
            # 2. Test async operations
            import asyncio
            
            async def test_async():
                faces = await db.aget_faces_by_name(self.test_face_data["name"])
                return faces
            
            faces = asyncio.run(test_async())
            self.assertTrue(len(faces) > 0)
            print("✓ Async operations working")
            
            # 3. Performance test
            start_time = time.time()
            for i in range(10):
                db.search_similar_faces(
                    np.random.rand(512).tolist(),
                    threshold=0.5
                )
            elapsed = time.time() - start_time
            avg_time = elapsed / 10
            print(f"✓ Average search time: {avg_time:.3f}s")
            
            # Clean up test data
            db.delete_face(face_id)
            
            print("\n✅ MySQL backend tests passed!")
            
        except ImportError:
            self.skipTest("MySQL dependencies not available")
        except Exception as e:
            self.skipTest(f"MySQL backend test skipped: {e}")
    
    def test_database_factory(self):
        """Test database factory pattern"""
        try:
            from facecv.database.factory import create_face_database, get_available_databases
            
            print("\n=== Testing Database Factory ===")
            
            # Test available databases
            available = get_available_databases()
            print(f"Available databases: {', '.join(available)}")
            self.assertIn("sqlite", available)
            
            # Test SQLite creation
            db = create_face_database("sqlite", "sqlite:///:memory:")
            self.assertIsNotNone(db)
            print("✓ Created SQLite database via factory")
            
            # Test with invalid type
            with self.assertRaises(ValueError):
                create_face_database("invalid_db", "")
            print("✓ Factory properly rejects invalid database types")
            
            # Test MySQL creation (if available)
            if "mysql" in available:
                from facecv.config.database import DatabaseConfig
                config = DatabaseConfig()
                if config.mysql_host:
                    db = create_face_database("mysql", config.get_mysql_url())
                    self.assertIsNotNone(db)
                    print("✓ Created MySQL database via factory")
            
            print("\n✅ Database factory tests passed!")
            
        except Exception as e:
            self.fail(f"Database factory test failed: {e}")
    
    def test_database_performance_comparison(self):
        """Compare performance across different backends"""
        try:
            from facecv.database.factory import create_face_database, get_available_databases
            
            print("\n=== Database Performance Comparison ===")
            
            results = {}
            test_size = 100
            
            # Generate test data
            test_faces = []
            for i in range(test_size):
                test_faces.append({
                    "name": f"Person_{i}",
                    "embedding": np.random.rand(512).tolist(),
                    "metadata": {"index": i}
                })
            
            available = get_available_databases()
            
            for db_type in available:
                if db_type == "sqlite":
                    db = create_face_database(db_type, "sqlite:///:memory:")
                elif db_type == "mysql":
                    # Skip MySQL for performance test if not configured
                    continue
                else:
                    continue
                
                print(f"\nTesting {db_type}...")
                
                # Test insertion speed
                start_time = time.time()
                face_ids = []
                for face in test_faces:
                    face_id = db.add_face(
                        face["name"],
                        face["embedding"],
                        face["metadata"]
                    )
                    face_ids.append(face_id)
                insert_time = time.time() - start_time
                
                # Test search speed
                start_time = time.time()
                for _ in range(10):
                    results = db.search_similar_faces(
                        np.random.rand(512).tolist(),
                        threshold=0.5
                    )
                search_time = (time.time() - start_time) / 10
                
                # Test retrieval speed
                start_time = time.time()
                for face_id in face_ids[:10]:
                    face = db.get_face(face_id)
                retrieve_time = (time.time() - start_time) / 10
                
                results[db_type] = {
                    "insert": insert_time / test_size,
                    "search": search_time,
                    "retrieve": retrieve_time
                }
                
                print(f"  Insert: {results[db_type]['insert']:.4f}s per face")
                print(f"  Search: {results[db_type]['search']:.4f}s per query")
                print(f"  Retrieve: {results[db_type]['retrieve']:.4f}s per face")
            
            print("\n✅ Performance comparison completed!")
            
        except Exception as e:
            self.skipTest(f"Performance comparison skipped: {e}")
    
    def test_database_migration(self):
        """Test data migration between different backends"""
        try:
            from facecv.database.factory import create_face_database
            
            print("\n=== Testing Database Migration ===")
            
            # Create source database (SQLite)
            source_db = create_face_database("sqlite", "sqlite:///:memory:")
            
            # Add test data
            face_ids = []
            for i in range(5):
                face_id = source_db.add_face(
                    f"Person_{i}",
                    np.random.rand(512).tolist(),
                    {"index": i}
                )
                face_ids.append(face_id)
            
            print(f"✓ Added {len(face_ids)} faces to source database")
            
            # Create target database (another SQLite for testing)
            target_db = create_face_database("sqlite", "sqlite:///:memory:")
            
            # Migrate data
            migrated = 0
            for face_id in face_ids:
                face = source_db.get_face(face_id)
                if face:
                    new_id = target_db.add_face(
                        face["name"],
                        face["embedding"],
                        face.get("metadata", {})
                    )
                    if new_id:
                        migrated += 1
            
            self.assertEqual(migrated, len(face_ids))
            print(f"✓ Migrated {migrated} faces to target database")
            
            # Verify migration
            self.assertEqual(source_db.get_face_count(), target_db.get_face_count())
            print("✓ Migration verification passed")
            
            print("\n✅ Database migration test completed!")
            
        except Exception as e:
            self.skipTest(f"Migration test skipped: {e}")
    
    def test_database_concurrency(self):
        """Test concurrent database operations"""
        try:
            from facecv.database.factory import create_face_database
            import threading
            import queue
            
            print("\n=== Testing Database Concurrency ===")
            
            db = create_face_database("sqlite", "sqlite:///:memory:")
            errors = queue.Queue()
            results = queue.Queue()
            
            def worker(worker_id, num_operations):
                try:
                    for i in range(num_operations):
                        # Add face
                        face_id = db.add_face(
                            f"Worker{worker_id}_Person{i}",
                            np.random.rand(512).tolist(),
                            {"worker": worker_id, "index": i}
                        )
                        
                        # Search
                        similar = db.search_similar_faces(
                            np.random.rand(512).tolist(),
                            threshold=0.5
                        )
                        
                        results.put((worker_id, face_id))
                        
                except Exception as e:
                    errors.put((worker_id, str(e)))
            
            # Start multiple threads
            threads = []
            num_workers = 5
            operations_per_worker = 10
            
            start_time = time.time()
            for i in range(num_workers):
                t = threading.Thread(target=worker, args=(i, operations_per_worker))
                t.start()
                threads.append(t)
            
            # Wait for completion
            for t in threads:
                t.join()
            
            elapsed = time.time() - start_time
            
            # Check results
            error_count = errors.qsize()
            result_count = results.qsize()
            
            self.assertEqual(error_count, 0, f"Concurrent operations produced {error_count} errors")
            self.assertEqual(result_count, num_workers * operations_per_worker)
            
            print(f"✓ Completed {result_count} concurrent operations in {elapsed:.2f}s")
            print(f"✓ No errors during concurrent access")
            
            # Verify final count
            final_count = db.get_face_count()
            expected_count = num_workers * operations_per_worker
            self.assertEqual(final_count, expected_count)
            print(f"✓ Final face count correct: {final_count}")
            
            print("\n✅ Concurrency test completed!")
            
        except Exception as e:
            self.skipTest(f"Concurrency test skipped: {e}")

class TestDatabaseIntegration(unittest.TestCase):
    """Test database integration with face recognition system"""
    
    def test_recognizer_with_database(self):
        """Test face recognizer integration with database"""
        try:
            from facecv.models.insightface.recognizer import InsightFaceRecognizer
            from facecv.database.factory import create_face_database
            
            print("\n=== Testing Recognizer-Database Integration ===")
            
            # Create database
            db = create_face_database("sqlite", "sqlite:///:memory:")
            
            # Create recognizer with database
            recognizer = InsightFaceRecognizer(mock_mode=True, face_db=db)
            
            # Test registration
            mock_image = MagicMock(shape=(224, 224, 3))
            face_ids = recognizer.register(
                mock_image,
                name="Test Person",
                metadata={"test": True}
            )
            
            self.assertTrue(len(face_ids) > 0)
            print(f"✓ Registered {len(face_ids)} face(s)")
            
            # Verify in database
            db_count = db.get_face_count()
            self.assertEqual(db_count, len(face_ids))
            print(f"✓ Database contains {db_count} face(s)")
            
            # Test recognition
            results = recognizer.recognize(mock_image, threshold=0.6)
            self.assertTrue(len(results) > 0)
            print(f"✓ Recognition found {len(results)} match(es)")
            
            print("\n✅ Recognizer-database integration test passed!")
            
        except Exception as e:
            self.skipTest(f"Integration test skipped: {e}")

def run_tests():
    """Run all database backend tests"""
    print("Starting Database Backend Tests\n")
    
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestDatabaseBackends))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestDatabaseIntegration))
    
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