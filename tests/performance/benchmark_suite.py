#!/usr/bin/env python3
"""Performance Benchmarking Suite for FaceCV"""

import time
import psutil
import numpy as np
import json
import threading
from datetime import datetime
from typing import Dict, List, Any
import concurrent.futures
import os
import sys
import gc

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

class PerformanceBenchmark:
    """Performance benchmarking tools for FaceCV"""
    
    def __init__(self):
        self.results = {}
        self.start_memory = None
        
    def measure_memory(self):
        """Get current memory usage"""
        process = psutil.Process()
        return {
            "rss_mb": process.memory_info().rss / 1024 / 1024,
            "vms_mb": process.memory_info().vms / 1024 / 1024,
            "percent": process.memory_percent(),
            "available_mb": psutil.virtual_memory().available / 1024 / 1024
        }
    
    def measure_cpu(self):
        """Get CPU usage"""
        return {
            "percent": psutil.cpu_percent(interval=0.1),
            "count": psutil.cpu_count(),
            "per_cpu": psutil.cpu_percent(interval=0.1, percpu=True)
        }
    
    def benchmark_face_detection(self, iterations: int = 100):
        """Benchmark face detection performance"""
        print("\n=== Benchmarking Face Detection ===")
        
        from facecv.models.insightface.recognizer import InsightFaceRecognizer
        recognizer = InsightFaceRecognizer(mock_mode=True)
        
        # Generate test images of different sizes
        test_sizes = [(640, 480), (1280, 720), (1920, 1080)]
        results = {}
        
        for size in test_sizes:
            print(f"\nTesting with image size: {size}")
            
            # Create mock image
            mock_image = np.random.randint(0, 255, (*size, 3), dtype=np.uint8)
            
            # Warm up
            for _ in range(5):
                recognizer.detect_faces(mock_image)
            
            # Measure
            times = []
            memory_before = self.measure_memory()
            
            for i in range(iterations):
                start = time.time()
                faces = recognizer.detect_faces(mock_image)
                elapsed = time.time() - start
                times.append(elapsed)
                
                if i % 20 == 0:
                    print(f"  Progress: {i}/{iterations}")
            
            memory_after = self.measure_memory()
            
            results[f"{size[0]}x{size[1]}"] = {
                "avg_time_ms": np.mean(times) * 1000,
                "min_time_ms": np.min(times) * 1000,
                "max_time_ms": np.max(times) * 1000,
                "std_time_ms": np.std(times) * 1000,
                "fps": 1.0 / np.mean(times),
                "memory_delta_mb": memory_after["rss_mb"] - memory_before["rss_mb"]
            }
            
            print(f"  Average: {results[f'{size[0]}x{size[1]}']['avg_time_ms']:.2f}ms")
            print(f"  FPS: {results[f'{size[0]}x{size[1]}']['fps']:.2f}")
        
        self.results["face_detection"] = results
        return results
    
    def benchmark_face_recognition(self, num_faces: List[int] = [100, 500, 1000]):
        """Benchmark face recognition with different database sizes"""
        print("\n=== Benchmarking Face Recognition ===")
        
        from facecv.database.factory import create_face_database
        from facecv.models.insightface.recognizer import InsightFaceRecognizer
        
        results = {}
        
        for db_size in num_faces:
            print(f"\nTesting with {db_size} faces in database")
            
            # Create database
            db = create_face_database("sqlite", "sqlite:///:memory:")
            recognizer = InsightFaceRecognizer(mock_mode=True, face_db=db)
            
            # Populate database
            print(f"  Populating database...")
            start_populate = time.time()
            for i in range(db_size):
                embedding = np.random.rand(512).tolist()
                db.add_face(f"Person_{i}", embedding, {"index": i})
            populate_time = time.time() - start_populate
            
            # Test recognition speed
            test_embedding = np.random.rand(512).tolist()
            
            # Warm up
            for _ in range(5):
                db.search_similar_faces(test_embedding, threshold=0.6)
            
            # Measure
            times = []
            for i in range(50):
                start = time.time()
                results_found = db.search_similar_faces(test_embedding, threshold=0.6)
                elapsed = time.time() - start
                times.append(elapsed)
            
            results[f"{db_size}_faces"] = {
                "populate_time_s": populate_time,
                "avg_search_ms": np.mean(times) * 1000,
                "min_search_ms": np.min(times) * 1000,
                "max_search_ms": np.max(times) * 1000,
                "searches_per_second": 1.0 / np.mean(times)
            }
            
            print(f"  Populate time: {populate_time:.2f}s")
            print(f"  Average search: {results[f'{db_size}_faces']['avg_search_ms']:.2f}ms")
            
            # Clean up
            del db
            del recognizer
            gc.collect()
        
        self.results["face_recognition"] = results
        return results
    
    def benchmark_database_operations(self):
        """Benchmark different database backends"""
        print("\n=== Benchmarking Database Operations ===")
        
        from facecv.database.factory import create_face_database, get_available_databases
        
        available_dbs = get_available_databases()
        results = {}
        
        operations = 1000
        test_embedding = np.random.rand(512).tolist()
        
        for db_type in available_dbs:
            if not available_dbs[db_type]:
                continue
                
            print(f"\nTesting {db_type} database")
            
            try:
                if db_type == "sqlite":
                    db = create_face_database(db_type, "sqlite:///:memory:")
                elif db_type == "chromadb":
                    db = create_face_database(db_type)
                else:
                    continue
                
                times = {
                    "insert": [],
                    "search": [],
                    "update": [],
                    "delete": []
                }
                
                face_ids = []
                
                # Insert operations
                for i in range(operations):
                    start = time.time()
                    face_id = db.add_face(f"Person_{i}", test_embedding, {"index": i})
                    times["insert"].append(time.time() - start)
                    face_ids.append(face_id)
                
                # Search operations
                for _ in range(100):
                    start = time.time()
                    db.search_similar_faces(test_embedding, threshold=0.6)
                    times["search"].append(time.time() - start)
                
                # Update operations
                for i in range(min(100, len(face_ids))):
                    start = time.time()
                    db.update_face(face_ids[i], embedding=test_embedding)
                    times["update"].append(time.time() - start)
                
                # Delete operations
                for i in range(min(100, len(face_ids))):
                    start = time.time()
                    db.delete_face(face_ids[i])
                    times["delete"].append(time.time() - start)
                
                results[db_type] = {
                    "insert_avg_ms": np.mean(times["insert"]) * 1000,
                    "search_avg_ms": np.mean(times["search"]) * 1000,
                    "update_avg_ms": np.mean(times["update"]) * 1000,
                    "delete_avg_ms": np.mean(times["delete"]) * 1000,
                    "total_operations": operations
                }
                
                print(f"  Insert: {results[db_type]['insert_avg_ms']:.3f}ms")
                print(f"  Search: {results[db_type]['search_avg_ms']:.3f}ms")
                print(f"  Update: {results[db_type]['update_avg_ms']:.3f}ms")
                print(f"  Delete: {results[db_type]['delete_avg_ms']:.3f}ms")
                
            except Exception as e:
                print(f"  Error testing {db_type}: {e}")
                results[db_type] = {"error": str(e)}
        
        self.results["database_operations"] = results
        return results
    
    def benchmark_concurrent_requests(self, num_threads: List[int] = [1, 5, 10, 20]):
        """Benchmark concurrent request handling"""
        print("\n=== Benchmarking Concurrent Requests ===")
        
        from facecv.models.insightface.recognizer import InsightFaceRecognizer
        
        recognizer = InsightFaceRecognizer(mock_mode=True)
        mock_image = np.random.randint(0, 255, (640, 480, 3), dtype=np.uint8)
        
        results = {}
        
        def process_request():
            """Simulate API request processing"""
            start = time.time()
            recognizer.recognize(mock_image, threshold=0.6)
            return time.time() - start
        
        for threads in num_threads:
            print(f"\nTesting with {threads} concurrent threads")
            
            total_requests = threads * 50
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
                start_time = time.time()
                futures = [executor.submit(process_request) for _ in range(total_requests)]
                
                response_times = []
                for future in concurrent.futures.as_completed(futures):
                    response_times.append(future.result())
                
                total_time = time.time() - start_time
            
            results[f"{threads}_threads"] = {
                "total_requests": total_requests,
                "total_time_s": total_time,
                "requests_per_second": total_requests / total_time,
                "avg_response_ms": np.mean(response_times) * 1000,
                "min_response_ms": np.min(response_times) * 1000,
                "max_response_ms": np.max(response_times) * 1000,
                "p95_response_ms": np.percentile(response_times, 95) * 1000,
                "p99_response_ms": np.percentile(response_times, 99) * 1000
            }
            
            print(f"  RPS: {results[f'{threads}_threads']['requests_per_second']:.2f}")
            print(f"  Avg response: {results[f'{threads}_threads']['avg_response_ms']:.2f}ms")
            print(f"  P95 response: {results[f'{threads}_threads']['p95_response_ms']:.2f}ms")
        
        self.results["concurrent_requests"] = results
        return results
    
    def benchmark_memory_usage(self):
        """Benchmark memory usage patterns"""
        print("\n=== Benchmarking Memory Usage ===")
        
        from facecv.models.insightface.recognizer import InsightFaceRecognizer
        from facecv.database.factory import create_face_database
        
        results = {}
        
        # Initial memory
        gc.collect()
        initial_memory = self.measure_memory()
        results["initial"] = initial_memory
        
        # After loading recognizer
        recognizer = InsightFaceRecognizer(mock_mode=True)
        after_recognizer = self.measure_memory()
        results["after_recognizer"] = after_recognizer
        results["recognizer_delta_mb"] = after_recognizer["rss_mb"] - initial_memory["rss_mb"]
        
        # After creating database
        db = create_face_database("sqlite", "sqlite:///:memory:")
        recognizer.face_db = db
        after_db = self.measure_memory()
        results["after_database"] = after_db
        results["database_delta_mb"] = after_db["rss_mb"] - after_recognizer["rss_mb"]
        
        # After processing images
        memory_timeline = []
        for i in range(100):
            mock_image = np.random.randint(0, 255, (640, 480, 3), dtype=np.uint8)
            recognizer.detect_faces(mock_image)
            
            if i % 10 == 0:
                memory_timeline.append({
                    "iteration": i,
                    "memory": self.measure_memory()
                })
        
        results["processing_timeline"] = memory_timeline
        results["final_memory"] = self.measure_memory()
        results["total_delta_mb"] = results["final_memory"]["rss_mb"] - initial_memory["rss_mb"]
        
        print(f"  Initial memory: {initial_memory['rss_mb']:.2f}MB")
        print(f"  After recognizer: +{results['recognizer_delta_mb']:.2f}MB")
        print(f"  After database: +{results['database_delta_mb']:.2f}MB")
        print(f"  Total delta: +{results['total_delta_mb']:.2f}MB")
        
        self.results["memory_usage"] = results
        return results
    
    def generate_report(self):
        """Generate comprehensive performance report"""
        report = {
            "timestamp": datetime.now().isoformat(),
            "system_info": {
                "cpu_count": psutil.cpu_count(),
                "cpu_freq": psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None,
                "memory_total_gb": psutil.virtual_memory().total / 1024 / 1024 / 1024,
                "python_version": sys.version
            },
            "benchmarks": self.results
        }
        
        # Save report
        report_path = f"performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n=== Performance Report Saved ===")
        print(f"Report saved to: {report_path}")
        
        # Print summary
        print("\n=== Performance Summary ===")
        
        if "face_detection" in self.results:
            print("\nFace Detection:")
            for size, metrics in self.results["face_detection"].items():
                print(f"  {size}: {metrics['fps']:.2f} FPS ({metrics['avg_time_ms']:.2f}ms)")
        
        if "face_recognition" in self.results:
            print("\nFace Recognition:")
            for db_size, metrics in self.results["face_recognition"].items():
                print(f"  {db_size}: {metrics['searches_per_second']:.2f} searches/sec")
        
        if "concurrent_requests" in self.results:
            print("\nConcurrent Requests:")
            for threads, metrics in self.results["concurrent_requests"].items():
                print(f"  {threads}: {metrics['requests_per_second']:.2f} RPS")
        
        return report

def main():
    """Run complete benchmark suite"""
    print("Starting FaceCV Performance Benchmark Suite")
    print("=" * 60)
    
    benchmark = PerformanceBenchmark()
    
    try:
        # Run benchmarks
        benchmark.benchmark_face_detection(iterations=50)
        benchmark.benchmark_face_recognition(num_faces=[100, 500])
        benchmark.benchmark_database_operations()
        benchmark.benchmark_concurrent_requests(num_threads=[1, 5, 10])
        benchmark.benchmark_memory_usage()
        
        # Generate report
        report = benchmark.generate_report()
        
        print("\n✅ Benchmark completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()