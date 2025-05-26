#!/usr/bin/env python3
"""Memory Profiling Tools for FaceCV"""

import os
import sys
import gc
import tracemalloc
import psutil
from datetime import datetime
import json
from typing import Dict, List, Any

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

class MemoryProfiler:
    """Memory profiling and optimization tools"""
    
    def __init__(self):
        self.snapshots = []
        self.process = psutil.Process()
        
    def start_profiling(self):
        """Start memory profiling"""
        gc.collect()
        tracemalloc.start()
        self.initial_memory = self._get_memory_info()
        print(f"Memory profiling started. Initial RSS: {self.initial_memory['rss_mb']:.2f}MB")
    
    def take_snapshot(self, label: str):
        """Take a memory snapshot"""
        snapshot = tracemalloc.take_snapshot()
        memory_info = self._get_memory_info()
        
        self.snapshots.append({
            "label": label,
            "timestamp": datetime.now().isoformat(),
            "snapshot": snapshot,
            "memory_info": memory_info
        })
        
        print(f"Snapshot '{label}': RSS={memory_info['rss_mb']:.2f}MB, VMS={memory_info['vms_mb']:.2f}MB")
        return memory_info
    
    def _get_memory_info(self) -> Dict[str, float]:
        """Get current memory information"""
        mem_info = self.process.memory_info()
        return {
            "rss_mb": mem_info.rss / 1024 / 1024,
            "vms_mb": mem_info.vms / 1024 / 1024,
            "percent": self.process.memory_percent(),
            "available_system_mb": psutil.virtual_memory().available / 1024 / 1024
        }
    
    def compare_snapshots(self, idx1: int, idx2: int, limit: int = 10):
        """Compare two snapshots and show differences"""
        if idx1 >= len(self.snapshots) or idx2 >= len(self.snapshots):
            print("Invalid snapshot indices")
            return
        
        snap1 = self.snapshots[idx1]
        snap2 = self.snapshots[idx2]
        
        print(f"\nComparing '{snap1['label']}' -> '{snap2['label']}'")
        print(f"Memory delta: {snap2['memory_info']['rss_mb'] - snap1['memory_info']['rss_mb']:.2f}MB")
        
        # Compare tracemalloc snapshots
        top_stats = snap2["snapshot"].compare_to(snap1["snapshot"], 'lineno')
        
        print(f"\nTop {limit} memory allocations:")
        for stat in top_stats[:limit]:
            print(f"  {stat}")
    
    def profile_face_recognition_pipeline(self):
        """Profile memory usage in face recognition pipeline"""
        print("\n=== Profiling Face Recognition Pipeline ===")
        
        self.start_profiling()
        self.take_snapshot("start")
        
        # Import modules
        from facecv.models.insightface.recognizer import InsightFaceRecognizer
        from facecv.database.factory import create_face_database
        import numpy as np
        
        self.take_snapshot("after_imports")
        
        # Create recognizer
        recognizer = InsightFaceRecognizer(mock_mode=True)
        self.take_snapshot("after_recognizer_init")
        
        # Create database
        db = create_face_database("sqlite", "sqlite:///:memory:")
        recognizer.face_db = db
        self.take_snapshot("after_database_init")
        
        # Process some images
        for i in range(10):
            image = np.random.randint(0, 255, (640, 480, 3), dtype=np.uint8)
            faces = recognizer.detect_faces(image)
            
            # Register face
            if i < 5:
                embedding = np.random.rand(512).tolist()
                db.add_face(f"Person_{i}", embedding)
        
        self.take_snapshot("after_processing_10_images")
        
        # Process more images
        for i in range(90):
            image = np.random.randint(0, 255, (640, 480, 3), dtype=np.uint8)
            results = recognizer.recognize(image)
        
        self.take_snapshot("after_processing_100_images")
        
        # Cleanup
        del recognizer
        del db
        gc.collect()
        
        self.take_snapshot("after_cleanup")
        
        # Generate report
        self.generate_memory_report()
    
    def find_memory_leaks(self):
        """Look for potential memory leaks"""
        print("\n=== Checking for Memory Leaks ===")
        
        from facecv.models.insightface.recognizer import InsightFaceRecognizer
        import numpy as np
        
        # Monitor memory over repeated operations
        memory_timeline = []
        recognizer = InsightFaceRecognizer(mock_mode=True)
        
        for i in range(100):
            # Simulate processing
            image = np.random.randint(0, 255, (640, 480, 3), dtype=np.uint8)
            recognizer.detect_faces(image)
            
            if i % 10 == 0:
                gc.collect()
                mem_info = self._get_memory_info()
                memory_timeline.append({
                    "iteration": i,
                    "rss_mb": mem_info["rss_mb"]
                })
                print(f"Iteration {i}: {mem_info['rss_mb']:.2f}MB")
        
        # Analyze for leaks
        memory_values = [m["rss_mb"] for m in memory_timeline]
        memory_growth = memory_values[-1] - memory_values[0]
        avg_growth_per_iteration = memory_growth / 100
        
        print(f"\nMemory growth: {memory_growth:.2f}MB over 100 iterations")
        print(f"Average growth per iteration: {avg_growth_per_iteration:.4f}MB")
        
        if avg_growth_per_iteration > 0.1:
            print("⚠️  Potential memory leak detected!")
        else:
            print("✅ No significant memory leak detected")
        
        return memory_timeline
    
    def optimize_memory_usage(self):
        """Provide memory optimization recommendations"""
        print("\n=== Memory Optimization Recommendations ===")
        
        recommendations = []
        
        # Check current memory usage
        current_memory = self._get_memory_info()
        
        if current_memory["percent"] > 50:
            recommendations.append({
                "issue": "High memory usage",
                "current": f"{current_memory['percent']:.1f}%",
                "recommendation": "Consider reducing batch sizes or using memory-efficient models"
            })
        
        # Check for large objects
        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics('lineno')
        
        large_allocations = []
        for stat in top_stats[:20]:
            size_mb = stat.size / 1024 / 1024
            if size_mb > 10:
                large_allocations.append({
                    "file": stat.traceback.format()[0],
                    "size_mb": size_mb
                })
        
        if large_allocations:
            recommendations.append({
                "issue": "Large memory allocations detected",
                "allocations": large_allocations,
                "recommendation": "Review these allocations for optimization opportunities"
            })
        
        # Model-specific recommendations
        recommendations.extend([
            {
                "category": "Face Detection",
                "tips": [
                    "Use smaller input image sizes when possible",
                    "Process images in batches to amortize overhead",
                    "Clear detection cache periodically"
                ]
            },
            {
                "category": "Database",
                "tips": [
                    "Use pagination for large result sets",
                    "Implement connection pooling",
                    "Consider using memory-mapped files for large embeddings"
                ]
            },
            {
                "category": "Video Processing",
                "tips": [
                    "Limit frame buffer sizes",
                    "Skip frames to reduce memory usage",
                    "Release video captures properly"
                ]
            }
        ])
        
        return recommendations
    
    def generate_memory_report(self):
        """Generate comprehensive memory report"""
        report = {
            "timestamp": datetime.now().isoformat(),
            "system_memory": {
                "total_gb": psutil.virtual_memory().total / 1024 / 1024 / 1024,
                "available_gb": psutil.virtual_memory().available / 1024 / 1024 / 1024,
                "percent_used": psutil.virtual_memory().percent
            },
            "process_memory": self._get_memory_info(),
            "snapshots": []
        }
        
        # Add snapshot summaries
        for i, snap in enumerate(self.snapshots):
            report["snapshots"].append({
                "index": i,
                "label": snap["label"],
                "timestamp": snap["timestamp"],
                "memory_info": snap["memory_info"]
            })
        
        # Calculate deltas
        if len(self.snapshots) > 1:
            deltas = []
            for i in range(1, len(self.snapshots)):
                delta = {
                    "from": self.snapshots[i-1]["label"],
                    "to": self.snapshots[i]["label"],
                    "delta_mb": (self.snapshots[i]["memory_info"]["rss_mb"] - 
                               self.snapshots[i-1]["memory_info"]["rss_mb"])
                }
                deltas.append(delta)
            report["memory_deltas"] = deltas
        
        # Add optimization recommendations
        report["optimizations"] = self.optimize_memory_usage()
        
        # Save report
        report_path = f"memory_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\nMemory report saved to: {report_path}")
        
        # Print summary
        print("\n=== Memory Usage Summary ===")
        for snap in report["snapshots"]:
            print(f"{snap['label']:30} RSS: {snap['memory_info']['rss_mb']:8.2f}MB")
        
        return report

def main():
    """Run memory profiling"""
    profiler = MemoryProfiler()
    
    # Profile the face recognition pipeline
    profiler.profile_face_recognition_pipeline()
    
    # Check for memory leaks
    profiler.find_memory_leaks()
    
    # Show snapshot comparisons
    if len(profiler.snapshots) >= 2:
        profiler.compare_snapshots(0, len(profiler.snapshots)-1)
    
    tracemalloc.stop()

if __name__ == "__main__":
    main()