#!/usr/bin/env python3
"""
Test InsightFace API with different models and configurations
Verify MySQL and face recognition functionality
"""

import os
import sys
import json
import time
import requests
from typing import Dict, List, Any
import mysql.connector
from datetime import datetime
from tabulate import tabulate

# Configuration
BASE_URL = "http://localhost:7003/api/v1/insightface"
DB_CONFIG = {
    "host": "eurekailab.mysql.rds.aliyuncs.com",
    "port": 3306,
    "user": "root", 
    "password": "Zsg20010115_",
    "database": "facecv"
}

# Test image path
TEST_IMAGE = "/home/a/PycharmProjects/facecv/test_images/test_face.jpg"

# InsightFace models to test
MODELS = [
    "buffalo_l",     # Large model (best accuracy)
    "buffalo_m",     # Medium model (balanced)
    "buffalo_s",     # Small model (fastest)
    "buffalo_sc",    # Small model for mobile
]

# Detection configurations
DETECTORS = [
    {"det_size": (640, 640)},   # Standard detection
    {"det_size": (320, 320)},   # Fast detection
    {"det_size": (160, 160)},   # Ultra-fast detection
]


def test_mysql_connection():
    """Test MySQL database connection"""
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM faces")
        count = cursor.fetchone()[0]
        print(f"✓ MySQL connected: {count} faces in database")
        cursor.close()
        conn.close()
        return True
    except Exception as e:
        print(f"✗ MySQL connection failed: {e}")
        return False


def test_registration(model: str, det_size: tuple) -> Dict[str, Any]:
    """Test face registration with specific model"""
    result = {
        "model": model,
        "det_size": det_size,
        "registration": False,
        "recognition": False,
        "mysql_sync": False,
        "error": None,
        "time": 0
    }
    
    start_time = time.time()
    
    try:
        # Configure the model
        config_data = {
            "model_name": model,
            "det_size": list(det_size),
            "gpu_id": -1  # CPU mode
        }
        
        config_resp = requests.post(
            f"{BASE_URL}/configure",
            json=config_data
        )
        
        if config_resp.status_code != 200:
            print(f"Configuration failed for {model}")
        
        # Register face
        test_name = f"test_{model}_{det_size[0]}"
        
        with open(TEST_IMAGE, "rb") as f:
            files = {"file": ("test.jpg", f, "image/jpeg")}
            data = {
                "name": test_name,
                "metadata": json.dumps({"model": model, "det_size": str(det_size)})
            }
            
            reg_resp = requests.post(
                f"{BASE_URL}/faces/",
                files=files,
                data=data
            )
        
        if reg_resp.status_code == 200:
            result["registration"] = True
            reg_data = reg_resp.json()
            face_id = reg_data.get("face_id")
            
            # Test recognition
            with open(TEST_IMAGE, "rb") as f:
                files = {"file": ("test.jpg", f, "image/jpeg")}
                rec_resp = requests.post(
                    f"{BASE_URL}/recognize",
                    files=files
                )
                
            if rec_resp.status_code == 200:
                rec_data = rec_resp.json()
                if rec_data.get("faces"):
                    for face in rec_data["faces"]:
                        if face.get("name") == test_name:
                            result["recognition"] = True
                            break
            
            # Check MySQL sync
            conn = mysql.connector.connect(**DB_CONFIG)
            cursor = conn.cursor()
            cursor.execute(
                "SELECT COUNT(*) FROM faces WHERE person_name = %s",
                (test_name,)
            )
            count = cursor.fetchone()[0]
            result["mysql_sync"] = count > 0
            
            # Cleanup
            cursor.execute(
                "DELETE FROM faces WHERE person_name = %s",
                (test_name,)
            )
            conn.commit()
            cursor.close()
            conn.close()
            
        else:
            result["error"] = reg_resp.text
            
    except Exception as e:
        result["error"] = str(e)
    
    result["time"] = time.time() - start_time
    return result


def run_matrix_test():
    """Run comprehensive matrix test"""
    print("🚀 InsightFace Model x Configuration Matrix Test")
    print("This will test different models and detection sizes")
    print("Each test includes: Registration, Recognition, MySQL Sync\n")
    
    # Check MySQL connection first
    if not test_mysql_connection():
        print("❌ MySQL connection required for tests")
        return
    
    # Check if test image exists
    if not os.path.exists(TEST_IMAGE):
        print(f"❌ Test image not found: {TEST_IMAGE}")
        return
    
    results = []
    total_tests = len(MODELS) * len(DETECTORS)
    
    print(f"🔬 Testing {total_tests} model-configuration combinations")
    print(f"Models: {', '.join(MODELS)}")
    print(f"Detection sizes: {', '.join([str(d['det_size']) for d in DETECTORS])}\n")
    
    test_num = 0
    
    for model in MODELS:
        for detector in DETECTORS:
            test_num += 1
            det_size = detector["det_size"]
            
            print(f"\n[{test_num}/{total_tests}] Testing {model} + {det_size}")
            print("=" * 60)
            
            result = test_registration(model, det_size)
            results.append(result)
            
            # Print result
            if result["registration"]:
                print(f"✅ Registration: Success")
            else:
                print(f"❌ Registration: Failed - {result['error']}")
                
            if result["recognition"]:
                print(f"✅ Recognition: Success")
            else:
                print(f"❌ Recognition: Failed")
                
            if result["mysql_sync"]:
                print(f"✅ MySQL Sync: Success")
            else:
                print(f"❌ MySQL Sync: Failed")
                
            print(f"⏱️  Time: {result['time']:.2f}s")
    
    # Summary report
    print("\n" + "=" * 80)
    print("INSIGHTFACE MATRIX TEST REPORT")
    print("=" * 80)
    
    # Calculate statistics
    total = len(results)
    reg_success = sum(1 for r in results if r["registration"])
    rec_success = sum(1 for r in results if r["recognition"])
    mysql_success = sum(1 for r in results if r["mysql_sync"])
    
    print(f"\nSummary:")
    print(f"  Total Combinations Tested: {total}")
    print(f"  Registration Success: {reg_success}/{total} ({reg_success/total*100:.1f}%)")
    print(f"  Recognition Success: {rec_success}/{total} ({rec_success/total*100:.1f}%)")
    print(f"  MySQL Sync Success: {mysql_success}/{total} ({mysql_success/total*100:.1f}%)")
    
    # Detailed table
    print(f"\n\nDetailed Results:")
    
    table_data = []
    for r in results:
        table_data.append([
            r["model"],
            str(r["det_size"]),
            "✅" if r["registration"] else "❌",
            "✅" if r["recognition"] else "❌",
            "✅" if r["mysql_sync"] else "❌",
            f"{r['time']:.2f}s",
            r["error"][:50] + "..." if r["error"] and len(r["error"]) > 50 else r["error"] or ""
        ])
    
    headers = ["Model", "Det Size", "Register", "Recognize", "MySQL", "Time", "Error"]
    print(tabulate(table_data, headers=headers, tablefmt="grid"))
    
    # Save results
    results_file = f"insightface_matrix_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_tests": total,
                "registration_success": reg_success,
                "recognition_success": rec_success,
                "mysql_sync_success": mysql_success
            },
            "results": results
        }, f, indent=2)
    
    print(f"\n💾 Detailed results saved to: {results_file}")
    
    # Recommendations
    print("\n📊 Recommendations:")
    
    best_accuracy = None
    best_speed = None
    best_balanced = None
    
    for r in results:
        if r["registration"] and r["recognition"] and r["mysql_sync"]:
            if r["model"] == "buffalo_l":
                best_accuracy = r
            elif r["model"] == "buffalo_s" and r["det_size"] == (160, 160):
                best_speed = r
            elif r["model"] == "buffalo_m" and r["det_size"] == (320, 320):
                best_balanced = r
    
    if best_accuracy:
        print(f"  🎯 Best Accuracy: {best_accuracy['model']} with {best_accuracy['det_size']} detection")
    if best_speed:
        print(f"  ⚡ Best Speed: {best_speed['model']} with {best_speed['det_size']} detection ({best_speed['time']:.2f}s)")
    if best_balanced:
        print(f"  ⚖️  Best Balanced: {best_balanced['model']} with {best_balanced['det_size']} detection")
    
    print("\n✅ Matrix test complete!")


if __name__ == "__main__":
    # Make sure server is running
    try:
        resp = requests.get("http://localhost:7003/api/v1/health")
        if resp.status_code != 200:
            print("❌ Server is not running. Start with: python main.py")
            sys.exit(1)
    except:
        print("❌ Cannot connect to server at http://localhost:7003")
        print("Start the server with: python main.py")
        sys.exit(1)
    
    run_matrix_test()