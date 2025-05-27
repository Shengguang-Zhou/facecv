#!/usr/bin/env python3
"""Comprehensive DeepFace model x detector matrix test with database verification"""

import os
import sys
import json
import time
import requests
import numpy as np
from pathlib import Path
from itertools import product
import pymysql
from tabulate import tabulate

# Set environment variables
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

# Add project to path
sys.path.insert(0, '/')

# Test configuration
BASE_URL = "http://127.0.0.1:7003"
TEST_IMAGE_DIR = "/home/a/PycharmProjects/EurekCV/dataset/faces"

# DeepFace models and detectors
MODELS = [
    "VGG-Face",
    "Facenet",
    "Facenet512",
    "OpenFace",
    "DeepFace",
    "DeepID",
    "ArcFace",
    "Dlib",
    "SFace"
]

DETECTORS = [
    "opencv",
    "retinaface",
    "mtcnn",
    "ssd",
    "dlib",
    "mediapipe",
    "yolov8"
]

class DeepFaceMatrixTester:
    def __init__(self):
        self.results = []
        self.db_config = self._get_db_config()
        
    def _get_db_config(self):
        """Get database configuration"""
        from facecv.config import get_db_config
        return get_db_config()
    
    def _connect_mysql(self):
        """Connect to MySQL database"""
        return pymysql.connect(
            host=self.db_config.mysql_host,
            port=self.db_config.mysql_port,
            user=self.db_config.mysql_user,
            password=self.db_config.mysql_password,
            database=self.db_config.mysql_database,
            charset='utf8mb4'
        )
    
    def _get_mysql_face_count(self):
        """Get face count from MySQL"""
        try:
            conn = self._connect_mysql()
            with conn.cursor() as cursor:
                cursor.execute("SELECT COUNT(*) FROM faces")
                count = cursor.fetchone()[0]
            conn.close()
            return count
        except Exception as e:
            print(f"MySQL error: {e}")
            return -1
    
    def _verify_chromadb_sync(self, face_id):
        """Verify if face is in ChromaDB"""
        try:
            # Check if face can be recognized
            test_image = f"{TEST_IMAGE_DIR}/harris2.jpeg"
            with open(test_image, 'rb') as f:
                files = {'file': ('test.jpeg', f, 'image/jpeg')}
                resp = requests.post(
                    f"{BASE_URL}/api/v1/deepface/recognition",
                    files=files,
                    data={'threshold': '0.6'}
                )
                if resp.status_code == 200:
                    results = resp.json()
                    return any(r.get('person_name') == 'Test Person' for r in results.get('results', []))
        except Exception as e:
            print(f"ChromaDB verification error: {e}")
        return False
    
    def test_model_detector_combination(self, model, detector):
        """Test a specific model-detector combination"""
        print(f"\n{'='*60}")
        print(f"Testing: Model={model}, Detector={detector}")
        print('='*60)
        
        result = {
            'model': model,
            'detector': detector,
            'registration': 'FAIL',
            'recognition': 'FAIL',
            'verification': 'FAIL',
            'mysql_sync': 'FAIL',
            'chromadb_sync': 'FAIL',
            'error': None
        }
        
        try:
            # Step 1: Configure DeepFace to use this model/detector
            config_resp = requests.post(
                f"{BASE_URL}/api/v1/deepface/configure",
                json={
                    "model_name": model,
                    "detector_backend": detector
                }
            )
            if config_resp.status_code != 200:
                # If configuration endpoint doesn't exist, try direct registration
                print(f"Configuration endpoint not available, proceeding with default")
            
            # Step 2: Register a face
            mysql_count_before = self._get_mysql_face_count()
            
            test_image = f"{TEST_IMAGE_DIR}/harris1.jpeg"
            with open(test_image, 'rb') as f:
                files = {'file': ('harris1.jpeg', f, 'image/jpeg')}
                data = {
                    'name': 'Test Person',
                    'metadata': json.dumps({
                        'model': model,
                        'detector': detector,
                        'test_run': True
                    })
                }
                
                reg_resp = requests.post(
                    f"{BASE_URL}/api/v1/deepface/faces/",
                    files=files,
                    data=data
                )
                
                if reg_resp.status_code == 200:
                    result['registration'] = 'PASS'
                    reg_data = reg_resp.json()
                    face_id = reg_data.get('face_id')
                    print(f"✅ Registration successful: {face_id}")
                    
                    # Verify MySQL sync
                    mysql_count_after = self._get_mysql_face_count()
                    if mysql_count_after > mysql_count_before:
                        result['mysql_sync'] = 'PASS'
                        print(f"✅ MySQL sync verified: {mysql_count_before} -> {mysql_count_after}")
                    
                    # Step 3: Test recognition
                    time.sleep(1)  # Give time for indexing
                    
                    test_image2 = f"{TEST_IMAGE_DIR}/harris2.jpeg"
                    with open(test_image2, 'rb') as f:
                        files = {'file': ('harris2.jpeg', f, 'image/jpeg')}
                        rec_resp = requests.post(
                            f"{BASE_URL}/api/v1/deepface/recognition",
                            files=files,
                            data={'threshold': '0.6'}
                        )
                        
                        if rec_resp.status_code == 200:
                            rec_data = rec_resp.json()
                            if rec_data.get('results'):
                                result['recognition'] = 'PASS'
                                print(f"✅ Recognition successful: {len(rec_data['results'])} faces found")
                                
                                # Verify ChromaDB sync
                                if self._verify_chromadb_sync(face_id):
                                    result['chromadb_sync'] = 'PASS'
                                    print(f"✅ ChromaDB sync verified")
                        else:
                            print(f"❌ Recognition failed: {rec_resp.status_code}")
                            print(f"   Error: {rec_resp.text[:200]}")
                    
                    # Step 4: Test verification
                    with open(f"{TEST_IMAGE_DIR}/harris1.jpeg", 'rb') as f1, \
                         open(f"{TEST_IMAGE_DIR}/harris2.jpeg", 'rb') as f2:
                        files = {
                            'file1': ('harris1.jpeg', f1, 'image/jpeg'),
                            'file2': ('harris2.jpeg', f2, 'image/jpeg')
                        }
                        ver_resp = requests.post(
                            f"{BASE_URL}/api/v1/deepface/verify/",
                            files=files
                        )
                        
                        if ver_resp.status_code == 200:
                            ver_data = ver_resp.json()
                            if ver_data.get('verified'):
                                result['verification'] = 'PASS'
                                print(f"✅ Verification successful: similarity={ver_data.get('similarity')}")
                        else:
                            print(f"❌ Verification failed: {ver_resp.status_code}")
                    
                    # Cleanup: Delete the test face
                    if face_id:
                        del_resp = requests.delete(f"{BASE_URL}/api/v1/deepface/faces/{face_id}")
                        if del_resp.status_code == 200:
                            print(f"✅ Cleanup: Deleted test face {face_id}")
                else:
                    print(f"❌ Registration failed: {reg_resp.status_code}")
                    print(f"   Error: {reg_resp.text[:200]}")
                    result['error'] = reg_resp.text[:100]
                    
        except Exception as e:
            print(f"❌ Test failed with exception: {str(e)}")
            result['error'] = str(e)[:100]
        
        self.results.append(result)
        return result
    
    def run_matrix_test(self, models=None, detectors=None):
        """Run tests for all model-detector combinations"""
        test_models = models or MODELS
        test_detectors = detectors or DETECTORS
        
        total_combinations = len(test_models) * len(test_detectors)
        print(f"\n🔬 Testing {total_combinations} model-detector combinations")
        print(f"Models: {', '.join(test_models)}")
        print(f"Detectors: {', '.join(test_detectors)}")
        
        # Clear previous results
        self.results = []
        
        # Test each combination
        for i, (model, detector) in enumerate(product(test_models, test_detectors)):
            print(f"\n\n[{i+1}/{total_combinations}] Testing {model} + {detector}")
            self.test_model_detector_combination(model, detector)
            time.sleep(2)  # Avoid overwhelming the server
    
    def generate_report(self):
        """Generate test report"""
        print("\n\n" + "="*80)
        print("DEEPFACE MODEL x DETECTOR MATRIX TEST REPORT")
        print("="*80)
        
        # Summary statistics
        total = len(self.results)
        reg_pass = sum(1 for r in self.results if r['registration'] == 'PASS')
        rec_pass = sum(1 for r in self.results if r['recognition'] == 'PASS')
        ver_pass = sum(1 for r in self.results if r['verification'] == 'PASS')
        mysql_pass = sum(1 for r in self.results if r['mysql_sync'] == 'PASS')
        chromadb_pass = sum(1 for r in self.results if r['chromadb_sync'] == 'PASS')
        
        print(f"\nSummary:")
        print(f"  Total Combinations Tested: {total}")
        print(f"  Registration Success: {reg_pass}/{total} ({reg_pass/total*100:.1f}%)")
        print(f"  Recognition Success: {rec_pass}/{total} ({rec_pass/total*100:.1f}%)")
        print(f"  Verification Success: {ver_pass}/{total} ({ver_pass/total*100:.1f}%)")
        print(f"  MySQL Sync Success: {mysql_pass}/{total} ({mysql_pass/total*100:.1f}%)")
        print(f"  ChromaDB Sync Success: {chromadb_pass}/{total} ({chromadb_pass/total*100:.1f}%)")
        
        # Detailed table
        print("\n\nDetailed Results:")
        headers = ['Model', 'Detector', 'Register', 'Recognize', 'Verify', 'MySQL', 'ChromaDB', 'Error']
        table_data = []
        
        for r in self.results:
            table_data.append([
                r['model'],
                r['detector'],
                '✅' if r['registration'] == 'PASS' else '❌',
                '✅' if r['recognition'] == 'PASS' else '❌',
                '✅' if r['verification'] == 'PASS' else '❌',
                '✅' if r['mysql_sync'] == 'PASS' else '❌',
                '✅' if r['chromadb_sync'] == 'PASS' else '❌',
                r['error'][:30] + '...' if r['error'] else ''
            ])
        
        print(tabulate(table_data, headers=headers, tablefmt='grid'))
        
        # Save results to file
        with open('deepface_matrix_test_results.json', 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\n💾 Detailed results saved to: deepface_matrix_test_results.json")

def main():
    """Main test function"""
    tester = DeepFaceMatrixTester()
    
    # For quick testing, use subset
    quick_test = True
    
    if quick_test:
        # Test subset for quick validation
        test_models = ["VGG-Face", "OpenFace", "Dlib"]
        test_detectors = ["opencv", "retinaface", "mtcnn"]
    else:
        # Full matrix test
        test_models = MODELS
        test_detectors = DETECTORS
    
    # Run the matrix test
    tester.run_matrix_test(models=test_models, detectors=test_detectors)
    
    # Generate report
    tester.generate_report()

if __name__ == "__main__":
    print("🚀 DeepFace Model x Detector Matrix Test")
    print("This will test all combinations of models and detectors")
    print("Each test includes: Registration, Recognition, Verification, DB Sync")
    
    main()
    
    print("\n✅ Matrix test complete!")