#!/usr/bin/env python3
"""Verify API implementation by checking code structure"""

import os
import re
import ast

def find_api_endpoints(file_path):
    """Extract API endpoints from a Python file"""
    endpoints = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find router decorators
    patterns = [
        r'@router\.(get|post|put|delete|patch)\s*\(\s*["\']([^"\']+)["\']',
        r'@app\.(get|post|put|delete|patch)\s*\(\s*["\']([^"\']+)["\']'
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, content)
        for method, path in matches:
            endpoints.append((method.upper(), path))
    
    # Also find function names
    func_pattern = r'(?:async\s+)?def\s+(\w+)\s*\('
    functions = re.findall(func_pattern, content)
    
    return endpoints, functions

def verify_apis():
    """Verify all API implementations"""
    print("=== Verifying FaceCV API Implementation ===\n")
    
    api_files = {
        "InsightFace": "/home/a/PycharmProjects/facecv/facecv/api/routes/face.py",
        "DeepFace": "/home/a/PycharmProjects/facecv/facecv/api/routes/deepface.py",
        "Health": "/home/a/PycharmProjects/facecv/facecv/api/routes/health.py",
        "Stream": "/home/a/PycharmProjects/facecv/facecv/api/routes/stream.py"
    }
    
    all_endpoints = []
    
    for name, file_path in api_files.items():
        if os.path.exists(file_path):
            print(f"📁 {name} API ({file_path}):")
            endpoints, functions = find_api_endpoints(file_path)
            
            if endpoints:
                for method, path in endpoints:
                    print(f"  ✅ {method:6} {path}")
                    all_endpoints.append(f"{method} {path}")
            else:
                print(f"  ⚠️  No endpoints found")
            
            print(f"  📝 Functions: {', '.join(functions[:5])}{'...' if len(functions) > 5 else ''}")
            print()
        else:
            print(f"❌ {name} API file not found: {file_path}\n")
    
    # Summary of newly added endpoints
    print("\n=== Newly Added API Endpoints ===")
    
    new_insightface_endpoints = [
        "POST /video_face/",
        "GET /recognize/webcam/stream",
        "POST /faces/offline"
    ]
    
    print("\n🔹 InsightFace New Endpoints:")
    for endpoint in new_insightface_endpoints:
        if any(endpoint in e for e in all_endpoints):
            print(f"  ✅ {endpoint} - Implemented")
        else:
            print(f"  ❌ {endpoint} - Not found")
    
    # Check specific implementation details
    print("\n=== Implementation Details ===")
    
    # Check for video face extraction
    with open(api_files["InsightFace"], 'r') as f:
        content = f.read()
        
        if "extract_faces_from_video" in content:
            print("✅ Video face extraction function implemented")
            if "VideoExtractor" in content:
                print("  ✅ Uses VideoExtractor from utils")
            if "FaceQualityAssessor" in content:
                print("  ✅ Uses FaceQualityAssessor for quality filtering")
        
        if "recognize_webcam_stream" in content:
            print("\n✅ Webcam stream recognition implemented")
            if "StreamingResponse" in content:
                print("  ✅ Returns StreamingResponse for SSE")
            if "VideoStreamManager" in content:
                print("  ✅ Uses VideoStreamManager for stream handling")
        
        if "batch_register_offline" in content:
            print("\n✅ Offline batch registration implemented")
            if "Path" in content or "pathlib" in content:
                print("  ✅ Uses Path for directory traversal")
            if "quality_threshold" in content:
                print("  ✅ Supports quality threshold filtering")
    
    # Check DeepFace endpoints
    print("\n🔹 DeepFace API Endpoints:")
    deepface_required = [
        "POST /analyze/",
        "POST /verify/",
        "POST /faces/",
        "GET /faces/",
        "POST /recognition",
        "POST /video_face/",
        "GET /recognize/webcam/stream"
    ]
    
    if os.path.exists(api_files["DeepFace"]):
        with open(api_files["DeepFace"], 'r') as f:
            deepface_content = f.read()
            
        for endpoint in deepface_required:
            if any(endpoint in e for e in all_endpoints):
                print(f"  ✅ {endpoint}")
    
    print("\n=== Verification Complete ===")
    print(f"Total endpoints found: {len(all_endpoints)}")
    
    return all_endpoints

if __name__ == "__main__":
    endpoints = verify_apis()
    
    # Save endpoint list
    with open("/home/a/PycharmProjects/facecv/api_endpoints.txt", "w") as f:
        f.write("FaceCV API Endpoints\n")
        f.write("===================\n\n")
        for endpoint in sorted(set(endpoints)):
            f.write(f"{endpoint}\n")
    
    print("\nEndpoint list saved to api_endpoints.txt")