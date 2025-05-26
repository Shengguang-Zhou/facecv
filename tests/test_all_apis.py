#!/usr/bin/env python3.10
import httpx
import json
import base64
import asyncio
from pathlib import Path
import sys

BASE_URL = "http://localhost:7000"

# Test face image paths
FACE_DIR = Path("/home/a/PycharmProjects/EurekCV/dataset/faces")
TEST_IMAGES = list(FACE_DIR.glob("*.jpg"))[:3] if FACE_DIR.exists() else []

def encode_image(image_path):
    """Encode image to base64"""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')

async def test_api():
    """Test all API endpoints"""
    async with httpx.AsyncClient(timeout=30.0) as client:
        results = {}
        
        # 1. Test health endpoint
        print("Testing health endpoint...")
        try:
            resp = await client.get(f"{BASE_URL}/health")
            results["health"] = {"status": resp.status_code, "response": resp.json()}
        except Exception as e:
            results["health"] = {"status": "error", "error": str(e)}
        
        # 2. Test face detection
        if TEST_IMAGES:
            print(f"Testing face detection with {TEST_IMAGES[0].name}...")
            try:
                image_base64 = encode_image(TEST_IMAGES[0])
                resp = await client.post(
                    f"{BASE_URL}/api/face/detect",
                    json={"image": image_base64}
                )
                results["face_detect"] = {"status": resp.status_code, "response": resp.json()}
            except Exception as e:
                results["face_detect"] = {"status": "error", "error": str(e)}
        
        # 3. Test face embedding
        if TEST_IMAGES:
            print(f"Testing face embedding...")
            try:
                image_base64 = encode_image(TEST_IMAGES[0])
                resp = await client.post(
                    f"{BASE_URL}/api/face/embedding",
                    json={"image": image_base64}
                )
                results["face_embedding"] = {"status": resp.status_code, "response": resp.json()}
            except Exception as e:
                results["face_embedding"] = {"status": "error", "error": str(e)}
        
        # 4. Test face database operations
        print("Testing face database operations...")
        
        # 4a. Add face
        if TEST_IMAGES:
            try:
                image_base64 = encode_image(TEST_IMAGES[0])
                resp = await client.post(
                    f"{BASE_URL}/api/face/database/add",
                    json={
                        "name": "test_user",
                        "image": image_base64,
                        "metadata": {"department": "test"}
                    }
                )
                results["face_add"] = {"status": resp.status_code, "response": resp.json()}
            except Exception as e:
                results["face_add"] = {"status": "error", "error": str(e)}
        
        # 4b. Get all faces
        try:
            resp = await client.get(f"{BASE_URL}/api/face/database/all")
            results["face_get_all"] = {"status": resp.status_code, "response": resp.json()}
        except Exception as e:
            results["face_get_all"] = {"status": "error", "error": str(e)}
        
        # 4c. Get face by name
        try:
            resp = await client.get(f"{BASE_URL}/api/face/database/get/test_user")
            results["face_get_by_name"] = {"status": resp.status_code, "response": resp.json()}
        except Exception as e:
            results["face_get_by_name"] = {"status": "error", "error": str(e)}
        
        # 5. Test face recognition
        if TEST_IMAGES and len(TEST_IMAGES) > 1:
            print("Testing face recognition...")
            try:
                image_base64 = encode_image(TEST_IMAGES[1])
                resp = await client.post(
                    f"{BASE_URL}/api/face/recognize",
                    json={"image": image_base64}
                )
                results["face_recognize"] = {"status": resp.status_code, "response": resp.json()}
            except Exception as e:
                results["face_recognize"] = {"status": "error", "error": str(e)}
        
        # 6. Test face verification
        if TEST_IMAGES and len(TEST_IMAGES) > 1:
            print("Testing face verification...")
            try:
                resp = await client.post(
                    f"{BASE_URL}/api/face/verify",
                    json={
                        "image1": encode_image(TEST_IMAGES[0]),
                        "image2": encode_image(TEST_IMAGES[1])
                    }
                )
                results["face_verify"] = {"status": resp.status_code, "response": resp.json()}
            except Exception as e:
                results["face_verify"] = {"status": "error", "error": str(e)}
        
        # 7. Test DeepFace endpoints
        print("Testing DeepFace endpoints...")
        
        # 7a. DeepFace verify
        if TEST_IMAGES and len(TEST_IMAGES) > 1:
            try:
                resp = await client.post(
                    f"{BASE_URL}/api/deepface/verify",
                    json={
                        "img1_base64": encode_image(TEST_IMAGES[0]),
                        "img2_base64": encode_image(TEST_IMAGES[1])
                    }
                )
                results["deepface_verify"] = {"status": resp.status_code, "response": resp.json()}
            except Exception as e:
                results["deepface_verify"] = {"status": "error", "error": str(e)}
        
        # 7b. DeepFace analyze
        if TEST_IMAGES:
            try:
                resp = await client.post(
                    f"{BASE_URL}/api/deepface/analyze",
                    json={
                        "img_base64": encode_image(TEST_IMAGES[0]),
                        "actions": ["age", "gender", "emotion", "race"]
                    }
                )
                results["deepface_analyze"] = {"status": resp.status_code, "response": resp.json()}
            except Exception as e:
                results["deepface_analyze"] = {"status": "error", "error": str(e)}
        
        # 8. Test webhook endpoints
        print("Testing webhook endpoints...")
        
        # 8a. Register webhook
        try:
            resp = await client.post(
                f"{BASE_URL}/api/webhooks/register",
                json={
                    "name": "test_webhook",
                    "url": "http://localhost:8080/webhook",
                    "events": ["face_detected", "face_recognized"],
                    "headers": {"X-API-Key": "test123"}
                }
            )
            results["webhook_register"] = {"status": resp.status_code, "response": resp.json()}
        except Exception as e:
            results["webhook_register"] = {"status": "error", "error": str(e)}
        
        # 8b. List webhooks
        try:
            resp = await client.get(f"{BASE_URL}/api/webhooks/")
            results["webhook_list"] = {"status": resp.status_code, "response": resp.json()}
        except Exception as e:
            results["webhook_list"] = {"status": "error", "error": str(e)}
        
        # 9. Test InsightFace endpoints
        print("Testing InsightFace endpoints...")
        
        # 9a. InsightFace detect
        if TEST_IMAGES:
            try:
                resp = await client.post(
                    f"{BASE_URL}/api/insightface/detect",
                    json={"image": encode_image(TEST_IMAGES[0])}
                )
                results["insightface_detect"] = {"status": resp.status_code, "response": resp.json()}
            except Exception as e:
                results["insightface_detect"] = {"status": "error", "error": str(e)}
        
        # 9b. InsightFace verify
        if TEST_IMAGES and len(TEST_IMAGES) > 1:
            try:
                resp = await client.post(
                    f"{BASE_URL}/api/insightface/verify",
                    json={
                        "image1": encode_image(TEST_IMAGES[0]),
                        "image2": encode_image(TEST_IMAGES[1])
                    }
                )
                results["insightface_verify"] = {"status": resp.status_code, "response": resp.json()}
            except Exception as e:
                results["insightface_verify"] = {"status": "error", "error": str(e)}
        
        # Print results summary
        print("\n" + "="*60)
        print("API TEST RESULTS SUMMARY")
        print("="*60)
        
        success_count = 0
        total_count = 0
        
        for endpoint, result in results.items():
            total_count += 1
            status = result.get("status", "error")
            if isinstance(status, int) and 200 <= status < 300:
                success_count += 1
                print(f"✓ {endpoint}: SUCCESS (Status: {status})")
            else:
                print(f"✗ {endpoint}: FAILED")
                if "error" in result:
                    print(f"  Error: {result['error']}")
                elif isinstance(status, int):
                    print(f"  Status: {status}")
                    if "response" in result:
                        print(f"  Response: {result['response']}")
        
        print(f"\nTotal: {success_count}/{total_count} tests passed")
        
        # Save detailed results
        with open("/test_results.json", "w") as f:
            json.dump(results, f, indent=2)
        print("\nDetailed results saved to test_results.json")
        
        return success_count, total_count

if __name__ == "__main__":
    success, total = asyncio.run(test_api())
    sys.exit(0 if success == total else 1)