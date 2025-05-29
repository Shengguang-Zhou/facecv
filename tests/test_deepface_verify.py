#!/usr/bin/env python3
"""Test DeepFace verify functionality directly"""

import asyncio
import sys
sys.path.append('..')

from facecv.models.deepface.core.verification import FaceVerification
import cv2

async def test_verify():
    """Test the verification directly"""
    try:
        # Initialize verification using singleton
        from facecv.models.deepface import face_verification as verification
        
        # Load test images
        img1 = cv2.imread("/home/a/PycharmProjects/EurekCV/dataset/faces/trump.jpeg")
        img2 = cv2.imread("/home/a/PycharmProjects/EurekCV/dataset/faces/trump2.jpeg")
        
        print(f"Image 1 shape: {img1.shape}")
        print(f"Image 2 shape: {img2.shape}")
        
        # Test verification
        result = await verification.face_verification(
            image_1=img1,
            image_2=img2,
            model_name="ArcFace",
            threshold=0.6,
            anti_spoofing=False
        )
        
        print(f"\nVerification result:")
        print(f"Verified: {result.get('verified')}")
        print(f"Distance: {result.get('distance')}")
        print(f"Threshold: {result.get('threshold')}")
        print(f"Similarity: {1.0 - result.get('distance', 1.0)}")
        
        # Test the response format
        from facecv.schemas.face import FaceVerificationResponse
        response = FaceVerificationResponse(
            verified=result.get("verified", False),
            similarity=1.0 - result.get("distance", 1.0),
            threshold=result.get("threshold", 0.6),
            model="ArcFace",
            detector_used="mtcnn"
        )
        print(f"\nResponse object created successfully: {response}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_verify())