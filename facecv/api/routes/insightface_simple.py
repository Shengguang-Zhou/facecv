"""Simple InsightFace API Routes for Testing"""

from fastapi import APIRouter, File, UploadFile, HTTPException, Query
from typing import List
import numpy as np
from PIL import Image
import io
import logging

from facecv.models.insightface import InsightFaceRecognizer, InsightFaceVerifier, InsightFaceDetector
from facecv.schemas.face import FaceDetection, VerificationResult
from facecv.config import get_settings

router = APIRouter(prefix="/api/v1/insightface", tags=["InsightFace"])
logger = logging.getLogger(__name__)

# Global instances
_detector = None

def get_detector():
    """Get InsightFace detector instance"""
    global _detector
    if _detector is None:
        _detector = InsightFaceDetector(mock_mode=True)  # Use mock mode for now
    return _detector

async def process_upload_file(file: UploadFile) -> np.ndarray:
    """Process uploaded image file"""
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    if image.mode != 'RGB':
        image = image.convert('RGB')
    return np.array(image)

@router.get("/health")
async def health_check():
    """Health check for InsightFace services"""
    return {"status": "healthy", "mock_mode": True}

@router.post("/detect", response_model=List[FaceDetection])
async def detect_faces(
    file: UploadFile = File(...),
    min_confidence: float = Query(0.5, description="Minimum detection confidence")
):
    """Detect faces in uploaded image"""
    detector = get_detector()
    image = await process_upload_file(file)
    
    try:
        faces = detector.detect_faces(image=image, include_embeddings=False, include_quality=True)
        filtered_faces = [f for f in faces if f.confidence >= min_confidence]
        return filtered_faces
    except Exception as e:
        logger.error(f"Error detecting faces: {e}")
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")

@router.post("/verify", response_model=VerificationResult)
async def verify_faces(
    file1: UploadFile = File(...),
    file2: UploadFile = File(...),
    threshold: float = Query(0.4, description="Similarity threshold")
):
    """Verify if two faces are the same person"""
    verifier = InsightFaceVerifier(mock_mode=True)
    image1 = await process_upload_file(file1)
    image2 = await process_upload_file(file2)
    
    try:
        result = verifier.verify_faces(image1=image1, image2=image2, threshold=threshold)
        return result
    except Exception as e:
        logger.error(f"Error verifying faces: {e}")
        raise HTTPException(status_code=500, detail=f"Verification failed: {str(e)}")