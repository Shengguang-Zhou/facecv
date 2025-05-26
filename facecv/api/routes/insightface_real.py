"""Real InsightFace API Routes (Non-Mock)"""

from fastapi import APIRouter, File, UploadFile, HTTPException, Form, Query
from typing import List, Optional
import numpy as np
from PIL import Image
import io
import logging
import cv2
from datetime import datetime

from facecv.models.insightface.onnx_recognizer import ONNXFaceRecognizer
from facecv.models.insightface.real_recognizer import RealInsightFaceRecognizer
from facecv.schemas.face import FaceDetection, VerificationResult, RecognitionResult, FaceRegisterResponse
from facecv.config import get_settings
from facecv.database.factory import get_default_database

router = APIRouter(tags=["InsightFace"])
logger = logging.getLogger(__name__)

# Global instance
_recognizer = None

def get_recognizer():
    """Get Real InsightFace recognizer instance"""
    global _recognizer
    if _recognizer is None:
        settings = get_settings()
        # Use SQLite database directly - no factory dependency
        from facecv.database.sqlite_facedb import SQLiteFaceDB
        import os
        db_path = os.path.join(os.getcwd(), "facecv_production.db")
        face_db = SQLiteFaceDB(db_path=db_path)
        
        # Use Real InsightFace recognizer with buffalo_l model pack
        _recognizer = RealInsightFaceRecognizer(
            face_db=face_db,
            model_pack="buffalo_l",
            similarity_threshold=getattr(settings, 'similarity_threshold', 0.4),
            det_thresh=getattr(settings, 'detection_threshold', 0.5)
        )
        logger.info(f"Real InsightFace recognizer initialized with SQLite at: {db_path}")
    return _recognizer

async def process_upload_file(file: UploadFile) -> np.ndarray:
    """Process uploaded image file"""
    contents = await file.read()
    
    try:
        image = Image.open(io.BytesIO(contents))
        if image.mode == 'RGBA':
            rgb_image = Image.new('RGB', image.size, (255, 255, 255))
            rgb_image.paste(image, mask=image.split()[3])
            image = rgb_image
        elif image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert to BGR for OpenCV
        image_array = np.array(image)
        image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        return image_bgr
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Cannot process image: {str(e)}")

# ==================== Detection Endpoints ====================

@router.post("/detect", response_model=List[FaceDetection])
async def detect_faces(
    file: UploadFile = File(...),
    min_confidence: float = Query(0.5, description="Minimum detection confidence")
):
    """
    Detect faces in uploaded image using real face detection
    
    - **file**: Image file containing faces
    - **min_confidence**: Minimum detection confidence
    """
    recognizer = get_recognizer()
    image = await process_upload_file(file)
    
    try:
        faces = recognizer.detect_faces(image)
        
        # Filter by confidence
        filtered_faces = [f for f in faces if f.confidence >= min_confidence]
        
        logger.info(f"Detected {len(filtered_faces)} faces (filtered from {len(faces)})")
        return filtered_faces
        
    except Exception as e:
        logger.error(f"Error detecting faces: {e}")
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")

# ==================== Verification Endpoints ====================

@router.post("/verify", response_model=VerificationResult)
async def verify_faces(
    file1: UploadFile = File(..., description="First face image"),
    file2: UploadFile = File(..., description="Second face image"),
    threshold: float = Query(0.4, description="Similarity threshold (0-1)")
):
    """
    Verify if two faces are the same person using real face verification
    
    - **file1**: First face image
    - **file2**: Second face image
    - **threshold**: Similarity threshold for verification
    """
    recognizer = get_recognizer()
    image1 = await process_upload_file(file1)
    image2 = await process_upload_file(file2)
    
    try:
        result = recognizer.verify(image1=image1, image2=image2, threshold=threshold)
        
        logger.info(f"Verification result: {result.is_same_person} (confidence: {result.confidence:.3f})")
        return result
        
    except Exception as e:
        logger.error(f"Error verifying faces: {e}")
        raise HTTPException(status_code=500, detail=f"Verification failed: {str(e)}")

# ==================== Recognition Endpoints ====================

@router.post("/recognize", response_model=List[RecognitionResult])
async def recognize_faces(
    file: UploadFile = File(..., description="Image containing faces to recognize"),
    threshold: float = Query(0.4, description="Recognition similarity threshold (0-1)")
):
    """
    Recognize faces in uploaded image using real face recognition
    
    - **file**: Image file containing faces to recognize
    - **threshold**: Similarity threshold for recognition
    """
    recognizer = get_recognizer()
    image = await process_upload_file(file)
    
    try:
        results = recognizer.recognize(image=image, threshold=threshold)
        
        logger.info(f"Recognized {len(results)} faces")
        return results
        
    except Exception as e:
        logger.error(f"Error recognizing faces: {e}")
        raise HTTPException(status_code=500, detail=f"Recognition failed: {str(e)}")

@router.post("/register", response_model=FaceRegisterResponse)
async def register_face(
    file: UploadFile = File(..., description="Image containing face to register"),
    name: str = Form(..., description="Person's name"),
    department: Optional[str] = Form(None, description="Department"),
    employee_id: Optional[str] = Form(None, description="Employee ID")
):
    """
    Register face(s) in uploaded image using real face registration
    
    - **file**: Image file containing face to register
    - **name**: Person's name
    - **department**: Department (optional)
    - **employee_id**: Employee ID (optional)
    """
    recognizer = get_recognizer()
    image = await process_upload_file(file)
    
    try:
        # Prepare metadata
        face_metadata = {}
        if department:
            face_metadata["department"] = department
        if employee_id:
            face_metadata["employee_id"] = employee_id
        
        # Register faces
        face_ids = recognizer.register(image=image, name=name, metadata=face_metadata)
        
        if not face_ids:
            raise HTTPException(status_code=400, detail="No faces detected for registration")
        
        logger.info(f"Successfully registered {len(face_ids)} face(s) for {name}")
        
        return FaceRegisterResponse(
            success=True,
            message=f"Successfully registered {len(face_ids)} face(s)",
            person_name=name,
            face_id=face_ids[0] if face_ids else None
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error registering face: {e}")
        raise HTTPException(status_code=500, detail=f"Registration failed: {str(e)}")

# ==================== Database Endpoints ====================

@router.get("/faces")
async def list_faces(
    name: Optional[str] = Query(None, description="Filter by name"),
    limit: int = Query(100, description="Maximum number of faces to return")
):
    """
    List faces in database
    
    - **name**: Filter by person name (optional)
    - **limit**: Maximum number of faces to return
    """
    recognizer = get_recognizer()
    
    try:
        faces = recognizer.list_faces(name)
        
        # Limit results and clean data for JSON serialization
        limited_faces = faces[:limit]
        
        # Clean faces data - remove or convert numpy arrays
        clean_faces = []
        for face in limited_faces:
            clean_face = {
                'id': face.get('id'),
                'name': face.get('name'),
                'metadata': face.get('metadata', {}),
                'created_at': face.get('created_at'),
                'updated_at': face.get('updated_at')
            }
            # Convert embedding to list if it exists
            if 'embedding' in face and face['embedding'] is not None:
                if hasattr(face['embedding'], 'tolist'):
                    clean_face['embedding_size'] = len(face['embedding'])
                else:
                    clean_face['embedding_size'] = 0
            clean_faces.append(clean_face)
        
        return {
            "faces": clean_faces,
            "total": len(faces),
            "returned": len(limited_faces)
        }
        
    except Exception as e:
        logger.error(f"Error listing faces: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list faces: {str(e)}")

@router.delete("/faces/{face_id}")
async def delete_face(face_id: str):
    """
    Delete a face by ID
    
    - **face_id**: Face ID to delete
    """
    recognizer = get_recognizer()
    
    try:
        success = recognizer.delete(face_id=face_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="Face not found")
        
        return {"message": f"Successfully deleted face {face_id}"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting face: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete face: {str(e)}")

@router.delete("/faces/by-name/{name}")
async def delete_faces_by_name(name: str):
    """
    Delete all faces for a person
    
    - **name**: Person's name
    """
    recognizer = get_recognizer()
    
    try:
        success = recognizer.delete(name=name)
        
        if not success:
            raise HTTPException(status_code=404, detail="No faces found for this name")
        
        return {"message": f"Successfully deleted all faces for {name}"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting faces by name: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete faces: {str(e)}")

@router.get("/faces/count")
async def get_face_count():
    """
    Get total number of faces in database
    """
    recognizer = get_recognizer()
    
    try:
        count = recognizer.get_face_count()
        return {"total_faces": count}
        
    except Exception as e:
        logger.error(f"Error getting face count: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get face count: {str(e)}")

# ==================== Utility Endpoints ====================

@router.get("/models/info")
async def get_model_info():
    """
    Get information about loaded models
    """
    try:
        recognizer = get_recognizer()
        
        # Get model info from real recognizer
        model_info = recognizer.get_model_info()
        
        return {
            "status": "active",
            "initialized": model_info.get("initialized", False),
            "model_pack": model_info.get("model_pack", "buffalo_l"),
            "similarity_threshold": recognizer.similarity_threshold,
            "detection_threshold": recognizer.det_thresh,
            "face_database_connected": recognizer.face_db is not None,
            "face_count": recognizer.get_face_count() if recognizer.face_db else 0,
            "insightface_available": model_info.get("insightface_available", False)
        }
        
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get model info: {str(e)}")

@router.get("/health")
async def health_check():
    """
    Health check for InsightFace services
    """
    try:
        recognizer = get_recognizer()
        
        return {
            "status": "healthy",
            "service": "Real InsightFace API",
            "initialized": recognizer.initialized,
            "model_pack": recognizer.model_pack,
            "database_connected": recognizer.face_db is not None,
            "timestamp": str(datetime.now())
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")