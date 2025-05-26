"""Batch Processing API Routes for InsightFace"""

from fastapi import APIRouter, File, UploadFile, HTTPException, Form, Query
from typing import List, Optional, Dict, Any
import numpy as np
from PIL import Image
import io
import logging
import cv2
from datetime import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor

from facecv.models.insightface.real_recognizer import RealInsightFaceRecognizer
from facecv.schemas.face import FaceDetection, VerificationResult, RecognitionResult, FaceRegisterResponse
from facecv.config import get_settings
from facecv.database.sqlite_facedb import SQLiteFaceDB

router = APIRouter(prefix="/api/v1/batch", tags=["Batch Processing"])
logger = logging.getLogger(__name__)

# Thread pool for parallel processing
executor = ThreadPoolExecutor(max_workers=4)

# Global instance
_recognizer = None

def get_recognizer():
    """Get Real InsightFace recognizer instance"""
    global _recognizer
    if _recognizer is None:
        settings = get_settings()
        import os
        db_path = os.path.join(os.getcwd(), "facecv_production.db")
        face_db = SQLiteFaceDB(db_path=db_path)
        
        _recognizer = RealInsightFaceRecognizer(
            face_db=face_db,
            model_pack="buffalo_l",
            similarity_threshold=getattr(settings, 'similarity_threshold', 0.4),
            det_thresh=getattr(settings, 'detection_threshold', 0.5)
        )
        logger.info(f"Batch processing recognizer initialized with SQLite at: {db_path}")
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

# ==================== Batch Detection ====================

@router.post("/detect", response_model=Dict[str, List[FaceDetection]])
async def batch_detect_faces(
    files: List[UploadFile] = File(..., description="Multiple image files"),
    min_confidence: float = Query(0.5, description="Minimum detection confidence")
):
    """
    Detect faces in multiple images simultaneously
    
    - **files**: List of image files
    - **min_confidence**: Minimum detection confidence
    """
    recognizer = get_recognizer()
    results = {}
    
    # Process images in parallel
    async def process_image(idx: int, file: UploadFile):
        try:
            image = await process_upload_file(file)
            faces = await asyncio.get_event_loop().run_in_executor(
                executor, recognizer.detect_faces, image
            )
            
            # Filter by confidence
            filtered_faces = [f for f in faces if f.confidence >= min_confidence]
            results[f"image_{idx}_{file.filename}"] = filtered_faces
            
        except Exception as e:
            logger.error(f"Error processing {file.filename}: {e}")
            results[f"image_{idx}_{file.filename}"] = []
    
    # Process all images concurrently
    await asyncio.gather(*[
        process_image(idx, file) for idx, file in enumerate(files)
    ])
    
    return results

# ==================== Batch Registration ====================

@router.post("/register", response_model=Dict[str, Any])
async def batch_register_faces(
    files: List[UploadFile] = File(..., description="Multiple face images"),
    names: str = Form(..., description="Comma-separated names corresponding to images"),
    department: Optional[str] = Form(None, description="Department for all faces"),
    metadata: Optional[str] = Form(None, description="JSON metadata for all faces")
):
    """
    Register multiple faces from different images
    
    - **files**: List of image files (one face per image expected)
    - **names**: Comma-separated names matching the order of images
    - **department**: Optional department for all registrations
    - **metadata**: Optional JSON metadata
    """
    recognizer = get_recognizer()
    name_list = [n.strip() for n in names.split(',')]
    
    if len(name_list) != len(files):
        raise HTTPException(
            status_code=400, 
            detail=f"Number of names ({len(name_list)}) must match number of images ({len(files)})"
        )
    
    results = {
        "successful": [],
        "failed": [],
        "total": len(files)
    }
    
    # Process registrations in parallel
    async def register_face(idx: int, file: UploadFile, name: str):
        try:
            image = await process_upload_file(file)
            
            # Prepare metadata
            face_metadata = {}
            if department:
                face_metadata["department"] = department
            if metadata:
                import json
                try:
                    face_metadata.update(json.loads(metadata))
                except:
                    pass
            
            # Register face
            face_ids = await asyncio.get_event_loop().run_in_executor(
                executor, recognizer.register, image, name, face_metadata
            )
            
            if face_ids:
                results["successful"].append({
                    "name": name,
                    "face_id": face_ids[0],
                    "filename": file.filename
                })
            else:
                results["failed"].append({
                    "name": name,
                    "filename": file.filename,
                    "reason": "No face detected"
                })
                
        except Exception as e:
            logger.error(f"Error registering {name} from {file.filename}: {e}")
            results["failed"].append({
                "name": name,
                "filename": file.filename,
                "reason": str(e)
            })
    
    # Process all registrations concurrently
    await asyncio.gather(*[
        register_face(idx, file, name) 
        for idx, (file, name) in enumerate(zip(files, name_list))
    ])
    
    results["success_rate"] = len(results["successful"]) / results["total"]
    return results

# ==================== Batch Recognition ====================

@router.post("/recognize", response_model=Dict[str, List[RecognitionResult]])
async def batch_recognize_faces(
    files: List[UploadFile] = File(..., description="Multiple images to recognize faces in"),
    threshold: float = Query(0.4, description="Recognition similarity threshold")
):
    """
    Recognize faces in multiple images
    
    - **files**: List of image files
    - **threshold**: Similarity threshold for recognition
    """
    recognizer = get_recognizer()
    results = {}
    
    # Process images in parallel
    async def recognize_image(idx: int, file: UploadFile):
        try:
            image = await process_upload_file(file)
            recognition_results = await asyncio.get_event_loop().run_in_executor(
                executor, recognizer.recognize, image, threshold
            )
            results[f"image_{idx}_{file.filename}"] = recognition_results
            
        except Exception as e:
            logger.error(f"Error recognizing faces in {file.filename}: {e}")
            results[f"image_{idx}_{file.filename}"] = []
    
    # Process all images concurrently
    await asyncio.gather(*[
        recognize_image(idx, file) for idx, file in enumerate(files)
    ])
    
    return results

# ==================== Batch Verification ====================

@router.post("/verify", response_model=List[Dict[str, Any]])
async def batch_verify_faces(
    reference_images: List[UploadFile] = File(..., description="Reference face images"),
    comparison_images: List[UploadFile] = File(..., description="Images to compare against references"),
    threshold: float = Query(0.4, description="Similarity threshold"),
    cross_compare: bool = Query(False, description="Compare all references against all comparisons")
):
    """
    Verify multiple face pairs or cross-compare face sets
    
    - **reference_images**: List of reference face images
    - **comparison_images**: List of comparison face images
    - **threshold**: Similarity threshold for verification
    - **cross_compare**: If True, compare all references against all comparisons
    """
    recognizer = get_recognizer()
    results = []
    
    # Process verification pairs
    async def verify_pair(ref_idx: int, ref_file: UploadFile, 
                         comp_idx: int, comp_file: UploadFile):
        try:
            ref_image = await process_upload_file(ref_file)
            comp_image = await process_upload_file(comp_file)
            
            verification = await asyncio.get_event_loop().run_in_executor(
                executor, recognizer.verify, ref_image, comp_image, threshold
            )
            
            return {
                "reference": f"{ref_idx}_{ref_file.filename}",
                "comparison": f"{comp_idx}_{comp_file.filename}",
                "is_same_person": verification.is_same_person,
                "confidence": verification.confidence,
                "distance": verification.distance,
                "threshold": verification.threshold
            }
            
        except Exception as e:
            logger.error(f"Error verifying {ref_file.filename} vs {comp_file.filename}: {e}")
            return {
                "reference": f"{ref_idx}_{ref_file.filename}",
                "comparison": f"{comp_idx}_{comp_file.filename}",
                "error": str(e)
            }
    
    # Create verification pairs
    if cross_compare:
        # Compare all references against all comparisons
        pairs = [
            (ref_idx, ref_file, comp_idx, comp_file)
            for ref_idx, ref_file in enumerate(reference_images)
            for comp_idx, comp_file in enumerate(comparison_images)
        ]
    else:
        # Pair-wise comparison
        if len(reference_images) != len(comparison_images):
            raise HTTPException(
                status_code=400,
                detail=f"Number of reference images ({len(reference_images)}) must match comparison images ({len(comparison_images)}) for pair-wise comparison"
            )
        pairs = [
            (idx, ref_file, idx, comp_file)
            for idx, (ref_file, comp_file) in enumerate(zip(reference_images, comparison_images))
        ]
    
    # Process all verifications concurrently
    verification_results = await asyncio.gather(*[
        verify_pair(ref_idx, ref_file, comp_idx, comp_file)
        for ref_idx, ref_file, comp_idx, comp_file in pairs
    ])
    
    return verification_results

# ==================== Batch Analysis ====================

@router.post("/analyze", response_model=Dict[str, Any])
async def batch_analyze_faces(
    files: List[UploadFile] = File(..., description="Multiple images to analyze"),
    include_embeddings: bool = Query(False, description="Include face embeddings in response")
):
    """
    Analyze faces in multiple images for attributes
    
    - **files**: List of image files
    - **include_embeddings**: Whether to include face embeddings
    """
    recognizer = get_recognizer()
    results = {}
    
    # Process images in parallel
    async def analyze_image(idx: int, file: UploadFile):
        try:
            image = await process_upload_file(file)
            faces = await asyncio.get_event_loop().run_in_executor(
                executor, recognizer.detect_faces, image
            )
            
            face_analyses = []
            for face in faces:
                analysis = {
                    "bbox": face.bbox,
                    "confidence": face.confidence,
                    "quality_score": face.quality_score
                }
                
                # Add attributes if available
                if face.age is not None:
                    analysis["age"] = face.age
                if face.gender is not None:
                    analysis["gender"] = face.gender
                if face.landmarks is not None:
                    analysis["landmarks"] = face.landmarks
                if include_embeddings and face.embedding is not None:
                    analysis["embedding_size"] = len(face.embedding)
                    
                face_analyses.append(analysis)
            
            results[f"image_{idx}_{file.filename}"] = {
                "faces_detected": len(faces),
                "analyses": face_analyses
            }
            
        except Exception as e:
            logger.error(f"Error analyzing {file.filename}: {e}")
            results[f"image_{idx}_{file.filename}"] = {
                "error": str(e)
            }
    
    # Process all images concurrently
    await asyncio.gather(*[
        analyze_image(idx, file) for idx, file in enumerate(files)
    ])
    
    # Add summary statistics
    total_faces = sum(
        result.get("faces_detected", 0) 
        for result in results.values() 
        if "faces_detected" in result
    )
    
    results["summary"] = {
        "total_images": len(files),
        "total_faces": total_faces,
        "average_faces_per_image": total_faces / len(files) if files else 0
    }
    
    return results