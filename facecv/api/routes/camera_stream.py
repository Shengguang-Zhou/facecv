"""Camera Streaming API with Real-time Face Recognition"""

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse
from typing import Dict, Any, Optional
import cv2
import numpy as np
import json
import asyncio
import logging
import threading
import time
from datetime import datetime

from facecv.models.insightface.onnx_recognizer import ONNXFaceRecognizer
from facecv.models.insightface.real_recognizer import RealInsightFaceRecognizer
from facecv.database.sqlite_facedb import SQLiteFaceDB
from facecv.config import get_settings

router = APIRouter(prefix="/api/v1/camera", tags=["Camera Streaming"])
logger = logging.getLogger(__name__)

# Global camera manager
class CameraManager:
    def __init__(self):
        self.cameras: Dict[str, cv2.VideoCapture] = {}
        self.camera_threads: Dict[str, threading.Thread] = {}
        self.camera_active: Dict[str, bool] = {}
        self.camera_frames: Dict[str, np.ndarray] = {}
        self.camera_results: Dict[str, Dict[str, Any]] = {}
        self.recognizer = None
        self._init_recognizer()
    
    def _init_recognizer(self):
        """Initialize Real InsightFace recognizer"""
        try:
            face_db = SQLiteFaceDB(db_path="facecv_production.db")
            self.recognizer = RealInsightFaceRecognizer(
                face_db=face_db,
                model_pack="buffalo_l",
                similarity_threshold=0.4,
                det_thresh=0.5
            )
            logger.info("Real InsightFace camera recognizer initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Real InsightFace recognizer: {e}")
            # Fallback to ONNX recognizer
            try:
                face_db = SQLiteFaceDB(db_path="facecv_production.db")
                self.recognizer = ONNXFaceRecognizer(
                    face_db=face_db,
                    similarity_threshold=0.4,
                    det_thresh=0.5,
                    mock_mode=False
                )
                logger.info("Fallback to ONNX recognizer")
            except Exception as e2:
                logger.error(f"Failed to initialize any recognizer: {e2}")
                self.recognizer = None
    
    def connect_camera(self, camera_id: str, source: str) -> bool:
        """Connect to camera source"""
        try:
            if camera_id in self.cameras:
                self.disconnect_camera(camera_id)
            
            # Try to parse source as int for local cameras
            try:
                source_int = int(source)
                cap = cv2.VideoCapture(source_int)
            except ValueError:
                # RTSP or other string source
                cap = cv2.VideoCapture(source)
            
            # Test if camera is accessible
            if not cap.isOpened():
                logger.error(f"Failed to open camera {camera_id} with source {source}")
                return False
            
            # Test reading a frame
            ret, frame = cap.read()
            if not ret:
                logger.error(f"Failed to read frame from camera {camera_id}")
                cap.release()
                return False
            
            # Store camera
            self.cameras[camera_id] = cap
            self.camera_active[camera_id] = True
            self.camera_frames[camera_id] = frame
            self.camera_results[camera_id] = {
                "camera_id": camera_id,
                "source": source,
                "status": "connected",
                "timestamp": datetime.now().isoformat(),
                "faces": []
            }
            
            # Start processing thread
            thread = threading.Thread(target=self._process_camera, args=(camera_id,))
            thread.daemon = True
            thread.start()
            self.camera_threads[camera_id] = thread
            
            logger.info(f"Camera {camera_id} connected successfully to {source}")
            return True
            
        except Exception as e:
            logger.error(f"Error connecting camera {camera_id}: {e}")
            return False
    
    def disconnect_camera(self, camera_id: str) -> bool:
        """Disconnect camera and cleanup resources"""
        try:
            # Stop processing
            if camera_id in self.camera_active:
                self.camera_active[camera_id] = False
            
            # Wait for thread to finish
            if camera_id in self.camera_threads:
                thread = self.camera_threads[camera_id]
                thread.join(timeout=2)
                del self.camera_threads[camera_id]
            
            # Release camera
            if camera_id in self.cameras:
                self.cameras[camera_id].release()
                del self.cameras[camera_id]
            
            # Clean up data
            if camera_id in self.camera_frames:
                del self.camera_frames[camera_id]
            if camera_id in self.camera_results:
                del self.camera_results[camera_id]
            
            logger.info(f"Camera {camera_id} disconnected successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error disconnecting camera {camera_id}: {e}")
            return False
    
    def _process_camera(self, camera_id: str):
        """Process camera frames in separate thread"""
        cap = self.cameras[camera_id]
        
        while self.camera_active.get(camera_id, False):
            try:
                ret, frame = cap.read()
                if not ret:
                    logger.warning(f"Failed to read frame from camera {camera_id}")
                    time.sleep(0.1)
                    continue
                
                # Store frame
                self.camera_frames[camera_id] = frame.copy()
                
                # Process with face recognition
                if self.recognizer:
                    try:
                        faces = self.recognizer.detect_faces(frame)
                        face_results = []
                        
                        for face in faces:
                            # Try recognition
                            try:
                                recognition_results = self.recognizer.recognize(frame, threshold=0.4)
                                if recognition_results:
                                    for result in recognition_results:
                                        if self._bbox_overlap(face.bbox, result.bbox):
                                            face_result = {
                                                "bbox": result.bbox,
                                                "name": result.name,
                                                "confidence": result.confidence,
                                                "detection_score": face.confidence,
                                                "quality_score": getattr(face, 'quality_score', 0.7),
                                                "face_id": face.face_id
                                            }
                                            # Add InsightFace attributes if available
                                            if hasattr(face, 'age'):
                                                face_result["age"] = face.age
                                            if hasattr(face, 'gender'):
                                                face_result["gender"] = face.gender
                                            if hasattr(face, 'landmarks'):
                                                face_result["landmarks"] = face.landmarks
                                            face_results.append(face_result)
                                            break
                                else:
                                    # Detection only
                                    face_result = {
                                        "bbox": face.bbox,
                                        "name": "Unknown",
                                        "confidence": 0.0,
                                        "detection_score": face.confidence,
                                        "quality_score": getattr(face, 'quality_score', 0.7),
                                        "face_id": face.face_id
                                    }
                                    # Add InsightFace attributes if available
                                    if hasattr(face, 'age'):
                                        face_result["age"] = face.age
                                    if hasattr(face, 'gender'):
                                        face_result["gender"] = face.gender
                                    if hasattr(face, 'landmarks'):
                                        face_result["landmarks"] = face.landmarks
                                    face_results.append(face_result)
                            except Exception as e:
                                logger.error(f"Recognition error for camera {camera_id}: {e}")
                                # Fallback to detection only
                                face_result = {
                                    "bbox": face.bbox,
                                    "name": "Unknown",
                                    "confidence": 0.0,
                                    "detection_score": face.confidence,
                                    "quality_score": getattr(face, 'quality_score', 0.7),
                                    "face_id": face.face_id
                                }
                                # Add InsightFace attributes if available
                                if hasattr(face, 'age'):
                                    face_result["age"] = face.age
                                if hasattr(face, 'gender'):
                                    face_result["gender"] = face.gender
                                if hasattr(face, 'landmarks'):
                                    face_result["landmarks"] = face.landmarks
                                face_results.append(face_result)
                        
                        # Update results
                        self.camera_results[camera_id] = {
                            "camera_id": camera_id,
                            "timestamp": datetime.now().isoformat(),
                            "status": "active",
                            "faces": face_results,
                            "total_faces": len(face_results)
                        }
                        
                    except Exception as e:
                        logger.error(f"Face processing error for camera {camera_id}: {e}")
                
                # Control frame rate
                time.sleep(0.1)  # ~10 FPS
                
            except Exception as e:
                logger.error(f"Camera processing error {camera_id}: {e}")
                time.sleep(0.5)
        
        logger.info(f"Camera processing stopped for {camera_id}")
    
    def _bbox_overlap(self, bbox1, bbox2, threshold=0.5):
        """Check if two bounding boxes overlap significantly"""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return False
        
        # Calculate areas
        area_i = (x2_i - x1_i) * (y2_i - y1_i)
        area_1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area_2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        
        # Calculate IoU
        iou = area_i / (area_1 + area_2 - area_i)
        return iou > threshold
    
    def get_camera_status(self, camera_id: str = None):
        """Get status of cameras"""
        if camera_id:
            return self.camera_results.get(camera_id, {"error": "Camera not found"})
        else:
            return {
                "active_cameras": list(self.camera_active.keys()),
                "total_cameras": len(self.cameras),
                "cameras": {cid: self.camera_results.get(cid, {}) for cid in self.camera_active.keys()}
            }
    
    def get_latest_results(self, camera_id: str):
        """Get latest recognition results for camera"""
        return self.camera_results.get(camera_id, {"error": "Camera not found"})

# Global camera manager instance
camera_manager = CameraManager()

@router.post("/connect")
async def connect_camera(
    camera_id: str = Query(..., description="Unique camera identifier"),
    source: str = Query(..., description="Camera source (0 for webcam, RTSP URL for IP camera)")
):
    """
    Connect to a camera (local or RTSP)
    
    - **camera_id**: Unique identifier for the camera
    - **source**: Camera source (0 for local camera, RTSP URL for IP camera)
    """
    try:
        success = camera_manager.connect_camera(camera_id, source)
        
        if success:
            return {
                "success": True,
                "message": f"Camera {camera_id} connected successfully",
                "camera_id": camera_id,
                "source": source,
                "timestamp": datetime.now().isoformat()
            }
        else:
            raise HTTPException(
                status_code=400, 
                detail=f"Failed to connect to camera {camera_id} with source {source}"
            )
    except Exception as e:
        logger.error(f"Error connecting camera {camera_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Connection error: {str(e)}")

@router.post("/disconnect")
async def disconnect_camera(
    camera_id: str = Query(..., description="Camera identifier to disconnect")
):
    """
    Disconnect a camera and cleanup resources
    
    - **camera_id**: Identifier of camera to disconnect
    """
    try:
        success = camera_manager.disconnect_camera(camera_id)
        
        if success:
            return {
                "success": True,
                "message": f"Camera {camera_id} disconnected successfully",
                "camera_id": camera_id,
                "timestamp": datetime.now().isoformat()
            }
        else:
            raise HTTPException(
                status_code=404,
                detail=f"Camera {camera_id} not found or already disconnected"
            )
    except Exception as e:
        logger.error(f"Error disconnecting camera {camera_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Disconnection error: {str(e)}")

@router.get("/status")
async def get_camera_status(
    camera_id: Optional[str] = Query(None, description="Specific camera ID (optional)")
):
    """
    Get status of cameras
    
    - **camera_id**: Specific camera ID to check (optional, returns all if not specified)
    """
    try:
        status = camera_manager.get_camera_status(camera_id)
        return status
    except Exception as e:
        logger.error(f"Error getting camera status: {e}")
        raise HTTPException(status_code=500, detail=f"Status error: {str(e)}")

@router.get("/stream")
async def stream_recognition_results(
    camera_id: str = Query(..., description="Camera identifier"),
    format: str = Query("json", description="Output format (json or sse)")
):
    """
    Stream real-time recognition results from camera
    
    - **camera_id**: Camera identifier
    - **format**: Output format (json or sse for Server-Sent Events)
    """
    if camera_id not in camera_manager.camera_active:
        raise HTTPException(status_code=404, detail=f"Camera {camera_id} not found or not active")
    
    if format == "sse":
        async def generate_sse():
            while camera_manager.camera_active.get(camera_id, False):
                try:
                    results = camera_manager.get_latest_results(camera_id)
                    yield f"data: {json.dumps(results)}\n\n"
                    await asyncio.sleep(0.5)  # 2 FPS for streaming
                except Exception as e:
                    logger.error(f"SSE streaming error for camera {camera_id}: {e}")
                    yield f"data: {json.dumps({'error': str(e), 'camera_id': camera_id})}\n\n"
                    break
        
        return StreamingResponse(
            generate_sse(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )
    else:
        # Return current results
        results = camera_manager.get_latest_results(camera_id)
        return results

@router.get("/test/rtsp")
async def test_rtsp_connection(
    rtsp_url: str = Query(..., description="RTSP URL to test")
):
    """
    Test RTSP connection without starting recognition
    
    - **rtsp_url**: RTSP URL to test
    """
    try:
        cap = cv2.VideoCapture(rtsp_url)
        
        if not cap.isOpened():
            return {
                "success": False,
                "message": "Failed to connect to RTSP stream",
                "url": rtsp_url
            }
        
        # Try to read a frame
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            return {
                "success": False,
                "message": "Connected but failed to read frame",
                "url": rtsp_url
            }
        
        return {
            "success": True,
            "message": "RTSP connection successful",
            "url": rtsp_url,
            "frame_shape": frame.shape,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"RTSP test error: {e}")
        return {
            "success": False,
            "message": f"Connection error: {str(e)}",
            "url": rtsp_url
        }

@router.get("/test/local")
async def test_local_camera(
    camera_index: int = Query(0, description="Local camera index (usually 0)")
):
    """
    Test local camera connection
    
    - **camera_index**: Camera index (usually 0 for default camera)
    """
    try:
        cap = cv2.VideoCapture(camera_index)
        
        if not cap.isOpened():
            return {
                "success": False,
                "message": f"Failed to connect to local camera {camera_index}",
                "camera_index": camera_index
            }
        
        # Try to read a frame
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            return {
                "success": False,
                "message": f"Connected but failed to read frame from camera {camera_index}",
                "camera_index": camera_index
            }
        
        return {
            "success": True,
            "message": f"Local camera {camera_index} connection successful",
            "camera_index": camera_index,
            "frame_shape": frame.shape,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Local camera test error: {e}")
        return {
            "success": False,
            "message": f"Camera error: {str(e)}",
            "camera_index": camera_index
        }