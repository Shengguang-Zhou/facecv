"""InsightFace API Routes with Lazy Model Loading"""

from fastapi import APIRouter, File, UploadFile, HTTPException, Form, Query, BackgroundTasks, Depends
from typing import List, Optional, Union, Dict, Any
import numpy as np
from PIL import Image
import io
import logging
import cv2
import uuid
from datetime import datetime
import asyncio
import base64
import threading
import time

from facecv.models.lazy_model_pool import lazy_model_pool as model_pool
from facecv.schemas.face import (
    FaceDetection, VerificationResult, RecognitionResult, FaceRegisterResponse,
    StreamRecognitionRequest, StreamVerificationRequest, StreamProcessResponse,
    StreamWebhookPayload
)
from facecv.config import get_settings
from facecv.database.factory import get_default_database
from facecv.core.webhook import webhook_manager, WebhookConfig, send_recognition_event
from facecv.core.video_stream import VideoStreamProcessor, StreamConfig

router = APIRouter(tags=["InsightFace-Lazy"])
logger = logging.getLogger(__name__)

async def process_upload_file(file: UploadFile) -> np.ndarray:
    """Process uploaded image file"""
    logger.info(f"Processing uploaded file: {file.filename}, content_type: {file.content_type}")
    
    try:
        contents = await file.read()
        logger.info(f"Read file contents, size: {len(contents)} bytes")
        
        nparr = np.frombuffer(contents, np.uint8)
        image_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image_bgr is None:
            logger.error("Failed to decode image with OpenCV")
            raise ValueError("Failed to decode image with OpenCV")
            
        logger.info(f"Successfully decoded image with OpenCV, shape: {image_bgr.shape}")
        
        max_size = 1024
        h, w = image_bgr.shape[:2]
        if h > max_size or w > max_size:
            scale = min(max_size / h, max_size / w)
            new_h, new_w = int(h * scale), int(w * scale)
            logger.info(f"Resizing image from {h}x{w} to {new_h}x{new_w}")
            image_bgr = cv2.resize(image_bgr, (new_w, new_h))
            
        return image_bgr
        
    except Exception as e:
        import traceback
        logger.error(f"Error processing image: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=400, detail=f"Cannot process image: {str(e)}")

# ==================== Detection Endpoints ====================

@router.post("/detect", response_model=List[FaceDetection], summary="人脸检测 (支持SCRFD检测模型)")
async def detect_faces(
    file: UploadFile = File(..., description="待检测的图像文件"),
    model_name: str = Form("scrfd_10g", description="检测模型: scrfd_10g, scrfd_2.5g, scrfd_500m, buffalo_l/m/s"),
    min_confidence: float = Form(0.5, description="最低检测置信度阈值")
):
    """
    使用指定模型检测上传图像中的人脸
    
    **参数:**
    - file: 包含待检测人脸的图像文件
    - model_name: 使用的模型名称
    - min_confidence: 最低检测置信度阈值
    
    **检测模型说明 (SCRFD):**
    - scrfd_10g: SCRFD-10GF，最高精度检测，适合生产环境
    - scrfd_2.5g: SCRFD-2.5GF，平衡精度和速度
    - scrfd_500m: SCRFD-500MF，速度优先，适合移动端
    - scrfd_10g_bnkps: SCRFD-10G with keypoints
    - scrfd_2.5g_bnkps: SCRFD-2.5G with keypoints
    
    **完整模型包 (包含检测+识别):**
    - buffalo_l: 包含SCRFD-10GF检测器
    - buffalo_m: 包含SCRFD-2.5GF检测器
    - buffalo_s: 包含SCRFD-500MF检测器
    - antelopev2: 包含SCRFD-10G-BNKPS检测器
    """
    logger.info(f"Detect endpoint called with model: {model_name}, min_confidence: {min_confidence}")
    
    try:
        image = await process_upload_file(file)
        logger.info(f"Image processed successfully, shape: {image.shape}")
        
        # Get model from pool (lazy loading)
        recognizer = model_pool.get_model(model_name)
        logger.info(f"Using model: {model_name}")
        
        # Detect faces
        faces = recognizer.detect_faces(image)
        logger.info(f"Detected {len(faces)} faces")
        
        # Filter by confidence
        filtered_faces = [f for f in faces if f.confidence >= min_confidence]
        logger.info(f"Filtered to {len(filtered_faces)} faces with confidence >= {min_confidence}")
        
        # Format response
        response_faces = []
        for face in filtered_faces:
            response_face = {
                "bbox": face.bbox,
                "confidence": face.confidence,
                "id": getattr(face, 'id', str(uuid.uuid4())),
                "landmarks": getattr(face, 'landmarks', []),
                "quality_score": getattr(face, 'quality_score', 1.0),
                "name": "Unknown",  # Detection only, no recognition
                "similarity": 0.0
            }
            response_faces.append(response_face)
        
        return response_faces
        
    except Exception as e:
        import traceback
        logger.error(f"Error detecting faces: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")

# ==================== Verification Endpoints ====================

@router.post("/verify", response_model=VerificationResult, summary="人脸验证 (支持模型选择)")
async def verify_faces(
    file1: UploadFile = File(..., description="用于比较的第一张人脸图像"),
    file2: UploadFile = File(..., description="用于比较的第二张人脸图像"),
    model_name: str = Form("buffalo_l", description="模型名称: buffalo_l, buffalo_m, buffalo_s, antelopev2"),
    threshold: float = Query(0.4, description="验证判断的相似度阈值")
):
    """
    使用指定模型验证两张人脸是否为同一人
    
    **参数:**
    - file1: 第一张人脸图像文件
    - file2: 第二张人脸图像文件
    - model_name: 使用的模型名称
    - threshold: 验证判断的相似度阈值
    """
    logger.info(f"Verify endpoint called with model: {model_name}, threshold: {threshold}")
    
    try:
        image1 = await process_upload_file(file1)
        image2 = await process_upload_file(file2)
        
        # Get model from pool
        recognizer = model_pool.get_model(model_name)
        logger.info(f"Using model: {model_name}")
        
        # Verify faces
        result = recognizer.verify(image1=image1, image2=image2, threshold=threshold)
        
        logger.info(f"Verification result: {result.is_same_person} (confidence: {result.confidence:.3f})")
        return result
        
    except Exception as e:
        logger.error(f"Error verifying faces: {e}")
        raise HTTPException(status_code=500, detail=f"Verification failed: {str(e)}")

# ==================== Recognition Endpoints ====================

@router.post("/recognize", response_model=List[RecognitionResult], summary="人脸识别 (支持模型选择)")
async def recognize_faces(
    file: UploadFile = File(..., description="包含待识别人脸的图像"),
    model_name: str = Form("buffalo_l", description="模型名称: buffalo_l, buffalo_m, buffalo_s, antelopev2"),
    threshold: float = Form(0.35, description="识别匹配的相似度阈值")
):
    """
    使用指定模型识别上传图像中的人脸
    
    **参数:**
    - file: 包含待识别人脸的图像文件
    - model_name: 使用的模型名称
    - threshold: 识别匹配的相似度阈值
    
    **说明:**
    - 如果之前使用了相同的模型，将复用已加载的模型实例
    - 切换模型时，之前的模型会被自动卸载以节省内存
    """
    logger.info(f"Recognize endpoint called with model: {model_name}, threshold: {threshold}")
    
    try:
        image = await process_upload_file(file)
        logger.info(f"Image processed successfully, shape: {image.shape}")
        
        # Get model from pool (will reuse if same model)
        recognizer = model_pool.get_model(model_name)
        logger.info(f"Using model: {model_name}")
        
        # Recognize faces
        results = recognizer.recognize(image, threshold=threshold)
        logger.info(f"Recognized {len(results)} faces")
        
        # Format response
        response_results = []
        for result in results:
            # Convert to dict for easier manipulation
            result_dict = result.dict() if hasattr(result, 'dict') else result
            
            # Clean up fields
            if 'face_id' in result_dict and 'id' not in result_dict:
                result_dict['id'] = result_dict.pop('face_id')
            if 'person_id' in result_dict:
                result_dict.pop('person_id')
                
            # Ensure required fields
            result_dict.setdefault('name', 'Unknown')
            result_dict.setdefault('similarity', 0.0)
            result_dict.setdefault('quality_score', 1.0)
                
            response_results.append(result_dict)
            
        return response_results
        
    except Exception as e:
        import traceback
        logger.error(f"Error recognizing faces: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Recognition failed: {str(e)}")

@router.post("/register", response_model=FaceRegisterResponse, summary="人脸注册 (支持模型选择)")
async def register_face(
    file: UploadFile = File(..., description="包含待注册人脸的图像"),
    name: str = Form(..., description="注册人员的完整姓名"),
    model_name: str = Form("buffalo_l", description="模型名称: buffalo_l, buffalo_m, buffalo_s, antelopev2"),
    department: Optional[str] = Form(None, description="部门或组织单位"),
    employee_id: Optional[str] = Form(None, description="唯一的员工或人员标识符")
):
    """
    使用指定模型注册上传图像中的人脸
    
    **参数:**
    - file: 包含待注册人脸的图像文件
    - name: 注册人员的完整姓名
    - model_name: 使用的模型名称
    - department: 部门或组织单位 (可选)
    - employee_id: 唯一的员工或人员标识符 (可选)
    """
    logger.info(f"Register endpoint called with model: {model_name}, name: {name}")
    
    try:
        image = await process_upload_file(file)
        
        # Get model from pool
        recognizer = model_pool.get_model(model_name)
        logger.info(f"Using model: {model_name}")
        
        # Prepare metadata
        face_metadata = {}
        if department:
            face_metadata["department"] = department
        if employee_id:
            face_metadata["employee_id"] = employee_id
        
        # First detect faces to check how many are present
        faces = recognizer.detect_faces(image)
        
        if not faces:
            raise HTTPException(status_code=400, detail="No faces detected in the image")
        
        if len(faces) > 1:
            raise HTTPException(
                status_code=400, 
                detail=f"Multiple faces detected ({len(faces)} faces). Please provide an image with only one clear face for registration."
            )
        
        # Register the single face
        face_ids = recognizer.register(image=image, name=name, metadata=face_metadata)
        
        if not face_ids:
            raise HTTPException(status_code=400, detail="Failed to register face")
        
        logger.info(f"Successfully registered {len(face_ids)} face(s) for {name}")
        
        return FaceRegisterResponse(
            success=True,
            message=f"Successfully registered {len(face_ids)} face(s)",
            person_name=name,
            face_id=face_ids[0] if len(face_ids) == 1 else None,
            face_ids=face_ids if len(face_ids) > 1 else None,
            face_count=len(face_ids)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error registering face: {e}")
        raise HTTPException(status_code=500, detail=f"Registration failed: {str(e)}")

# ==================== Model Management Endpoints ====================

@router.get("/models/status", summary="获取模型池状态")
async def get_model_pool_status():
    """
    获取模型池的当前状态
    
    显示当前加载的模型、内存使用情况等信息。
    """
    current_model = model_pool._current_model_name
    loaded_models = list(model_pool._models.keys())
    
    # Get available models categorized
    detection_only_models = model_pool.get_available_models(detection_only=True)
    full_models = list(model_pool._model_configs.keys())
    
    return {
        "status": "active",
        "current_model": current_model,
        "loaded_models": loaded_models,
        "model_count": len(loaded_models),
        "lazy_loading_enabled": True,
        "auto_unload_enabled": True,
        "available_models": {
            "detection_only": detection_only_models,
            "full_packages": full_models,
            "all": model_pool.get_available_models()
        },
        "model_details": {
            "scrfd_10g": "SCRFD-10GF - Highest accuracy detection",
            "scrfd_2.5g": "SCRFD-2.5GF - Balanced detection", 
            "scrfd_500m": "SCRFD-500MF - Fast detection",
            "buffalo_l": "Full package with SCRFD-10GF + ResNet50",
            "buffalo_m": "Full package with SCRFD-2.5GF + MobileFaceNet",
            "buffalo_s": "Full package with SCRFD-500MF + MobileFaceNet",
            "antelopev2": "Full package with SCRFD-10G-BNKPS + GlinTR100"
        }
    }

@router.post("/models/preload", summary="预加载模型")
async def preload_model(model_name: str = Form(..., description="要预加载的模型名称")):
    """
    预加载指定的模型到内存中
    
    **参数:**
    - model_name: 模型名称 (buffalo_l/buffalo_m/buffalo_s/antelopev2)
    """
    try:
        # Load model into pool
        model = model_pool.get_model(model_name)
        
        return {
            "success": True,
            "message": f"Model {model_name} loaded successfully",
            "model_name": model_name,
            "current_models": list(model_pool._models.keys())
        }
        
    except Exception as e:
        logger.error(f"Error preloading model: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to preload model: {str(e)}")

@router.post("/models/clear", summary="清除所有模型")
async def clear_models():
    """
    清除模型池中的所有已加载模型，释放内存
    """
    try:
        model_pool.clear_all()
        
        return {
            "success": True,
            "message": "All models cleared from pool",
            "current_models": []
        }
        
    except Exception as e:
        logger.error(f"Error clearing models: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to clear models: {str(e)}")

# ==================== Database Endpoints ====================

@router.get("/faces", summary="获取人脸列表")
async def list_faces(
    name: Optional[str] = Query(None, description="按人员姓名过滤结果"),
    limit: int = Query(100, description="返回的最大人脸数量")
):
    """获取数据库中的人脸列表"""
    try:
        # Get database directly (no model needed)
        face_db = get_default_database()
        
        # Get faces from database
        if name:
            faces = face_db.search_by_name(name)
        else:
            faces = face_db.get_all_faces()
        
        # Limit results and clean data
        limited_faces = faces[:limit]
        
        # Clean faces data
        clean_faces = []
        for face in limited_faces:
            clean_face = {
                'id': face.get('id'),
                'name': face.get('name'),
                'metadata': face.get('metadata', {}),
                'created_at': face.get('created_at'),
                'updated_at': face.get('updated_at')
            }
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

@router.delete("/faces/{face_id}", summary="删除人脸")
async def delete_face(face_id: str):
    """根据ID删除人脸"""
    try:
        # Get any model to access database
        recognizer = model_pool.get_model()
        success = recognizer.delete(face_id=face_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="Face not found")
        
        return {"message": f"Successfully deleted face {face_id}"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting face: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete face: {str(e)}")

@router.delete("/faces/by-name/{name}", summary="按姓名删除人脸")
async def delete_faces_by_name(name: str):
    """删除某个人的所有人脸"""
    try:
        # Get any model to access database
        recognizer = model_pool.get_model()
        deleted_count = recognizer.delete(name=name)
        
        if deleted_count == 0:
            raise HTTPException(status_code=404, detail="No faces found for this name")
        
        return {
            "message": f"Successfully deleted all faces for {name}",
            "deleted_count": deleted_count
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting faces by name: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete faces: {str(e)}")

# ==================== Health Check ====================

@router.get("/health", summary="健康检查")
async def health_check():
    """InsightFace服务健康检查 (带延迟加载)"""
    try:
        # Don't load models during health check
        return {
            "status": "healthy",
            "service": "InsightFace API with Lazy Loading",
            "lazy_loading": True,
            "loaded_models": list(model_pool._models.keys()),
            "current_model": model_pool._current_model_name,
            "database_connected": True,  # Database is always available
            "timestamp": str(datetime.now())
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "service": "InsightFace API with Lazy Loading",
            "error": str(e),
            "timestamp": str(datetime.now())
        }

# ==================== Stream Processing Endpoints ====================

# Global dictionary to track active streams
_active_streams = {}

async def process_stream_with_webhook(
    stream_id: str,
    source: Union[int, str],
    webhook_url: str,
    recognizer,
    request_params: Dict[str, Any],
    event_type: str = "face_recognized"
):
    """Background task to process video stream and send results to webhook"""
    try:
        # Configure webhook
        webhook_config = WebhookConfig(
            url=webhook_url,
            timeout=30,
            retry_count=3,
            retry_delay=1,
            batch_size=10,
            batch_timeout=1.0
        )
        webhook_manager.add_webhook(stream_id, webhook_config)
        
        # Start webhook manager if not running
        if not webhook_manager.running:
            webhook_manager.start()
        
        # Configure stream processor
        config = StreamConfig(
            skip_frames=request_params.get('skip_frames', 1),
            show_preview=False,  # Never show preview in API mode
            enable_tracking=True
        )
        
        processor = VideoStreamProcessor(recognizer, config)
        
        # Open video source
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            error_msg = f"Cannot open video source: {source}"
            logger.error(error_msg)
            # Send error to webhook
            webhook_manager.send_event(stream_id, {
                "stream_id": stream_id,
                "event_type": "error",
                "error": error_msg,
                "camera_id": str(source),
                "timestamp": datetime.now().isoformat()
            })
            # Update stream status
            if stream_id in _active_streams:
                _active_streams[stream_id]["status"] = "error"
                _active_streams[stream_id]["error"] = error_msg
            raise Exception(error_msg)
        
        # Set camera properties for better performance
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer to get latest frames
        cap.set(cv2.CAP_PROP_FPS, 30)  # Set FPS
        
        # Try to read a test frame
        test_read_success = False
        for _ in range(5):
            ret, _ = cap.read()
            if ret:
                test_read_success = True
                break
            import time
            time.sleep(0.1)
        
        if not test_read_success:
            error_msg = f"Camera {source} opened but cannot read frames. Please check camera permissions or try unplugging and replugging the camera."
            logger.error(error_msg)
            cap.release()
            # Send error to webhook
            webhook_manager.send_event(stream_id, {
                "stream_id": stream_id,
                "event_type": "error",
                "error": error_msg,
                "camera_id": str(source),
                "timestamp": datetime.now().isoformat()
            })
            # Update stream status
            if stream_id in _active_streams:
                _active_streams[stream_id]["status"] = "error"
                _active_streams[stream_id]["error"] = error_msg
            raise Exception(error_msg)
        
        _active_streams[stream_id] = {
            "processor": processor,
            "cap": cap,
            "status": "processing"
        }
        
        frame_count = 0
        
        consecutive_failures = 0
        max_consecutive_failures = 10
        
        while stream_id in _active_streams:
            ret, frame = cap.read()
            if not ret:
                consecutive_failures += 1
                if consecutive_failures >= max_consecutive_failures:
                    error_msg = f"Failed to read {max_consecutive_failures} consecutive frames from camera {source}"
                    logger.error(error_msg)
                    webhook_manager.send_event(stream_id, {
                        "stream_id": stream_id,
                        "event_type": "error",
                        "error": error_msg,
                        "camera_id": str(source),
                        "timestamp": datetime.now().isoformat()
                    })
                    break
                time.sleep(0.1)  # Wait a bit before retrying
                continue
            
            # Reset failure counter on successful read
            consecutive_failures = 0
            
            # Skip frames
            if frame_count % request_params.get('skip_frames', 1) != 0:
                frame_count += 1
                continue
            
            # Process frame based on event type
            if event_type == "face_recognized":
                # Recognition mode
                results = recognizer.recognize(frame, threshold=request_params.get('threshold', 0.35))
                
                if results:
                    # Prepare webhook payload
                    faces_data = []
                    for result in results:
                        face_dict = {
                            "name": result.name,
                            "confidence": result.confidence,
                            "bbox": result.bbox,
                            "id": getattr(result, 'id', str(uuid.uuid4())),
                            "similarity": getattr(result, 'similarity', result.confidence)
                        }
                        faces_data.append(face_dict)
                    
                    # Encode frame if requested
                    frame_base64 = None
                    if request_params.get('return_frame', False):
                        # Draw bboxes if requested
                        if request_params.get('draw_bbox', True):
                            for result in results:
                                x1, y1, x2, y2 = result.bbox
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                label = f"{result.name}: {result.confidence:.2f}"
                                cv2.putText(frame, label, (x1, y1-10), 
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        
                        _, buffer = cv2.imencode('.jpg', frame)
                        frame_base64 = base64.b64encode(buffer).decode('utf-8')
                    
                    # Send to webhook
                    send_recognition_event(
                        camera_id=str(source),
                        recognized_faces=faces_data,
                        metadata={
                            "stream_id": stream_id,
                            "frame_count": frame_count,
                            "timestamp": datetime.now().isoformat(),
                            "frame_base64": frame_base64
                        }
                    )
                    
            elif event_type == "face_verified":
                # Verification mode
                target_name = request_params.get('target_name')
                verification_threshold = request_params.get('verification_threshold', 0.4)
                
                results = recognizer.recognize(frame, threshold=verification_threshold)
                
                verified_faces = []
                non_verified_faces = []
                
                for result in results:
                    if result.name == target_name and result.confidence >= verification_threshold:
                        verified_faces.append(result)
                    else:
                        non_verified_faces.append(result)
                
                # Send verification results
                if verified_faces or (non_verified_faces and request_params.get('alert_on_mismatch', False)):
                    faces_data = []
                    for result in (verified_faces + non_verified_faces):
                        face_dict = {
                            "name": result.name,
                            "confidence": result.confidence,
                            "bbox": result.bbox,
                            "verified": result.name == target_name and result.confidence >= verification_threshold,
                            "target_name": target_name
                        }
                        faces_data.append(face_dict)
                    
                    # Encode frame if requested
                    frame_base64 = None
                    if request_params.get('return_frame', False):
                        if request_params.get('draw_bbox', True):
                            for result in verified_faces:
                                x1, y1, x2, y2 = result.bbox
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                cv2.putText(frame, f"Verified: {target_name}", (x1, y1-10), 
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                            
                            for result in non_verified_faces:
                                x1, y1, x2, y2 = result.bbox
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                                cv2.putText(frame, f"Not {target_name}", (x1, y1-10), 
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                        
                        _, buffer = cv2.imencode('.jpg', frame)
                        frame_base64 = base64.b64encode(buffer).decode('utf-8')
                    
                    send_recognition_event(
                        camera_id=str(source),
                        recognized_faces=faces_data,
                        metadata={
                            "stream_id": stream_id,
                            "event_type": "face_verified",
                            "target_name": target_name,
                            "frame_count": frame_count,
                            "timestamp": datetime.now().isoformat(),
                            "frame_base64": frame_base64
                        }
                    )
            
            frame_count += 1
        
        # Cleanup
        cap.release()
        webhook_manager.remove_webhook(stream_id)
        if stream_id in _active_streams:
            _active_streams[stream_id]["status"] = "completed"
            del _active_streams[stream_id]
            
    except Exception as e:
        logger.error(f"Stream processing error: {e}")
        if stream_id in _active_streams:
            _active_streams[stream_id]["status"] = "error"
            del _active_streams[stream_id]
        webhook_manager.remove_webhook(stream_id)


@router.get(
    "/stream/process_recognition",
    response_model=StreamProcessResponse,
    summary="视频流人脸识别",
    description="处理视频流进行实时人脸识别并通过Webhook发送结果"
)
async def process_stream_recognition(
    background_tasks: BackgroundTasks,
    camera_id: Union[int, str] = Query(..., description="摄像头索引(0,1,2...)或RTSP URL"),
    webhook_url: str = Query(..., description="接收识别结果的Webhook URL"),
    skip_frames: int = Query(1, description="跳帧数，1=每帧处理，2=隔帧处理"),
    model: str = Query("buffalo_l", description="使用的模型(默认buffalo_l)"),
    use_scrfd: bool = Query(True, description="是否使用SCRFD检测器"),
    return_frame: bool = Query(False, description="是否在Webhook中返回处理后的帧图像"),
    draw_bbox: bool = Query(True, description="是否在返回帧上绘制边界框"),
    threshold: float = Query(0.35, description="识别阈值"),
    return_all_candidates: bool = Query(False, description="是否返回所有候选人"),
    max_candidates: int = Query(5, description="最大候选人数")
):
    """
    处理视频流进行实时人脸识别
    
    此接口启动一个后台任务来处理视频流，检测并识别人脸，
    然后通过Webhook URL实时发送识别结果。
    
    **参数:**
    - camera_id: 摄像头索引(0,1,2...)或RTSP URL
    - webhook_url: 接收识别结果的Webhook URL
    - skip_frames: 跳帧数，1=每帧处理，2=隔帧处理
    - model: 使用的模型(默认buffalo_l)
    - use_scrfd: 是否使用SCRFD检测器
    - return_frame: 是否在Webhook中返回处理后的帧图像
    - draw_bbox: 是否在返回帧上绘制边界框
    - threshold: 识别阈值
    - return_all_candidates: 是否返回所有候选人
    - max_candidates: 最大候选人数
    
    **Webhook数据格式:**
    ```json
    {
        "stream_id": "uuid",
        "timestamp": "2024-01-15T10:00:00",
        "camera_id": "0",
        "event_type": "face_recognized",
        "faces": [
            {
                "name": "张三",
                "confidence": 0.95,
                "bbox": [100, 100, 200, 200],
                "id": "face_id",
                "similarity": 0.95
            }
        ],
        "frame_base64": "base64_encoded_image_if_requested",
        "metadata": {
            "frame_count": 150,
            "stream_id": "uuid"
        }
    }
    ```
    """
    # Generate stream ID
    stream_id = str(uuid.uuid4())
    
    # Convert camera_id to proper format
    source = camera_id
    if isinstance(source, str) and source.isdigit():
        source = int(source)
    
    # Prepare request parameters
    request_params = {
        "skip_frames": skip_frames,
        "threshold": threshold,
        "return_frame": return_frame,
        "draw_bbox": draw_bbox,
        "return_all_candidates": return_all_candidates,
        "max_candidates": max_candidates
    }
    
    # Get recognizer instance from model pool
    recognizer = model_pool.get_model(model)
    
    # Start background processing
    background_tasks.add_task(
        process_stream_with_webhook,
        stream_id=stream_id,
        source=source,
        webhook_url=webhook_url,
        recognizer=recognizer,
        request_params=request_params,
        event_type="face_recognized"
    )
    
    return StreamProcessResponse(
        stream_id=stream_id,
        status="started",
        message=f"Stream processing started for camera {camera_id}",
        camera_id=camera_id,
        webhook_url=webhook_url,
        start_time=datetime.now().isoformat()
    )


@router.get(
    "/stream/process_verification",
    response_model=StreamProcessResponse,
    summary="视频流人脸验证",
    description="处理视频流进行特定人员的人脸验证并通过Webhook发送结果"
)
async def process_stream_verification(
    background_tasks: BackgroundTasks,
    camera_id: Union[int, str] = Query(..., description="摄像头索引(0,1,2...)或RTSP URL"),
    webhook_url: str = Query(..., description="接收验证结果的Webhook URL"),
    target_name: str = Query(..., description="目标人员姓名"),
    verification_threshold: float = Query(0.4, description="验证阈值"),
    alert_on_mismatch: bool = Query(False, description="不匹配时是否发送警报"),
    skip_frames: int = Query(1, description="跳帧数"),
    model: str = Query("buffalo_l", description="使用的模型"),
    use_scrfd: bool = Query(True, description="是否使用SCRFD检测器"),
    return_frame: bool = Query(False, description="是否返回处理后的帧图像"),
    draw_bbox: bool = Query(True, description="是否绘制边界框"),
    threshold: float = Query(0.35, description="识别阈值")
):
    """
    处理视频流进行特定人员的人脸验证
    
    此接口启动一个后台任务来处理视频流，验证是否为特定目标人员，
    并通过Webhook URL实时发送验证结果。
    
    **参数:**
    - camera_id: 摄像头索引(0,1,2...)或RTSP URL
    - webhook_url: 接收验证结果的Webhook URL
    - target_name: 目标人员姓名
    - verification_threshold: 验证阈值
    - alert_on_mismatch: 不匹配时是否发送警报
    - skip_frames: 跳帧数
    - return_frame: 是否返回处理后的帧图像
    - draw_bbox: 是否绘制边界框
    
    **Webhook数据格式:**
    ```json
    {
        "stream_id": "uuid",
        "timestamp": "2024-01-15T10:00:00",
        "camera_id": "0",
        "event_type": "face_verified",
        "faces": [
            {
                "name": "张三",
                "confidence": 0.95,
                "bbox": [100, 100, 200, 200],
                "verified": true,
                "target_name": "张三"
            }
        ],
        "frame_base64": "base64_encoded_image_if_requested",
        "metadata": {
            "target_name": "张三",
            "frame_count": 150
        }
    }
    ```
    """
    # Generate stream ID
    stream_id = str(uuid.uuid4())
    
    # Convert camera_id to proper format
    source = camera_id
    if isinstance(source, str) and source.isdigit():
        source = int(source)
    
    # Prepare request parameters
    request_params = {
        "skip_frames": skip_frames,
        "threshold": threshold,
        "return_frame": return_frame,
        "draw_bbox": draw_bbox,
        "target_name": target_name,
        "verification_threshold": verification_threshold,
        "alert_on_mismatch": alert_on_mismatch
    }
    
    # Get recognizer instance from model pool
    recognizer = model_pool.get_model(model)
    
    # Start background processing
    background_tasks.add_task(
        process_stream_with_webhook,
        stream_id=stream_id,
        source=source,
        webhook_url=webhook_url,
        recognizer=recognizer,
        request_params=request_params,
        event_type="face_verified"
    )
    
    return StreamProcessResponse(
        stream_id=stream_id,
        status="started",
        message=f"Stream verification started for camera {camera_id}, target: {target_name}",
        camera_id=camera_id,
        webhook_url=webhook_url,
        start_time=datetime.now().isoformat()
    )


@router.get("/stream/status/{stream_id}", summary="获取流处理状态")
async def get_stream_status(stream_id: str):
    """
    获取视频流处理状态
    
    **参数:**
    - stream_id: 流处理会话ID
    
    **返回:**
    流处理状态信息
    """
    if stream_id in _active_streams:
        response = {
            "stream_id": stream_id,
            "status": _active_streams[stream_id]["status"],
            "active": True
        }
        # Include error message if there's an error
        if "error" in _active_streams[stream_id]:
            response["error"] = _active_streams[stream_id]["error"]
        return response
    else:
        return {
            "stream_id": stream_id,
            "status": "not_found",
            "active": False
        }


@router.post("/stream/stop/{stream_id}", summary="停止流处理")
async def stop_stream(stream_id: str):
    """
    停止视频流处理
    
    **参数:**
    - stream_id: 流处理会话ID
    
    **返回:**
    停止操作结果
    """
    if stream_id in _active_streams:
        # Release resources
        if "cap" in _active_streams[stream_id]:
            _active_streams[stream_id]["cap"].release()
        
        # Remove from active streams
        del _active_streams[stream_id]
        
        # Remove webhook
        webhook_manager.remove_webhook(stream_id)
        
        return {
            "stream_id": stream_id,
            "status": "stopped",
            "message": "Stream processing stopped successfully"
        }
    else:
        raise HTTPException(status_code=404, detail="Stream not found")


@router.get("/stream/test_camera/{camera_id}", summary="测试摄像头")
async def test_camera(camera_id: int):
    """
    快速测试摄像头是否可用
    
    **参数:**
    - camera_id: 摄像头索引
    
    **返回:**
    摄像头测试结果
    """
    try:
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            return {
                "camera_id": camera_id,
                "status": "error",
                "message": "Cannot open camera"
            }
        
        # Try to read frames
        success_count = 0
        errors = []
        for i in range(5):
            ret, frame = cap.read()
            if ret:
                success_count += 1
            else:
                errors.append(f"Frame {i+1} failed")
            time.sleep(0.1)
        
        cap.release()
        
        return {
            "camera_id": camera_id,
            "status": "ok" if success_count > 0 else "error",
            "frames_read": f"{success_count}/5",
            "message": f"Successfully read {success_count} frames" if success_count > 0 else "Cannot read frames",
            "errors": errors
        }
        
    except Exception as e:
        return {
            "camera_id": camera_id,
            "status": "error",
            "message": str(e)
        }

@router.get("/stream/sources", summary="获取可用视频源")
async def get_video_sources():
    """
    获取系统可用的视频源列表
    
    **返回:**
    - cameras: 本地摄像头列表
    - sample_rtsp: RTSP URL 示例
    """
    cameras = []
    
    # 检测本地摄像头
    for i in range(5):  # 检查前5个摄像头
        try:
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                # Try to get camera properties
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = int(cap.get(cv2.CAP_PROP_FPS))
                
                cameras.append({
                    "index": i,
                    "name": f"摄像头 {i}",
                    "available": True,
                    "resolution": f"{width}x{height}",
                    "fps": fps
                })
                cap.release()
            else:
                # Camera index exists but can't open
                cameras.append({
                    "index": i,
                    "name": f"摄像头 {i}",
                    "available": False,
                    "error": "Cannot open camera"
                })
        except Exception as e:
            cameras.append({
                "index": i,
                "name": f"摄像头 {i}",
                "available": False,
                "error": str(e)
            })
    
    return {
        "cameras": cameras,
        "sample_rtsp": [
            "rtsp://username:password@192.168.1.100:554/stream1",
            "rtsp://192.168.1.100:8554/live.sdp"
        ],
        "sample_files": [
            "/path/to/video.mp4",
            "/path/to/video.avi"
        ]
    }