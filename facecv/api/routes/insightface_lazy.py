"""InsightFace API Routes with Lazy Model Loading"""

import asyncio
import base64
import io
import logging
import threading
import time
import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

import cv2
import numpy as np
from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    Form,
    HTTPException,
    Query,
    UploadFile,
)
from PIL import Image

from facecv.config import get_settings
from facecv.config.runtime_config import get_runtime_config

# Webhook functionality removed
from facecv.core.video_stream import StreamConfig, VideoStreamProcessor
from facecv.database.factory import get_default_database
from facecv.models.lazy_model_pool import lazy_model_pool as model_pool
from facecv.schemas.face import (
    FaceDetection,
    FaceRegisterResponse,
    RecognitionResult,
    VerificationResult,
)

router = APIRouter(tags=["InsightFace"])
logger = logging.getLogger(__name__)

# Optional FastRTC support
FASTRTC_AVAILABLE = False
try:
    # Temporarily disable proxy for FastRTC import
    import os

    old_proxies = {k: os.environ.get(k) for k in ["http_proxy", "https_proxy", "all_proxy"] if k in os.environ}
    for k in old_proxies:
        os.environ.pop(k, None)

    import gradio as gr
    from fastrtc import Stream, VideoStreamHandler

    FASTRTC_AVAILABLE = True

    # Restore proxies
    for k, v in old_proxies.items():
        if v:
            os.environ[k] = v
except (ImportError, ValueError) as e:
    logger.warning(f"FastRTC not available: {e}")
except Exception as e:
    logger.warning(f"FastRTC initialization error: {e}")

# Get runtime configuration
runtime_config = get_runtime_config()


# Model name enum for Swagger dropdown
class ModelName(str, Enum):
    """Available InsightFace models"""

    # Buffalo packages (full face analysis)
    BUFFALO_L = "buffalo_l"
    BUFFALO_M = "buffalo_m"
    BUFFALO_S = "buffalo_s"
    ANTELOPEV2 = "antelopev2"

    # SCRFD detection-only models
    SCRFD_10G_BNKPS = "scrfd_10g_bnkps"
    SCRFD_10G_KPS = "scrfd_10g_kps"
    SCRFD_2P5G_BNKPS = "scrfd_2.5g_bnkps"
    SCRFD_2P5G_KPS = "scrfd_2.5g_kps"
    SCRFD_500M_BNKPS = "scrfd_500m_bnkps"
    SCRFD_500M_KPS = "scrfd_500m_kps"


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

        # Get adaptive max size
        runtime_config = get_runtime_config()
        cuda_available = runtime_config.get("cuda_available", False)
        if cuda_available:
            max_size = 1920  # GPU can handle larger images
        else:
            # CPU mode - use smaller size
            import psutil

            ram_gb = psutil.virtual_memory().total / (1024**3)
            max_size = 1280 if ram_gb >= 8 else 960

        h, w = image_bgr.shape[:2]
        if h > max_size or w > max_size:
            scale = min(max_size / h, max_size / w)
            new_h, new_w = int(h * scale), int(w * scale)
            logger.info(f"Resizing image from {h}x{w} to {new_h}x{new_w} (adaptive max: {max_size})")
            image_bgr = cv2.resize(image_bgr, (new_w, new_h))

        return image_bgr

    except Exception as e:
        import traceback

        logger.error(f"Error processing image: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=400, detail=f"Cannot process image: {str(e)}")


# ==================== System Info Endpoints ====================


@router.get("/system/info", summary="获取系统配置信息")
async def get_system_info():
    """获取自适应系统配置信息"""
    runtime_config = get_runtime_config()

    return {
        "hardware": {
            "cuda_available": runtime_config.get("cuda_available", False),
            "cuda_version": runtime_config.get("cuda_version"),
            "execution_providers": runtime_config.get("execution_providers", ["CPUExecutionProvider"]),
        },
        "configuration": {
            "model": runtime_config.get("insightface_model_pack"),
            "detection_size": runtime_config.get("insightface_det_size"),
            "detection_threshold": runtime_config.get("insightface_det_thresh"),
            "max_faces": runtime_config.get("max_faces_per_image"),
            "enable_emotion": runtime_config.get("enable_emotion"),
            "enable_mask": runtime_config.get("enable_mask"),
        },
        "features": {"fastrtc_available": FASTRTC_AVAILABLE, "adaptive_mode": True},
    }


@router.get("/stream/fastrtc/available", summary="检查 FastRTC 可用性")
async def check_fastrtc():
    """检查 FastRTC WebRTC 支持是否可用"""
    return {
        "available": FASTRTC_AVAILABLE,
        "message": (
            "FastRTC is available for WebRTC streaming"
            if FASTRTC_AVAILABLE
            else "FastRTC not installed, install with: pip install fastrtc gradio"
        ),
    }


# ==================== Detection Endpoints ====================


@router.post(
    "/detect",
    response_model=List[FaceDetection],
    summary="人脸检测 (支持SCRFD检测模型)",
)
async def detect_faces(
    file: UploadFile = File(..., description="待检测的图像文件"),
    model_name: ModelName = Form(ModelName.SCRFD_10G_BNKPS, description="检测模型"),
    min_confidence: float = Form(0.5, description="最低检测置信度阈值"),
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
        recognizer = model_pool.get_model(model_name.value)
        logger.info(f"Using model: {model_name.value}")

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
                "id": getattr(face, "id", str(uuid.uuid4())),
                "landmarks": getattr(face, "landmarks", []),
                "quality_score": getattr(face, "quality_score", 1.0),
                "name": "Unknown",  # Detection only, no recognition
                "similarity": 0.0,
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
    model_name: ModelName = Form(ModelName.BUFFALO_L, description="模型名称"),
    threshold: float = Query(0.4, description="验证判断的相似度阈值"),
):
    """
    使用指定模型验证两张人脸是否为同一人

    **参数:**
    - file1: 第一张人脸图像文件
    - file2: 第二张人脸图像文件
    - model_name: 使用的模型名称
    - threshold: 验证判断的相似度阈值
    """
    logger.info(f"Verify endpoint called with model: {model_name.value}, threshold: {threshold}")

    try:
        image1 = await process_upload_file(file1)
        image2 = await process_upload_file(file2)

        # Get model from pool
        recognizer = model_pool.get_model(model_name.value)
        logger.info(f"Using model: {model_name.value}")

        # Verify faces
        result = recognizer.verify(image1=image1, image2=image2, threshold=threshold)

        logger.info(f"Verification result: {result.is_same_person} (confidence: {result.confidence:.3f})")
        return result

    except Exception as e:
        logger.error(f"Error verifying faces: {e}")
        raise HTTPException(status_code=500, detail=f"Verification failed: {str(e)}")


# ==================== Recognition Endpoints ====================


@router.post(
    "/recognize",
    response_model=List[RecognitionResult],
    summary="人脸识别 (支持模型选择)",
)
async def recognize_faces(
    file: UploadFile = File(..., description="包含待识别人脸的图像"),
    model_name: ModelName = Form(ModelName.BUFFALO_L, description="模型名称"),
    threshold: float = Form(0.35, description="识别匹配的相似度阈值"),
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
    logger.info(f"Recognize endpoint called with model: {model_name.value}, threshold: {threshold}")

    try:
        image = await process_upload_file(file)
        logger.info(f"Image processed successfully, shape: {image.shape}")

        # Get model from pool (will reuse if same model)
        recognizer = model_pool.get_model(model_name.value)
        logger.info(f"Using model: {model_name.value}")

        # Recognize faces
        results = recognizer.recognize(image, threshold=threshold)
        logger.info(f"Recognized {len(results)} faces")

        # Format response
        response_results = []
        for result in results:
            # Convert to dict for easier manipulation
            result_dict = result.dict() if hasattr(result, "dict") else result

            # Clean up fields
            if "face_id" in result_dict and "id" not in result_dict:
                result_dict["id"] = result_dict.pop("face_id")
            if "person_id" in result_dict:
                result_dict.pop("person_id")

            # Ensure required fields
            result_dict.setdefault("name", "Unknown")
            result_dict.setdefault("similarity", 0.0)
            result_dict.setdefault("quality_score", 1.0)

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
    model_name: ModelName = Form(ModelName.BUFFALO_L, description="模型名称"),
    department: Optional[str] = Form(None, description="部门或组织单位"),
    employee_id: Optional[str] = Form(None, description="唯一的员工或人员标识符"),
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
    logger.info(f"Register endpoint called with model: {model_name.value}, name: {name}")

    try:
        image = await process_upload_file(file)

        # Get model from pool
        recognizer = model_pool.get_model(model_name.value)
        logger.info(f"Using model: {model_name.value}")

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
                detail=f"Multiple faces detected ({len(faces)} faces). Please provide an image with only one clear face for registration.",
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
            face_count=len(face_ids),
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
            "all": model_pool.get_available_models(),
        },
        "model_details": {
            "scrfd_10g": "SCRFD-10GF - Highest accuracy detection",
            "scrfd_2.5g": "SCRFD-2.5GF - Balanced detection",
            "scrfd_500m": "SCRFD-500MF - Fast detection",
            "buffalo_l": "Full package with SCRFD-10GF + ResNet50",
            "buffalo_m": "Full package with SCRFD-2.5GF + MobileFaceNet",
            "buffalo_s": "Full package with SCRFD-500MF + MobileFaceNet",
            "antelopev2": "Full package with SCRFD-10G-BNKPS + GlinTR100",
        },
    }


@router.post("/models/preload", summary="预加载模型")
async def preload_model(model_names: List[ModelName] = Form(..., description="要预加载的模型名称列表")):
    """
    预加载指定的模型到内存中。用于提前加载模型以避免首次请求时的延迟。

    **用途:**
    - 在系统启动后预加载常用模型
    - 在业务高峰前预热模型缓存
    - 批量加载多个模型以支持不同精度需求

    **参数:**
    - model_names: 模型名称列表 (buffalo_l/buffalo_m/buffalo_s/antelopev2/scrfd_*)

    **示例:**
    ```
    model_names=buffalo_l&model_names=scrfd_10g_bnkps
    ```
    """
    try:
        loaded_models = []
        failed_models = []

        for model_name in model_names:
            try:
                # Load model into pool
                model = model_pool.get_model(model_name.value)
                loaded_models.append(model_name.value)
                logger.info(f"Successfully preloaded model: {model_name.value}")
            except Exception as e:
                logger.error(f"Failed to preload model {model_name.value}: {e}")
                failed_models.append({"model": model_name.value, "error": str(e)})

        return {
            "success": len(failed_models) == 0,
            "message": f"Loaded {len(loaded_models)} models, {len(failed_models)} failed",
            "loaded_models": loaded_models,
            "failed_models": failed_models,
            "current_models": list(model_pool._models.keys()),
        }

    except Exception as e:
        logger.error(f"Error in preload operation: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to preload models: {str(e)}")


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
            "current_models": [],
        }

    except Exception as e:
        logger.error(f"Error clearing models: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to clear models: {str(e)}")


# ==================== Database Endpoints ====================


@router.get("/faces", summary="获取人脸列表")
async def list_faces(
    name: Optional[str] = Query(None, description="按人员姓名过滤结果"),
    limit: int = Query(100, description="返回的最大人脸数量"),
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
                "id": face.get("id"),
                "name": face.get("name"),
                "metadata": face.get("metadata", {}),
                "created_at": face.get("created_at"),
                "updated_at": face.get("updated_at"),
            }
            if "embedding" in face and face["embedding"] is not None:
                if hasattr(face["embedding"], "tolist"):
                    clean_face["embedding_size"] = len(face["embedding"])
                else:
                    clean_face["embedding_size"] = 0
            clean_faces.append(clean_face)

        return {
            "faces": clean_faces,
            "total": len(faces),
            "returned": len(limited_faces),
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
            "deleted_count": deleted_count,
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
            "timestamp": str(datetime.now()),
        }

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "service": "InsightFace API with Lazy Loading",
            "error": str(e),
            "timestamp": str(datetime.now()),
        }


# ==================== Stream Processing Endpoints ====================


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

                cameras.append(
                    {
                        "index": i,
                        "name": f"摄像头 {i}",
                        "available": True,
                        "resolution": f"{width}x{height}",
                        "fps": fps,
                    }
                )
                cap.release()
            else:
                # Camera index exists but can't open
                cameras.append(
                    {
                        "index": i,
                        "name": f"摄像头 {i}",
                        "available": False,
                        "error": "Cannot open camera",
                    }
                )
        except Exception as e:
            cameras.append({"index": i, "name": f"摄像头 {i}", "available": False, "error": str(e)})

    return {
        "cameras": cameras,
        "sample_rtsp": [
            "rtsp://username:password@192.168.1.100:554/stream1",
            "rtsp://192.168.1.100:8554/live.sdp",
        ],
        "sample_files": ["/path/to/video.mp4", "/path/to/video.avi"],
    }
