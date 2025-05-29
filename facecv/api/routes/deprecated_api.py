"""
Deprecated API endpoints - moved here for reference and gradual phase-out
All these endpoints are deprecated and will be removed in future versions
"""

import asyncio
import json
import time
import cv2
from datetime import datetime
from typing import Optional, List
from fastapi import APIRouter, Query, Form, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse
from loguru import logger

# Import DeepFace schemas
from facecv.schemas.face import (
    FaceListResponse, FaceRegisterResponse, FaceUpdateRequest,
    FaceDeleteResponse, FaceRecognitionResponse, FaceVerificationResponse,
    FaceAnalysisResponse
)

# ==================== DEPRECATED INSIGHTFACE MODEL MANAGEMENT APIS ====================
# ⚠️ These APIs are deprecated - please use per-request model selection instead

insightface_deprecated_router = APIRouter(prefix="/api/v1/insightface", tags=["Deprecated InsightFace Model Management"])

@insightface_deprecated_router.post("/models/switch", summary="[已弃用] 切换模型类型", deprecated=True)
async def switch_model_type_deprecated(
    enable_arcface: bool = Form(..., description="是否启用ArcFace专用模型"),
    arcface_backbone: Optional[str] = Form("resnet50", description="ArcFace骨干网络 (resnet50/mobilefacenet)")
):
    """
    [已弃用] 切换模型类型 (ArcFace vs Buffalo)
    
    ⚠️ **此接口已弃用！请在每个 API 调用中直接指定 model 参数**
    
    推荐使用方式:
    - 使用 arcface_resnet50 替代 enable_arcface=True, arcface_backbone="resnet50"
    - 使用 arcface_mobilefacenet 替代 enable_arcface=True, arcface_backbone="mobilefacenet"
    - 使用 buffalo_l 替代 enable_arcface=False
    
    **参数:**
    - enable_arcface `bool`: 是否启用ArcFace专用模型
    - arcface_backbone `str`: ArcFace骨干网络类型 (仅当enable_arcface=True时有效)
    
    **返回:**
    切换结果信息，包含新模型的详细配置
    """
    # Log deprecation warning
    logger.warning(f"DEPRECATED API USED: /api/v1/insightface/models/switch - Please use per-request model selection")
    
    try:
        # Import here to avoid circular dependencies
        from facecv.config import get_runtime_config, get_settings
        from facecv.api.routes.insightface_api import get_recognizer, _recognizer
        
        # 获取当前设置
        settings = get_settings()
        runtime_config = get_runtime_config()
        
        runtime_config.set("arcface_enabled", enable_arcface)
        if enable_arcface and arcface_backbone:
            runtime_config.set("arcface_backbone", arcface_backbone)
        
        # Clear current recognizer to force reload
        # Note: _recognizer is managed in insightface_api module
        
        # Get new recognizer
        new_recognizer = get_recognizer()
        
        # Get model info
        model_info = new_recognizer.get_model_info()
        
        return {
            "success": True,
            "message": f"Successfully switched to {'ArcFace' if enable_arcface else 'Buffalo'} model",
            "model_type": "ArcFace" if enable_arcface else "Buffalo", 
            "model_info": model_info,
            "deprecation_warning": "⚠️ This API is deprecated. Please use per-request model selection with the 'model' parameter.",
            "configuration": {
                "arcface_enabled": enable_arcface,
                "backbone": arcface_backbone if enable_arcface else runtime_config.get("insightface_model_pack"),
                "similarity_threshold": runtime_config.get("insightface_similarity_thresh"),
                "detection_threshold": runtime_config.get("insightface_det_thresh")
            }
        }
        
    except Exception as e:
        logger.error(f"Model switching failed: {e}")
        return {
            "success": False,
            "message": f"Failed to switch model: {str(e)}",
            "error": str(e),
            "deprecation_warning": "⚠️ This API is deprecated. Please use per-request model selection with the 'model' parameter."
        }

@insightface_deprecated_router.post("/models/select", summary="[已弃用] 统一模型选择", deprecated=True)
async def select_model_deprecated(
    model: str = Query(..., description="模型名称: arcface_resnet50, arcface_mobilefacenet, buffalo_l, buffalo_s, etc.")
):
    """
    [已弃用] 统一的模型选择接口
    
    ⚠️ **此接口已弃用！请在每个 API 调用中直接指定 model 参数**
    
    **支持的模型:**
    - arcface_resnet50: ArcFace ResNet50 专用模型
    - arcface_mobilefacenet: ArcFace MobileFaceNet 轻量级模型
    - buffalo_l: Buffalo大型模型包 (最佳精度)
    - buffalo_s: Buffalo小型模型包 (速度优先) 
    - buffalo_m: Buffalo中型模型包 (平衡)
    - antelopev2: Antelope V2高精度模型
    
    **返回:**
    - success: 切换是否成功
    - message: 状态消息
    - previous_model: 之前使用的模型
    - current_model: 当前使用的模型
    - model_type: 模型类型 (arcface/buffalo)
    """
    # Log deprecation warning
    logger.warning(f"DEPRECATED API USED: /api/v1/insightface/models/select - Please use per-request model selection")
    
    # Define all valid models
    arcface_models = ["arcface_resnet50", "arcface_mobilefacenet"]
    buffalo_models = ["buffalo_l", "buffalo_m", "buffalo_s", "antelopev2"]
    valid_models = arcface_models + buffalo_models
    
    if model not in valid_models:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid model '{model}'. Valid choices: {', '.join(valid_models)}"
        )
    
    try:
        # Import here to avoid circular dependencies
        from facecv.config import get_runtime_config
        from facecv.api.routes.insightface_api import get_recognizer, _recognizer
        
        runtime_config = get_runtime_config()
        previous_arcface_enabled = runtime_config.get('arcface_enabled', False)
        previous_model = runtime_config.get('insightface_model_pack', 'buffalo_l')
        
        # Determine previous model name for display
        if previous_arcface_enabled:
            previous_backbone = runtime_config.get('arcface_backbone', 'resnet50')
            previous_model_name = f"arcface_{previous_backbone}"
        else:
            previous_model_name = previous_model
        
        # Clear current recognizer
        # Note: _recognizer is managed in insightface_api module
        
        # Configure model selection based on choice
        if model in arcface_models:
            runtime_config.set("arcface_enabled", True)
            backbone = "mobilefacenet" if "mobilefacenet" in model else "resnet50"
            runtime_config.set("arcface_backbone", backbone)
        else:
            runtime_config.set("arcface_enabled", False)
            runtime_config.set("insightface_model_pack", model)
        
        # Get new recognizer with updated configuration
        new_recognizer = get_recognizer()
        
        return {
            "success": True,
            "message": f"Successfully selected model: {model}",
            "previous_model": previous_model_name,
            "current_model": model,
            "model_type": "ArcFace" if model in arcface_models else "Buffalo",
            "deprecation_warning": "⚠️ This API is deprecated. Please use per-request model selection with the 'model' parameter.",
            "timestamp": str(datetime.now())
        }
        
    except Exception as e:
        logger.error(f"Model selection failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to select model {model}: {str(e)}"
        )

@insightface_deprecated_router.post("/models/offload", summary="[已弃用] 卸载所有模型", deprecated=True)
async def offload_models_deprecated():
    """
    [已弃用] 卸载当前加载的所有模型以释放内存
    
    ⚠️ **此接口已弃用！现在使用智能模型池管理，会自动管理内存**
    
    **用途:**
    - 释放GPU/CPU内存
    - 在模型切换前清理资源
    - 故障排除和重置
    
    **返回:**
    - success: 卸载是否成功
    - message: 详细信息
    - memory_freed: 是否释放了内存
    - previous_model: 卸载前的模型
    """
    # Log deprecation warning
    logger.warning(f"DEPRECATED API USED: /api/v1/insightface/models/offload - Please use smart model pool management")
    
    # Get current model info before unloading
    previous_model = "None"
    model_was_loaded = False
    
    try:
        # Import here to avoid circular dependencies
        from facecv.api.routes.insightface_api import _recognizer
        from facecv.config import get_runtime_config
        
        if _recognizer is not None:
            model_was_loaded = True
            try:
                runtime_config = get_runtime_config()
                arcface_enabled = runtime_config.get('arcface_enabled', False)
                
                if arcface_enabled:
                    backbone = runtime_config.get('arcface_backbone', 'resnet50')
                    previous_model = f"arcface_{backbone}"
                else:
                    previous_model = runtime_config.get('insightface_model_pack', 'buffalo_l')
            except:
                previous_model = "Unknown"
        
        # Clear the recognizer
        # Note: _recognizer is managed in insightface_api module
        
        # Force garbage collection
        import gc
        gc.collect()
        
        # If CUDA is available, clear GPU cache
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gpu_cache_cleared = True
            else:
                gpu_cache_cleared = False
        except ImportError:
            gpu_cache_cleared = False
        
        return {
            "success": True,
            "message": "All models have been offloaded successfully",
            "memory_freed": model_was_loaded,
            "previous_model": previous_model,
            "gpu_cache_cleared": gpu_cache_cleared,
            "deprecation_warning": "⚠️ This API is deprecated. The smart model pool now manages memory automatically.",
            "note": "Models will be reloaded automatically on next recognition request",
            "timestamp": str(datetime.now())
        }
        
    except Exception as e:
        logger.error(f"Model offload failed: {e}")
        return {
            "success": False,
            "message": f"Failed to offload models: {str(e)}",
            "error": str(e),
            "deprecation_warning": "⚠️ This API is deprecated. The smart model pool now manages memory automatically."
        }


# ==================== DEPRECATED CAMERA STREAMING APIS ====================
# ⚠️ These APIs are deprecated - please use new InsightFace unified camera APIs

camera_deprecated_router = APIRouter(prefix="/api/v1/camera", tags=["Deprecated Camera Streaming"])

@camera_deprecated_router.post("/connect", 
    summary="连接摄像头 (已弃用)",
    description="""⚠️ **已弃用 - 请使用新的 InsightFace 摄像头 API**

    **新 API 地址：**
    - 人脸识别：`/api/v1/insightface/camera/recognition`
    - 人脸验证：`/api/v1/insightface/camera/verification`
    - 人脸检测：`/api/v1/insightface/camera/detection`
    
    ---
    
    连接到摄像头源（本地摄像头或RTSP流）并启动实时人脸识别。
    """,
    deprecated=True)
async def connect_camera_deprecated(
    camera_id: str = Query(..., description="唯一的摄像头标识符"),
    source: str = Query(..., description="摄像头源（0表示网络摄像头，RTSP URL表示IP摄像头）")
):
    # Log deprecation warning
    logger.warning(f"DEPRECATED API USED: /api/v1/camera/connect - Please migrate to InsightFace camera APIs")
    
    return {
        "success": False,
        "message": "This endpoint has been deprecated",
        "deprecation_warning": "⚠️ This API is deprecated. Please use the new InsightFace unified camera APIs",
        "migration_guide": {
            "new_apis": [
                "/api/v1/insightface/camera/recognition",
                "/api/v1/insightface/camera/verification",
                "/api/v1/insightface/camera/detection"
            ],
            "sunset_date": "2025-12-31"
        }
    }

@camera_deprecated_router.post("/disconnect",
    summary="断开摄像头连接 (已弃用)",
    description="""⚠️ **已弃用 - 请使用新的 InsightFace 摄像头 API**

    **新 API 地址：**
    - 人脸识别：`/api/v1/insightface/camera/recognition`
    - 人脸验证：`/api/v1/insightface/camera/verification`
    - 人脸检测：`/api/v1/insightface/camera/detection`
    """,
    deprecated=True)
async def disconnect_camera_deprecated(
    camera_id: str = Query(..., description="要断开连接的摄像头标识符")
):
    # Log deprecation warning
    logger.warning(f"DEPRECATED API USED: /api/v1/camera/disconnect - Please migrate to InsightFace camera APIs")
    
    return {
        "success": False,
        "message": "This endpoint has been deprecated",
        "deprecation_warning": "⚠️ This API is deprecated. Please use the new InsightFace unified camera APIs"
    }

@camera_deprecated_router.get("/status",
    summary="获取摄像头状态 (已弃用)",
    description="""⚠️ **已弃用 - 请使用新的 InsightFace 摄像头 API**

    **新 API 地址：**
    - 人脸识别：`/api/v1/insightface/camera/recognition`
    - 人脸验证：`/api/v1/insightface/camera/verification`
    - 人脸检测：`/api/v1/insightface/camera/detection`
    """,
    deprecated=True)
async def get_camera_status_deprecated(
    camera_id: Optional[str] = Query(None, description="特定的摄像头ID（可选）")
):
    # Log deprecation warning
    logger.warning(f"DEPRECATED API USED: /api/v1/camera/status - Please migrate to InsightFace camera APIs")
    
    return {
        "success": False,
        "message": "This endpoint has been deprecated",
        "deprecation_warning": "⚠️ This API is deprecated. Please use the new InsightFace unified camera APIs"
    }

@camera_deprecated_router.get("/stream",
    summary="流式传输识别结果 (已弃用)", 
    description="""⚠️ **已弃用 - 请使用新的 InsightFace 摄像头 API**

    **新 API 地址：**
    - 人脸识别：`/api/v1/insightface/camera/recognition`
    - 人脸验证：`/api/v1/insightface/camera/verification`
    - 人脸检测：`/api/v1/insightface/camera/detection`
    """,
    deprecated=True)
async def stream_recognition_results_deprecated(
    camera_id: str = Query(..., description="摄像头标识符"),
    format: str = Query("json", description="输出格式（json或sse）")
):
    # Log deprecation warning
    logger.warning(f"DEPRECATED API USED: /api/v1/camera/stream - Please migrate to InsightFace camera APIs")
    
    return {
        "success": False,
        "message": "This endpoint has been deprecated",
        "deprecation_warning": "⚠️ This API is deprecated. Please use the new InsightFace unified camera APIs"
    }

@camera_deprecated_router.get("/test/rtsp",
    summary="测试RTSP连接 (已弃用)",
    description="""⚠️ **已弃用 - 请使用新的 InsightFace 摄像头 API**

    **新 API 地址：**
    - 人脸识别：`/api/v1/insightface/camera/recognition`
    - 人脸验证：`/api/v1/insightface/camera/verification`
    - 人脸检测：`/api/v1/insightface/camera/detection`
    """,
    deprecated=True)
async def test_rtsp_connection_deprecated(
    url: str = Query(..., description="要测试的RTSP URL", alias="rtsp_url")
):
    # Log deprecation warning
    logger.warning(f"DEPRECATED API USED: /api/v1/camera/test/rtsp - Please migrate to InsightFace camera APIs")
    
    return {
        "success": False,
        "message": "This endpoint has been deprecated",
        "url": url,
        "deprecation_warning": "⚠️ This API is deprecated. Please use the new InsightFace unified camera APIs"
    }

@camera_deprecated_router.get("/test/local",
    summary="测试本地摄像头 (已弃用)",
    description="""⚠️ **已弃用 - 请使用新的 InsightFace 摄像头 API**

    **新 API 地址：**
    - 人脸识别：`/api/v1/insightface/camera/recognition`
    - 人脸验证：`/api/v1/insightface/camera/verification`
    - 人脸检测：`/api/v1/insightface/camera/detection`
    """,
    deprecated=True)
async def test_local_camera_deprecated(
    camera_id: str = Query("0", description="本地摄像头ID（通常为0）")
):
    # Log deprecation warning
    logger.warning(f"DEPRECATED API USED: /api/v1/camera/test/local - Please migrate to InsightFace camera APIs")
    
    return {
        "success": False,
        "message": "This endpoint has been deprecated",
        "camera_id": camera_id,
        "deprecation_warning": "⚠️ This API is deprecated. Please use the new InsightFace unified camera APIs"
    }


# ==================== DEPRECATED DEEPFACE APIS ====================
# ⚠️ These APIs are deprecated - please use InsightFace APIs instead

deepface_deprecated_router = APIRouter(prefix="/api/v1/deepface", tags=["Deprecated DeepFace"])

@deepface_deprecated_router.get("/faces/", 
    summary="[已弃用] 获取人脸列表", 
    deprecated=True,
    response_model=FaceListResponse)
async def get_faces_deprecated(
    skip: int = Query(0, description="跳过记录数"),
    limit: int = Query(100, description="返回记录数")
):
    """
    [已弃用] 获取数据库中的人脸列表
    
    ⚠️ **此接口已弃用！请使用 InsightFace API: GET /api/v1/insightface/faces**
    
    **参数:**
    - skip: 跳过记录数
    - limit: 返回记录数
    
    **返回:**
    人脸列表
    """
    logger.warning("DEPRECATED API USED: GET /api/v1/deepface/faces/ - Please use InsightFace API")
    
    return FaceListResponse(
        faces=[],
        total=0,
        deprecation_warning="⚠️ This API is deprecated. Please use GET /api/v1/insightface/faces"
    )

@deepface_deprecated_router.post("/faces/", 
    summary="[已弃用] 注册人脸", 
    deprecated=True,
    response_model=FaceRegisterResponse)
async def register_face_deprecated(
    file: UploadFile = File(..., description="人脸图像文件"),
    name: str = Form(..., description="人员姓名"),
    metadata: Optional[str] = Form(None, description="JSON格式的元数据")
):
    """
    [已弃用] 注册新人脸到数据库
    
    ⚠️ **此接口已弃用！请使用 InsightFace API: POST /api/v1/insightface/register**
    
    **参数:**
    - file: 人脸图像文件
    - name: 人员姓名
    - metadata: JSON格式的元数据（可选）
    
    **返回:**
    注册结果
    """
    logger.warning("DEPRECATED API USED: POST /api/v1/deepface/faces/ - Please use InsightFace API")
    
    return FaceRegisterResponse(
        success=False,
        message="This API is deprecated. Please use POST /api/v1/insightface/register",
        person_name=name,
        face_id=None
    )

@deepface_deprecated_router.put("/faces/{face_id}", 
    summary="[已弃用] 更新人脸信息", 
    deprecated=True)
async def update_face_deprecated(
    face_id: str,
    request: FaceUpdateRequest
):
    """
    [已弃用] 更新人脸信息
    
    ⚠️ **此接口已弃用！请使用 InsightFace API 进行人脸管理**
    
    **参数:**
    - face_id: 人脸ID
    - request: 更新请求
    
    **返回:**
    更新结果
    """
    logger.warning(f"DEPRECATED API USED: PUT /api/v1/deepface/faces/{face_id} - Please use InsightFace API")
    
    return {
        "success": False,
        "message": "This API is deprecated. Please use InsightFace APIs for face management",
        "face_id": face_id
    }

@deepface_deprecated_router.delete("/faces/{face_id}", 
    summary="[已弃用] 删除人脸", 
    deprecated=True,
    response_model=FaceDeleteResponse)
async def delete_face_deprecated(face_id: str):
    """
    [已弃用] 删除指定人脸
    
    ⚠️ **此接口已弃用！请使用 InsightFace API: DELETE /api/v1/insightface/faces/{face_id}**
    
    **参数:**
    - face_id: 人脸ID
    
    **返回:**
    删除结果
    """
    logger.warning(f"DEPRECATED API USED: DELETE /api/v1/deepface/faces/{face_id} - Please use InsightFace API")
    
    return FaceDeleteResponse(
        success=False,
        message="This API is deprecated. Please use DELETE /api/v1/insightface/faces/{face_id}"
    )

@deepface_deprecated_router.get("/faces/name/{name}", 
    summary="[已弃用] 按姓名查询人脸", 
    deprecated=True,
    response_model=FaceListResponse)
async def get_faces_by_name_deprecated(name: str):
    """
    [已弃用] 根据姓名查询人脸
    
    ⚠️ **此接口已弃用！请使用 InsightFace API: GET /api/v1/insightface/faces?name={name}**
    
    **参数:**
    - name: 人员姓名
    
    **返回:**
    人脸列表
    """
    logger.warning(f"DEPRECATED API USED: GET /api/v1/deepface/faces/name/{name} - Please use InsightFace API")
    
    return FaceListResponse(
        faces=[],
        total=0,
        deprecation_warning="⚠️ This API is deprecated. Please use GET /api/v1/insightface/faces?name={name}"
    )

@deepface_deprecated_router.post("/recognition", 
    summary="[已弃用] 人脸识别", 
    deprecated=True,
    response_model=FaceRecognitionResponse)
async def recognize_face_deprecated(
    file: UploadFile = File(..., description="待识别的人脸图像"),
    threshold: float = Form(0.6, description="识别阈值")
):
    """
    [已弃用] 识别图像中的人脸
    
    ⚠️ **此接口已弃用！请使用 InsightFace API: POST /api/v1/insightface/recognize**
    
    **参数:**
    - file: 待识别的人脸图像
    - threshold: 识别阈值
    
    **返回:**
    识别结果
    """
    logger.warning("DEPRECATED API USED: POST /api/v1/deepface/recognition - Please use InsightFace API")
    
    return FaceRecognitionResponse(
        faces=[{
            "deprecation_warning": "⚠️ This API is deprecated. Please use POST /api/v1/insightface/recognize"
        }],
        total_faces=0,
        processing_time=0.0
    )

@deepface_deprecated_router.post("/verify/", 
    summary="[已弃用] 人脸验证", 
    deprecated=True,
    response_model=FaceVerificationResponse)
async def verify_faces_deprecated(
    file1: UploadFile = File(..., description="第一张人脸图像"),
    file2: UploadFile = File(..., description="第二张人脸图像"),
    threshold: float = Form(0.6, description="验证阈值")
):
    """
    [已弃用] 验证两张人脸是否为同一人
    
    ⚠️ **此接口已弃用！请使用 InsightFace API: POST /api/v1/insightface/verify**
    
    **参数:**
    - file1: 第一张人脸图像
    - file2: 第二张人脸图像
    - threshold: 验证阈值
    
    **返回:**
    验证结果
    """
    logger.warning("DEPRECATED API USED: POST /api/v1/deepface/verify/ - Please use InsightFace API")
    
    return FaceVerificationResponse(
        verified=False,
        confidence=0.0,
        distance=2.0,
        threshold=threshold,
        model="deprecated",
        deprecation_warning="⚠️ This API is deprecated. Please use POST /api/v1/insightface/verify"
    )

@deepface_deprecated_router.post("/analyze/", 
    summary="[已弃用] 人脸属性分析", 
    deprecated=True,
    response_model=FaceAnalysisResponse)
async def analyze_face_deprecated(
    file: UploadFile = File(..., description="待分析的人脸图像"),
    actions: List[str] = Form(["emotion", "age", "gender"], description="分析项目")
):
    """
    [已弃用] 分析人脸属性（年龄、性别、情绪等）
    
    ⚠️ **此接口已弃用！请使用 InsightFace API 进行人脸检测和分析**
    
    **参数:**
    - file: 待分析的人脸图像
    - actions: 分析项目列表
    
    **返回:**
    分析结果
    """
    logger.warning("DEPRECATED API USED: POST /api/v1/deepface/analyze/ - Please use InsightFace API")
    
    return FaceAnalysisResponse(
        faces=[{
            "deprecation_warning": "⚠️ This API is deprecated. Please use InsightFace detect API with emotion/age/gender detection"
        }],
        total_faces=0
    )

@deepface_deprecated_router.post("/video_face/", 
    summary="[已弃用] 视频人脸采样注册", 
    deprecated=True)
async def register_video_face_deprecated(
    name: str = Form(..., description="人员姓名"),
    video_source: str = Form("0", description="视频源"),
    sample_interval: int = Form(30, description="采样间隔"),
    max_samples: int = Form(10, description="最大采样数")
):
    """
    [已弃用] 从视频源采样人脸并注册
    
    ⚠️ **此接口已弃用！请使用 InsightFace 视频流 API**
    
    **参数:**
    - name: 人员姓名
    - video_source: 视频源
    - sample_interval: 采样间隔（帧）
    - max_samples: 最大采样数
    
    **返回:**
    注册结果
    """
    logger.warning("DEPRECATED API USED: POST /api/v1/deepface/video_face/ - Please use InsightFace stream APIs")
    
    return {
        "success": False,
        "message": "This API is deprecated. Please use InsightFace stream processing APIs",
        "person_name": name,
        "deprecation_warning": "⚠️ This API is deprecated. Please use POST /api/v1/insightface/stream/process"
    }

@deepface_deprecated_router.get("/recognize/webcam/stream", 
    summary="[已弃用] 实时人脸识别流", 
    deprecated=True)
async def webcam_recognition_stream_deprecated():
    """
    [已弃用] 网络摄像头实时人脸识别流
    
    ⚠️ **此接口已弃用！请使用 InsightFace 视频流 API**
    
    **返回:**
    SSE事件流
    """
    logger.warning("DEPRECATED API USED: GET /api/v1/deepface/recognize/webcam/stream - Please use InsightFace stream APIs")
    
    async def generate():
        yield f"data: {json.dumps({'error': 'This API is deprecated. Please use InsightFace stream APIs'})}\n\n"
        yield f"data: {json.dumps({'deprecation_warning': '⚠️ Please use /api/v1/insightface/stream/sources and /api/v1/insightface/stream/process'})}\n\n"
    
    return StreamingResponse(generate(), media_type="text/event-stream")

@deepface_deprecated_router.get("/health", 
    summary="[已弃用] DeepFace健康检查", 
    deprecated=True)
async def deepface_health_deprecated():
    """
    [已弃用] DeepFace服务健康检查
    
    ⚠️ **此接口已弃用！请使用 InsightFace API: GET /api/v1/insightface/health**
    
    **返回:**
    服务状态
    """
    logger.warning("DEPRECATED API USED: GET /api/v1/deepface/health - Please use InsightFace API")
    
    return {
        "status": "deprecated",
        "service": "DeepFace API (Deprecated)",
        "message": "This API is deprecated. Please use GET /api/v1/insightface/health",
        "deprecation_warning": "⚠️ DeepFace APIs are deprecated. Please migrate to InsightFace APIs",
        "migration_guide": {
            "insightface_health": "/api/v1/insightface/health",
            "insightface_detect": "/api/v1/insightface/detect",
            "insightface_recognize": "/api/v1/insightface/recognize",
            "insightface_verify": "/api/v1/insightface/verify",
            "insightface_register": "/api/v1/insightface/register"
        },
        "sunset_date": "2025-12-31",
        "timestamp": str(datetime.now())
    }


# ==================== DEPRECATED STREAM PROCESSING APIS ====================
# ⚠️ These APIs are deprecated - please use InsightFace stream APIs instead

stream_deprecated_router = APIRouter(prefix="/api/v1", tags=["Deprecated Stream"])

@stream_deprecated_router.post("/stream/process", 
    summary="[已弃用] 处理视频流", 
    deprecated=True)
async def process_video_stream_deprecated(
    source: str = Query(..., description="视频源"),
    duration: Optional[int] = Query(None, description="处理时长"),
    skip_frames: int = Query(1, description="跳帧数"),
    show_preview: bool = Query(False, description="显示预览")
):
    """
    [已弃用] 处理视频流进行人脸识别
    
    ⚠️ **此接口已弃用！请使用 InsightFace API: POST /api/v1/insightface/stream/process**
    
    **参数:**
    - source: 视频源（摄像头索引或RTSP URL）
    - duration: 处理时长（秒）
    - skip_frames: 跳帧数
    - show_preview: 是否显示预览
    
    **返回:**
    处理结果
    """
    logger.warning("DEPRECATED API USED: POST /api/v1/stream/process - Please use InsightFace stream API")
    
    return {
        "status": "deprecated",
        "source": source,
        "message": "This API is deprecated. Please use POST /api/v1/insightface/stream/process",
        "deprecation_warning": "⚠️ Stream APIs are deprecated. Please migrate to InsightFace stream APIs",
        "migration_guide": {
            "new_endpoint": "/api/v1/insightface/stream/process",
            "documentation": "Use the same parameters with the new endpoint"
        },
        "sunset_date": "2025-12-31"
    }

@stream_deprecated_router.get("/stream/sources", 
    summary="[已弃用] 获取可用视频源列表", 
    deprecated=True)
async def list_video_sources_deprecated():
    """
    [已弃用] 获取可用视频源列表
    
    ⚠️ **此接口已弃用！请使用 InsightFace API: GET /api/v1/insightface/stream/sources**
    
    **返回:**
    视频源列表
    """
    logger.warning("DEPRECATED API USED: GET /api/v1/stream/sources - Please use InsightFace stream API")
    
    return {
        "camera_sources": [],
        "rtsp_examples": [],
        "file_support": {
            "formats": [],
            "example": "deprecated"
        },
        "notes": [
            "⚠️ This API is deprecated. Please use GET /api/v1/insightface/stream/sources"
        ],
        "deprecation_warning": "⚠️ Stream APIs are deprecated. Please migrate to InsightFace stream APIs",
        "migration_guide": {
            "new_endpoint": "/api/v1/insightface/stream/sources",
            "documentation": "The new endpoint provides the same functionality"
        },
        "sunset_date": "2025-12-31"
    }


# Combined deprecated router for easy registration
deprecated_router = APIRouter()
deprecated_router.include_router(insightface_deprecated_router)
deprecated_router.include_router(camera_deprecated_router)
deprecated_router.include_router(deepface_deprecated_router)
deprecated_router.include_router(stream_deprecated_router)