"""Real InsightFace API Routes (Non-Mock)"""

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

from facecv.models.insightface.onnx_recognizer import ONNXFaceRecognizer
from facecv.models.insightface.real_recognizer import RealInsightFaceRecognizer
from facecv.models.insightface.arcface_recognizer import ArcFaceRecognizer
from facecv.schemas.face import (
    FaceDetection, VerificationResult, RecognitionResult, FaceRegisterResponse,
    StreamRecognitionRequest, StreamVerificationRequest, StreamProcessResponse,
    StreamWebhookPayload
)
from facecv.config import get_settings
from facecv.database.factory import get_default_database
from facecv.core.webhook import webhook_manager, WebhookConfig, send_recognition_event
from facecv.core.video_stream import VideoStreamProcessor, StreamConfig

router = APIRouter(tags=["InsightFace"])
logger = logging.getLogger(__name__)

# Global instance - cache recognizer to avoid reloading
_recognizer = None
_recognizer_cache = {}  # Cache for different model configurations

def get_recognizer():
    """
    获取InsightFace识别器实例，支持ArcFace和Buffalo模型
    
    支持的模型类型:
    1. ArcFace专用模型 (arcface_enabled=True):
       - buffalo_l_resnet50: ResNet50 ArcFace (生产推荐)
       - buffalo_s_mobilefacenet: MobileFaceNet (移动端)
    
    2. Buffalo模型包 (arcface_enabled=False):
       - buffalo_l: 最佳精度，适合生产环境 (推荐)
       - buffalo_m: 平衡精度和速度
       - buffalo_s: 速度优先，较低精度
       - antelopev2: 高精度模型
    
    Returns:
        Union[ArcFaceRecognizer, RealInsightFaceRecognizer]: 配置好的识别器实例
    """
    global _recognizer
    if _recognizer is not None:
        logger.info("Returning cached recognizer instance")
        return _recognizer
        
    # Create new recognizer only if not cached
    logger.info("Creating new recognizer instance (not cached)")
    from facecv.config import get_settings, get_runtime_config
    
    settings = get_settings()
    runtime_config = get_runtime_config()
    
    # Use standardized database configuration
    face_db = get_default_database()
    
    arcface_enabled = runtime_config.get('arcface_enabled', 
                                       getattr(settings, 'arcface_enabled', False))
    
    # Check if ArcFace is enabled
    if arcface_enabled:
        # Use dedicated ArcFace recognizer
        logger.info("Initializing dedicated ArcFace recognizer...")
        
        det_size = tuple(runtime_config.get('insightface_det_size', 
                                          getattr(settings, 'insightface_det_size', [640, 640])))
        det_thresh = runtime_config.get('insightface_det_thresh', 
                                      getattr(settings, 'insightface_det_thresh', 0.5))
        similarity_threshold = runtime_config.get('insightface_similarity_thresh', 
                                                getattr(settings, 'insightface_similarity_thresh', 0.35))
        
        # Auto-select ArcFace model based on backbone preference
        backbone = runtime_config.get('arcface_backbone', 
                                    getattr(settings, 'arcface_backbone', 'resnet50'))
        
        if backbone == 'mobilefacenet':
            model_name = 'buffalo_s_mobilefacenet'
        else:
            model_name = 'buffalo_l_resnet50'  # Default to ResNet50
        
        _recognizer = ArcFaceRecognizer(
            face_db=face_db,
            model_name=model_name,
            similarity_threshold=similarity_threshold,
            det_thresh=det_thresh,
            det_size=det_size,
            enable_detection=True
        )
        
        logger.info(f"ArcFace recognizer initialized:")
        logger.info(f"  Model: {model_name}")
        logger.info(f"  Backbone: {backbone}")
        logger.info(f"  Dataset: {runtime_config.get('arcface_dataset', getattr(settings, 'arcface_dataset', 'webface600k'))}")
        logger.info(f"  Embedding Size: {runtime_config.get('arcface_embedding_size', getattr(settings, 'arcface_embedding_size', 512))}")
        
    else:
        # Use traditional buffalo models
        logger.info("Initializing Real InsightFace recognizer...")
        
        model_pack = runtime_config.get('insightface_model_pack', 
                                      getattr(settings, 'insightface_model_pack', 'buffalo_l'))
        det_size = tuple(runtime_config.get('insightface_det_size', 
                                              getattr(settings, 'insightface_det_size', [640, 640])))
        det_thresh = runtime_config.get('insightface_det_thresh', 
                                      getattr(settings, 'insightface_det_thresh', 0.5))
        similarity_threshold = runtime_config.get('insightface_similarity_thresh', 
                                                getattr(settings, 'insightface_similarity_thresh', 0.35))
        enable_emotion = runtime_config.get('enable_emotion', 
                                          getattr(settings, 'insightface_enable_emotion', True))
        enable_mask = runtime_config.get('enable_mask', 
                                       getattr(settings, 'insightface_enable_mask', True))
        prefer_gpu = runtime_config.get('prefer_gpu', 
                                      getattr(settings, 'insightface_prefer_gpu', True))
        
        # Use Real InsightFace recognizer with configurable parameters
        _recognizer = RealInsightFaceRecognizer(
                face_db=face_db,
                model_pack=model_pack,
                similarity_threshold=similarity_threshold,
                det_thresh=det_thresh,
                det_size=det_size,
                enable_emotion=enable_emotion,
                enable_mask_detection=enable_mask,
                prefer_gpu=prefer_gpu
            )
            
        logger.info(f"  Detection Size: {det_size}")
        logger.info(f"  Detection Threshold: {det_thresh}")
        logger.info(f"  Similarity Threshold: {similarity_threshold}")
        logger.info(f"  GPU Acceleration: {prefer_gpu}")
        logger.info(f"  Emotion Recognition: {enable_emotion}")
        logger.info(f"  Mask Detection: {enable_mask}")
    
    logger.info(f"  Database: {face_db.__class__.__name__}")
    logger.info(f"  Recognizer Type: {'ArcFace' if arcface_enabled else 'RealInsightFace'}")
    return _recognizer

# ==================== Model Management Endpoints ====================
@router.get("/models/available", summary="获取可用模型列表")
async def get_available_models():
    """
    获取可用的InsightFace模型包列表及其特性说明
    
    返回支持的所有模型包信息，包括精度、速度、大小等特性对比，
    帮助用户根据需求选择最适合的模型。
    
    **返回:**
    模型列表，包含每个模型的详细信息:
    - name: 模型名称
    - description: 模型描述
    - accuracy: 精度等级
    - speed: 速度等级  
    - size: 模型大小
    - recommended_use: 推荐使用场景
    """
    models = {
        "buffalo_l": {
            "name": "buffalo_l",
            "description": "Buffalo-L 大型模型包 - 最佳精度，生产环境推荐",
            "accuracy": "最高 (★★★★★)",
            "speed": "中等 (★★★☆☆)",
            "size": "大 (~1.5GB)",
            "models_included": ["SCRFD-10GF", "ResNet100", "2d106det", "genderage"],
            "recommended_use": "生产环境、高精度要求、服务器部署",
            "pros": ["最高识别精度", "完整功能支持", "生产稳定"],
            "cons": ["模型较大", "加载时间长", "内存占用高"]
        },
        "buffalo_m": {
            "name": "buffalo_m", 
            "description": "Buffalo-M 中型模型包 - 精度与速度平衡",
            "accuracy": "高 (★★★★☆)",
            "speed": "快 (★★★★☆)",
            "size": "中 (~800MB)",
            "models_included": ["SCRFD-2.5GF", "ResNet50", "2d106det", "genderage"],
            "recommended_use": "边缘设备、实时应用、平衡性能",
            "pros": ["良好精度", "适中速度", "适合边缘部署"],
            "cons": ["精度略低于buffalo_l", "功能可能受限"]
        },
        "buffalo_s": {
            "name": "buffalo_s",
            "description": "Buffalo-S 小型模型包 - 速度优先，移动端适用",
            "accuracy": "中等 (★★★☆☆)", 
            "speed": "最快 (★★★★★)",
            "size": "小 (~300MB)",
            "models_included": ["SCRFD-500MF", "MobileFaceNet", "2d106det"],
            "recommended_use": "移动设备、实时处理、资源受限环境",
            "pros": ["启动快速", "内存占用小", "适合移动端"],
            "cons": ["精度较低", "功能较少", "复杂场景表现差"]
        },
        "antelopev2": {
            "name": "antelopev2",
            "description": "Antelope-V2 高精度模型包 - 研究和高要求场景",
            "accuracy": "最高 (★★★★★)",
            "speed": "慢 (★★☆☆☆)", 
            "size": "最大 (~2GB+)",
            "models_included": ["SCRFD-34GF", "ResNet100+", "高精度检测器"],
            "recommended_use": "研究环境、高精度需求、离线处理",
            "pros": ["极高精度", "最新算法", "研究友好"],
            "cons": ["模型巨大", "速度最慢", "资源需求高"]
        }
    }
    
    # 添加ArcFace专用模型信息
    settings = get_settings()
    arcface_models = {}
    
    if getattr(settings, 'arcface_enabled', False):
        from facecv.models.insightface.arcface_models import ArcFaceModelLoader
        try:
            loader = ArcFaceModelLoader()
            available_arcface = loader.list_available_models()
            
            for model_name in available_arcface:
                model_info = loader.get_model_info(model_name)
                if model_info:
                    arcface_models[model_name] = {
                        "name": model_name,
                        "type": "ArcFace",
                        "description": f"ArcFace {model_info.backbone} - {model_info.dataset}数据集",
                        "backbone": model_info.backbone,
                        "dataset": model_info.dataset,
                        "embedding_size": f"{model_info.embedding_size}D",
                        "accuracy": "高 (★★★★☆)" if model_info.backbone == "mobilefacenet" else "极高 (★★★★★)",
                        "speed": "快 (★★★★☆)" if model_info.backbone == "mobilefacenet" else "中等 (★★★☆☆)",
                        "recommended_use": "移动端、边缘计算" if model_info.backbone == "mobilefacenet" else "生产环境、高精度识别"
                    }
        except Exception as e:
            logger.warning(f"Failed to load ArcFace models: {e}")
    
    current_model = getattr(settings, 'insightface_model_pack', 'buffalo_l')
    
    return {
        "available_models": models,
        "arcface_models": arcface_models,
        "current_model": current_model,
        "arcface_enabled": getattr(settings, 'arcface_enabled', False),
        "recommendation": {
            "production": "buffalo_l - 生产环境首选，精度和稳定性最佳",
            "development": "buffalo_m - 开发测试阶段，速度与精度平衡",
            "mobile": "buffalo_s - 移动端或资源受限环境", 
            "research": "antelopev2 - 研究和极高精度要求"
        },
        "selection_guide": {
            "高精度要求": "buffalo_l 或 antelopev2",
            "实时处理": "buffalo_m 或 buffalo_s",
            "资源受限": "buffalo_s",
            "生产部署": "buffalo_l",
            "开发测试": "buffalo_m"
        }
    }

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

@router.post("/detect", response_model=List[FaceDetection], summary="人脸检测")
async def detect_faces(
    file: UploadFile = File(..., description="待检测的图像文件"),
    model: str = Form("buffalo_s", description="人脸检测模型"),
    min_confidence: float = Form(0.5, description="最低检测置信度阈值")
):
    """
    使用真实人脸检测模型检测上传图像中的人脸
    
    此接口使用InsightFace深度学习模型分析上传的图像以检测人脸。
    返回每个检测到的人脸的边界框坐标、置信度分数和面部关键点。
    
    **参数:**
    - file `UploadFile`: 包含待检测人脸的图像文件 (JPG, PNG)
    - model `str`: 使用的模型名称 (默认: buffalo_s)
    - min_confidence `float`: 最低检测置信度阈值 (0.0-1.0, 默认: 0.5)
    
    **返回:**
    FaceDetection对象列表，包含:
    - bbox `List[int]`: 边界框坐标 [x1, y1, x2, y2]
    - confidence `float`: 检测置信度分数 (0.0-1.0)
    - id `str`: 检测到的人脸唯一标识符 (MySQL UUID)
    - landmarks `List[List[float]]`: 面部关键点坐标 (可选)
    - quality_score `float`: 人脸质量评估分数 (可选)
    - name `str`: 识别到的人员姓名 (如未识别则为"Unknown")
    - similarity `float`: 相似度分数 (0.0-1.0)
    """
    logger.info(f"Detect endpoint called with model: {model}, min_confidence: {min_confidence}")
    
    try:
        image = await process_upload_file(file)
        logger.info(f"Image processed successfully, shape: {image.shape}")
        
        max_size = 320
        h, w = image.shape[:2]
        if h > max_size or w > max_size:
            scale = min(max_size / h, max_size / w)
            new_h, new_w = int(h * scale), int(w * scale)
            logger.info(f"Resizing image from {h}x{w} to {new_h}x{new_w}")
            image = cv2.resize(image, (new_w, new_h))
        
        # Get the cached recognizer
        recognizer = get_recognizer()
        logger.info(f"Using cached recognizer: {type(recognizer).__name__}")
        
        faces = recognizer.detect_faces(image)
        logger.info(f"Detected {len(faces)} faces")
        
        # Filter by confidence
        filtered_faces = [f for f in faces if f.confidence >= min_confidence]
        logger.info(f"Filtered to {len(filtered_faces)} faces with confidence >= {min_confidence}")
        
        # Ensure all required fields are set for detection (no recognition)
        for face in filtered_faces:
            if not hasattr(face, 'id') or not face.id:
                face.id = str(uuid.uuid4())
            if not hasattr(face, 'quality_score') or face.quality_score is None:
                face.quality_score = 1.0
        
        response_faces = []
        for face in filtered_faces:
            response_face = {
                "bbox": face.bbox,
                "confidence": face.confidence,
                "id": face.id,
                "landmarks": getattr(face, 'landmarks', []),
                "quality_score": getattr(face, 'quality_score', 1.0),
                "name": getattr(face, 'name', "Unknown"),
                "similarity": getattr(face, 'similarity', 0.0)
            }
            response_faces.append(response_face)
        
        logger.info(f"Returning {len(response_faces)} faces with clean format")
        return response_faces
        
    except Exception as e:
        import traceback
        logger.error(f"Error detecting faces: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")

# ==================== Verification Endpoints ====================

@router.post("/verify", response_model=VerificationResult, summary="人脸验证")
async def verify_faces(
    file1: UploadFile = File(..., description="用于比较的第一张人脸图像"),
    file2: UploadFile = File(..., description="用于比较的第二张人脸图像"),
    threshold: float = Query(0.4, description="验证判断的相似度阈值")
):
    """
    使用真实人脸验证技术验证两张人脸是否为同一人
    
    此接口比较两张人脸图像以确定它们是否属于同一人。
    使用InsightFace深度学习模型提取面部特征并计算相似度分数。
    
    **参数:**
    - file1 `UploadFile`: 用于比较的第一张人脸图像文件 (JPG, PNG)
    - file2 `UploadFile`: 用于比较的第二张人脸图像文件 (JPG, PNG)
    - threshold `float`: 验证判断的相似度阈值 (0.0-1.0, 默认: 0.4)
    
    **返回:**
    VerificationResult对象，包含:
    - is_same_person `bool`: 人脸是否属于同一人
    - confidence `float`: 相似度置信度分数 (0.0-1.0)
    - distance `float`: 人脸特征向量间的距离度量 (0.0-2.0, 越小越相似)
    - threshold `float`: 用于判断的阈值
    - message `str`: 额外的验证信息 (可选)
    - face1_bbox `List[int]`: 第一张图像中人脸的边界框 (可选)
    - face2_bbox `List[int]`: 第二张图像中人脸的边界框 (可选)
    - face1_quality `float`: 第一张人脸的质量分数 (可选)
    - face2_quality `float`: 第二张人脸的质量分数 (可选)
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

@router.post("/recognize", response_model=List[RecognitionResult], summary="人脸识别")
async def recognize_faces(
    file: UploadFile = File(..., description="包含待识别人脸的图像"),
    threshold: float = Form(0.35, description="识别匹配的相似度阈值"),
    model: str = Form("buffalo_s", description="使用的模型名称")
):
    """
    使用真实人脸识别技术识别上传图像中的人脸
    
    此接口通过将检测到的人脸与已注册的人脸数据库进行比较来识别上传图像中的已知人脸。
    返回包含人员姓名和置信度分数的识别结果。
    
    **参数:**
    - file `UploadFile`: 包含待识别人脸的图像文件 (JPG, PNG)
    - threshold `float`: 识别匹配的相似度阈值 (0.0-1.0, 默认: 0.35)
    - model `str`: 使用的模型名称 (默认: buffalo_s)
    
    **返回:**
    RecognitionResult对象列表，包含:
    - name `str`: 从数据库中识别出的人员姓名
    - confidence `float`: 识别置信度分数 (0.0-1.0)
    - bbox `List[int]`: 人脸边界框坐标 [x1, y1, x2, y2]
    - id `str`: 匹配人员的数据库人脸ID (MySQL UUID)
    - quality_score `float`: 人脸质量评估分数 (可选)
    - similarity `float`: 相似度分数 (0.0-1.0)
    """
    logger.info(f"Recognize endpoint called with model: {model}, threshold: {threshold}")
    
    try:
        image = await process_upload_file(file)
        logger.info(f"Image processed successfully, shape: {image.shape}")
        
        max_size = 320
        h, w = image.shape[:2]
        if h > max_size or w > max_size:
            scale = min(max_size / h, max_size / w)
            new_h, new_w = int(h * scale), int(w * scale)
            logger.info(f"Resizing image from {h}x{w} to {new_h}x{new_w}")
            image = cv2.resize(image, (new_w, new_h))
        
        # Get the cached recognizer
        recognizer = get_recognizer()
        logger.info(f"Using cached recognizer: {type(recognizer).__name__}")
        
        results = recognizer.recognize(image, threshold=threshold)
        logger.info(f"Recognized {len(results)} faces")
        
        response_results = []
        for result in results:
            # Convert to dict for easier manipulation
            result_dict = result.dict() if hasattr(result, 'dict') else result
            
            if 'face_id' in result_dict and 'id' not in result_dict:
                result_dict['id'] = result_dict.pop('face_id')
            
            if 'person_id' in result_dict:
                result_dict.pop('person_id')
                
            if 'name' not in result_dict or not result_dict['name']:
                result_dict['name'] = "Unknown"
            if 'similarity' not in result_dict or result_dict['similarity'] is None:
                result_dict['similarity'] = 0.0
            if 'quality_score' not in result_dict or result_dict['quality_score'] is None:
                result_dict['quality_score'] = 1.0
                
            response_results.append(result_dict)
            
        logger.info(f"Returning {len(response_results)} recognition results with clean format")
        return response_results
        
    except Exception as e:
        import traceback
        logger.error(f"Error recognizing faces: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Recognition failed: {str(e)}")

@router.post("/register", response_model=FaceRegisterResponse, summary="人脸注册")
async def register_face(
    file: UploadFile = File(..., description="包含待注册人脸的图像"),
    name: str = Form(..., description="注册人员的完整姓名"),
    department: Optional[str] = Form(None, description="部门或组织单位"),
    employee_id: Optional[str] = Form(None, description="唯一的员工或人员标识符")
):
    """
    使用真实人脸注册技术注册上传图像中的人脸
    
    此接口将新人脸注册到人脸识别数据库中。检测上传图像中的人脸，
    提取面部特征，并将其与提供的人员信息一起存储以供将来识别。
    
    **参数:**
    - file `UploadFile`: 包含待注册人脸的图像文件 (JPG, PNG)
    - name `str`: 注册人员的完整姓名
    - department `str`: 部门或组织单位 (可选)
    - employee_id `str`: 唯一的员工或人员标识符 (可选)
    
    **返回:**
    FaceRegisterResponse对象，包含:
    - success `bool`: 注册是否成功
    - message `str`: 注册状态消息或错误详情
    - person_name `str`: 已注册人员的姓名
    - face_id `str`: 在数据库中生成的唯一人脸ID (可选)
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

# ==================== Database Endpoints ====================

@router.get("/faces", summary="获取人脸列表")
async def list_faces(
    name: Optional[str] = Query(None, description="按人员姓名过滤结果"),
    limit: int = Query(100, description="返回的最大人脸数量")
):
    """
    获取数据库中的人脸列表
    
    此接口从数据库中检索所有已注册人脸的列表，可选择按人员姓名过滤。
    返回包括姓名、ID、时间戳和关联信息的人脸元数据。
    
    **参数:**
    - name `str`: 按人员姓名过滤结果 (可选, 区分大小写)
    - limit `int`: 返回的最大人脸数量 (默认: 100, 最大: 1000)
    
    **返回:**
    包含以下内容的对象:
    - faces `List[Dict]`: 人脸记录列表，包含:
      - id `str`: 唯一人脸标识符
      - name `str`: 人员的注册姓名
      - metadata `Dict[str, Any]`: 额外的人员元数据 (部门、员工ID等)
      - created_at `str`: 注册时间戳
      - updated_at `str`: 最后更新时间戳 (可选)
      - embedding_size `int`: 人脸特征向量的大小
    - total `int`: 符合过滤条件的人脸总数
    - returned `int`: 此响应中返回的人脸数量
    """
    recognizer = get_recognizer()
    
    try:
        # Get faces directly from the database
        if name:
            faces = recognizer.face_db.search_by_name(name)
        else:
            faces = recognizer.face_db.get_all_faces()
        
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

@router.delete("/faces/{face_id}", summary="删除人脸")
async def delete_face(face_id: str):
    """
    根据ID删除人脸
    
    此接口使用唯一的人脸ID从数据库中删除特定的人脸记录。
    删除后，该人脸将不再在未来的识别请求中被识别。
    
    **参数:**
    - face_id `str`: 要从数据库中删除的唯一人脸标识符
    
    **返回:**
    包含以下内容的对象:
    - message `str`: 包含已删除人脸ID的成功确认消息
    
    **错误:**
    - 404: 数据库中未找到人脸ID
    - 500: 数据库删除错误
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

@router.delete("/faces/by-name/{name}", summary="按姓名删除人脸")
async def delete_faces_by_name(name: str):
    """
    删除某个人的所有人脸
    
    此接口从数据库中删除与特定人员姓名关联的所有人脸记录。
    这对于从识别系统中完全移除某个人很有用。
    
    **参数:**
    - name `str`: 要删除所有关联人脸的人员姓名
    
    **返回:**
    包含以下内容的对象:
    - message `str`: 包含人员姓名的成功确认消息
    - deleted_count `int`: 删除的人脸数量
    
    **错误:**
    - 404: 未找到指定姓名的人脸
    - 500: 数据库删除错误
    """
    recognizer = get_recognizer()
    
    try:
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

# ==================== Utility Endpoints ====================

@router.get("/models/info", summary="获取模型信息")
async def get_model_info():
    """
    获取已加载模型的信息
    
    此接口提供当前已加载的InsightFace模型的详细信息、
    配置设置和系统状态，用于调试和监控目的。
    
    **返回:**
    包含以下内容的对象:
    - status `str`: 模型状态 (active/inactive)
    - initialized `bool`: 模型是否正确初始化
    - model_pack `str`: 已加载模型包的名称 (例如: "buffalo_l")
    - similarity_threshold `float`: 当前识别的相似度阈值
    - detection_threshold `float`: 当前检测置信度阈值
    - face_database_connected `bool`: 人脸数据库是否可访问
    - face_count `int`: 数据库中人脸的总数
    - insightface_available `bool`: InsightFace库是否正确加载
    
    **错误:**
    - 500: 模型信息获取错误
    """
    try:
        recognizer = get_recognizer()
        
        # Get model info from real recognizer
        model_info = recognizer.get_model_info()
        
        # 获取当前设置
        settings = get_settings()
        
        # 获取GPU/CPU状态信息
        try:
            import torch
            gpu_available = torch.cuda.is_available()
            gpu_count = torch.cuda.device_count() if gpu_available else 0
            gpu_name = torch.cuda.get_device_name(0) if gpu_available and gpu_count > 0 else None
        except ImportError:
            gpu_available = False
            gpu_count = 0 
            gpu_name = None
        
        return {
            "status": "active",
            "initialized": recognizer.app is not None,
            "model_pack": model_info.get("model_pack", "buffalo_l"),
            "similarity_threshold": recognizer.similarity_threshold,
            "detection_threshold": recognizer.det_thresh,
            "detection_size": list(recognizer.det_size),
            "face_database_connected": recognizer.face_db is not None,
            "face_count": recognizer.face_db.get_face_count() if recognizer.face_db else 0,
            "insightface_available": True,
            
            # 扩展配置信息
            "configuration": {
                "emotion_recognition": recognizer.enable_emotion,
                "mask_detection": recognizer.enable_mask_detection,
                "prefer_gpu": getattr(recognizer, 'prefer_gpu', True),
                "model_device": getattr(settings, 'model_device', 'auto'),
                "backend": getattr(settings, 'model_backend', 'insightface')
            },
            
            # 系统资源信息
            "system": {
                "gpu_available": gpu_available,
                "gpu_count": gpu_count,
                "gpu_name": gpu_name,
                "pytorch_available": 'torch' in locals()
            },
            
            # 数据库信息
            "database": {
                "type": recognizer.face_db.__class__.__name__ if recognizer.face_db else "None",
                "connected": recognizer.face_db is not None,
                "face_count": recognizer.face_db.get_face_count() if recognizer.face_db else 0
            },
            
            # 性能建议
            "recommendations": {
                "current_setup": "生产就绪" if recognizer.app is not None else "需要安装InsightFace",
                "gpu_recommendation": "建议使用GPU加速以获得更好性能" if not gpu_available and getattr(recognizer, 'prefer_gpu', True) else "GPU配置正常",
                "model_recommendation": f"当前使用 {model_info.get('model_pack', 'buffalo_l')} - 适合生产环境" if model_info.get('model_pack', 'buffalo_l') == 'buffalo_l' else f"当前使用 {model_info.get('model_pack', 'buffalo_l')} - 考虑使用buffalo_l获得更高精度"
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get model info: {str(e)}")

@router.get("/status", summary="详细状态检查")
async def detailed_status():
    """
    InsightFace服务详细状态检查
    
    提供详细的状态信息，包括智能模型池和数据库状态。
    显示当前加载的模型、使用统计和最后API调用信息。
    
    **返回:**
    - status: 整体状态
    - service: 服务名称
    - model_pool: 模型池状态
    - loaded_models: 当前加载的模型列表
    - database_status: 数据库状态
    - memory_usage: 内存使用情况
    - timestamp: 时间戳
    """
    try:
        recognizer = get_recognizer()
        
        # 获取模型信息
        model_info = recognizer.get_model_info()
        
        # 获取数据库状态
        face_count = 0
        db_status = "disconnected"
        try:
            faces = recognizer.list_all_faces()
            face_count = len(faces)
            db_status = "connected"
        except:
            pass
        
        # 构建详细状态
        status_info = {
            "status": "healthy",
            "service": "InsightFace API",
            "model_pool": {
                "initialized": recognizer.app is not None,
                "model_pack": recognizer.model_pack,
                "detection_threshold": recognizer.det_thresh,
                "similarity_threshold": recognizer.similarity_threshold
            },
            "loaded_models": [
                getattr(model, 'taskname', str(model)) for model in recognizer.app.models
            ] if recognizer.app and hasattr(recognizer.app, 'models') else [],
            "database_status": {
                "connected": db_status == "connected",
                "type": recognizer.face_db.__class__.__name__ if recognizer.face_db else "None",
                "face_count": face_count
            },
            "memory_usage": {
                "gpu_enabled": getattr(recognizer, 'prefer_gpu', True),
                "emotion_model": getattr(recognizer, 'enable_emotion', True),
                "mask_detection": getattr(recognizer, 'enable_mask_detection', True)
            },
            "timestamp": str(datetime.now())
        }
        
        return status_info
        
    except Exception as e:
        logger.error(f"Status check failed: {e}")
        return {
            "status": "unhealthy",
            "service": "InsightFace API",
            "error": str(e),
            "timestamp": str(datetime.now())
        }

@router.get("/health", summary="健康检查")
async def health_check():
    """
    InsightFace服务健康检查
    
    此接口提供InsightFace服务的快速健康状态检查，
    包括模型初始化状态和数据库连接性。
    
    **返回:**
    包含以下内容的对象:
    - status `str`: 整体服务健康状态 (healthy/unhealthy)
    - service `str`: 服务名称标识符
    - initialized `bool`: 人脸识别模型是否已初始化
    - model_pack `str`: 当前加载的模型包名称
    - database_connected `bool`: 人脸数据库是否可访问
    - timestamp `str`: 当前服务器时间戳
    
    **错误:**
    - 500: 服务健康检查失败
    """
    try:
        recognizer = get_recognizer()
        
        return {
            "status": "healthy",
            "service": "Real InsightFace API",
            "initialized": recognizer.app is not None,
            "model_pack": recognizer.model_pack,
            "database_connected": recognizer.face_db is not None,
            "timestamp": str(datetime.now())
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

# ==================== 视频流处理端点 ====================

@router.post("/stream/process", summary="处理视频流")
async def process_video_stream(
    source: str = Form(..., description="视频源 (0 表示摄像头, RTSP URL 表示网络流)"),
    duration: Optional[int] = Form(None, description="处理时长(秒)"),
    skip_frames: int = Form(1, description="跳帧数"),
    show_preview: bool = Form(False, description="是否显示预览")
):
    """
    处理视频流进行人脸识别
    
    支持本地摄像头和 RTSP 网络流。
    
    **参数:**
    - source: 视频源 (0/1/2 表示本地摄像头索引, rtsp:// 开头表示网络流)
    - duration: 处理时长，None 表示持续处理
    - skip_frames: 每隔几帧处理一次 (1=每帧, 2=隔帧)
    - show_preview: 是否显示预览窗口 (仅本地有效)
    
    **返回:**
    处理结果包含检测到的人脸信息
    """
    try:
        recognizer = get_recognizer()
        
        # 尝试将 source 转换为整数（摄像头索引）
        try:
            source_int = int(source)
            cap = cv2.VideoCapture(source_int)
        except ValueError:
            # RTSP 或文件路径
            cap = cv2.VideoCapture(source)
        
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail=f"无法打开视频源: {source}")
        
        results = {
            "source": source,
            "status": "processing",
            "total_faces": 0,
            "unique_persons": set(),
            "detections": []
        }
        
        frame_count = 0
        start_time = datetime.now()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 跳帧处理
            if frame_count % skip_frames != 0:
                frame_count += 1
                continue
            
            # 检测和识别人脸
            faces = recognizer.detect_faces(frame)
            
            for face in faces:
                # 识别人脸
                recognition_results = recognizer.recognize(frame, threshold=0.35)
                
                for result in recognition_results:
                    results["total_faces"] += 1
                    if result.name != "Unknown":
                        results["unique_persons"].add(result.name)
                    
                    results["detections"].append({
                        "timestamp": str(datetime.now()),
                        "name": result.name,
                        "confidence": result.confidence,
                        "bbox": result.bbox
                    })
            
            # 显示预览
            if show_preview:
                cv2.imshow("Face Recognition", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            # 检查是否超时
            if duration and (datetime.now() - start_time).seconds >= duration:
                break
            
            frame_count += 1
        
        cap.release()
        if show_preview:
            cv2.destroyAllWindows()
        
        results["status"] = "completed"
        results["unique_persons"] = list(results["unique_persons"])
        results["duration"] = (datetime.now() - start_time).seconds
        
        return results
        
    except Exception as e:
        logger.error(f"视频流处理失败: {e}")
        raise HTTPException(status_code=500, detail=f"视频流处理失败: {str(e)}")

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
    for i in range(3):  # 检查前3个摄像头
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            cameras.append({
                "index": i,
                "name": f"摄像头 {i}",
                "available": True
            })
            cap.release()
    
    return {
        "cameras": cameras,
        "sample_rtsp": [
            "rtsp://username:password@192.168.1.100:554/stream1",
            "rtsp://192.168.1.100:8554/live.sdp"
        ]
    }

# ==================== New Stream Processing Endpoints ====================

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
        # Configure webhook only if URL is provided
        has_webhook = webhook_url and webhook_url.strip()
        if has_webhook:
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
            raise Exception(f"Cannot open video source: {source}")
        
        _active_streams[stream_id] = {
            "processor": processor,
            "cap": cap,
            "status": "processing"
        }
        
        frame_count = 0
        
        while stream_id in _active_streams:
            ret, frame = cap.read()
            if not ret:
                break
            
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
                    
                    # Send to webhook if configured
                    if has_webhook:
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
                    
                    if has_webhook:
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
        if has_webhook:
            webhook_manager.remove_webhook(stream_id)
        if stream_id in _active_streams:
            _active_streams[stream_id]["status"] = "completed"
            del _active_streams[stream_id]
            
    except Exception as e:
        logger.error(f"Stream processing error: {e}")
        if stream_id in _active_streams:
            _active_streams[stream_id]["status"] = "error"
            del _active_streams[stream_id]
        if has_webhook:
            webhook_manager.remove_webhook(stream_id)


@router.post(
    "/stream/process_recognition",
    response_model=StreamProcessResponse,
    summary="视频流人脸识别",
    description="处理视频流进行实时人脸识别并通过Webhook发送结果"
)
async def process_stream_recognition(
    request: StreamRecognitionRequest,
    background_tasks: BackgroundTasks
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
    source = request.camera_id
    if isinstance(source, str) and source.isdigit():
        source = int(source)
    
    # Prepare request parameters
    request_params = {
        "skip_frames": request.skip_frames,
        "threshold": request.threshold,
        "return_frame": request.return_frame,
        "draw_bbox": request.draw_bbox,
        "return_all_candidates": request.return_all_candidates,
        "max_candidates": request.max_candidates
    }
    
    # Get recognizer instance
    recognizer = get_recognizer()
    
    # Start background processing
    background_tasks.add_task(
        process_stream_with_webhook,
        stream_id=stream_id,
        source=source,
        webhook_url=request.webhook_url,
        recognizer=recognizer,
        request_params=request_params,
        event_type="face_recognized"
    )
    
    return StreamProcessResponse(
        stream_id=stream_id,
        status="started",
        message=f"Stream processing started for camera {request.camera_id}",
        camera_id=request.camera_id,
        webhook_url=request.webhook_url,
        start_time=datetime.now().isoformat()
    )


@router.post(
    "/stream/process_verification",
    response_model=StreamProcessResponse,
    summary="视频流人脸验证",
    description="处理视频流进行特定人员的人脸验证并通过Webhook发送结果"
)
async def process_stream_verification(
    request: StreamVerificationRequest,
    background_tasks: BackgroundTasks
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
    source = request.camera_id
    if isinstance(source, str) and source.isdigit():
        source = int(source)
    
    # Prepare request parameters
    request_params = {
        "skip_frames": request.skip_frames,
        "threshold": request.threshold,
        "return_frame": request.return_frame,
        "draw_bbox": request.draw_bbox,
        "target_name": request.target_name,
        "verification_threshold": request.verification_threshold,
        "alert_on_mismatch": request.alert_on_mismatch
    }
    
    # Get recognizer instance
    recognizer = get_recognizer()
    
    # Start background processing
    background_tasks.add_task(
        process_stream_with_webhook,
        stream_id=stream_id,
        source=source,
        webhook_url=request.webhook_url,
        recognizer=recognizer,
        request_params=request_params,
        event_type="face_verified"
    )
    
    return StreamProcessResponse(
        stream_id=stream_id,
        status="started",
        message=f"Stream verification started for camera {request.camera_id}, target: {request.target_name}",
        camera_id=request.camera_id,
        webhook_url=request.webhook_url,
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
        return {
            "stream_id": stream_id,
            "status": _active_streams[stream_id]["status"],
            "active": True
        }
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
        
        # Remove webhook if exists
        if stream_id in webhook_manager.webhooks:
            webhook_manager.remove_webhook(stream_id)
        
        return {
            "stream_id": stream_id,
            "status": "stopped",
            "message": "Stream processing stopped successfully"
        }
    else:
        raise HTTPException(status_code=404, detail="Stream not found")
