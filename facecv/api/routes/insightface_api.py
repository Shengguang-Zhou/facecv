"""Real InsightFace API Routes (Non-Mock)"""

from fastapi import APIRouter, File, UploadFile, HTTPException, Form, Query
from typing import List, Optional
import numpy as np
from PIL import Image
import io
import logging
import cv2
import uuid
from datetime import datetime

from facecv.models.insightface.onnx_recognizer import ONNXFaceRecognizer
from facecv.models.insightface.real_recognizer import RealInsightFaceRecognizer
from facecv.models.insightface.arcface_recognizer import ArcFaceRecognizer
from facecv.schemas.face import FaceDetection, VerificationResult, RecognitionResult, FaceRegisterResponse
from facecv.config import get_settings
from facecv.database.factory import get_default_database

router = APIRouter(tags=["InsightFace"])
logger = logging.getLogger(__name__)

# Global instance
_recognizer = None

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
    if _recognizer is None:
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

@router.post("/models/switch", summary="切换模型类型")
async def switch_model_type(
    enable_arcface: bool = Form(..., description="是否启用ArcFace专用模型"),
    arcface_backbone: Optional[str] = Form("resnet50", description="ArcFace骨干网络 (resnet50/mobilefacenet)")
):
    """
    切换模型类型 (ArcFace vs Buffalo)
    
    此接口允许在运行时切换模型类型，支持：
    1. 启用ArcFace专用模型 (enable_arcface=True)
    2. 使用传统Buffalo模型 (enable_arcface=False)
    
    **参数:**
    - enable_arcface `bool`: 是否启用ArcFace专用模型
    - arcface_backbone `str`: ArcFace骨干网络类型 (仅当enable_arcface=True时有效)
    
    **返回:**
    切换结果信息，包含新模型的详细配置
    """
    global _recognizer
    
    try:
        # Clear current recognizer to force reload
        _recognizer = None
        
        from facecv.config import get_runtime_config, get_settings
        
        # 获取当前设置
        settings = get_settings()
        runtime_config = get_runtime_config()
        
        runtime_config.set("arcface_enabled", enable_arcface)
        if enable_arcface and arcface_backbone:
            runtime_config.set("arcface_backbone", arcface_backbone)
        
        # Get new recognizer
        new_recognizer = get_recognizer()
        
        # Get model info
        model_info = new_recognizer.get_model_info()
        
        return {
            "success": True,
            "message": f"Successfully switched to {'ArcFace' if enable_arcface else 'Buffalo'} model",
            "model_type": "ArcFace" if enable_arcface else "Buffalo", 
            "model_info": model_info,
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
            "error": str(e)
        }

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
        
        # Get the recognizer - use smaller model for better performance
        from facecv.models.model_pool import get_model_recognizer
        actual_model = "buffalo_s"  # Use smaller model for better performance
        recognizer = get_model_recognizer(actual_model, "detect")
        logger.info(f"Using recognizer: {type(recognizer).__name__} with model {actual_model}")
        
        faces = recognizer.detect_faces(image)
        logger.info(f"Detected {len(faces)} faces")
        
        # Filter by confidence
        filtered_faces = [f for f in faces if f.confidence >= min_confidence]
        logger.info(f"Filtered to {len(filtered_faces)} faces with confidence >= {min_confidence}")
        
        for face in filtered_faces:
            if not hasattr(face, 'id') or not face.id:
                face.id = str(uuid.uuid4())
            if not hasattr(face, 'name') or not face.name:
                face.name = "Unknown"
            if not hasattr(face, 'similarity') or face.similarity is None:
                face.similarity = 0.0
            if not hasattr(face, 'quality_score') or face.quality_score is None:
                face.quality_score = 1.0
        
        if filtered_faces:
            try:
                from facecv.database.factory import create_face_database
                face_db = create_face_database('hybrid')
                logger.info(f"Created hybrid face database: {type(face_db).__name__}")
                
                for face in filtered_faces:
                    x1, y1, x2, y2 = face.bbox
                    if x1 < 0: x1 = 0
                    if y1 < 0: y1 = 0
                    if x2 > image.shape[1]: x2 = image.shape[1]
                    if y2 > image.shape[0]: y2 = image.shape[0]
                    
                    if x2 <= x1 or y2 <= y1:
                        logger.warning(f"Invalid bbox: {face.bbox}, skipping recognition")
                        continue
                    
                    face_img = image[y1:y2, x1:x2]
                    
                    embedding = recognizer.extract_embedding(face_img)
                    if embedding is None:
                        logger.warning(f"Failed to extract embedding for face at {face.bbox}")
                        continue
                    
                    similar_faces = face_db.query_faces_by_embedding(embedding, top_k=1, threshold=0.35)
                    
                    if similar_faces and similar_faces[0].get('similarity', 0) >= 0.35:
                        match = similar_faces[0]
                        face.id = match.get('id', face.id)
                        face.name = match.get('name', "Unknown")
                        face.similarity = match.get('similarity', 0.0)
                        logger.info(f"Match found: {face.name} (ID: {face.id}, similarity: {face.similarity})")
                    else:
                        logger.info(f"No match found for face at {face.bbox}")
                
            except Exception as e:
                logger.error(f"Error during recognition in detect endpoint: {e}")
                import traceback
                logger.error(traceback.format_exc())
        
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
        
        from facecv.database.factory import create_face_database
        face_db = create_face_database('hybrid')
        logger.info(f"Created hybrid face database: {type(face_db).__name__}")
        
        # Get the recognizer with the hybrid database
        from facecv.models.model_pool import get_model_recognizer
        actual_model = "buffalo_s"  # Use smaller model for better performance
        recognizer = get_model_recognizer(actual_model, "recognize", face_db=face_db)
        logger.info(f"Using recognizer: {type(recognizer).__name__} with model {actual_model}")
        
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
    
    **错误:**
    - 404: 未找到指定姓名的人脸
    - 500: 数据库删除错误
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

@router.get("/faces/count", summary="获取人脸数量")
async def get_face_count():
    """
    获取数据库中人脸的总数
    
    此接口返回数据库中已注册人脸的总数。
    对于监控数据库大小和使用统计很有用。
    
    **返回:**
    包含以下内容的对象:
    - total_faces `int`: 数据库中已注册人脸的总数
    
    **错误:**
    - 500: 数据库查询错误
    """
    recognizer = get_recognizer()
    
    try:
        count = recognizer.get_face_count()
        return {"total_faces": count}
        
    except Exception as e:
        logger.error(f"Error getting face count: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get face count: {str(e)}")

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
            "initialized": model_info.get("initialized", False),
            "model_pack": model_info.get("model_pack", "buffalo_l"),
            "similarity_threshold": recognizer.similarity_threshold,
            "detection_threshold": recognizer.det_thresh,
            "detection_size": list(recognizer.det_size),
            "face_database_connected": recognizer.face_db is not None,
            "face_count": recognizer.get_face_count() if recognizer.face_db else 0,
            "insightface_available": model_info.get("insightface_available", False),
            
            # 扩展配置信息
            "configuration": {
                "emotion_recognition": recognizer.enable_emotion,
                "mask_detection": recognizer.enable_mask_detection,
                "prefer_gpu": recognizer.prefer_gpu,
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
                "face_count": recognizer.get_face_count() if recognizer.face_db else 0
            },
            
            # 性能建议
            "recommendations": {
                "current_setup": "生产就绪" if model_info.get("initialized", False) and model_info.get("insightface_available", False) else "需要安装InsightFace",
                "gpu_recommendation": "建议使用GPU加速以获得更好性能" if not gpu_available and recognizer.prefer_gpu else "GPU配置正常",
                "model_recommendation": f"当前使用 {model_info.get('model_pack', 'buffalo_l')} - 适合生产环境" if model_info.get('model_pack', 'buffalo_l') == 'buffalo_l' else f"当前使用 {model_info.get('model_pack', 'buffalo_l')} - 考虑使用buffalo_l获得更高精度"
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get model info: {str(e)}")

@router.post("/models/select", summary="选择模型")
async def select_model(
    model: str = Query(..., description="要选择的模型名称: buffalo_l, buffalo_m, buffalo_s, antelopev2")
):
    """
    动态切换InsightFace模型
    
    此接口允许在运行时切换使用的InsightFace模型包，
    支持在精度和速度之间进行权衡。
    
    **参数:**
    - model `str`: 要选择的模型名称
      - buffalo_l: 最佳精度 (推荐生产环境)
      - buffalo_m: 平衡精度和速度
      - buffalo_s: 速度优先
      - antelopev2: 最高精度
    
    **返回:**
    包含以下内容的对象:
    - success `bool`: 切换是否成功
    - message `str`: 状态消息
    - previous_model `str`: 之前使用的模型
    - current_model `str`: 当前使用的模型
    """
    global _recognizer
    
    # Validate model choice
    valid_models = ["buffalo_l", "buffalo_m", "buffalo_s", "antelopev2"]
    if model not in valid_models:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid model '{model}'. Valid choices: {', '.join(valid_models)}"
        )
    
    try:
        from facecv.config import get_runtime_config
        
        runtime_config = get_runtime_config()
        
        previous_model = runtime_config.get('insightface_model_pack', 'buffalo_l')
        
        if previous_model == model:
            return {
                "success": True,
                "message": f"Model '{model}' is already active",
                "previous_model": previous_model,
                "current_model": model
            }
        
        runtime_config.set("insightface_model_pack", model)
        
        # Reset recognizer to force reinitialization with new model
        _recognizer = None
        
        # Initialize with new model
        new_recognizer = get_recognizer()
        
        logger.info(f"Successfully switched model from {previous_model} to {model}")
        
        return {
            "success": True,
            "message": f"Successfully switched from {previous_model} to {model}",
            "previous_model": previous_model,
            "current_model": model,
            "model_info": new_recognizer.get_model_info()
        }
        
    except Exception as e:
        logger.error(f"Error switching model to {model}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to switch model: {str(e)}")

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
            "initialized": recognizer.initialized,
            "model_pack": recognizer.model_pack,
            "database_connected": recognizer.face_db is not None,
            "timestamp": str(datetime.now())
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")
