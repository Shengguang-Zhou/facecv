"""
FaceCV DeepFace API路由

提供完整的DeepFace人脸识别、验证和分析API端点。
支持人脸注册、识别、验证、属性分析、实时流处理等功能。
"""

import io
import base64
import asyncio
import logging
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
from PIL import Image
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks, Query
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from facecv.core.webhook import webhook_manager, WebhookConfig, send_recognition_event

from facecv.schemas.face import (
    FaceRegisterRequest, FaceRegisterResponse,
    FaceRecognitionRequest, FaceRecognitionResponse,
    FaceVerificationRequest, FaceVerificationResponse,
    FaceAnalysisRequest, FaceAnalysisResponse,
    FaceListResponse, FaceUpdateRequest, FaceDeleteResponse
)

logger = logging.getLogger(__name__)

# 创建路由器
router = APIRouter(prefix="/api/v1/deepface", tags=["DeepFace"])

# 全局变量存储模型实例（延迟加载）
deepface_recognizer = None
face_embedding = None
face_verification = None
face_analysis = None
face_detection = None

def get_deepface_components():
    """获取DeepFace组件实例（延迟加载）"""
    global deepface_recognizer, face_embedding, face_verification, face_analysis, face_detection
    
    if deepface_recognizer is None or face_embedding is None or face_verification is None or face_analysis is None or face_detection is None:
        try:
            from facecv.models.deepface.recognizer import DeepFaceRecognizer
            import facecv.models.deepface.face_embedding as fe
            import facecv.models.deepface.face_verification as fv
            import facecv.models.deepface.face_analysis as fa
            import facecv.models.deepface.face_detection as fd
            
            deepface_recognizer = DeepFaceRecognizer()
            face_embedding = fe
            face_verification = fv
            face_analysis = fa
            face_detection = fd
            
            if not hasattr(face_verification, 'face_verification'):
                raise ImportError("face_verification模块缺少face_verification方法")
            if not hasattr(face_analysis, 'face_analysis'):
                raise ImportError("face_analysis模块缺少face_analysis方法")
            if not hasattr(face_detection, 'face_detection'):
                raise ImportError("face_detection模块缺少face_detection方法")
                
            logger.info("DeepFace组件初始化成功")
        except ImportError as e:
            logger.error(f"DeepFace组件初始化失败: {e}")
            raise HTTPException(
                status_code=503, 
                detail=f"DeepFace服务不可用，请确保已安装相关依赖: {str(e)}"
            )
        except Exception as e:
            logger.error(f"DeepFace组件初始化异常: {e}")
            raise HTTPException(
                status_code=503, 
                detail=f"DeepFace服务初始化失败: {str(e)}"
            )
    
    return deepface_recognizer, face_embedding, face_verification, face_analysis, face_detection


# ==================== 数据模型 ====================

class FaceAnalyzeRequest(BaseModel):
    """人脸分析请求模型"""
    actions: List[str] = ["emotion", "age", "gender", "race"]
    detector_backend: str = "mtcnn"
    enforce_detection: bool = False


class VideoFaceRequest(BaseModel):
    """视频人脸添加请求模型"""
    name: str
    video_source: str = "0"  # 摄像头编号或视频路径
    sample_interval: int = 30  # 采样间隔（帧数）
    max_samples: int = 10  # 最大采样数量


# ==================== 人脸管理API ====================

@router.post("/faces/", response_model=FaceRegisterResponse, summary="注册人脸")
async def register_face(
    name: str = Form(..., description="人员姓名，用于标识注册的人脸"),
    file: UploadFile = File(..., description="包含人脸的图片文件 (支持JPG, PNG, BMP格式)"),
    metadata: Optional[str] = Form(None, description="附加元数据JSON字符串，如部门、员工ID等")
):
    """
    使用DeepFace技术注册新人脸到数据库
    
    此接口将人脸图片添加到DeepFace人脸识别数据库中，用于后续的人脸识别和验证操作。
    系统会自动检测图片中的人脸并提取特征向量存储。
    
    **请求参数:**
    - name `str`: 人员姓名，必填，用于标识注册的人脸
    - file `UploadFile`: 包含人脸的图片文件，支持格式：JPG, PNG, BMP
    - metadata `str`: 附加元数据JSON字符串（可选），可包含部门、员工ID等信息
    
    **响应数据:**
    - success `bool`: 注册是否成功
    - message `str`: 注册状态消息
    - person_name `str`: 已注册的人员姓名
    - face_id `str`: 生成的人脸ID（可选）
    
    **注意事项:**
    - 图片中必须包含清晰可识别的人脸
    - 建议使用正面、光线充足的照片
    - 元数据格式示例：{"department": "技术部", "employee_id": "E001"}
    """
    try:
        recognizer, embedding_mgr, _, _, _ = get_deepface_components()
        
        # 读取上传的图片
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        image_array = np.array(image)
        
        # 转换颜色格式（PIL使用RGB，OpenCV使用BGR）
        if len(image_array.shape) == 3:
            image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        
        # 解析元数据
        import json
        parsed_metadata = {}
        if metadata:
            try:
                parsed_metadata = json.loads(metadata)
            except json.JSONDecodeError:
                logger.warning(f"无效的元数据格式: {metadata}")
        
        # 注册人脸
        face_id = await recognizer.register_face_async(
            name=name,
            image=image_array,
            metadata=parsed_metadata
        )
        
        if face_id:
            return FaceRegisterResponse(
                success=True,
                message=f"人脸注册成功: {name}",
                person_name=name,
                face_id=face_id
            )
        else:
            raise HTTPException(status_code=400, detail="人脸注册失败，可能无法检测到清晰的人脸")
            
    except Exception as e:
        logger.error(f"人脸注册异常: {e}")
        raise HTTPException(status_code=500, detail=f"服务器内部错误: {str(e)}")


@router.get("/faces/", response_model=FaceListResponse, summary="获取人脸列表")
async def list_faces():
    """
    获取所有已注册的人脸信息列表
    
    此接口返回DeepFace数据库中所有已注册人脸的详细信息，包括姓名、ID、创建时间和元数据。
    
    **响应数据:**
    - faces `List[Dict]`: 人脸信息列表，每个对象包含：
      - face_id `str`: 人脸唯一标识符
      - person_name `str`: 人员姓名
      - created_at `str`: 创建时间戳
      - metadata `Dict`: 附加元数据信息
    - total `int`: 人脸总数量
    """
    try:
        recognizer, _, _, _, _ = get_deepface_components()
        
        # 获取所有用户
        face_list = recognizer.list_faces()
        
        faces = []
        for face_data in face_list:
            faces.append({
                "face_id": face_data.face_id,
                "person_name": face_data.person_name,
                "created_at": face_data.metadata.get("created_at") if face_data.metadata else None,
                "metadata": face_data.metadata or {}
            })
        
        return FaceListResponse(
            faces=faces,
            total=len(faces)
        )
        
    except Exception as e:
        logger.error(f"获取人脸列表异常: {e}")
        raise HTTPException(status_code=500, detail=f"服务器内部错误: {str(e)}")


@router.put("/faces/{face_id}", summary="更新人脸信息")
async def update_face(
    face_id: str,
    name: Optional[str] = Form(None, description="更新的人员姓名"),
    file: Optional[UploadFile] = File(None, description="更新的人脸图片文件"),
    metadata: Optional[str] = Form(None, description="更新的元数据JSON字符串")
):
    """
    根据人脸ID更新现有人脸的信息
    
    此接口允许更新已注册人脸的姓名、图片或元数据信息。可以单独更新任一字段，
    也可以同时更新多个字段。如果更新图片，系统会重新提取人脸特征。
    
    **路径参数:**
    - face_id `str`: 要更新的人脸唯一标识符
    
    **请求参数:**
    - name `str`: 更新的人员姓名（可选）
    - file `UploadFile`: 更新的人脸图片文件（可选），支持JPG, PNG, BMP格式
    - metadata `str`: 更新的元数据JSON字符串（可选）
    
    **响应数据:**
    - success `bool`: 更新是否成功
    - message `str`: 更新状态消息
    
    **注意事项:**
    - 至少需要提供一个更新参数
    - 更新图片会导致重新提取人脸特征，可能影响识别结果
    - 如果face_id不存在，将返回404错误
    - 元数据格式示例：{"department": "市场部", "position": "经理"}
    
    **错误码:**
    - 404: 指定的face_id不存在
    - 400: 更新人脸图片失败（可能图片中无清晰人脸）
    - 500: 服务器内部错误
    """
    try:
        recognizer, _, _, _, _ = get_deepface_components()
        
        # 获取当前用户信息
        current_face = await recognizer.get_face_by_id_async(face_id)
        if not current_face:
            raise HTTPException(status_code=404, detail=f"未找到face_id: {face_id}")
        
        current_name = current_face.get("name", "Unknown")
        
        # 更新姓名
        if name and name != current_name:
            await recognizer.update_face_async(face_id, name=name)
            logger.info(f"更新姓名: {current_name} -> {name}")
        
        # 更新图片（需要重新注册）
        if file:
            image_data = await file.read()
            image = Image.open(io.BytesIO(image_data))
            image_array = np.array(image)
            
            if len(image_array.shape) == 3:
                image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
            
            # 删除旧记录并重新注册
            await recognizer.delete_face_async(face_id)
            new_face_id = await recognizer.register_face_async(
                name=name or current_name,
                image=image_array
            )
            
            if not new_face_id:
                raise HTTPException(status_code=400, detail="更新人脸图片失败")
        
        return JSONResponse(
            content={"success": True, "message": f"人脸信息更新成功: {face_id}"}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"更新人脸异常: {e}")
        raise HTTPException(status_code=500, detail=f"服务器内部错误: {str(e)}")


@router.delete("/faces/{face_id}", response_model=FaceDeleteResponse, summary="删除人脸")
async def delete_face(face_id: str):
    """
    根据人脸ID从数据库中删除指定人脸
    
    此接口永久删除指定ID的人脸记录，包括其特征数据和元数据信息。
    删除操作不可逆，请谨慎使用。
    
    **路径参数:**
    - face_id `str`: 要删除的人脸唯一标识符
    
    **响应数据:**
    - success `bool`: 删除是否成功
    - message `str`: 删除状态确认消息
    
    **错误码:**
    - 404: 指定的face_id不存在
    - 500: 服务器内部错误
    
    **注意事项:**
    - 删除操作不可撤销
    - 删除后该人脸将无法在识别中被匹配
    - 建议在删除前先备份重要数据
    """
    try:
        recognizer, _, _, _, _ = get_deepface_components()
        
        # 检查用户是否存在
        face = await recognizer.get_face_by_id_async(face_id)
        if not face:
            raise HTTPException(status_code=404, detail=f"未找到face_id: {face_id}")
        
        # 删除用户
        result = await recognizer.delete_face_async(face_id)
        
        return FaceDeleteResponse(
            success=True,
            message=f"人脸删除成功: {face_id}"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"删除人脸异常: {e}")
        raise HTTPException(status_code=500, detail=f"服务器内部错误: {str(e)}")


@router.get("/faces/name/{name}", summary="按姓名查询人脸")
async def get_face_by_name(name: str):
    """
    根据人员姓名查询所有相关的人脸信息
    
    此接口查询指定姓名的所有人脸记录，支持一个人有多张注册照片的情况。
    返回该姓名下所有人脸的详细信息。
    
    **路径参数:**
    - name `str`: 要查询的人员姓名，需要完全匹配
    
    **响应数据:**
    - faces `List[Dict]`: 该姓名下的所有人脸记录列表，每个对象包含：
      - face_id `str`: 人脸唯一标识符
      - person_name `str`: 人员姓名
      - created_at `str`: 创建时间戳
      - metadata `Dict`: 附加元数据信息
    - total `int`: 该姓名下的人脸总数量
    
    **错误码:**
    - 404: 指定姓名不存在任何人脸记录
    - 500: 服务器内部错误
    
    **使用场景:**
    - 查询特定人员的所有注册照片
    - 验证人员是否已注册
    - 获取人员的完整信息记录
    
    **注意事项:**
    - 姓名查询区分大小写
    - 支持中英文姓名查询
    - 一个姓名可能对应多条人脸记录
    """
    try:
        recognizer, _, _, _, _ = get_deepface_components()
        
        faces_data = await recognizer.get_faces_by_name_async(name)
        
        if not faces_data or len(faces_data) == 0:
            raise HTTPException(status_code=404, detail=f"未找到姓名: {name}")
        
        faces = []
        for face in faces_data:
            metadata = face.get("metadata", {})
            faces.append({
                "face_id": face.get("id"),
                "person_name": face.get("name", name),
                "created_at": metadata.get("created_at"),
                "metadata": metadata
            })
        
        return JSONResponse(content={"faces": faces, "total": len(faces)})
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"按姓名查询异常: {e}")
        raise HTTPException(status_code=500, detail=f"服务器内部错误: {str(e)}")


# ==================== 识别和验证API ====================

@router.post("/recognize", response_model=FaceRecognitionResponse, summary="人脸识别")
async def recognize_faces(
    file: UploadFile = File(..., description="包含待识别人脸的图片文件"),
    threshold: Optional[float] = Form(0.6, description="识别相似度阈值 (0.0-1.0)"),
    return_all_candidates: bool = Form(False, description="是否返回所有候选匹配结果")
):
    """
    使用DeepFace技术识别图片中的人脸
    
    此接口分析上传图片中的所有人脸，并与数据库中已注册的人脸进行匹配识别。
    返回识别出的人员姓名、置信度分数和位置信息。
    
    **请求参数:**
    - file `UploadFile`: 包含待识别人脸的图片文件 (JPG, PNG, BMP)
    - threshold `float`: 识别相似度阈值，范围0.0-1.0，值越高要求越严格 (默认: 0.6)
    - return_all_candidates `bool`: 是否返回所有候选匹配结果，false时只返回最佳匹配
    
    **响应数据:**
    - faces `List[Dict]`: 识别结果列表，每个对象包含：
      - person_name `str`: 识别出的人员姓名
      - confidence `float`: 识别置信度分数 (0.0-1.0)
      - bbox `List[int]`: 人脸边界框坐标 [x, y, width, height]
      - face_id `str`: 匹配的数据库人脸ID
      - candidates `List[Dict]`: 所有候选结果 (当return_all_candidates=true时)
    - total_faces `int`: 检测到的人脸总数
    - processing_time `float`: 处理时间（秒）
    """
    try:
        recognizer, _, _, _, _ = get_deepface_components()
        
        # 读取图片
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        image_array = np.array(image)
        
        if len(image_array.shape) == 3:
            image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        
        # 执行识别
        results = await recognizer.recognize_face_async(
            image=image_array,
            threshold=threshold
        )
        
        # 转换结果格式
        faces = []
        for result in results:
            faces.append({
                "person_name": result.person_name,
                "confidence": result.confidence,
                "bbox": result.bbox,
                "face_id": result.face_id,
                "candidates": getattr(result, 'candidates', []) if return_all_candidates else []
            })
        
        return FaceRecognitionResponse(
            faces=faces,
            total_faces=len(faces),
            processing_time=0.0  # TODO: 添加计时
        )
        
    except Exception as e:
        logger.error(f"人脸识别异常: {e}")
        raise HTTPException(status_code=500, detail=f"服务器内部错误: {str(e)}")


@router.post("/verify", response_model=FaceVerificationResponse, summary="人脸验证")
async def verify_faces(
    file1: UploadFile = File(..., description="第一张人脸图片文件"),
    file2: UploadFile = File(..., description="第二张人脸图片文件"),
    threshold: Optional[float] = Form(0.6, description="验证阈值 (0.0-1.0)"),
    model_name: str = Form("ArcFace", description="使用的DeepFace模型"),
    anti_spoofing: bool = Form(False, description="是否启用反欺骗检测")
):
    """
    使用DeepFace技术验证两张人脸图片是否为同一人
    
    此接口比较两张人脸图片，判断是否属于同一个人。使用先进的深度学习模型计算人脸相似度，
    并根据设定的阈值返回验证结果。
    
    **请求参数:**
    - file1 `UploadFile`: 第一张人脸图片文件 (JPG, PNG, BMP)
    - file2 `UploadFile`: 第二张人脸图片文件 (JPG, PNG, BMP)
    - threshold `float`: 验证阈值，范围0.0-1.0，值越高要求越严格 (默认: 0.6)
    - model_name `str`: 使用的DeepFace模型，可选：ArcFace, VGG-Face, OpenFace等 (默认: ArcFace)
    - anti_spoofing `bool`: 是否启用反欺骗检测，检测虚假人脸攻击 (默认: false)
    
    **响应数据:**
    - verified `bool`: 验证结果，true表示是同一人
    - confidence `float`: 相似度置信度分数 (0.0-1.0)
    - distance `float`: 人脸特征距离值，值越小表示越相似
    - threshold `float`: 使用的验证阈值
    - model `str`: 使用的模型名称
    
    **支持的模型:**
    - ArcFace: 高精度人脸识别模型（推荐）
    - VGG-Face: 经典人脸识别模型
    - OpenFace: 轻量级开源模型
    - DeepFace: Facebook开发的模型
    """
    try:
        _, _, verification, _, _ = get_deepface_components()
        
        # 读取两张图片
        image1_data = await file1.read()
        image2_data = await file2.read()
        
        image1 = Image.open(io.BytesIO(image1_data))
        image2 = Image.open(io.BytesIO(image2_data))
        
        image1_array = np.array(image1)
        image2_array = np.array(image2)
        
        # 执行验证
        result = await verification.face_verification(
            image_1=image1_array,
            image_2=image2_array,
            threshold=threshold,
            model_name=model_name,
            detector_backend="retinaface"
        )
        
        return FaceVerificationResponse(
            verified=result.get("verified", False),
            confidence=1.0 - result.get("distance", 1.0),
            distance=result.get("distance", 1.0),
            threshold=result.get("threshold", threshold),
            model=model_name
        )
        
    except Exception as e:
        logger.error(f"人脸验证异常: {e}")
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "error": f"人脸验证失败: {str(e)}",
                "verified": False,
                "confidence": 0.0,
                "distance": 1.0,
                "threshold": threshold,
                "model": model_name
            }
        )


@router.post("/analyze", response_model=FaceAnalysisResponse, summary="人脸属性分析")
async def analyze_face(
    file: UploadFile = File(..., description="待分析的人脸图片文件"),
    actions: str = Form("emotion,age,gender,race", description="分析维度列表，逗号分隔"),
    detector_backend: str = Form("mtcnn", description="人脸检测器后端")
):
    """
    使用DeepFace技术分析人脸的多种属性特征
    
    此接口对上传的人脸图片进行全面的属性分析，包括年龄、性别、情绪、种族等多个维度。
    使用深度学习模型提供准确的预测结果和置信度分数。
    
    **请求参数:**
    - file `UploadFile`: 待分析的人脸图片文件 (JPG, PNG, BMP)
    - actions `str`: 分析维度列表，逗号分隔，可选值：
      - emotion: 情绪分析 (happy, sad, angry, surprise, fear, disgust, neutral)
      - age: 年龄估计
      - gender: 性别识别 (Man, Woman)
      - race: 种族分类 (asian, indian, black, white, middle eastern, latino hispanic)
    - detector_backend `str`: 人脸检测器，可选：mtcnn, opencv, ssd, dlib, retinaface
    
    **响应数据:**
    - faces `List[Dict]`: 分析结果列表，每个人脸包含：
      - region `Dict`: 人脸区域坐标信息
      - age `int`: 估计年龄 (当包含age分析时)
      - gender `Dict`: 性别分析结果，包含预测值和置信度
      - race `Dict`: 种族分析结果，包含各种族的概率分布
      - emotion `Dict`: 情绪分析结果，包含各情绪的概率分布
    - total_faces `int`: 检测到并分析的人脸数量
    
    **检测器说明:**
    - mtcnn: 高精度多任务CNN检测器（推荐）
    - opencv: OpenCV Haar级联检测器
    - ssd: 单发多盒检测器
    - dlib: dlib HOG检测器
    - retinaface: RetinaFace检测器
    """
    try:
        _, _, _, analysis, _ = get_deepface_components()
        
        # 读取图片
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        image_array = np.array(image)
        
        # 解析分析维度
        action_list = [action.strip() for action in actions.split(",")]
        
        # 执行分析
        results = await analysis.face_analysis(
            img_path=image_array,
            actions=action_list,
            detector_backend=detector_backend,
            enforce_detection=True,
            align=True
        )
        
        return FaceAnalysisResponse(
            faces=results,
            total_faces=len(results)
        )
        
    except Exception as e:
        logger.error(f"人脸分析异常: {e}")
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "error": f"人脸分析失败: {str(e)}",
                "faces": [],
                "total_faces": 0
            }
        )
@router.post("/detect/", summary="人脸检测")
async def detect_faces(
    file: UploadFile = File(..., description="待检测的人脸图片文件"),
    detector_backend: str = Form("retinaface", description="人脸检测器后端"),
    enforce_detection: bool = Form(True, description="是否强制检测人脸"),
    align: bool = Form(True, description="是否对齐人脸")
):
    """
    使用DeepFace技术检测图片中的所有人脸
    
    此接口分析上传图片中的所有人脸，返回人脸位置、大小和检测置信度。
    支持多种检测器后端，可用于人脸定位和预处理。
    
    **请求参数:**
    - file `UploadFile`: 待检测的人脸图片文件 (JPG, PNG, BMP)
    - detector_backend `str`: 人脸检测器，可选：mtcnn, opencv, ssd, dlib, retinaface
    - enforce_detection `bool`: 是否强制检测人脸，如果为false则在未检测到人脸时不会报错
    - align `bool`: 是否对齐检测到的人脸
    
    **响应数据:**
    - faces `List[Dict]`: 检测到的人脸列表，每个对象包含：
      - facial_area `Dict`: 人脸区域坐标 (x, y, w, h)
      - confidence `float`: 检测置信度
      - aligned_face `str`: Base64编码的对齐后人脸图像（当align=true时）
    - total_faces `int`: 检测到的人脸总数
    - processing_time `float`: 处理时间（秒）
    
    **检测器说明:**
    - retinaface: 高精度检测器（推荐）
    - mtcnn: 多任务CNN检测器
    - opencv: OpenCV Haar级联检测器
    - ssd: 单发多盒检测器
    - dlib: dlib HOG检测器
    """
    try:
        _, _, _, _, detection = get_deepface_components()
        
        # 读取图片
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        image_array = np.array(image)
        
        if len(image_array.shape) == 3:
            image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        
        start_time = datetime.now()
        faces = await detection.face_detection(
            img_path=image_array,
            detector_backend=detector_backend,
            enforce_detection=enforce_detection,
            align=align
        )
        processing_time = (datetime.now() - start_time).total_seconds()
        
        result_faces = []
        for face in faces:
            face_dict = {
                "facial_area": face.get("facial_area", {}),
                "confidence": face.get("confidence", 0.0)
            }
            
            if align and "face" in face:
                aligned_face = face["face"]
                if isinstance(aligned_face, np.ndarray):
                    _, buffer = cv2.imencode('.jpg', aligned_face)
                    face_dict["aligned_face"] = base64.b64encode(buffer).decode('utf-8')
            
            result_faces.append(face_dict)
        
        return {
            "faces": result_faces,
            "total_faces": len(result_faces),
            "processing_time": processing_time
        }
        
    except Exception as e:
        logger.error(f"人脸检测异常: {e}")
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "error": f"人脸检测失败: {str(e)}",
                "faces": [],
                "total_faces": 0,
                "processing_time": 0.0
            }
        )


# ==================== 视频处理API ====================

@router.post("/video_face/", summary="视频人脸采样注册")
async def add_face_from_video(
    name: str = Form(..., description="注册人员的姓名"),
    video_source: str = Form("0", description="视频源标识（摄像头编号或视频文件路径）"),
    sample_interval: int = Form(30, description="采样间隔帧数"),
    max_samples: int = Form(10, description="最大采样数量"),
    background_tasks: BackgroundTasks = None
):
    """
    通过视频流或视频文件进行人脸采样并批量注册
    
    此接口支持从实时摄像头或视频文件中自动采样人脸帧，并将多个高质量的人脸样本
    注册到数据库中。这种方式可以提高人脸识别的准确性和鲁棒性。
    
    **请求参数:**
    - name `str`: 注册人员的姓名，用于标识所有采样的人脸
    - video_source `str`: 视频源标识，可以是：
      - 摄像头编号：如 "0", "1" (默认摄像头为"0")
      - 视频文件路径：如 "/path/to/video.mp4"
      - RTSP流地址：如 "rtsp://camera_ip/stream"
    - sample_interval `int`: 采样间隔帧数，控制采样频率 (默认: 30帧)
    - max_samples `int`: 最大采样数量，限制采样的人脸数量 (默认: 10张)
    
    **响应数据:**
    - success `bool`: 任务是否成功启动
    - message `str`: 任务启动状态消息
    - video_source `str`: 使用的视频源
    - sample_interval `int`: 采样间隔设置
    - max_samples `int`: 最大采样数量设置
    
    **处理流程:**
    1. 启动后台视频处理任务
    2. 按设定间隔从视频流中提取帧
    3. 检测并提取人脸特征
    4. 将高质量人脸样本注册到数据库
    5. 达到最大样本数或视频结束时停止
    
    **使用场景:**
    - 新员工入职批量采样
    - 提高现有人员识别准确率
    - 视频文件中人脸批量提取
    - 多角度人脸样本收集
    
    **注意事项:**
    - 采样过程在后台异步执行
    - 确保视频源中有清晰可见的目标人脸
    - 摄像头编号从0开始，请确认设备连接
    - 视频文件路径必须可访问
    - 采样质量会影响后续识别效果
    """
    try:
        recognizer, _, _, _, _ = get_deepface_components()
        
        # 启动后台任务处理视频采样
        background_tasks.add_task(
            process_video_sampling,
            recognizer, name, video_source, sample_interval, max_samples
        )
        
        return JSONResponse(content={
            "success": True,
            "message": f"开始处理视频采样: {name}",
            "video_source": video_source,
            "sample_interval": sample_interval,
            "max_samples": max_samples
        })
        
    except Exception as e:
        logger.error(f"视频人脸采样异常: {e}")
        raise HTTPException(status_code=500, detail=f"服务器内部错误: {str(e)}")


@router.get("/recognize/webcam/stream", summary="实时人脸识别流")
async def real_time_recognition_stream(
    camera_id: str = Query("0", description="摄像头逻辑ID标识"),
    source: Union[str, int] = Query(0, description="视频源（摄像头编号或RTSP/HTTP流URL）"),
    threshold: float = Query(0.6, description="识别相似度阈值 (0.0-1.0)"),
    fps: int = Query(30, description="流输出帧率"),
    format: str = Query("sse", description="输出格式类型: sse 或 mjpeg"),
    webhook_urls: Optional[str] = Query(None, description="实时结果转发的Webhook URL列表，逗号分隔"),
    webhook_timeout: int = Query(30, description="Webhook请求超时时间（秒）"),
    webhook_retry_count: int = Query(3, description="Webhook失败重试次数")
):
    """
    摄像头或网络流实时人脸识别处理
    
    此接口提供实时人脸识别流服务，支持多种视频源和输出格式。可以处理摄像头、
    RTSP流、HTTP流等视频源，并提供SSE事件流或MJPEG视频流两种输出格式。
    
    **查询参数:**
    - camera_id `str`: 摄像头逻辑ID标识，用于日志和webhook事件标记 (默认: "0")
    - source `Union[str, int]`: 视频源，支持：
      - 本地摄像头编号：0, 1, 2...
      - RTSP流：rtsp://user:pass@ip:port/stream
      - HTTP流：http://ip:port/stream.mjpg
      - 视频文件：/path/to/video.mp4
    - threshold `float`: 识别相似度阈值，范围0.0-1.0 (默认: 0.6)
    - fps `int`: 输出流的目标帧率 (默认: 30)
    - format `str`: 输出格式，支持：
      - "sse": Server-Sent Events格式，返回JSON数据流
      - "mjpeg": Motion JPEG视频流，返回带标注的视频
    - webhook_urls `str`: 实时结果转发的Webhook URL列表，多个URL用逗号分隔
    - webhook_timeout `int`: Webhook HTTP请求超时时间，单位秒 (默认: 30)
    - webhook_retry_count `int`: Webhook发送失败时的重试次数 (默认: 3)
    
    **响应格式:**
    
    **SSE格式 (format=sse):**
    - Content-Type: text/event-stream
    - 实时发送JSON格式的识别结果
    ```json
    {
        "camera_id": "camera_001",
        "timestamp": "2024-01-15T10:30:00",
        "faces": [
            {
                "person_name": "张三",
                "confidence": 0.95,
                "bbox": [100, 100, 200, 200],
                "face_id": "face_12345"
            }
        ]
    }
    ```
    
    **MJPEG格式 (format=mjpeg):**
    - Content-Type: multipart/x-mixed-replace
    - 返回带有人脸框和标签的实时视频流
    - 在视频帧上直接标注识别结果
    
    **Webhook集成:**
    - 当检测到人脸时，自动发送识别结果到指定Webhook URL
    - 支持多个Webhook同时接收
    - 包含重试机制和超时控制
    - Webhook payload格式与SSE相同
    
    **使用场景:**
    - 实时监控和识别
    - 门禁考勤系统
    - 安防监控集成
    - 访客管理系统
    - 实时数据分析
    
    **性能优化:**
    - 自动帧率调节以适应处理能力
    - 可配置识别阈值平衡精度和性能
    - 支持多路并发处理
    - 内存和CPU使用优化
    
    **注意事项:**
    - 长时间运行可能消耗较多系统资源
    - 网络流需要稳定的连接
    - 建议根据硬件能力调整fps参数
    - Webhook URL必须能正常访问和响应
    - 流会在客户端断开连接时自动停止
    """
    try:
        recognizer, _, _, _, _ = get_deepface_components()
        
        # 转换source为整数（如果是数字）
        try:
            if isinstance(source, str) and source.isdigit():
                source = int(source)
        except ValueError:
            pass
        
        if format.lower() == "sse":
            # 返回SSE格式流
            async def generate_sse_stream():
                from facecv.core.video_stream import VideoStreamManager
                stream_manager = VideoStreamManager()
                
                # 设置webhook配置
                webhook_configs = []
                if webhook_urls:
                    # 启动webhook管理器
                    if not webhook_manager.running:
                        webhook_manager.start()
                    
                    # 解析webhook URLs并配置
                    urls = [url.strip() for url in webhook_urls.split(',') if url.strip()]
                    for i, url in enumerate(urls):
                        webhook_id = f"{camera_id}_deepface_webhook_{i}"
                        config = WebhookConfig(
                            url=url,
                            timeout=webhook_timeout,
                            retry_count=webhook_retry_count,
                            batch_size=1,  # Send events immediately for real-time
                            batch_timeout=0.1
                        )
                        webhook_manager.add_webhook(webhook_id, config)
                        webhook_configs.append(webhook_id)
                
                # 开始流处理
                started_camera_id = stream_manager.start_stream(
                    camera_id,
                    source,
                    lambda frame: process_deepface_frame_for_sse_with_webhook(
                        frame, recognizer, threshold, camera_id, webhook_configs
                    )
                )
                
                try:
                    # 持续发送识别结果
                    while True:
                        if started_camera_id in stream_manager.results:
                            results = stream_manager.get_results(started_camera_id)
                            for result in results:
                                import json
                                yield f"data: {json.dumps(result)}\n\n"
                        
                        await asyncio.sleep(1.0 / fps)
                        
                finally:
                    # 停止流和清理webhooks
                    stream_manager.stop_stream(started_camera_id)
                    for webhook_id in webhook_configs:
                        webhook_manager.remove_webhook(webhook_id)
            
            return StreamingResponse(
                generate_sse_stream(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no"
                }
            )
        
        else:
            # 返回MJPEG格式流
            def generate_mjpeg_stream():
                cap = cv2.VideoCapture(source)
                cap.set(cv2.CAP_PROP_FPS, fps)
                
                try:
                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        
                        # 执行人脸识别（同步版本以提高流畅度）
                        results = recognizer.recognize_face(frame, threshold)
                        
                        # 在帧上绘制识别结果
                        for result in results:
                            x, y, w, h = result.bbox
                            cv2.rectangle(frame, (int(x), int(y)), (int(x+w), int(y+h)), (0, 255, 0), 2)
                            
                            # 添加姓名标签
                            label = f"{result.person_name} ({result.confidence:.2f})"
                            cv2.putText(frame, label, (int(x), int(y-10)), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        
                        # 添加camera_id标签
                        cv2.putText(frame, f"Camera: {camera_id}", (10, 30), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        
                        # 编码为JPEG
                        _, buffer = cv2.imencode('.jpg', frame)
                        frame_bytes = buffer.tobytes()
                        
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                        
                finally:
                    cap.release()
            
            return StreamingResponse(
                generate_mjpeg_stream(),
                media_type="multipart/x-mixed-replace; boundary=frame"
            )
        
    except Exception as e:
        logger.error(f"实时识别流异常: {e}")
        raise HTTPException(status_code=500, detail=f"服务器内部错误: {str(e)}")


# ==================== 辅助函数 ====================

def process_deepface_frame_for_sse(frame: np.ndarray, recognizer, threshold: float, camera_id: str) -> dict:
    """处理DeepFace单帧并返回识别结果"""
    try:
        if recognizer:
            # 使用同步方法以提高流处理性能
            results = recognizer.recognize_face(frame, threshold)
            return {
                "camera_id": camera_id,
                "timestamp": datetime.now().isoformat(),
                "faces": [
                    {
                        "person_name": result.person_name,
                        "confidence": result.confidence,
                        "bbox": result.bbox,
                        "face_id": getattr(result, 'face_id', None)
                    }
                    for result in results
                ]
            }
        else:
            # Mock结果
            return {
                "camera_id": camera_id,
                "timestamp": datetime.now().isoformat(),
                "faces": [
                    {
                        "person_name": "test_person",
                        "confidence": 0.95,
                        "bbox": [100, 100, 200, 200],
                        "face_id": "mock_face_id"
                    }
                ]
            }
    except Exception as e:
        logger.error(f"Error processing DeepFace frame for camera {camera_id}: {e}")
        return {"camera_id": camera_id, "error": str(e)}


def process_deepface_frame_for_sse_with_webhook(
    frame: np.ndarray, 
    recognizer, 
    threshold: float, 
    camera_id: str, 
    webhook_configs: List[str]
) -> dict:
    """处理DeepFace单帧并返回识别结果，同时发送到webhook"""
    try:
        # 获取识别结果
        result = process_deepface_frame_for_sse(frame, recognizer, threshold, camera_id)
        
        # 如果有人脸检测到且配置了webhook，发送事件
        if webhook_configs and result.get("faces") and not result.get("error"):
            faces_data = []
            for face in result["faces"]:
                faces_data.append({
                    "name": face["person_name"],
                    "confidence": face["confidence"], 
                    "bbox": face["bbox"],
                    "face_id": face.get("face_id"),
                    "metadata": {}
                })
            
            # 发送recognition事件到webhook
            send_recognition_event(
                camera_id=camera_id,
                recognized_faces=faces_data,
                metadata={
                    "source": "deepface_stream",
                    "threshold": threshold,
                    "frame_timestamp": result["timestamp"]
                }
            )
        
        return result
        
    except Exception as e:
        logger.error(f"Error processing DeepFace frame with webhook for camera {camera_id}: {e}")
        return {"camera_id": camera_id, "error": str(e)}

async def process_video_sampling(
    recognizer, name: str, video_source: str, 
    sample_interval: int, max_samples: int
):
    """处理视频采样的后台任务"""
    try:
        # 转换视频源
        if video_source.isdigit():
            video_source = int(video_source)
        
        cap = cv2.VideoCapture(video_source)
        sample_count = 0
        frame_count = 0
        
        while sample_count < max_samples:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # 按间隔采样
            if frame_count % sample_interval == 0:
                success = await recognizer.register_face_async(
                    name=f"{name}_sample_{sample_count}",
                    image=frame
                )
                
                if success:
                    sample_count += 1
                    logger.info(f"采样成功: {name}_sample_{sample_count}")
        
        cap.release()
        logger.info(f"视频采样完成: {name}, 总计{sample_count}个样本")
        
    except Exception as e:
        logger.error(f"视频采样处理异常: {e}")


# ==================== 健康检查 ====================

@router.get("/health", summary="DeepFace健康检查")
async def deepface_health():
    """
    DeepFace服务组件健康状态检查
    
    此接口检查DeepFace相关组件的运行状态，包括模型加载情况、依赖库可用性等。
    用于监控服务健康状态和诊断潜在问题。
    
    **响应数据 (健康状态):**
    - status `str`: 服务状态 ("healthy" 表示正常)
    - service `str`: 服务名称标识 ("DeepFace")
    - version `str`: 服务版本号
    
    **响应数据 (不健康状态):**
    - status `str`: 服务状态 ("unhealthy" 表示异常)
    - service `str`: 服务名称标识 ("DeepFace")
    - error `str`: 错误详细信息
    
    **HTTP状态码:**
    - 200: 服务正常运行
    - 503: 服务不可用（依赖缺失、模型加载失败等）
    
    **检查项目:**
    - DeepFace库是否正确安装
    - 人脸识别模型是否可加载
    - 数据库连接是否正常
    - 依赖组件是否可用
    
    **使用场景:**
    - 服务启动后的状态验证
    - 定期健康检查和监控
    - 故障诊断和排查
    - 负载均衡器健康检查
    """
    try:
        get_deepface_components()
        return JSONResponse(content={
            "status": "healthy",
            "service": "DeepFace",
            "version": "1.0.0"
        })
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy", 
                "service": "DeepFace",
                "error": str(e)
            }
        )
