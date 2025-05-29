"""Batch Processing API Routes for InsightFace

DEPRECATED: These batch processing endpoints are deprecated and will be removed in a future version.
Please use the individual endpoints with proper client-side batching instead.
"""

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
import warnings

from facecv.models.insightface.real_recognizer import RealInsightFaceRecognizer
from facecv.schemas.face import FaceDetection, VerificationResult, RecognitionResult, FaceRegisterResponse
from facecv.config import get_settings
from facecv.database.sqlite_facedb import SQLiteFaceDB

router = APIRouter(prefix="/api/v1/batch", tags=["Batch Processing (Deprecated)"])
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

@router.post("/detect", 
    response_model=Dict[str, List[FaceDetection]],
    summary="批量检测人脸 (已弃用)",
    description="""**⚠️ DEPRECATED**: 此接口已被弃用，将在未来版本中移除。请使用单个检测接口进行客户端批量处理。
    
    在多张图片中同时检测人脸。
    
    该接口支持并行处理多张图片，高效检测每张图片中的所有人脸。
    返回每张图片的检测结果，包括人脸边界框、置信度和质量分数。
    
    **处理流程：**
    1. 并行处理所有上传的图片
    2. 检测每张图片中的人脸
    3. 根据最小置信度过滤结果
    4. 返回按图片分组的检测结果
    
    **返回格式：**
    - 键：image_{序号}_{文件名}
    - 值：该图片中检测到的人脸列表
    """,
    deprecated=True)
async def batch_detect_faces(
    files: List[UploadFile] = File(..., description="多个图片文件"),
    min_confidence: float = Query(0.5, description="最小检测置信度")
):
    warnings.warn(
        "The batch detect endpoint is deprecated and will be removed in a future version. "
        "Please use the individual detect endpoint with client-side batching.",
        DeprecationWarning,
        stacklevel=2
    )
    logger.warning("Deprecated batch detect endpoint called")
    
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

@router.post("/register", 
    response_model=Dict[str, Any],
    summary="批量注册人脸 (已弃用)",
    description="""**⚠️ DEPRECATED**: 此接口已被弃用，将在未来版本中移除。请使用单个注册接口进行客户端批量处理。
    
    从多张图片中批量注册人脸。
    
    该接口支持一次性注册多个人脸，每张图片对应一个人脸（建议每张图片只包含一个人脸）。
    所有注册操作并行处理，提高效率。
    
    **使用说明：**
    - 图片数量必须与姓名数量匹配
    - 每张图片建议只包含一个人脸
    - 支持为所有人脸设置相同的部门和元数据
    
    **返回信息：**
    - successful: 成功注册的人脸列表
    - failed: 注册失败的人脸列表（包含失败原因）
    - total: 总处理数量
    - success_rate: 成功率
    """,
    deprecated=True)
async def batch_register_faces(
    files: List[UploadFile] = File(..., description="多个人脸图片"),
    names: str = Form(..., description="逗号分隔的姓名，对应图片顺序"),
    department: Optional[str] = Form(None, description="所有人脸的部门（可选）"),
    metadata: Optional[str] = Form(None, description="JSON格式的元数据（可选）")
):
    warnings.warn(
        "The batch register endpoint is deprecated and will be removed in a future version. "
        "Please use the individual register endpoint with client-side batching.",
        DeprecationWarning,
        stacklevel=2
    )
    logger.warning("Deprecated batch register endpoint called")
    
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

@router.post("/recognize", 
    response_model=Dict[str, List[RecognitionResult]],
    summary="批量识别人脸 (已弃用)",
    description="""**⚠️ DEPRECATED**: 此接口已被弃用，将在未来版本中移除。请使用单个识别接口进行客户端批量处理。
    
    在多张图片中批量识别人脸。
    
    该接口并行处理多张图片，识别每张图片中的所有人脸，
    并与数据库中已注册的人脸进行匹配。
    
    **识别流程：**
    1. 并行处理所有上传的图片
    2. 检测每张图片中的所有人脸
    3. 将检测到的人脸与数据库进行匹配
    4. 返回识别结果（包括姓名和相似度）
    
    **返回格式：**
    - 键：image_{序号}_{文件名}
    - 值：该图片中识别到的人脸结果列表
    """,
    deprecated=True)
async def batch_recognize_faces(
    files: List[UploadFile] = File(..., description="多个要识别人脸的图片"),
    threshold: float = Query(0.4, description="识别相似度阈值")
):
    warnings.warn(
        "The batch recognize endpoint is deprecated and will be removed in a future version. "
        "Please use the individual recognize endpoint with client-side batching.",
        DeprecationWarning,
        stacklevel=2
    )
    logger.warning("Deprecated batch recognize endpoint called")
    
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

@router.post("/verify", 
    response_model=List[Dict[str, Any]],
    summary="批量验证人脸 (已弃用)",
    description="""**⚠️ DEPRECATED**: 此接口已被弃用，将在未来版本中移除。请使用单个验证接口进行客户端批量处理。
    
    批量验证多张图片中的人脸。
    
    该接口支持批量验证，可以：
    1. **成对比对**：将所有图片分成两组进行一一对应比对
    2. **自我交叉比对**：所有图片相互之间进行比对（去重）
    
    **使用场景：**
    - 批量验证身份证照片与现场照片
    - 人脸库去重（交叉比对模式）
    - 批量身份验证
    
    **参数说明：**
    - files: 要比对的图片文件列表
    - threshold: 相似度阈值
    - cross_compare: 如果为true，所有图片相互比对；如果为false，前一半与后一半比对
    
    **返回结果包括：**
    - reference: 参考图片标识
    - comparison: 比对图片标识
    - is_same_person: 是否为同一人
    - confidence: 相似度置信度
    - distance: 特征向量距离
    - threshold: 使用的阈值
    """,
    deprecated=True)
async def batch_verify_faces(
    files: List[UploadFile] = File(..., description="要比对的图片文件"),
    threshold: float = Query(0.4, description="相似度阈值"),
    cross_compare: bool = Query(False, description="是否进行交叉比对（所有图片相互比对）")
):
    warnings.warn(
        "The batch verify endpoint is deprecated and will be removed in a future version. "
        "Please use the individual verify endpoint with client-side batching.",
        DeprecationWarning,
        stacklevel=2
    )
    logger.warning("Deprecated batch verify endpoint called")
    
    recognizer = get_recognizer()
    results = []
    
    if len(files) < 2:
        raise HTTPException(
            status_code=400,
            detail="At least 2 images are required for verification"
        )
    
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
        # Compare all images with each other (excluding self-comparison)
        pairs = [
            (i, files[i], j, files[j])
            for i in range(len(files))
            for j in range(i + 1, len(files))
        ]
    else:
        # Split files into two halves and compare them pairwise
        mid = len(files) // 2
        if len(files) % 2 != 0:
            raise HTTPException(
                status_code=400,
                detail=f"For pairwise comparison, even number of images required. Got {len(files)} images."
            )
        
        reference_images = files[:mid]
        comparison_images = files[mid:]
        
        pairs = [
            (i, ref_file, mid + i, comp_file)
            for i, (ref_file, comp_file) in enumerate(zip(reference_images, comparison_images))
        ]
    
    # Process all verifications concurrently
    verification_results = await asyncio.gather(*[
        verify_pair(ref_idx, ref_file, comp_idx, comp_file)
        for ref_idx, ref_file, comp_idx, comp_file in pairs
    ])
    
    return verification_results

# ==================== Batch Analysis ====================

@router.post("/analyze", 
    response_model=Dict[str, Any],
    summary="批量分析人脸 (已弃用)",
    description="""**⚠️ DEPRECATED**: 此接口已被弃用，将在未来版本中移除。请使用单个分析接口进行客户端批量处理。
    
    批量分析多张图片中的人脸属性。
    
    该接口检测并分析每张图片中的所有人脸，提取人脸属性信息。
    支持的属性包括：
    - 年龄估计
    - 性别识别
    - 人脸质量分数
    - 面部特征点（可选）
    - 人脸特征向量（可选）
    
    **分析流程：**
    1. 并行处理所有上传的图片
    2. 检测每张图片中的人脸
    3. 提取每个人脸的属性信息
    4. 汇总统计信息
    
    **返回内容：**
    - 每张图片的分析结果
    - 汇总统计（总图片数、总人脸数、平均每图人脸数）
    """,
    deprecated=True)
async def batch_analyze_faces(
    files: List[UploadFile] = File(..., description="多个要分析的图片"),
    include_embeddings: bool = Query(False, description="是否包含人脸特征向量")
):
    warnings.warn(
        "The batch analyze endpoint is deprecated and will be removed in a future version. "
        "Please use the individual analyze endpoint with client-side batching.",
        DeprecationWarning,
        stacklevel=2
    )
    logger.warning("Deprecated batch analyze endpoint called")
    
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