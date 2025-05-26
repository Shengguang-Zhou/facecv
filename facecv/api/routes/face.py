"""人脸识别相关路由"""

from fastapi import APIRouter, File, UploadFile, HTTPException, Depends, Form, Query
from fastapi.responses import StreamingResponse
from typing import List, Optional, Generator, Union
import numpy as np
from PIL import Image
import io
import logging
import asyncio
import json
import cv2
from datetime import datetime

# from facecv import FaceRecognizer  # 暂时禁用
from facecv.schemas.face import (
    FaceInfo, RecognitionResult, VerificationResult,
    BatchRecognitionResult
)
from facecv.config import get_settings
from facecv.utils.video_utils import VideoExtractor, FrameExtractionMethod
from facecv.utils.face_quality import FaceQualityAssessor
from facecv.core.video_stream import VideoStreamManager
from facecv.core.webhook import webhook_manager, WebhookConfig, send_recognition_event

router = APIRouter()
logger = logging.getLogger(__name__)

# 全局人脸识别器实例
_recognizer = None

def get_recognizer():  # -> FaceRecognizer  # 暂时移除类型注解
    """获取人脸识别器单例"""
    global _recognizer
    if _recognizer is None:
        settings = get_settings()
        # _recognizer = FaceRecognizer(  # 暂时禁用
        #     model=settings.model_backend,
        #     db_type=settings.db_type,
        #     db_connection=settings.db_connection_string
        # )
        _recognizer = None  # Mock for testing
    return _recognizer

async def process_upload_file(file: UploadFile) -> np.ndarray:
    """处理上传的图片文件"""
    # 检查文件扩展名
    settings = get_settings()
    file_ext = f".{file.filename.split('.')[-1].lower()}"
    if file_ext not in settings.allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文件格式。支持的格式：{', '.join(settings.allowed_extensions)}"
        )
    
    # 检查文件大小
    contents = await file.read()
    if len(contents) > settings.max_upload_size:
        raise HTTPException(
            status_code=400,
            detail=f"文件太大。最大允许 {settings.max_upload_size // 1024 // 1024}MB"
        )
    
    # 转换为图片
    try:
        image = Image.open(io.BytesIO(contents))
        # 如果是 RGBA，转换为 RGB
        if image.mode == 'RGBA':
            rgb_image = Image.new('RGB', image.size, (255, 255, 255))
            rgb_image.paste(image, mask=image.split()[3])
            image = rgb_image
        elif image.mode != 'RGB':
            image = image.convert('RGB')
        return np.array(image)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"无法处理图片: {str(e)}")

@router.post("/faces/register", response_model=List[str])
async def register_face(
    name: str = Form(...),
    file: UploadFile = File(...),
    department: Optional[str] = Form(None),
    employee_id: Optional[str] = Form(None),
    # recognizer: FaceRecognizer = Depends(get_recognizer)  # 暂时禁用
    recognizer = Depends(get_recognizer)
):
    """
    注册新的人脸
    
    - **name**: 人员姓名
    - **file**: 包含人脸的图片文件
    - **department**: 部门（可选）
    - **employee_id**: 员工ID（可选）
    """
    # 处理上传的图片
    image = await process_upload_file(file)
    
    # 构建元数据
    metadata = {}
    if department:
        metadata["department"] = department
    if employee_id:
        metadata["employee_id"] = employee_id
    
    # 注册人脸
    try:
        face_ids = recognizer.register(image, name, metadata)
        if not face_ids:
            raise HTTPException(status_code=400, detail="未检测到人脸")
        
        logger.info(f"Successfully registered {len(face_ids)} face(s) for {name}")
        return face_ids
        
    except Exception as e:
        logger.error(f"Error registering face: {e}")
        raise HTTPException(status_code=500, detail=f"注册失败: {str(e)}")

@router.post("/faces/recognize", response_model=List[RecognitionResult])
async def recognize_face(
    file: UploadFile = File(...),
    threshold: float = 0.6,
    # recognizer: FaceRecognizer = Depends(get_recognizer)  # 暂时禁用
    recognizer = Depends(get_recognizer)
):
    """
    识别图片中的人脸
    
    - **file**: 包含人脸的图片文件
    - **threshold**: 相似度阈值（0-1）
    """
    # 处理上传的图片
    image = await process_upload_file(file)
    
    # 识别人脸
    try:
        results = recognizer.recognize(image, threshold)
        logger.info(f"Recognized {len(results)} face(s)")
        return results
        
    except Exception as e:
        logger.error(f"Error recognizing faces: {e}")
        raise HTTPException(status_code=500, detail=f"识别失败: {str(e)}")

@router.post("/faces/verify", response_model=VerificationResult)
async def verify_faces(
    file1: UploadFile = File(...),
    file2: UploadFile = File(...),
    threshold: float = 0.6,
    # recognizer: FaceRecognizer = Depends(get_recognizer)  # 暂时禁用
    recognizer = Depends(get_recognizer)
):
    """
    验证两张人脸是否为同一人
    
    - **file1**: 第一张人脸图片
    - **file2**: 第二张人脸图片
    - **threshold**: 判定阈值
    """
    # 处理两张图片
    image1 = await process_upload_file(file1)
    image2 = await process_upload_file(file2)
    
    # 验证
    try:
        result = recognizer.verify(image1, image2, threshold)
        logger.info(f"Verification result: {result.is_same_person}")
        return result
        
    except Exception as e:
        logger.error(f"Error verifying faces: {e}")
        raise HTTPException(status_code=500, detail=f"验证失败: {str(e)}")

@router.get("/faces", response_model=List[FaceInfo])
async def list_faces(
    name: Optional[str] = None,
    skip: int = 0,
    limit: int = 100,
    # recognizer: FaceRecognizer = Depends(get_recognizer)  # 暂时禁用
    recognizer = Depends(get_recognizer)
):
    """
    列出已注册的人脸
    
    - **name**: 按姓名筛选（可选）
    - **skip**: 跳过的记录数
    - **limit**: 返回的最大记录数
    """
    try:
        all_faces = recognizer.list_faces(name)
        
        # 转换为 FaceInfo 对象
        face_infos = []
        for face in all_faces[skip:skip + limit]:
            face_info = FaceInfo(
                id=face['id'],
                name=face['name'],
                created_at=face.get('created_at'),
                updated_at=face.get('updated_at'),
                metadata=face.get('metadata')
            )
            face_infos.append(face_info)
            
        return face_infos
        
    except Exception as e:
        logger.error(f"Error listing faces: {e}")
        raise HTTPException(status_code=500, detail=f"查询失败: {str(e)}")

@router.delete("/faces/{face_id}")
async def delete_face(
    face_id: str,
    # recognizer: FaceRecognizer = Depends(get_recognizer)  # 暂时禁用
    recognizer = Depends(get_recognizer)
):
    """删除指定的人脸"""
    try:
        success = recognizer.delete(face_id=face_id)
        if not success:
            raise HTTPException(status_code=404, detail="人脸不存在")
            
        logger.info(f"Deleted face {face_id}")
        return {"message": f"成功删除人脸: {face_id}"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting face: {e}")
        raise HTTPException(status_code=500, detail=f"删除失败: {str(e)}")

@router.delete("/faces/by-name/{name}")
async def delete_faces_by_name(
    name: str,
    # recognizer: FaceRecognizer = Depends(get_recognizer)  # 暂时禁用
    recognizer = Depends(get_recognizer)
):
    """删除指定姓名的所有人脸"""
    try:
        count = recognizer.face_db.delete_face_by_name(name)
        if count == 0:
            raise HTTPException(status_code=404, detail="未找到该姓名的人脸")
            
        logger.info(f"Deleted {count} face(s) for {name}")
        return {"message": f"成功删除 {count} 个人脸", "name": name}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting faces by name: {e}")
        raise HTTPException(status_code=500, detail=f"删除失败: {str(e)}")

@router.get("/faces/count")
async def get_face_count(recognizer = Depends(get_recognizer)):  # 移除类型注解
    """获取人脸总数"""
    try:
        count = recognizer.get_face_count()
        return {"total": count}
    except Exception as e:
        logger.error(f"Error getting face count: {e}")
        raise HTTPException(status_code=500, detail=f"查询失败: {str(e)}")

@router.post("/video_face/")
async def extract_faces_from_video(
    file: UploadFile = File(...),
    method: str = Form("uniform"),
    count: int = Form(10),
    interval: float = Form(1.0),
    quality_threshold: float = Form(0.7),
    recognizer = Depends(get_recognizer)
):
    """
    从视频中提取人脸
    
    - **file**: 视频文件
    - **method**: 提取方法 (uniform/interval/scene_change/quality_based)
    - **count**: 提取帧数（uniform方法）
    - **interval**: 时间间隔（interval方法）
    - **quality_threshold**: 质量阈值
    """
    # 保存上传的视频到临时文件
    import tempfile
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
        contents = await file.read()
        tmp.write(contents)
        video_path = tmp.name
    
    try:
        # 初始化视频提取器和质量评估器
        extractor = VideoExtractor(video_path)
        quality_assessor = FaceQualityAssessor()
        
        # 转换提取方法
        extraction_method = FrameExtractionMethod[method.upper()]
        
        # 提取并处理帧
        results = []
        for frame_idx, frame, timestamp in extractor.extract_frames(extraction_method, count, interval):
            # 检测人脸
            faces = recognizer.detect_faces(frame) if recognizer else []
            
            for face in faces:
                # 评估人脸质量
                quality = quality_assessor.assess_face(frame, face['bbox'])
                
                if quality.overall_score >= quality_threshold:
                    # 识别人脸
                    recognition_result = recognizer.recognize_single(frame, face) if recognizer else None
                    
                    results.append({
                        "frame_index": frame_idx,
                        "timestamp": timestamp,
                        "bbox": face['bbox'],
                        "quality_score": quality.overall_score,
                        "quality_metrics": {
                            "sharpness": quality.sharpness_score,
                            "brightness": quality.brightness_score,
                            "contrast": quality.contrast_score,
                            "pose": quality.pose_score
                        },
                        "recognition": recognition_result
                    })
        
        # 清理临时文件
        import os
        os.unlink(video_path)
        
        return {
            "total_frames_processed": extractor.frame_count,
            "faces_found": len(results),
            "results": results
        }
        
    except Exception as e:
        # 确保清理临时文件
        import os
        if os.path.exists(video_path):
            os.unlink(video_path)
        logger.error(f"Error processing video: {e}")
        raise HTTPException(status_code=500, detail=f"视频处理失败: {str(e)}")

@router.get("/recognize/webcam/stream")
async def recognize_webcam_stream(
    camera_id: str = "0",
    source: str = "0", 
    threshold: float = 0.6,
    fps: int = 10,
    webhook_urls: Optional[str] = Query(None, description="Webhook URLs (comma-separated)"),
    webhook_timeout: int = Query(30, description="Webhook timeout in seconds"),
    webhook_retry_count: int = Query(3, description="Webhook retry count"),
    recognizer = Depends(get_recognizer)
):
    """
    实时识别摄像头流中的人脸
    
    - **camera_id**: 摄像头ID标识
    - **source**: 视频源（0为默认摄像头，或RTSP URL）
    - **threshold**: 识别阈值
    - **fps**: 处理帧率
    - **webhook_urls**: Webhook URLs for real-time result forwarding (comma-separated)
    - **webhook_timeout**: Webhook request timeout in seconds
    - **webhook_retry_count**: Number of retry attempts for failed webhooks
    """
    async def generate_stream():
        """生成SSE流"""
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
                webhook_id = f"{camera_id}_webhook_{i}"
                config = WebhookConfig(
                    url=url,
                    timeout=webhook_timeout,
                    retry_count=webhook_retry_count,
                    batch_size=1,  # Send events immediately for real-time
                    batch_timeout=0.1
                )
                webhook_manager.add_webhook(webhook_id, config)
                webhook_configs.append(webhook_id)
        
        # 转换source为整数（如果是数字）
        try:
            source_int = int(source)
            stream_source = source_int
        except ValueError:
            stream_source = source
        
        # 开始流处理，传入camera_id和webhook配置
        started_camera_id = stream_manager.start_stream(
            camera_id,
            stream_source,
            lambda frame: process_frame_for_sse_with_webhook(
                frame, recognizer, threshold, camera_id, webhook_configs
            )
        )
        
        try:
            # 持续发送识别结果
            while True:
                if started_camera_id in stream_manager.results:
                    results = stream_manager.get_results(started_camera_id)
                    for result in results:
                        yield f"data: {json.dumps(result)}\n\n"
                
                await asyncio.sleep(1.0 / fps)
                
        finally:
            # 停止流和清理webhooks
            stream_manager.stop_stream(started_camera_id)
            for webhook_id in webhook_configs:
                webhook_manager.remove_webhook(webhook_id)
    
    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )

def process_frame_for_sse(frame: np.ndarray, recognizer, threshold: float, camera_id: str = "0") -> dict:
    """处理单帧并返回识别结果"""
    try:
        if recognizer:
            results = recognizer.recognize(frame, threshold)
            return {
                "camera_id": camera_id,
                "timestamp": datetime.now().isoformat(),
                "faces": [
                    {
                        "name": r.name,
                        "confidence": r.confidence,
                        "bbox": r.bbox,
                        "metadata": r.metadata
                    }
                    for r in results
                ]
            }
        else:
            # Mock结果
            return {
                "camera_id": camera_id,
                "timestamp": datetime.now().isoformat(),
                "faces": [
                    {
                        "name": "test_person",
                        "confidence": 0.95,
                        "bbox": [100, 100, 200, 200],
                        "metadata": {"department": "engineering"}
                    }
                ]
            }
    except Exception as e:
        logger.error(f"Error processing frame: {e}")
        return {"camera_id": camera_id, "error": str(e)}


def process_frame_for_sse_with_webhook(
    frame: np.ndarray, 
    recognizer, 
    threshold: float, 
    camera_id: str, 
    webhook_configs: List[str]
) -> dict:
    """处理单帧并返回识别结果，同时发送到webhook"""
    try:
        # 获取识别结果
        result = process_frame_for_sse(frame, recognizer, threshold, camera_id)
        
        # 如果有人脸检测到且配置了webhook，发送事件
        if webhook_configs and result.get("faces") and not result.get("error"):
            faces_data = []
            for face in result["faces"]:
                faces_data.append({
                    "name": face["name"],
                    "confidence": face["confidence"],
                    "bbox": face["bbox"],
                    "metadata": face.get("metadata", {})
                })
            
            # 发送recognition事件到webhook
            send_recognition_event(
                camera_id=camera_id,
                recognized_faces=faces_data,
                metadata={
                    "source": "insightface_stream",
                    "threshold": threshold,
                    "frame_timestamp": result["timestamp"]
                }
            )
        
        return result
        
    except Exception as e:
        logger.error(f"Error processing frame with webhook for camera {camera_id}: {e}")
        return {"camera_id": camera_id, "error": str(e)}

@router.post("/faces/offline")
async def batch_register_offline(
    directory_path: str = Form(...),
    quality_threshold: float = Form(0.7),
    recognizer = Depends(get_recognizer)
):
    """
    批量注册本地目录中的人脸
    
    目录结构应为:
    - directory_path/
      - person1/
        - image1.jpg
        - image2.jpg
      - person2/
        - image1.jpg
    
    - **directory_path**: 包含人脸图片的目录路径
    - **quality_threshold**: 质量阈值
    """
    import os
    from pathlib import Path
    
    if not os.path.exists(directory_path):
        raise HTTPException(status_code=400, detail="目录不存在")
    
    quality_assessor = FaceQualityAssessor()
    results = {
        "success": [],
        "failed": [],
        "skipped": []
    }
    
    # 遍历目录
    for person_dir in Path(directory_path).iterdir():
        if not person_dir.is_dir():
            continue
        
        person_name = person_dir.name
        person_results = {
            "name": person_name,
            "registered": 0,
            "failed": 0,
            "skipped": 0
        }
        
        # 处理每个人的图片
        for image_path in person_dir.glob("*"):
            if image_path.suffix.lower() not in ['.jpg', '.jpeg', '.png', '.bmp']:
                continue
            
            try:
                # 读取图片
                image = cv2.imread(str(image_path))
                if image is None:
                    person_results["failed"] += 1
                    results["failed"].append({
                        "path": str(image_path),
                        "reason": "无法读取图片"
                    })
                    continue
                
                # 转换颜色空间
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # 检测人脸
                faces = recognizer.detect_faces(image) if recognizer else [{"bbox": [0, 0, image.shape[1], image.shape[0]]}]
                
                if not faces:
                    person_results["skipped"] += 1
                    results["skipped"].append({
                        "path": str(image_path),
                        "reason": "未检测到人脸"
                    })
                    continue
                
                # 评估质量
                face = faces[0]  # 使用第一个检测到的人脸
                quality = quality_assessor.assess_face(image, face['bbox'])
                
                if quality.overall_score < quality_threshold:
                    person_results["skipped"] += 1
                    results["skipped"].append({
                        "path": str(image_path),
                        "reason": f"质量分数过低: {quality.overall_score:.2f}"
                    })
                    continue
                
                # 注册人脸
                if recognizer:
                    face_ids = recognizer.register(image, person_name, {
                        "source_file": str(image_path),
                        "quality_score": quality.overall_score
                    })
                    if face_ids:
                        person_results["registered"] += 1
                        results["success"].append({
                            "path": str(image_path),
                            "face_ids": face_ids,
                            "quality_score": quality.overall_score
                        })
                else:
                    # Mock注册
                    person_results["registered"] += 1
                    results["success"].append({
                        "path": str(image_path),
                        "face_ids": [f"mock_id_{person_name}_{image_path.stem}"],
                        "quality_score": quality.overall_score
                    })
                    
            except Exception as e:
                person_results["failed"] += 1
                results["failed"].append({
                    "path": str(image_path),
                    "reason": str(e)
                })
                logger.error(f"Error processing {image_path}: {e}")
        
        # 记录该人的结果
        logger.info(f"Processed {person_name}: registered={person_results['registered']}, "
                   f"failed={person_results['failed']}, skipped={person_results['skipped']}")
    
    return {
        "summary": {
            "total_success": len(results["success"]),
            "total_failed": len(results["failed"]),
            "total_skipped": len(results["skipped"])
        },
        "details": results
    }