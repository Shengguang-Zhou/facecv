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

def get_deepface_components():
    """获取DeepFace组件实例（延迟加载）"""
    global deepface_recognizer, face_embedding, face_verification, face_analysis
    
    if deepface_recognizer is None:
        try:
            from facecv.models.deepface import (
                DeepFaceRecognizer, face_embedding as fe, 
                face_verification as fv, face_analysis as fa
            )
            deepface_recognizer = DeepFaceRecognizer()
            face_embedding = fe
            face_verification = fv
            face_analysis = fa
            logger.info("DeepFace组件初始化成功")
        except ImportError as e:
            logger.error(f"DeepFace组件初始化失败: {e}")
            raise HTTPException(
                status_code=503, 
                detail=f"DeepFace服务不可用，请确保已安装相关依赖: {str(e)}"
            )
    
    return deepface_recognizer, face_embedding, face_verification, face_analysis


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

@router.post("/faces/", response_model=FaceRegisterResponse)
async def register_face(
    name: str = Form(..., description="人员姓名"),
    file: UploadFile = File(..., description="人脸图片"),
    metadata: Optional[str] = Form(None, description="附加元数据（JSON格式）")
):
    """
    注册新人脸到数据库
    
    支持的图片格式：JPG, PNG, BMP
    """
    try:
        recognizer, embedding_mgr, _, _ = get_deepface_components()
        
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
        success = await recognizer.register_face_async(
            name=name,
            image=image_array,
            metadata=parsed_metadata
        )
        
        if success:
            return FaceRegisterResponse(
                success=True,
                message=f"人脸注册成功: {name}",
                person_name=name
            )
        else:
            raise HTTPException(status_code=400, detail="人脸注册失败，可能无法检测到清晰的人脸")
            
    except Exception as e:
        logger.error(f"人脸注册异常: {e}")
        raise HTTPException(status_code=500, detail=f"服务器内部错误: {str(e)}")


@router.get("/faces/", response_model=FaceListResponse)
async def list_faces():
    """获取所有已注册的人脸信息"""
    try:
        _, embedding_mgr, _, _ = get_deepface_components()
        
        # 获取所有用户
        all_users = await embedding_mgr.get_all_users()
        
        faces = []
        if all_users and "ids" in all_users:
            for i, face_id in enumerate(all_users["ids"]):
                metadata = all_users["metadatas"][i] if i < len(all_users["metadatas"]) else {}
                faces.append({
                    "face_id": face_id,
                    "person_name": metadata.get("name", "Unknown"),
                    "created_at": metadata.get("created_at"),
                    "metadata": metadata
                })
        
        return FaceListResponse(
            faces=faces,
            total=len(faces)
        )
        
    except Exception as e:
        logger.error(f"获取人脸列表异常: {e}")
        raise HTTPException(status_code=500, detail=f"服务器内部错误: {str(e)}")


@router.put("/faces/{face_id}")
async def update_face(
    face_id: str,
    name: Optional[str] = Form(None, description="新姓名"),
    file: Optional[UploadFile] = File(None, description="新人脸图片"),
    metadata: Optional[str] = Form(None, description="新元数据（JSON格式）")
):
    """根据ID更新现有人脸信息"""
    try:
        recognizer, embedding_mgr, _, _ = get_deepface_components()
        
        # 获取当前用户信息
        current_user = await embedding_mgr.get_user(face_id)
        if not current_user or not current_user.get("ids"):
            raise HTTPException(status_code=404, detail=f"未找到face_id: {face_id}")
        
        current_name = current_user["metadatas"][0].get("name", "Unknown")
        
        # 更新姓名
        if name and name != current_name:
            await embedding_mgr.update_name(face_id, name)
            logger.info(f"更新姓名: {current_name} -> {name}")
        
        # 更新图片（需要重新注册）
        if file:
            image_data = await file.read()
            image = Image.open(io.BytesIO(image_data))
            image_array = np.array(image)
            
            if len(image_array.shape) == 3:
                image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
            
            # 删除旧记录并重新注册
            await embedding_mgr.del_user(face_id)
            success = await recognizer.register_face_async(
                name=name or current_name,
                image=image_array
            )
            
            if not success:
                raise HTTPException(status_code=400, detail="更新人脸图片失败")
        
        return JSONResponse(
            content={"success": True, "message": f"人脸信息更新成功: {face_id}"}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"更新人脸异常: {e}")
        raise HTTPException(status_code=500, detail=f"服务器内部错误: {str(e)}")


@router.delete("/faces/{face_id}", response_model=FaceDeleteResponse)
async def delete_face(face_id: str):
    """根据ID删除人脸"""
    try:
        _, embedding_mgr, _, _ = get_deepface_components()
        
        # 检查用户是否存在
        user = await embedding_mgr.get_user(face_id)
        if not user or not user.get("ids"):
            raise HTTPException(status_code=404, detail=f"未找到face_id: {face_id}")
        
        # 删除用户
        result = await embedding_mgr.del_user(face_id)
        
        return FaceDeleteResponse(
            success=True,
            message=f"人脸删除成功: {face_id}"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"删除人脸异常: {e}")
        raise HTTPException(status_code=500, detail=f"服务器内部错误: {str(e)}")


@router.get("/faces/name/{name}")
async def get_face_by_name(name: str):
    """根据姓名获取人脸信息"""
    try:
        _, embedding_mgr, _, _ = get_deepface_components()
        
        users = await embedding_mgr.get_user_by_name(name)
        
        if not users or not users.get("ids"):
            raise HTTPException(status_code=404, detail=f"未找到姓名: {name}")
        
        faces = []
        for i, face_id in enumerate(users["ids"]):
            metadata = users["metadatas"][i] if i < len(users["metadatas"]) else {}
            faces.append({
                "face_id": face_id,
                "person_name": metadata.get("name", name),
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

@router.post("/recognition", response_model=FaceRecognitionResponse)
async def recognize_faces(
    file: UploadFile = File(..., description="待识别的图片"),
    threshold: Optional[float] = Form(0.6, description="识别阈值"),
    return_all_candidates: bool = Form(False, description="是否返回所有候选结果")
):
    """
    人脸识别
    
    识别图片中的所有人脸，返回匹配的人员信息
    """
    try:
        recognizer, _, _, _ = get_deepface_components()
        
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


@router.post("/verify/", response_model=FaceVerificationResponse)
async def verify_faces(
    file1: UploadFile = File(..., description="第一张人脸图片"),
    file2: UploadFile = File(..., description="第二张人脸图片"),
    threshold: Optional[float] = Form(0.6, description="验证阈值"),
    model_name: str = Form("ArcFace", description="使用的模型"),
    anti_spoofing: bool = Form(False, description="是否启用反欺骗检测")
):
    """
    验证两张图片是否是同一个人
    
    返回验证结果和相似度分数
    """
    try:
        _, _, verification, _ = get_deepface_components()
        
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
            anti_spoofing=anti_spoofing
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
        raise HTTPException(status_code=500, detail=f"服务器内部错误: {str(e)}")


@router.post("/analyze/", response_model=FaceAnalysisResponse)
async def analyze_face(
    file: UploadFile = File(..., description="待分析的图片"),
    actions: str = Form("emotion,age,gender,race", description="分析维度，逗号分隔"),
    detector_backend: str = Form("mtcnn", description="人脸检测器")
):
    """
    分析面部属性
    
    分析图片中人脸的年龄、性别、情绪、种族等属性
    """
    try:
        _, _, _, analysis = get_deepface_components()
        
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
            detector_backend=detector_backend
        )
        
        return FaceAnalysisResponse(
            faces=results,
            total_faces=len(results)
        )
        
    except Exception as e:
        logger.error(f"人脸分析异常: {e}")
        raise HTTPException(status_code=500, detail=f"服务器内部错误: {str(e)}")


# ==================== 视频处理API ====================

@router.post("/video_face/")
async def add_face_from_video(
    name: str = Form(..., description="人员姓名"),
    video_source: str = Form("0", description="视频源（摄像头编号或文件路径）"),
    sample_interval: int = Form(30, description="采样间隔（帧数）"),
    max_samples: int = Form(10, description="最大采样数量"),
    background_tasks: BackgroundTasks = None
):
    """
    通过视频帧采样添加人脸到数据库
    
    支持摄像头实时采样和视频文件处理
    """
    try:
        recognizer, _, _, _ = get_deepface_components()
        
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


@router.get("/recognize/webcam/stream")
async def real_time_recognition_stream(
    camera_id: str = Query("0", description="摄像头ID标识"),
    source: Union[str, int] = Query(0, description="视频源（摄像头编号或RTSP URL）"),
    threshold: float = Query(0.6, description="识别阈值"),
    fps: int = Query(30, description="帧率"),
    format: str = Query("sse", description="输出格式: sse 或 mjpeg"),
    webhook_urls: Optional[str] = Query(None, description="Webhook URLs for real-time result forwarding (comma-separated)"),
    webhook_timeout: int = Query(30, description="Webhook timeout in seconds"),
    webhook_retry_count: int = Query(3, description="Webhook retry count")
):
    """
    摄像头/RTSP实时人脸识别流
    
    支持SSE和MJPEG两种输出格式，并支持webhook实时转发
    
    - **webhook_urls**: Webhook URLs for real-time result forwarding (comma-separated)
    - **webhook_timeout**: Webhook request timeout in seconds  
    - **webhook_retry_count**: Number of retry attempts for failed webhooks
    """
    try:
        recognizer, _, _, _ = get_deepface_components()
        
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

@router.get("/health")
async def deepface_health():
    """DeepFace服务健康检查"""
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