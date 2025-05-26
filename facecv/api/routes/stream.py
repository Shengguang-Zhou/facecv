"""视频流处理路由"""

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect, Depends
from typing import Optional
import json
import asyncio
import logging

# from facecv import FaceRecognizer  # 暂时禁用
from facecv.core.video_stream import VideoStreamProcessor, StreamConfig
from facecv.api.routes.face import get_recognizer

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/stream/process")
async def process_video_stream(
    source: str,
    duration: Optional[int] = None,
    skip_frames: int = 1,
    show_preview: bool = False,
    recognizer = Depends(get_recognizer)  # 移除类型注解
):
    """
    处理视频流（摄像头或RTSP）
    
    - **source**: 视频源（摄像头索引如 "0" 或 RTSP URL）
    - **duration**: 处理时长（秒），不指定则持续处理
    - **skip_frames**: 跳帧数，1表示处理每一帧
    - **show_preview**: 是否显示预览窗口（仅本地有效）
    """
    try:
        # 转换摄像头索引
        if source.isdigit():
            source = int(source)
            
        # 配置流处理器
        config = StreamConfig(
            skip_frames=skip_frames,
            show_preview=show_preview
        )
        
        processor = VideoStreamProcessor(recognizer.model, config)
        
        # 异步处理视频流
        results = await processor.process_stream_async(
            source=source,
            duration=duration
        )
        
        # 统计结果
        total_faces = len(results)
        unique_names = list(set(r.recognized_name for r in results if r.recognized_name != "Unknown"))
        
        return {
            "status": "completed",
            "source": str(source),
            "duration": duration,
            "total_detections": total_faces,
            "unique_persons": len(unique_names),
            "persons": unique_names,
            "summary": [
                {
                    "name": r.recognized_name,
                    "detections": sum(1 for x in results if x.recognized_name == r.recognized_name),
                    "avg_similarity": sum(x.similarity_score for x in results if x.recognized_name == r.recognized_name) / 
                                    sum(1 for x in results if x.recognized_name == r.recognized_name)
                }
                for r in results
                if r.recognized_name in unique_names
            ][:10]  # 返回前10个人的统计
        }
        
    except Exception as e:
        logger.error(f"视频流处理失败: {e}")
        raise HTTPException(status_code=500, detail=f"视频流处理失败: {str(e)}")


@router.websocket("/stream/ws")
async def websocket_stream(
    websocket: WebSocket,
    recognizer = Depends(get_recognizer)  # 移除类型注解
):
    """
    WebSocket 实时视频流处理
    
    客户端发送：
    {
        "action": "start",
        "source": "0" 或 "rtsp://...",
        "skip_frames": 1
    }
    
    服务端返回：
    {
        "type": "recognition",
        "faces": [
            {
                "name": "张三",
                "similarity": 0.95,
                "bbox": [x1, y1, x2, y2]
            }
        ],
        "timestamp": "2023-05-26T10:00:00"
    }
    """
    await websocket.accept()
    processor = None
    
    try:
        while True:
            # 接收客户端消息
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message.get("action") == "start":
                source = message.get("source", "0")
                if source.isdigit():
                    source = int(source)
                    
                skip_frames = message.get("skip_frames", 1)
                
                # 配置流处理器
                config = StreamConfig(
                    skip_frames=skip_frames,
                    show_preview=False  # WebSocket 模式不显示预览
                )
                
                processor = VideoStreamProcessor(recognizer.model, config)
                
                # 定义回调函数
                async def send_results(results):
                    """发送识别结果到客户端"""
                    response = {
                        "type": "recognition",
                        "faces": [
                            {
                                "name": r.recognized_name,
                                "similarity": r.similarity_score,
                                "bbox": r.bbox
                            }
                            for r in results
                        ],
                        "timestamp": asyncio.get_event_loop().time()
                    }
                    await websocket.send_json(response)
                    
                # 开始处理
                await websocket.send_json({
                    "type": "status",
                    "message": "Stream processing started"
                })
                
                # 在后台处理视频流
                asyncio.create_task(
                    processor.process_stream_async(
                        source=source,
                        callback=lambda results: asyncio.create_task(send_results(results))
                    )
                )
                
            elif message.get("action") == "stop":
                if processor:
                    processor.is_running = False
                    processor.stop_event.set()
                    
                await websocket.send_json({
                    "type": "status",
                    "message": "Stream processing stopped"
                })
                break
                
    except WebSocketDisconnect:
        logger.info("WebSocket 连接断开")
    except Exception as e:
        logger.error(f"WebSocket 错误: {e}")
        await websocket.send_json({
            "type": "error",
            "message": str(e)
        })
    finally:
        if processor:
            processor.is_running = False
            processor.stop_event.set()


@router.get("/stream/sources")
async def list_video_sources():
    """
    列出可用的视频源
    
    返回常见的视频源类型说明
    """
    return {
        "camera_sources": [
            {"index": 0, "description": "默认摄像头"},
            {"index": 1, "description": "第二个摄像头（如果有）"}
        ],
        "rtsp_examples": [
            "rtsp://username:password@ip:port/stream",
            "rtsp://192.168.1.100:554/ch01/0",
            "rtsp://admin:12345@192.168.1.64:554/h264/ch1/main/av_stream"
        ],
        "file_support": {
            "formats": ["mp4", "avi", "mov", "mkv"],
            "example": "/path/to/video.mp4"
        },
        "notes": [
            "摄像头索引从0开始",
            "RTSP需要网络摄像头支持",
            "确保有权限访问摄像头设备"
        ]
    }