"""视频流处理路由"""

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect, Depends, Query
from typing import Optional, List, Union
from pydantic import BaseModel, Field
import json
import asyncio
import logging

# from facecv import FaceRecognizer  # 暂时禁用
from facecv.core.video_stream import VideoStreamProcessor, StreamConfig
from facecv.api.routes.insightface_api import get_recognizer
import cv2

router = APIRouter()
logger = logging.getLogger(__name__)




class StreamProcessRequest(BaseModel):
    """视频流处理请求模型"""
    source: str = Field(..., description="视频源标识符，支持摄像头索引、RTSP URL或视频文件路径", example="0")
    duration: Optional[int] = Field(None, description="处理持续时间（秒），空值表示处理到流结束", example=30, ge=1, le=3600)
    skip_frames: int = Field(1, description="帧采样率，1表示处理每帧，数值越大处理越快但精度降低", example=2, ge=1, le=30)
    show_preview: bool = Field(False, description="是否显示预览窗口，仅本地环境有效", example=False)


class PersonSummary(BaseModel):
    """人员检测统计摘要"""
    name: str = Field(..., description="识别出的人员姓名", example="张三")
    detections: int = Field(..., description="该人员在视频中被检测到的次数", example=15, ge=0)
    avg_similarity: float = Field(..., description="该人员识别的平均相似度分数", example=0.92, ge=0.0, le=1.0)


class StreamProcessResponse(BaseModel):
    """视频流处理响应模型"""
    status: str = Field(..., description="处理状态", example="completed", enum=["completed", "error", "processing"])
    source: str = Field(..., description="处理的视频源标识符", example="0")
    duration: Optional[int] = Field(None, description="实际处理时长（秒）", example=30, ge=0)
    total_detections: int = Field(..., description="总检测到的人脸数量", example=45, ge=0)
    unique_persons: int = Field(..., description="识别出的不同人员数量", example=3, ge=0)
    persons: List[str] = Field(..., description="识别出的所有人员姓名列表", example=["张三", "李四", "王五"])
    summary: List[PersonSummary] = Field(..., description="每个人员的详细统计信息", example=[
        {"name": "张三", "detections": 15, "avg_similarity": 0.92},
        {"name": "李四", "detections": 12, "avg_similarity": 0.88}
    ])


class CameraSource(BaseModel):
    """摄像头信息"""
    index: int = Field(..., description="摄像头索引号，从0开始", example=0, ge=0)
    description: str = Field(..., description="摄像头描述信息", example="默认摄像头")


class FileSupport(BaseModel):
    """文件格式支持信息"""
    formats: List[str] = Field(..., description="支持的视频文件格式列表", example=["mp4", "avi", "mov", "mkv"])
    example: str = Field(..., description="文件路径格式示例", example="/path/to/video.mp4")


class VideoSourcesResponse(BaseModel):
    """视频源列表响应模型"""
    camera_sources: List[CameraSource] = Field(..., description="可用的本地摄像头设备列表", example=[
        {"index": 0, "description": "默认摄像头"},
        {"index": 1, "description": "第二个摄像头"}
    ])
    rtsp_examples: List[str] = Field(..., description="RTSP URL格式示例列表", example=[
        "rtsp://username:password@192.168.1.100:554/stream",
        "rtsp://admin:12345@192.168.1.64:554/h264/ch1/main/av_stream"
    ])
    file_support: FileSupport = Field(..., description="支持的视频文件格式信息")
    notes: List[str] = Field(..., description="重要使用说明和注意事项", example=[
        "摄像头索引从0开始",
        "RTSP需要网络摄像头支持",
        "确保应用程序具有摄像头访问权限"
    ])


@router.post(
    "/stream/process",
    response_model=StreamProcessResponse,
    summary="处理视频流",
    description="处理来自摄像头、RTSP流或视频文件的视频流进行实时人脸识别分析",
    tags=["视频流"]
)
async def process_video_stream(
    source: str = Query(..., description="视频源标识符：摄像头索引(0,1,2...)、RTSP URL或视频文件路径", example="0"),
    duration: Optional[int] = Query(None, description="处理持续时间（秒），1-3600秒范围，空值表示处理到流结束", example=30, ge=1, le=3600),
    skip_frames: int = Query(1, description="帧采样率，1=处理每帧，2=每2帧，数值越大速度越快但精度降低", example=2, ge=1, le=30),
    show_preview: bool = Query(False, description="是否显示预览窗口，仅本地环境有效，服务器部署请设为false", example=False),
    recognizer = Depends(get_recognizer)
) -> StreamProcessResponse:
    """
    处理视频流进行实时人脸识别分析
    
    此接口处理来自各种视频源的实时视频流，包括本地摄像头、网络RTSP流和视频文件。
    系统会检测每一帧中的人脸，与已注册的人脸数据库进行识别比对，并返回详细的统计信息。
    
    **请求参数详细说明:**
    
    **source** `str` (必需):
    - 描述: 视频源标识符，支持多种输入格式
    - 格式选项:
      - 本地摄像头索引: "0", "1", "2" (字符串或数字)
      - RTSP网络流: "rtsp://用户名:密码@IP地址:端口/流路径"
      - HTTP流: "http://IP地址:端口/stream.mjpg"
      - 视频文件: "/绝对路径/video.mp4" 或 "相对路径/video.avi"
    - 示例值:
      - "0" - 默认摄像头
      - "rtsp://admin:12345@192.168.1.100:554/stream"
      - "/data/recordings/security_20240115.mp4"
    - 验证规则: 非空字符串，支持的格式
    
    **duration** `Optional[int]`:
    - 描述: 处理持续时间（秒）
    - 取值范围: 1-3600秒（1小时）
    - 默认值: None（处理到流结束）
    - 使用建议:
      - 测试: 30-60秒
      - 短视频分析: 300秒（5分钟）
      - 实时监控: 不设置（持续处理）
    - 注意: 文件处理会在文件结束时自动停止
    
    **skip_frames** `int`:
    - 描述: 帧采样率，控制处理频率和性能
    - 取值范围: 1-30
    - 默认值: 1
    - 性能影响:
      - 1: 处理每帧（最高精度，最慢，CPU密集）
      - 2: 处理每2帧（推荐平衡设置）
      - 5: 处理每5帧（快速模式，适合实时监控）
      - 10+: 处理每10帧以上（最快，低精度）
    - 选择建议:
      - 高精度要求: 1-2
      - 实时监控: 3-5
      - 快速预览: 8-15
    
    **show_preview** `bool`:
    - 描述: 是否显示实时预览窗口
    - 默认值: false
    - 适用环境:
      - true: 本地开发环境，有显示器
      - false: 服务器部署，无显示器（推荐）
    - 注意: 预览窗口会增加CPU使用率
        
    **响应数据详细说明:**
    
    **StreamProcessResponse** 对象包含以下字段:
    
    **status** `str`:
    - 描述: 处理状态码
    - 可能值:
      - "completed": 处理成功完成
      - "error": 处理过程中发生错误
      - "processing": 正在处理中（异步模式）
    - 示例: "completed"
    
    **source** `str`:
    - 描述: 输入视频源的回显确认
    - 作用: 确认系统实际处理的视频源
    - 示例: "0" 或 "rtsp://192.168.1.100:554/stream"
    
    **duration** `Optional[int]`:
    - 描述: 实际处理时长（秒）
    - 含义: 从开始到结束的总时间
    - 范围: 0或正整数
    - 示例: 30（处理了30秒）
    - 注意: 可能小于请求的duration（文件结束等原因）
    
    **total_detections** `int`:
    - 描述: 在所有处理帧中检测到的人脸总数
    - 计算: 包括重复检测同一人的次数
    - 范围: 0或正整数
    - 示例: 45（总共检测到45个人脸）
    - 用途: 评估视频中人脸出现频率
    
    **unique_persons** `int`:
    - 描述: 识别出的不同人员数量
    - 计算: 去重后的唯一人员数
    - 范围: 0到total_detections之间
    - 示例: 3（识别出3个不同的人）
    - 用途: 了解视频中涉及多少人
    
    **persons** `List[str]`:
    - 描述: 识别出的所有人员姓名列表
    - 格式: 字符串数组，已去重
    - 示例: ["张三", "李四", "王五"]
    - 排序: 按检测次数降序排列
    - 注意: 不包含"Unknown"未知人员
    
    **summary** `List[PersonSummary]`:
    - 描述: 每个人员的详细统计信息数组
    - 限制: 最多返回前10个人员的统计
    - 排序: 按检测次数降序排列
    - 每个PersonSummary对象包含:
      - **name** `str`: 人员姓名（来自注册数据库）
      - **detections** `int`: 该人员被检测到的总次数
      - **avg_similarity** `float`: 平均识别相似度分数(0.0-1.0)
    - 示例:
    ```json
    [
        {
            "name": "张三",
            "detections": 15,
            "avg_similarity": 0.92
        },
        {
            "name": "李四", 
            "detections": 12,
            "avg_similarity": 0.88
        }
    ]
    ```
    
    **完整使用示例:**
    
    **示例1: 处理默认摄像头30秒**
    ```bash
    curl -X POST "http://localhost:7003/api/v1/stream/process?source=0&duration=30&skip_frames=2&show_preview=false"
    ```
    响应:
    ```json
    {
        "status": "completed",
        "source": "0",
        "duration": 30,
        "total_detections": 45,
        "unique_persons": 2,
        "persons": ["张三", "李四"],
        "summary": [
            {"name": "张三", "detections": 28, "avg_similarity": 0.91},
            {"name": "李四", "detections": 17, "avg_similarity": 0.87}
        ]
    }
    ```
    
    **示例2: 处理RTSP流（持续处理）**
    ```bash
    curl -X POST "http://localhost:7003/api/v1/stream/process?source=rtsp://admin:password@192.168.1.100:554/stream&skip_frames=5"
    ```
    响应:
    ```json
    {
        "status": "completed",
        "source": "rtsp://admin:password@192.168.1.100:554/stream",
        "duration": null,
        "total_detections": 156,
        "unique_persons": 5,
        "persons": ["张三", "李四", "王五", "赵六", "孙七"],
        "summary": [
            {"name": "张三", "detections": 45, "avg_similarity": 0.93},
            {"name": "李四", "detections": 38, "avg_similarity": 0.89}
        ]
    }
    ```
    
    **示例3: 处理视频文件**
    ```bash
    curl -X POST "http://localhost:7003/api/v1/stream/process?source=/data/recordings/meeting_20240115.mp4&skip_frames=3"
    ```
    响应:
    ```json
    {
        "status": "completed",
        "source": "/data/recordings/meeting_20240115.mp4",
        "duration": 1800,
        "total_detections": 324,
        "unique_persons": 8,
        "persons": ["张三", "李四", "王五", "赵六", "孙七", "周八", "吴九", "郑十"],
        "summary": [
            {"name": "张三", "detections": 89, "avg_similarity": 0.94},
            {"name": "李四", "detections": 76, "avg_similarity": 0.91}
        ]
    }
    ```
    
    **应用场景:**
    - 安防监控系统中的人脸识别
    - 通过摄像头流进行考勤跟踪
    - 入口处访客身份识别
    - 录制视频footage的分析处理
    
    **性能优化建议:**
    - 使用skip_frames参数平衡精度与速度
    - 较长的duration可能影响响应时间
    - RTSP流需要稳定的网络连接
    - 建议实时处理时使用skip_frames=2-5
    
    **错误处理详细说明:**
    
    **HTTP 400 - Bad Request**
    - 触发条件:
      - 无效的视频源格式
      - 摄像头设备无法访问
      - RTSP URL格式错误或连接失败
      - 参数值超出允许范围
    - 响应示例:
    ```json
    {
        "detail": "无效的视频源格式或无法访问的摄像头/RTSP"
    }
    ```
    
    **HTTP 404 - Not Found**
    - 触发条件:
      - 视频文件路径不存在
      - 摄像头索引超出系统可用范围
      - 指定的网络摄像头不可达
    - 响应示例:
    ```json
    {
        "detail": "视频文件未找到或摄像头索引超出范围"
    }
    ```
    
    **HTTP 500 - Internal Server Error**
    - 触发条件:
      - 视频处理过程中发生内部错误
      - 系统资源不足（内存、CPU）
      - 人脸识别模型加载失败
      - 数据库连接错误
    - 响应示例:
    ```json
    {
        "detail": "视频流处理失败: 系统资源不足"
    }
    ```
    
    **HTTP 503 - Service Unavailable**
    - 触发条件:
      - 人脸识别服务未初始化
      - 系统维护模式
      - 依赖服务不可用
    - 响应示例:
    ```json
    {
        "detail": "人脸识别服务暂不可用，请稍后重试"
    }
    ```
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
        
        processor = VideoStreamProcessor(recognizer, config)
        
        # 异步处理视频流
        results = await processor.process_stream_async(
            source=source,
            duration=duration
        )
        
        # 统计结果
        total_faces = len(results)
        unique_names = list(set(r.name for r in results if r.name != "Unknown"))
        
        return StreamProcessResponse(
            status="completed",
            source=str(source),
            duration=duration,
            total_detections=total_faces,
            unique_persons=len(unique_names),
            persons=unique_names,
            summary=[
                PersonSummary(
                    name=r.name,
                    detections=sum(1 for x in results if x.name == r.name),
                    avg_similarity=sum(x.confidence for x in results if x.name == r.name) / 
                                    sum(1 for x in results if x.name == r.name)
                )
                for r in results
                if r.name in unique_names
            ][:10]  # 返回前10个人的统计
        )
        
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
                
                processor = VideoStreamProcessor(recognizer, config)
                
                # 定义回调函数
                async def send_results(results):
                    """发送识别结果到客户端"""
                    response = {
                        "type": "recognition",
                        "faces": [
                            {
                                "name": r.name,
                                "similarity": r.confidence,
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


@router.get(
    "/stream/sources",
    response_model=VideoSourcesResponse,
    summary="获取可用视频源列表",
    description="获取系统支持的所有视频源信息和格式说明",
    tags=["视频流"]
)
async def list_video_sources() -> VideoSourcesResponse:
    """
    获取系统支持的所有视频源信息和格式详细说明
    
    此接口提供FaceCV系统支持的视频输入选项的完整概览，帮助用户了解可用于
    人脸识别处理的各种视频源类型和配置要求。
    
    **响应数据结构详细说明:**
    
    **VideoSourcesResponse** 对象包含以下字段:
    
    **camera_sources** `List[CameraSource]`:
    - 描述: 系统检测到的本地摄像头设备列表
    - 数据结构: CameraSource对象数组
    - 每个CameraSource包含:
      - **index** `int`: 摄像头索引号，从0开始递增
      - **description** `str`: 摄像头的描述信息
    - 使用方法: 将index值作为stream/process接口的source参数
    - 示例:
    ```json
    [
        {"index": 0, "description": "默认摄像头"},
        {"index": 1, "description": "第二个摄像头（如果有）"}
    ]
    ```
    
    **rtsp_examples** `List[str]`:
    - 描述: 常见网络摄像头RTSP URL格式示例
    - 格式: 字符串数组，包含各种厂商的URL模式
    - 包含内容:
      - 基本认证格式: rtsp://用户名:密码@IP:端口/路径
      - 不同厂商的特定路径格式
      - 高清和标清流的不同端点
    - 使用方法: 替换占位符为实际的摄像头信息
    - 示例:
    ```json
    [
        "rtsp://username:password@192.168.1.100:554/stream",
        "rtsp://admin:12345@192.168.1.64:554/h264/ch1/main/av_stream",
        "rtsp://user:pass@camera-ip:554/live/ch00_0"
    ]
    ```
    
    **file_support** `FileSupport`:
    - 描述: 系统支持的视频文件格式信息
    - 数据结构: FileSupport对象
    - 包含字段:
      - **formats** `List[str]`: 支持的文件扩展名列表
      - **example** `str`: 文件路径格式示例
    - 支持的格式: MP4, AVI, MOV, MKV, WMV, FLV
    - 路径要求: 绝对路径或相对路径均可
    - 示例:
    ```json
    {
        "formats": ["mp4", "avi", "mov", "mkv", "wmv", "flv"],
        "example": "/path/to/video.mp4"
    }
    ```
    
    **notes** `List[str]`:
    - 描述: 重要的使用指南、限制和注意事项
    - 格式: 字符串数组，每项为一个重要提示
    - 内容包括:
      - 摄像头访问权限要求
      - 网络连接和RTSP配置说明
      - 性能优化建议
      - 平台特定注意事项
      - 相关API端点信息
    - 示例:
    ```json
    [
        "摄像头索引从0开始 - 使用 /api/v1/camera/test/local 测试",
        "RTSP需要网络摄像头支持 - 使用 /api/v1/camera/test/rtsp 测试",
        "确保应用程序具有摄像头访问权限",
        "建议实时处理时使用skip_frames=2-5以获得最佳性能",
        "文件处理支持定位 - duration参数对视频文件有效",
        "WebSocket端点 /api/v1/stream/ws 可用于实时流传输"
    ]
    ```
    
    **完整响应示例:**
    ```json
    {
        "camera_sources": [
            {
                "index": 0,
                "description": "默认摄像头"
            },
            {
                "index": 1,
                "description": "第二个摄像头（如果有）"
            }
        ],
        "rtsp_examples": [
            "rtsp://username:password@192.168.1.100:554/stream",
            "rtsp://admin:12345@192.168.1.64:554/h264/ch1/main/av_stream",
            "rtsp://user:pass@camera-ip:554/live/ch00_0",
            "rtsp://192.168.1.100:554/ch01/0",
            "rtsp://viewer:viewer@192.168.1.100:554/cam/realmonitor?channel=1&subtype=0"
        ],
        "file_support": {
            "formats": ["mp4", "avi", "mov", "mkv", "wmv", "flv"],
            "example": "/path/to/video.mp4"
        },
        "notes": [
            "摄像头索引从0开始 - 使用 /api/v1/camera/test/local 测试可用性",
            "RTSP需要网络摄像头支持 - 使用 /api/v1/camera/test/rtsp 测试连接",
            "确保应用程序具有摄像头访问权限",
            "建议实时处理时使用skip_frames=2-5以获得最佳性能",
            "文件处理支持定位 - duration参数对视频文件有效",
            "WebSocket端点 /api/v1/stream/ws 可用于实时流传输",
            "支持HTTP流格式: http://ip:port/stream.mjpg",
            "大文件处理建议分段处理以避免内存溢出"
        ]
    }
    ```
    
    **使用示例:**
    ```bash
    curl -X GET "http://localhost:7003/api/v1/stream/sources"
    ```
    
    **使用场景:**
    - 开始流处理前发现可用摄像头
    - 获取IP摄像头设置的RTSP URL格式示例
    - 上传前检查支持的视频文件格式
    - 了解系统要求和限制
    
    **此接口特点:**
    - 无需认证 - 这是一个发现性端点
    - 无需参数 - 返回静态配置信息
    - 实时检测可用的摄像头设备
    """
    # 简化摄像头检测 - 只检查设备文件
    available_cameras = []
    
    try:
        import glob
        video_devices = glob.glob('/dev/video*')
        
        if video_devices:
            video_devices.sort()
            logger.info(f"检测到视频设备: {video_devices}")
            
            for device_path in video_devices[:2]:  # 最多检查2个
                try:
                    device_num = int(device_path.replace('/dev/video', ''))
                    if device_num == 0:
                        description = f"默认摄像头 (检测到设备)"
                    else:
                        description = f"摄像头 #{device_num} (检测到设备)"
                    
                    available_cameras.append(CameraSource(
                        index=device_num,
                        description=description
                    ))
                except:
                    continue
        else:
            logger.info("未找到/dev/video*设备，提供默认配置")
            
    except Exception as e:
        logger.debug(f"设备检测失败: {e}")
    
    # 如果没有检测到设备，提供默认配置
    if not available_cameras:
        available_cameras = [
            CameraSource(index=0, description="摄像头 (请手动测试索引0)")
        ]
    
    return VideoSourcesResponse(
        camera_sources=available_cameras,
        rtsp_examples=[
            "rtsp://username:password@ip:port/stream",
            "rtsp://192.168.1.100:554/ch01/0", 
            "rtsp://admin:12345@192.168.1.64:554/h264/ch1/main/av_stream",
            "rtsp://user:pass@camera-ip:554/live/ch00_0",
            "rtsp://viewer:viewer@192.168.1.100:554/cam/realmonitor?channel=1&subtype=0"
        ],
        file_support=FileSupport(
            formats=["mp4", "avi", "mov", "mkv", "wmv", "flv"],
            example="/path/to/video.mp4"
        ),
        notes=[
            f"检测到 {len(available_cameras)} 个可用摄像头 - 使用 /api/v1/camera/test/local 测试可用性",
            "RTSP需要网络摄像头支持 - 使用 /api/v1/camera/test/rtsp 测试连接",
            "确保应用程序具有摄像头访问权限",
            "建议实时处理时使用skip_frames=2-5以获得最佳性能",
            "文件处理支持定位 - duration参数对视频文件有效",
            "WebSocket端点 /api/v1/stream/ws 可用于实时流传输",
            "支持HTTP流格式: http://ip:port/stream.mjpg",
            "大文件处理建议分段处理以避免内存溢出"
        ]
    )