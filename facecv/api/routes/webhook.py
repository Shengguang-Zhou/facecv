"""Webhook management API routes"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from typing import List, Dict, Optional
from pydantic import BaseModel, HttpUrl, Field
import logging
from datetime import datetime

from facecv.core.webhook import webhook_manager, WebhookConfig, WebhookEvent
from facecv.config import get_settings

router = APIRouter()
logger = logging.getLogger(__name__)


class WebhookConfigRequest(BaseModel):
    """Webhook配置请求模型"""
    webhook_id: str = Field(..., description="Webhook唯一标识符", example="webhook_001")
    url: HttpUrl = Field(..., description="接收Webhook事件的目标URL", example="https://api.example.com/webhooks/face-events")
    headers: Optional[Dict[str, str]] = Field(None, description="发送请求时附加的HTTP头", example={"Authorization": "Bearer token123", "X-API-Key": "abc123"})
    timeout: int = Field(30, description="HTTP请求超时时间（秒）", ge=1, le=300, example=30)
    retry_count: int = Field(3, description="失败时的重试次数", ge=0, le=10, example=3)
    retry_delay: int = Field(1, description="重试间隔时间（秒）", ge=1, le=60, example=2)
    batch_size: int = Field(10, description="每批次发送的最大事件数量", ge=1, le=100, example=5)
    batch_timeout: float = Field(1.0, description="批处理超时时间（秒）", ge=0.1, le=60.0, example=2.0)
    enabled: bool = Field(True, description="Webhook是否启用", example=True)
    event_types: Optional[List[str]] = Field(None, description="订阅的事件类型列表，为空则接收所有事件", example=["face_detected", "face_recognized", "stranger_alert"])


class WebhookConfigResponse(BaseModel):
    """Webhook配置响应模型"""
    webhook_id: str = Field(..., description="Webhook唯一标识符", example="webhook_001")
    url: str = Field(..., description="Webhook目标URL", example="https://api.example.com/webhooks/face-events")
    enabled: bool = Field(..., description="Webhook当前启用状态", example=True)
    created_at: datetime = Field(..., description="Webhook创建时间", example="2024-01-15T10:30:00")
    event_types: Optional[List[str]] = Field(None, description="订阅的事件类型", example=["face_detected", "face_recognized"])
    statistics: Optional[Dict[str, int]] = Field(None, description="Webhook统计信息", example={"total_sent": 150, "success_count": 145, "failed_count": 5})


class WebhookTestRequest(BaseModel):
    """Webhook测试请求模型"""
    url: HttpUrl = Field(..., description="要测试的Webhook URL", example="https://api.example.com/webhooks/test")
    test_data: Optional[Dict] = Field(None, description="自定义测试数据，如不提供则使用默认测试事件", example={"message": "Custom test event", "test_id": "12345"})


class WebhookDeleteResponse(BaseModel):
    """Webhook删除响应模型"""
    message: str = Field(..., description="删除操作确认消息", example="Webhook webhook_001 deleted successfully")


class WebhookStatusResponse(BaseModel):
    """Webhook状态更改响应模型"""
    message: str = Field(..., description="状态更改确认消息", example="Webhook webhook_001 enabled")


class WebhookTestResponse(BaseModel):
    """Webhook测试响应模型"""
    message: str = Field(..., description="测试状态消息", example="Test webhook sent")
    url: str = Field(..., description="测试的目标URL", example="https://api.example.com/webhooks/test")
    test_event: Dict = Field(..., description="发送的测试事件数据", example={
        "event_type": "test",
        "timestamp": "2024-01-15T10:30:00",
        "camera_id": "test_camera",
        "data": {
            "message": "This is a test webhook event",
            "faces": [{"name": "Test Person", "confidence": 0.95, "bbox": [100, 100, 200, 200]}]
        },
        "metadata": {"test": True}
    })


class WebhookStatsResponse(BaseModel):
    """Webhook统计响应模型"""
    total_webhooks: int = Field(..., description="已配置的Webhook总数", example=5)
    enabled_webhooks: int = Field(..., description="当前启用的Webhook数量", example=3)
    queue_size: int = Field(..., description="当前事件队列中的事件数量", example=12)
    manager_running: bool = Field(..., description="Webhook管理器是否正在运行", example=True)


@router.post(
    "/webhooks",
    response_model=WebhookConfigResponse,
    summary="创建Webhook",
    description="创建新的Webhook配置用于接收人脸识别事件。当检测到或识别出人脸时将触发Webhook。",
    tags=["Webhook"]
)
async def create_webhook(config: WebhookConfigRequest) -> WebhookConfigResponse:
    """
    创建新的Webhook配置用于接收人脸识别事件通知
    
    此接口创建一个新的Webhook配置，系统将向指定URL发送实时人脸识别事件。
    支持批量发送、重试机制、自定义头信息等高级配置选项。
    
    **请求参数:**
    - webhook_id `str`: Webhook的唯一标识符，用于管理和引用此配置
    - url `HttpUrl`: 接收Webhook事件的目标URL，必须是有效的HTTP/HTTPS地址
    - headers `Dict[str, str]`: 发送请求时附加的HTTP头信息（可选），如认证令牌等
    - timeout `int`: HTTP请求超时时间（秒，范围：1-300，默认：30）
    - retry_count `int`: 发送失败时的重试次数（范围：0-10，默认：3）
    - retry_delay `int`: 重试间隔时间（秒，范围：1-60，默认：1）
    - batch_size `int`: 每批次发送的最大事件数量（范围：1-100，默认：10）
    - batch_timeout `float`: 批处理超时时间（秒，范围：0.1-60.0，默认：1.0）
    - enabled `bool`: Webhook是否启用（默认：true）
    - event_types `List[str]`: 订阅的事件类型过滤器（可选），支持的类型：
      - "face_detected": 人脸检测事件
      - "face_recognized": 人脸识别事件  
      - "stranger_alert": 陌生人警报事件
      - "attendance_recorded": 考勤记录事件
            
    **响应数据:**
    - webhook_id `str`: Webhook唯一标识符
    - url `str`: 目标Webhook URL
    - enabled `bool`: Webhook当前启用状态
    - created_at `datetime`: 创建时间戳
    - event_types `List[str]`: 订阅的事件类型列表
    - statistics `Dict[str, int]`: Webhook统计信息（可选）
    
    **Webhook事件格式:**
    系统将发送如下格式的JSON数据到您的URL：
    ```json
    {
        "webhook_id": "webhook_001",
        "timestamp": "2024-01-15T10:30:00",
        "events": [
            {
                "event_type": "face_detected",
                "timestamp": "2024-01-15T10:30:00",
                "camera_id": "camera_001",
                "data": {
                    "faces": [
                        {
                            "name": "张三",
                            "confidence": 0.95,
                            "bbox": [100, 100, 200, 200]
                        }
                    ],
                    "face_count": 1
                },
                "metadata": {"location": "entrance"}
            }
        ]
    }
    ```
    """
    try:
        webhook_config = WebhookConfig(
            url=str(config.url),
            headers=config.headers,
            timeout=config.timeout,
            retry_count=config.retry_count,
            retry_delay=config.retry_delay,
            batch_size=config.batch_size,
            batch_timeout=config.batch_timeout,
            enabled=config.enabled
        )
        
        webhook_manager.add_webhook(config.webhook_id, webhook_config)
        
        # Start webhook manager if not running
        if not webhook_manager.running:
            webhook_manager.start()
        
        logger.info(f"Created webhook {config.webhook_id}")
        
        return WebhookConfigResponse(
            webhook_id=config.webhook_id,
            url=str(config.url),
            enabled=config.enabled,
            created_at=datetime.now(),
            event_types=config.event_types
        )
        
    except Exception as e:
        logger.error(f"Error creating webhook: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create webhook: {str(e)}")


@router.get(
    "/webhooks",
    response_model=List[WebhookConfigResponse],
    summary="列出Webhook",
    description="获取所有已配置的Webhook列表及其当前状态和配置。",
    tags=["Webhook"]
)
async def list_webhooks() -> List[WebhookConfigResponse]:
    """
    获取所有已配置的Webhook列表及其当前状态
    
    此接口返回系统中所有已配置的Webhook列表，包括其配置信息、启用状态和基本统计数据。
    可用于监控和管理所有Webhook配置。
    
    **响应数据:**
    返回WebhookConfigResponse对象数组，每个对象包含：
    - webhook_id `str`: Webhook唯一标识符
    - url `str`: 目标Webhook URL地址
    - enabled `bool`: Webhook当前是否启用
    - created_at `datetime`: Webhook创建时间戳
    - event_types `List[str]`: 订阅的事件类型列表，null表示接收所有事件
    - statistics `Dict[str, int]`: Webhook统计信息（可选），包含：
      - total_sent: 总发送事件数
      - success_count: 成功发送数
      - failed_count: 发送失败数
    
    **使用场景:**
    - 查看所有已配置的Webhook
    - 监控Webhook状态和性能
    - 管理Webhook配置
    """
    webhooks = []
    
    for webhook_id, config in webhook_manager.webhooks.items():
        webhooks.append(WebhookConfigResponse(
            webhook_id=webhook_id,
            url=config.url,
            enabled=config.enabled,
            created_at=datetime.now(),  # Would need to track this properly
            event_types=None
        ))
    
    return webhooks


@router.get(
    "/webhooks/{webhook_id}",
    response_model=WebhookConfigResponse,
    summary="获取Webhook",
    description="根据ID获取特定Webhook的配置详情。",
    tags=["Webhook"]
)
async def get_webhook(webhook_id: str) -> WebhookConfigResponse:
    """
    获取特定Webhook配置。
    
    Args:
        webhook_id `str`: 要检索的Webhook的唯一标识符
        
    Returns:
        WebhookConfigResponse: Webhook配置详情
            - webhook_id `str`: Webhook唯一标识符
            - url `str`: 目标Webhook URL
            - enabled `bool`: Webhook激活状态
            - created_at `datetime`: 创建时间戳
            - event_types `List[str]`: 过滤的事件类型
            
    Raises:
        HTTPException: 如果找不到Webhook则返回404
    """
    if webhook_id not in webhook_manager.webhooks:
        raise HTTPException(status_code=404, detail="Webhook not found")
    
    config = webhook_manager.webhooks[webhook_id]
    
    return WebhookConfigResponse(
        webhook_id=webhook_id,
        url=config.url,
        enabled=config.enabled,
        created_at=datetime.now(),
        event_types=None
    )


@router.put(
    "/webhooks/{webhook_id}",
    response_model=WebhookConfigResponse,
    summary="更新Webhook",
    description="更新现有的Webhook配置。所有参数将被新值替换。",
    tags=["Webhook"]
)
async def update_webhook(webhook_id: str, config: WebhookConfigRequest) -> WebhookConfigResponse:
    """
    更新Webhook配置。
    
    Args:
        webhook_id `str`: 要更新的Webhook的唯一标识符
        config: 新的Webhook配置参数
            - webhook_id `str`: Webhook的唯一标识符
            - url `HttpUrl`: 发送Webhook事件的目标URL
            - headers `Dict[str, str]`: 可选的HTTP头信息
            - timeout `int`: 请求超时时间（秒）
            - retry_count `int`: 重试次数
            - retry_delay `int`: 重试间隔（秒）
            - batch_size `int`: 每批次最大事件数
            - batch_timeout `float`: 批处理超时（秒）
            - enabled `bool`: Webhook是否激活
            - event_types `List[str]`: 特定事件类型过滤器
            
    Returns:
        WebhookConfigResponse: 更新后的Webhook配置
            - webhook_id `str`: Webhook唯一标识符
            - url `str`: 目标Webhook URL
            - enabled `bool`: Webhook激活状态
            - created_at `datetime`: 创建时间戳
            - event_types `List[str]`: 过滤的事件类型
            
    Raises:
        HTTPException: 如果找不到Webhook则返回404
    """
    if webhook_id not in webhook_manager.webhooks:
        raise HTTPException(status_code=404, detail="Webhook not found")
    
    webhook_config = WebhookConfig(
        url=str(config.url),
        headers=config.headers,
        timeout=config.timeout,
        retry_count=config.retry_count,
        retry_delay=config.retry_delay,
        batch_size=config.batch_size,
        batch_timeout=config.batch_timeout,
        enabled=config.enabled
    )
    
    webhook_manager.update_webhook(webhook_id, webhook_config)
    
    return WebhookConfigResponse(
        webhook_id=webhook_id,
        url=str(config.url),
        enabled=config.enabled,
        created_at=datetime.now(),
        event_types=config.event_types
    )


@router.delete(
    "/webhooks/{webhook_id}",
    response_model=WebhookDeleteResponse,
    summary="删除Webhook",
    description="永久删除Webhook配置。此操作不可撤销。",
    tags=["Webhook"]
)
async def delete_webhook(webhook_id: str) -> WebhookDeleteResponse:
    """
    删除Webhook配置。
    
    Args:
        webhook_id `str`: 要删除的Webhook的唯一标识符
        
    Returns:
        WebhookDeleteResponse: 删除确认
            - message `str`: 成功确认消息
            
    Raises:
        HTTPException: 如果找不到Webhook则返回404
    """
    if webhook_id not in webhook_manager.webhooks:
        raise HTTPException(status_code=404, detail="Webhook not found")
    
    webhook_manager.remove_webhook(webhook_id)
    
    return WebhookDeleteResponse(message=f"Webhook {webhook_id} deleted successfully")


@router.post(
    "/webhooks/{webhook_id}/enable",
    response_model=WebhookStatusResponse,
    summary="启用Webhook",
    description="启用Webhook开始接收人脸识别事件。",
    tags=["Webhook"]
)
async def enable_webhook(webhook_id: str) -> WebhookStatusResponse:
    """
    启用Webhook。
    
    Args:
        webhook_id `str`: 要启用的Webhook的唯一标识符
        
    Returns:
        WebhookStatusResponse: 启用确认
            - message `str`: 成功确认消息
            
    Raises:
        HTTPException: 如果找不到Webhook则返回404
    """
    if webhook_id not in webhook_manager.webhooks:
        raise HTTPException(status_code=404, detail="Webhook not found")
    
    config = webhook_manager.webhooks[webhook_id]
    config.enabled = True
    webhook_manager.update_webhook(webhook_id, config)
    
    return WebhookStatusResponse(message=f"Webhook {webhook_id} enabled")


@router.post(
    "/webhooks/{webhook_id}/disable",
    response_model=WebhookStatusResponse,
    summary="禁用Webhook",
    description="禁用Webhook停止接收人脸识别事件。",
    tags=["Webhook"]
)
async def disable_webhook(webhook_id: str) -> WebhookStatusResponse:
    """
    禁用Webhook。
    
    Args:
        webhook_id `str`: 要禁用的Webhook的唯一标识符
        
    Returns:
        WebhookStatusResponse: 禁用确认
            - message `str`: 成功确认消息
            
    Raises:
        HTTPException: 如果找不到Webhook则返回404
    """
    if webhook_id not in webhook_manager.webhooks:
        raise HTTPException(status_code=404, detail="Webhook not found")
    
    config = webhook_manager.webhooks[webhook_id]
    config.enabled = False
    webhook_manager.update_webhook(webhook_id, config)
    
    return WebhookStatusResponse(message=f"Webhook {webhook_id} disabled")


@router.post(
    "/webhooks/test",
    response_model=WebhookTestResponse,
    summary="测试Webhook",
    description="向Webhook URL发送测试事件以验证连接性和载荷格式。",
    tags=["Webhook"]
)
async def test_webhook(test_request: WebhookTestRequest, background_tasks: BackgroundTasks) -> WebhookTestResponse:
    """
    向指定URL发送测试Webhook事件以验证连接和数据格式
    
    此接口允许您在正式配置Webhook之前测试目标URL的连接性和数据处理能力。
    将发送一个模拟的人脸识别事件到指定URL，用于验证您的服务器能否正确接收和处理Webhook数据。
    
    **请求参数:**
    - url `HttpUrl`: 要测试的目标Webhook URL，必须是可访问的HTTP/HTTPS地址
    - test_data `Dict`: 自定义测试数据（可选），如不提供则使用系统默认的测试事件数据
    
    **响应数据:**
    - message `str`: 测试状态消息，指示测试是否已发送
    - url `str`: 已测试的Webhook URL地址
    - test_event `Dict`: 发送的测试事件数据，格式与实际事件相同
    
    **默认测试事件格式:**
    ```json
    {
        "event_type": "test",
        "timestamp": "2024-01-15T10:30:00",
        "camera_id": "test_camera",
        "data": {
            "message": "This is a test webhook event",
            "faces": [
                {
                    "name": "Test Person",
                    "confidence": 0.95,
                    "bbox": [100, 100, 200, 200]
                }
            ]
        },
        "metadata": {"test": true}
    }
    ```
    
    **注意事项:**
    - 测试事件将在后台异步发送
    - 测试不会重试，仅发送一次
    - 建议在创建正式Webhook配置前使用此接口测试
    - 确保您的服务器能正确响应HTTP POST请求
    """
    test_event = WebhookEvent(
        event_type="test",
        timestamp=datetime.now().isoformat(),
        camera_id="test_camera",
        data=test_request.test_data or {
            "message": "This is a test webhook event",
            "faces": [
                {
                    "name": "Test Person",
                    "confidence": 0.95,
                    "bbox": [100, 100, 200, 200]
                }
            ]
        },
        metadata={"test": True}
    )
    
    # Create temporary webhook config
    test_config = WebhookConfig(
        url=str(test_request.url),
        timeout=10,
        retry_count=1
    )
    
    # Test the webhook in background
    async def send_test():
        try:
            if not webhook_manager._session:
                import aiohttp
                async with aiohttp.ClientSession() as session:
                    await webhook_manager._deliver_to_webhook(
                        "test",
                        test_config,
                        [test_event]
                    )
            else:
                await webhook_manager._deliver_to_webhook(
                    "test",
                    test_config,
                    [test_event]
                )
            logger.info(f"Test webhook sent to {test_request.url}")
        except Exception as e:
            logger.error(f"Test webhook failed: {e}")
    
    background_tasks.add_task(send_test)
    
    return WebhookTestResponse(
        message="Test webhook sent",
        url=str(test_request.url),
        test_event=test_event.to_dict()
    )


@router.get(
    "/webhooks/stats",
    response_model=WebhookStatsResponse,
    summary="获取Webhook统计",
    description="获取当前Webhook系统统计信息，包括队列状态和交付指标。",
    tags=["Webhook"]
)
async def get_webhook_statistics() -> WebhookStatsResponse:
    """
    获取Webhook系统运行状态和性能统计信息
    
    此接口提供Webhook系统的全面统计信息，包括配置数量、队列状态、管理器运行状态等。
    用于监控系统性能和诊断潜在问题。
    
    **响应数据:**
    - total_webhooks `int`: 系统中已配置的Webhook总数量
    - enabled_webhooks `int`: 当前处于启用状态的Webhook数量
    - queue_size `int`: 当前事件队列中等待发送的事件数量
    - manager_running `bool`: Webhook管理器是否正在运行
    
    **监控指标说明:**
    - 如果 queue_size 持续增长，可能表示Webhook发送速度跟不上事件产生速度
    - 如果 manager_running 为 false，表示Webhook系统未运行，不会发送任何事件
    - enabled_webhooks 与 total_webhooks 的比例反映了系统的活跃度
    
    **使用场景:**
    - 系统健康监控
    - 性能分析和优化
    - 故障诊断和排查
    - 运维状态检查
    """
    return WebhookStatsResponse(
        total_webhooks=len(webhook_manager.webhooks),
        enabled_webhooks=sum(1 for w in webhook_manager.webhooks.values() if w.enabled),
        queue_size=webhook_manager.event_queue.qsize(),
        manager_running=webhook_manager.running
    )


@router.on_event("startup")
async def startup_webhook_manager():
    """Start webhook manager on API startup"""
    webhook_manager.start()
    logger.info("Webhook manager started")


@router.on_event("shutdown")
async def shutdown_webhook_manager():
    """Stop webhook manager on API shutdown"""
    webhook_manager.stop()
    logger.info("Webhook manager stopped")