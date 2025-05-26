"""健康检查路由"""

from fastapi import APIRouter
from datetime import datetime
from pydantic import BaseModel
from typing import Dict, Any

router = APIRouter()


class HealthResponse(BaseModel):
    """健康检查响应模型"""
    status: str
    timestamp: str
    service: str
    version: str


class RootResponse(BaseModel):
    """根路径响应模型"""
    message: str
    docs: str
    health: str

@router.get(
    "/health",
    response_model=HealthResponse,
    summary="健康检查",
    description="检查FaceCV API服务的健康状态，返回当前时间戳和服务信息。",
    tags=["健康检查"]
)
async def health_check() -> HealthResponse:
    """
    健康检查端点，返回API服务的当前状态。
    
    Returns:
        HealthResponse: 服务健康状态信息
            - status `str`: 当前服务状态（如果能响应则始终为"healthy"）
            - timestamp `str`: 当前UTC时间戳（ISO格式）
            - service `str`: 服务名称标识符
            - version `str`: 当前API版本
    """
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "FaceCV API",
        "version": "0.1.0"
    }

@router.get(
    "/",
    response_model=RootResponse,
    summary="根路径",
    description="欢迎端点，提供基本的API信息和导航链接。",
    tags=["根路径"]
)
async def root() -> RootResponse:
    """
    根路径端点，提供欢迎信息和API导航。
    
    Returns:
        RootResponse: 基本API信息和链接
            - message `str`: 欢迎消息
            - docs `str`: API文档路径
            - health `str`: 健康检查端点路径
    """
    return {
        "message": "Welcome to FaceCV API",
        "docs": "/docs",
        "health": "/health"
    }