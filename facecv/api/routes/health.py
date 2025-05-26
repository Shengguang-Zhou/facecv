"""健康检查路由"""

from fastapi import APIRouter
from datetime import datetime

router = APIRouter()

@router.get("/health")
async def health_check():
    """健康检查端点"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "FaceCV API",
        "version": "0.1.0"
    }

@router.get("/")
async def root():
    """根路径"""
    return {
        "message": "Welcome to FaceCV API",
        "docs": "/docs",
        "health": "/health"
    }