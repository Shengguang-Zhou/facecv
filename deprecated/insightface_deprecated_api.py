"""
InsightFace 废弃的 API 端点

这些端点已被废弃，保留仅为向后兼容。
请勿在新项目中使用这些 API。
"""

from fastapi import APIRouter, HTTPException, Form, Query
from typing import Optional
import logging
from datetime import datetime

router = APIRouter(tags=["InsightFace Deprecated"])
logger = logging.getLogger(__name__)

# ==================== 废弃的模型管理端点 ====================

@router.post("/models/switch", summary="[废弃] 切换模型类型", deprecated=True)
async def switch_model_type(
    enable_arcface: bool = Form(..., description="是否启用ArcFace专用模型"),
    arcface_backbone: Optional[str] = Form("resnet50", description="ArcFace骨干网络 (resnet50/mobilefacenet)")
):
    """
    切换模型类型 (ArcFace vs Buffalo) - **已废弃**
    
    ⚠️ **此端点已废弃**：会破坏模型缓存，导致性能问题。
    建议在每个请求中指定 model_name 参数。
    
    此接口允许在运行时切换模型类型，支持：
    1. 启用ArcFace专用模型 (enable_arcface=True)
    2. 使用传统Buffalo模型 (enable_arcface=False)
    """
    raise HTTPException(
        status_code=410,
        detail={
            "error": "This endpoint has been deprecated",
            "message": "Model switching breaks the cache and causes performance issues",
            "recommendation": "Specify model_name in each request instead",
            "deprecated_since": "2024-01-01"
        }
    )

@router.post("/models/select", summary="[废弃] 选择模型", deprecated=True)
async def select_model(
    model: str = Query(..., description="要选择的模型名称: buffalo_l, buffalo_m, buffalo_s, antelopev2")
):
    """
    动态切换InsightFace模型 - **已废弃**
    
    ⚠️ **此端点已废弃**：会破坏模型缓存，导致性能问题。
    建议在每个请求中指定 model_name 参数。
    
    **参数:**
    - model `str`: 要选择的模型名称
    """
    raise HTTPException(
        status_code=410,
        detail={
            "error": "This endpoint has been deprecated",
            "message": "Model selection breaks the cache and causes performance issues",
            "recommendation": "Specify model_name in each request instead",
            "deprecated_since": "2024-01-01"
        }
    )

@router.get("/faces/count", summary="[废弃] 获取人脸数量", deprecated=True)
async def get_face_count():
    """
    获取数据库中人脸的总数 - **已废弃**
    
    ⚠️ **此端点已废弃**：请使用 GET /faces 端点，它返回的响应中包含 'total' 字段。
    
    **返回:**
    包含以下内容的对象:
    - total_faces `int`: 数据库中已注册人脸的总数
    """
    raise HTTPException(
        status_code=410,
        detail={
            "error": "This endpoint has been deprecated",
            "message": "This functionality is redundant",
            "recommendation": "Use GET /faces endpoint which includes 'total' field",
            "deprecated_since": "2024-01-01",
            "example": "GET /api/v1/insightface/faces returns {faces: [...], total: 150, returned: 20}"
        }
    )