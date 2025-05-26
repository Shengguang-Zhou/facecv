"""InsightFace API Routes - Simplified Version"""

from fastapi import APIRouter, HTTPException
from datetime import datetime

router = APIRouter()

@router.get("/health")
async def health_check():
    """Health check for InsightFace services"""
    return {
        "status": "healthy", 
        "service": "InsightFace API",
        "timestamp": str(datetime.now())
    }

@router.get("/models/info")
async def get_model_info():
    """Get information about InsightFace models"""
    return {
        "status": "mock_mode",
        "message": "InsightFace models not fully configured"
    }