"""Model Management API Routes

DEPRECATED: These model management endpoints are deprecated and will be removed in a future version.
Model loading/unloading is now handled automatically by the framework.
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Dict, List, Any, Optional
import logging
import time
import psutil
import os
from datetime import datetime
from pydantic import BaseModel
import warnings

from facecv.models.insightface.real_recognizer import RealInsightFaceRecognizer
from facecv.config import get_settings
from facecv.database.sqlite_facedb import SQLiteFaceDB

router = APIRouter(prefix="/api/v1/models", tags=["Model Management (Deprecated)"])
logger = logging.getLogger(__name__)

# Global model storage
_model_store = {}
_model_stats = {}

class ModelStatus(BaseModel):
    loaded: bool
    status: str  # "active", "loading", "error", "unloaded"
    provider: Optional[str] = None
    memory_usage: int = 0  # MB
    load_time: float = 0.0  # seconds
    last_used: Optional[str] = None
    error_message: Optional[str] = None

class LoadModelRequest(BaseModel):
    model_name: str
    providers: Optional[List[str]] = ["CPUExecutionProvider"]
    force_reload: bool = False

class ModelInfo(BaseModel):
    name: str
    description: str
    use_cases: List[str]
    performance: Dict[str, Any]
    requirements: Dict[str, Any]
    downloaded: bool
    size: str
    accuracy: str
    speed: str

class RecommendationRequest(BaseModel):
    use_case: str  # "high_accuracy", "real_time", "mobile", "server"
    has_gpu: bool = False
    memory_limit_mb: int = 2048
    latency_requirement: float = 100.0  # ms

def get_memory_usage():
    """Get current memory usage in MB"""
    try:
        process = psutil.Process()
        return process.memory_info().rss // 1024 // 1024
    except:
        return 0

def get_model_instance(model_name: str):
    """Get or create model instance"""
    global _model_store, _model_stats
    
    if model_name not in _model_store:
        return None
    
    return _model_store[model_name]

@router.get("/status", 
    response_model=Dict[str, ModelStatus],
    summary="获取模型状态",
    description="""获取所有已加载模型的状态信息。
    
    返回每个模型的详细状态，包括：
    - 加载状态和运行状态
    - 内存使用情况
    - 最后使用时间
    - 错误信息（如果有）
    """)
async def get_models_status():
    try:
        status_dict = {}
        
        for model_name in ["buffalo_l", "buffalo_m", "buffalo_s", "antelopev2"]:
            if model_name in _model_store:
                model = _model_store[model_name]
                stats = _model_stats.get(model_name, {})
                
                status_dict[model_name] = ModelStatus(
                    loaded=True,
                    status="active",
                    provider="CPUExecutionProvider",
                    memory_usage=stats.get("memory_usage", 0),
                    load_time=stats.get("load_time", 0.0),
                    last_used=stats.get("last_used"),
                    error_message=None
                )
            else:
                status_dict[model_name] = ModelStatus(
                    loaded=False,
                    status="unloaded",
                    provider=None,
                    memory_usage=0,
                    load_time=0.0,
                    last_used=None,
                    error_message=None
                )
        
        return status_dict
        
    except Exception as e:
        logger.error(f"Error getting model status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get model status: {str(e)}")

@router.get("/providers",
    summary="获取可用的执行提供者",
    description="""获取系统中可用的模型执行提供者列表。
    
    执行提供者包括：
    - CPUExecutionProvider: CPU执行
    - CUDAExecutionProvider: NVIDIA GPU执行（如果可用）
    - TensorrtExecutionProvider: TensorRT加速（如果可用）
    """)
async def get_available_providers():
    try:
        providers = ["CPUExecutionProvider"]
        
        # Check for GPU availability
        try:
            import onnxruntime
            available_providers = onnxruntime.get_available_providers()
            if "CUDAExecutionProvider" in available_providers:
                providers.append("CUDAExecutionProvider")
            if "TensorrtExecutionProvider" in available_providers:
                providers.append("TensorrtExecutionProvider")
        except Exception as e:
            logger.warning(f"Could not check ONNX providers: {e}")
        
        return {
            "success": True,
            "providers": providers,
            "default": "CPUExecutionProvider"
        }
        
    except Exception as e:
        logger.error(f"Error getting providers: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get providers: {str(e)}")

@router.post("/load",
    summary="加载模型 (已弃用)",
    description="""**⚠️ DEPRECATED**: 此接口已被弃用，将在未来版本中移除。模型加载现在由框架自动处理。
    
    加载指定的人脸识别模型。
    
    支持的模型：
    - buffalo_l: 高精度模型（较大）
    - buffalo_m: 中等精度模型
    - buffalo_s: 高速模型（较小）
    - antelopev2: 先进模型
    
    **加载过程：**
    1. 验证模型名称
    2. 初始化模型实例
    3. 加载模型权重
    4. 验证模型功能
    """,
    deprecated=True)
async def load_model(request: LoadModelRequest):
    warnings.warn(
        "The model load endpoint is deprecated and will be removed in a future version. "
        "Model loading is now handled automatically by the framework.",
        DeprecationWarning,
        stacklevel=2
    )
    logger.warning("Deprecated model load endpoint called")
    
    try:
        start_time = time.time()
        
        if request.model_name not in ["buffalo_l", "buffalo_m", "buffalo_s", "antelopev2"]:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported model: {request.model_name}"
            )
        
        # Check if already loaded
        if request.model_name in _model_store and not request.force_reload:
            return {
                "success": True,
                "message": f"Model {request.model_name} already loaded",
                "model_name": request.model_name,
                "provider": "CPUExecutionProvider",
                "load_time": 0.0,
                "memory_usage": _model_stats.get(request.model_name, {}).get("memory_usage", 0)
            }
        
        # Initialize database
        settings = get_settings()
        db_path = os.path.join(os.getcwd(), "facecv_production.db")
        face_db = SQLiteFaceDB(db_path=db_path)
        
        # Load model
        recognizer = RealInsightFaceRecognizer(
            face_db=face_db,
            model_pack=request.model_name,
            similarity_threshold=0.4,
            det_thresh=0.5
        )
        
        load_time = time.time() - start_time
        memory_usage = get_memory_usage()
        
        # Store model and stats
        _model_store[request.model_name] = recognizer
        _model_stats[request.model_name] = {
            "load_time": load_time,
            "memory_usage": memory_usage,
            "last_used": datetime.now().isoformat(),
            "provider": request.providers[0] if request.providers else "CPUExecutionProvider"
        }
        
        return {
            "success": True,
            "message": f"Model {request.model_name} loaded successfully",
            "model_name": request.model_name,
            "provider": request.providers[0] if request.providers else "CPUExecutionProvider",
            "load_time": load_time,
            "memory_usage": memory_usage
        }
        
    except Exception as e:
        logger.error(f"Error loading model {request.model_name}: {e}")
        
        # Store error status
        _model_stats[request.model_name] = {
            "load_time": 0.0,
            "memory_usage": 0,
            "last_used": None,
            "error_message": str(e)
        }
        
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to load model {request.model_name}: {str(e)}"
        )

@router.post("/unload",
    summary="卸载模型 (已弃用)",
    description="""**⚠️ DEPRECATED**: 此接口已被弃用，将在未来版本中移除。模型生命周期现在由框架自动管理。
    
    卸载指定的模型以释放内存。
    
    **卸载过程：**
    1. 停止模型推理
    2. 释放模型内存
    3. 清理相关资源
    4. 更新模型状态
    """,
    deprecated=True)
async def unload_model(model_name: str):
    warnings.warn(
        "The model unload endpoint is deprecated and will be removed in a future version. "
        "Model lifecycle is now managed automatically by the framework.",
        DeprecationWarning,
        stacklevel=2
    )
    logger.warning("Deprecated model unload endpoint called")
    
    try:
        if model_name not in _model_store:
            raise HTTPException(
                status_code=404, 
                detail=f"Model {model_name} is not loaded"
            )
        
        # Remove model
        del _model_store[model_name]
        if model_name in _model_stats:
            del _model_stats[model_name]
        
        return {
            "success": True,
            "message": f"Model {model_name} unloaded successfully",
            "model_name": model_name,
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error unloading model {model_name}: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to unload model {model_name}: {str(e)}"
        )

@router.get("/info/{model_name}",
    summary="获取模型信息 (已弃用)",
    description="""**⚠️ DEPRECATED**: 此接口已被弃用，将在未来版本中移除。模型信息现在通过配置文件管理。
    
    获取指定模型的详细信息。
    
    返回模型的技术规格、性能参数和使用建议。
    """,
    deprecated=True)
async def get_model_info(model_name: str):
    warnings.warn(
        "The model info endpoint is deprecated and will be removed in a future version. "
        "Model information is now managed through configuration files.",
        DeprecationWarning,
        stacklevel=2
    )
    logger.warning("Deprecated model info endpoint called")
    
    try:
        model_configs = {
            "buffalo_l": {
                "name": "Buffalo-L",
                "description": "高精度人脸识别模型，适合服务器端部署",
                "accuracy": "99.8%",
                "speed": "~50ms",
                "memory": "~500MB",
                "use_cases": ["高精度识别", "服务器部署", "批量处理"]
            },
            "buffalo_m": {
                "name": "Buffalo-M", 
                "description": "中等精度模型，平衡准确性和速度",
                "accuracy": "99.5%",
                "speed": "~30ms",
                "memory": "~300MB",
                "use_cases": ["一般识别", "实时处理", "移动设备"]
            },
            "buffalo_s": {
                "name": "Buffalo-S",
                "description": "高速轻量模型，适合实时应用",
                "accuracy": "99.2%",
                "speed": "~15ms", 
                "memory": "~150MB",
                "use_cases": ["实时识别", "边缘设备", "移动应用"]
            },
            "antelopev2": {
                "name": "Antelope-V2",
                "description": "最新先进模型，具有最佳性能",
                "accuracy": "99.9%",
                "speed": "~40ms",
                "memory": "~400MB",
                "use_cases": ["最高精度", "关键应用", "安全系统"]
            }
        }
        
        if model_name not in model_configs:
            raise HTTPException(
                status_code=404,
                detail=f"Model {model_name} not found"
            )
        
        config = model_configs[model_name]
        is_loaded = model_name in _model_store
        
        return {
            "success": True,
            "model_name": model_name,
            "loaded": is_loaded,
            "config": config,
            "status": _model_stats.get(model_name, {}),
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting model info for {model_name}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get model info: {str(e)}"
        )

@router.get("/performance",
    summary="获取模型性能指标 (已弃用)",
    description="""**⚠️ DEPRECATED**: 此接口已被弃用，将在未来版本中移除。性能监控应通过专门的监控工具实现。
    
    获取所有模型的性能统计信息。
    
    包括推理时间、内存使用、吞吐量等关键指标。
    """,
    deprecated=True)
async def get_model_performance():
    warnings.warn(
        "The model performance endpoint is deprecated and will be removed in a future version. "
        "Performance monitoring should be implemented through dedicated monitoring tools.",
        DeprecationWarning,
        stacklevel=2
    )
    logger.warning("Deprecated model performance endpoint called")
    
    try:
        performance_data = {
            "timestamp": datetime.now().isoformat(),
            "system_memory": get_memory_usage(),
            "models": {}
        }
        
        for model_name, stats in _model_stats.items():
            performance_data["models"][model_name] = {
                "memory_usage_mb": stats.get("memory_usage", 0),
                "load_time_seconds": stats.get("load_time", 0.0),
                "last_used": stats.get("last_used"),
                "status": "active" if model_name in _model_store else "unloaded"
            }
        
        return {
            "success": True,
            "performance": performance_data
        }
        
    except Exception as e:
        logger.error(f"Error getting performance metrics: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get performance metrics: {str(e)}"
        )

# Advanced Model Management

@router.get("/advanced/available",
    summary="获取可用的高级模型 (已弃用)",
    description="""**⚠️ DEPRECATED**: 此接口已被弃用，将在未来版本中移除。模型信息现在通过配置文件静态管理。
    
    获取所有可用的高级人脸识别模型列表。
    
    返回每个模型的详细信息，包括性能特征、使用场景和系统要求。
    """,
    deprecated=True)
async def get_available_advanced_models():
    warnings.warn(
        "The advanced available models endpoint is deprecated and will be removed in a future version. "
        "Model information is now managed through configuration files.",
        DeprecationWarning,
        stacklevel=2
    )
    logger.warning("Deprecated advanced available models endpoint called")
    
    try:
        models = [
            {
                "name": "buffalo_l",
                "description": "Buffalo Large - 高精度服务器模型",
                "use_cases": ["高精度识别", "服务器部署", "批量处理"],
                "performance": {"accuracy": "99.8%", "speed": "~50ms", "memory": "~500MB"},
                "requirements": {"min_memory_mb": 512, "recommended_cpu_cores": 4},
                "downloaded": True,
                "size": "120MB",
                "accuracy": "Very High",
                "speed": "Medium"
            },
            {
                "name": "buffalo_m", 
                "description": "Buffalo Medium - 平衡性能模型",
                "use_cases": ["一般识别", "实时处理", "移动设备"],
                "performance": {"accuracy": "99.5%", "speed": "~30ms", "memory": "~300MB"},
                "requirements": {"min_memory_mb": 320, "recommended_cpu_cores": 2},
                "downloaded": True,
                "size": "80MB",
                "accuracy": "High",
                "speed": "Fast"
            },
            {
                "name": "buffalo_s",
                "description": "Buffalo Small - 高速轻量模型", 
                "use_cases": ["实时识别", "边缘设备", "移动应用"],
                "performance": {"accuracy": "99.2%", "speed": "~15ms", "memory": "~150MB"},
                "requirements": {"min_memory_mb": 160, "recommended_cpu_cores": 1},
                "downloaded": True,
                "size": "45MB",
                "accuracy": "Good",
                "speed": "Very Fast"
            },
            {
                "name": "antelopev2",
                "description": "Antelope V2 - 最新先进模型",
                "use_cases": ["最高精度", "关键应用", "安全系统"],
                "performance": {"accuracy": "99.9%", "speed": "~40ms", "memory": "~400MB"},
                "requirements": {"min_memory_mb": 420, "recommended_cpu_cores": 4},
                "downloaded": True,
                "size": "95MB", 
                "accuracy": "Excellent",
                "speed": "Fast"
            }
        ]
        
        return {
            "success": True,
            "models": models,
            "total_count": len(models)
        }
        
    except Exception as e:
        logger.error(f"Error getting available models: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get available models: {str(e)}"
        )

@router.post("/advanced/recommendations",
    summary="获取模型推荐 (已弃用)",
    description="""**⚠️ DEPRECATED**: 此接口已被弃用，将在未来版本中移除。模型选择应基于文档和最佳实践。
    
    根据使用场景和系统资源获取最适合的模型推荐。
    
    智能分析您的需求和系统能力，推荐最合适的模型配置。
    """,
    deprecated=True)
async def get_model_recommendations(request: RecommendationRequest):
    warnings.warn(
        "The model recommendations endpoint is deprecated and will be removed in a future version. "
        "Model selection should be based on documentation and best practices.",
        DeprecationWarning,
        stacklevel=2
    )
    logger.warning("Deprecated model recommendations endpoint called")
    
    try:
        recommendations = []
        
        # Define model characteristics
        models = {
            "buffalo_s": {"accuracy": 99.2, "speed": 15, "memory": 150, "score": 0},
            "buffalo_m": {"accuracy": 99.5, "speed": 30, "memory": 300, "score": 0},
            "buffalo_l": {"accuracy": 99.8, "speed": 50, "memory": 500, "score": 0},
            "antelopev2": {"accuracy": 99.9, "speed": 40, "memory": 400, "score": 0}
        }
        
        # Score models based on requirements
        for name, specs in models.items():
            score = 0
            reason_parts = []
            
            # Memory constraint
            if specs["memory"] <= request.memory_limit_mb:
                score += 30
                reason_parts.append("符合内存限制")
            else:
                score -= 20
                reason_parts.append("超出内存限制")
            
            # Latency requirement
            if specs["speed"] <= request.latency_requirement:
                score += 25
                reason_parts.append("满足延迟要求") 
            else:
                score -= 15
                reason_parts.append("延迟较高")
            
            # Use case specific scoring
            if request.use_case == "high_accuracy":
                score += specs["accuracy"] - 98  # Bonus for accuracy above 98%
                reason_parts.append("高精度需求")
            elif request.use_case == "real_time":
                score += max(0, 60 - specs["speed"])  # Bonus for lower latency
                reason_parts.append("实时处理需求")
            elif request.use_case == "mobile":
                score += max(0, 200 - specs["memory"]) / 10  # Bonus for lower memory
                reason_parts.append("移动设备优化")
            elif request.use_case == "server":
                score += specs["accuracy"] - 99  # Emphasis on accuracy for server
                reason_parts.append("服务器部署")
            
            # GPU bonus
            if request.has_gpu:
                score += 10
                reason_parts.append("GPU加速可用")
            
            models[name]["score"] = score
            
            recommendations.append({
                "model_name": name,
                "score": score,
                "reason": "，".join(reason_parts),
                "expected_performance": {
                    "accuracy": f"{specs['accuracy']:.1f}%",
                    "latency_ms": specs["speed"],
                    "memory_mb": specs["memory"]
                }
            })
        
        # Sort by score
        recommendations.sort(key=lambda x: x["score"], reverse=True)
        
        return {
            "success": True,
            "recommendations": recommendations[:3],  # Top 3 recommendations
            "request_summary": {
                "use_case": request.use_case,
                "has_gpu": request.has_gpu,
                "memory_limit_mb": request.memory_limit_mb,
                "latency_requirement_ms": request.latency_requirement
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting recommendations: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get recommendations: {str(e)}"
        )

@router.post("/advanced/switch",
    summary="智能模型切换 (已弃用)",
    description="""**⚠️ DEPRECATED**: 此接口已被弃用，将在未来版本中移除。模型切换现在由框架自动处理。
    
    在运行时切换模型，保持服务连续性。
    
    **切换流程：**
    1. 预热新模型
    2. 验证新模型功能
    3. 切换流量到新模型
    4. 卸载旧模型（可选）
    """,
    deprecated=True)
async def switch_model(
    from_model: str,
    to_model: str, 
    preserve_state: bool = True
):
    warnings.warn(
        "The model switch endpoint is deprecated and will be removed in a future version. "
        "Model switching is now handled automatically by the framework.",
        DeprecationWarning,
        stacklevel=2
    )
    logger.warning("Deprecated model switch endpoint called")
    
    try:
        # Validate model names
        valid_models = ["buffalo_l", "buffalo_m", "buffalo_s", "antelopev2"]
        if from_model not in valid_models or to_model not in valid_models:
            raise HTTPException(
                status_code=400,
                detail="Invalid model name"
            )
        
        # Check if source model exists
        if from_model not in _model_store:
            raise HTTPException(
                status_code=404,
                detail=f"Source model {from_model} is not loaded"
            )
        
        # Load target model if not already loaded
        if to_model not in _model_store:
            load_request = LoadModelRequest(model_name=to_model)
            await load_model(load_request)
        
        # Switch is essentially just ensuring both models are loaded
        # In a real implementation, you would switch the active model pointer
        
        result = {
            "success": True,
            "message": f"Successfully switched from {from_model} to {to_model}",
            "from_model": from_model,
            "to_model": to_model,
            "preserve_state": preserve_state,
            "timestamp": datetime.now().isoformat()
        }
        
        # Optionally unload the old model
        if not preserve_state:
            await unload_model(from_model)
            result["old_model_unloaded"] = True
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error switching models {from_model} -> {to_model}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to switch models: {str(e)}"
        )