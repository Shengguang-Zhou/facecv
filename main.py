"""FaceCV API 主程序"""

import os
# Fix protobuf compatibility issue with TensorFlow/DeepFace
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from facecv.api.routes import health
from facecv.api.routes import insightface_lazy as insightface
from facecv.api.routes.deprecated_api import deprecated_router
from facecv.config import get_settings
from facecv.config.runtime_config import get_runtime_config
from facecv.database import get_default_database, test_database_availability
# from facecv.core.webhook import webhook_manager  # 已废弃
from facecv.utils.cuda_utils import setup_cuda_environment

# 获取配置
settings = get_settings()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动事件
    logger.info("FaceCV API 服务启动")
    logger.info(f"文档地址: http://localhost:7003/docs")
    
    # Initialize CUDA environment
    logger.info("初始化 CUDA 环境...")
    try:
        setup_cuda_environment()
        runtime_config = get_runtime_config()
        cuda_available = runtime_config.get('cuda_available', False)
        if cuda_available:
            cuda_version = runtime_config.get('cuda_version')
            providers = runtime_config.get('execution_providers', ['CPUExecutionProvider'])
            logger.info(f"CUDA {cuda_version[0]}.{cuda_version[1]} 已配置")
            logger.info(f"执行提供者: {providers}")
        else:
            logger.info("未检测到 CUDA，使用 CPU 模式")
    except Exception as e:
        logger.warning(f"CUDA 初始化警告: {e}")
    
    # 测试数据库连接
    logger.info("测试数据库连接...")
    try:
        availability = test_database_availability()
        logger.info("数据库可用性:")
        for db_type, is_available in availability.items():
            status = "可用" if is_available else "不可用"
            logger.info(f"  {db_type}: {status}")
        
        # 初始化默认数据库
        db = get_default_database()
        face_count = db.get_face_count()
        logger.info(f"数据库初始化成功，当前人脸数量: {face_count}")
        
    except Exception as e:
        logger.error(f"数据库初始化失败: {e}")

    
    # Start webhook manager for stream processing
    from facecv.core.webhook import webhook_manager
    webhook_manager.start()
    logger.info("Webhook manager 已启动")
    
    yield
    
    # 关闭事件
    logger.info("FaceCV API 服务关闭")
    
    # Stop webhook manager
    webhook_manager.stop()
    logger.info("Webhook manager 已停止")

# 创建FastAPI应用
app = FastAPI(
    title="FaceCV API",
    description="专业的人脸识别API服务",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(health.router)
# Clean route organization
# app.include_router(stream.router, prefix="/api/v1")  # Stream APIs deprecated - moved to deprecated_router
# app.include_router(deepface_api.router)  # DeepFace APIs deprecated - moved to deprecated_router
# app.include_router(webhook.router, prefix="/api/v1")  # Webhook API 已废弃
app.include_router(insightface.router, prefix="/api/v1/insightface")

# Register deprecated routes (includes DeepFace, old InsightFace, and old camera APIs)
# Commented out to hide from Swagger - these deprecated APIs still exist in codebase
# app.include_router(deprecated_router)
# logger.warning("Deprecated API routes loaded - these will be removed in future versions")
# app.include_router(camera_stream.router)  # Camera Streaming API 已废弃，使用 InsightFace 内置的流处理

try:
    from facecv.api.routes import simple_detect
    app.include_router(simple_detect.router, prefix="/api/v1/test")
    logger.info("Simple test detection API loaded")
except ImportError as e:
    logger.warning(f"Simple test detection API not loaded: {e}")

# Import and register batch processing routes (DEPRECATED)
# Commented out to hide from Swagger
# try:
#     from facecv.api.routes import batch_processing
#     app.include_router(batch_processing.router)
#     logger.warning("Batch processing APIs loaded - DEPRECATED: These endpoints will be removed in a future version")
# except ImportError as e:
#     logger.info(f"Batch processing APIs not loaded: {e}")

# Import and register model management routes (DEPRECATED)
# Commented out to hide from Swagger
# try:
#     from facecv.api.routes import models
#     app.include_router(models.router)
#     logger.warning("Model management APIs loaded - DEPRECATED: These endpoints will be removed in a future version")
# except ImportError as e:
#     logger.info(f"Model management APIs not loaded: {e}")

# Import and register system health routes
try:
    from facecv.api.routes import system_health
    app.include_router(system_health.router)
    logger.info("System health APIs loaded")
except ImportError as e:
    logger.warning(f"System health APIs not loaded: {e}")

# 废弃的 API（默认不加载）
# if settings.load_deprecated_apis:
#     try:
#         from deprecated import insightface_deprecated_api
#         app.include_router(insightface_deprecated_api.router, prefix="/api/v1/insightface")
#         logger.warning("已加载废弃的 InsightFace API - 仅用于向后兼容")
#     except ImportError:
#         pass

# Additional route modules can be added here in the future


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=settings.host, help="Host to bind to")
    parser.add_argument("--port", type=int, default=settings.port, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", default=settings.debug, help="Enable auto-reload")
    args = parser.parse_args()
    
    uvicorn.run(
        "main:app",
        host=args.host,
        port=args.port,
        reload=args.reload
    )
