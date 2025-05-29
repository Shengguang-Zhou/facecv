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
from facecv.config import get_settings
from facecv.config.runtime_config import get_runtime_config
from facecv.database import get_default_database, test_database_availability
from facecv.utils.cuda_utils import setup_cuda_environment

# 获取配置
settings = get_settings()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动事件
    logger.info("FaceCV API 服务启动")
    logger.info(f"文档地址: http://localhost:{settings.port}/docs")
    
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

    
    yield
    
    # 关闭事件
    logger.info("FaceCV API 服务关闭")

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
# Register required routes
app.include_router(insightface.router, prefix="/api/v1/insightface")

# Import and register system health routes
try:
    from facecv.api.routes import system_health
    app.include_router(system_health.router)
    logger.info("System health APIs loaded")
except ImportError as e:
    logger.warning(f"System health APIs not loaded: {e}")



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
