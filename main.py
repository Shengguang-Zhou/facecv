"""FaceCV API 主程序"""

import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from facecv.api.routes import health, stream, webhook, camera_stream, deepface_api
from facecv.api.routes import insightface_api as insightface
from facecv.config import get_settings
from facecv.database import get_default_database, test_database_availability
from facecv.core.webhook import webhook_manager

# 获取配置
settings = get_settings()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动事件
    logger.info("FaceCV API 服务启动")
    logger.info(f"文档地址: http://localhost:7003/docs")
    
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
        # 继续运行，使用mock模式
    
    # 启动 webhook manager
    webhook_manager.start()
    logger.info("Webhook manager 已启动")
    
    yield
    
    # 关闭事件
    logger.info("FaceCV API 服务关闭")
    
    # 停止 webhook manager
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
app.include_router(stream.router, prefix="/api/v1")
app.include_router(deepface_api.router)  # DeepFace APIs restored
app.include_router(webhook.router, prefix="/api/v1")
app.include_router(insightface.router, prefix="/api/v1/insightface")
app.include_router(camera_stream.router)

# Import and register batch processing routes
from facecv.api.routes import batch_processing
app.include_router(batch_processing.router)

# Import and register model management routes
try:
    from facecv.api.routes import models
    app.include_router(models.router)
    logger.info("Model management APIs loaded")
except ImportError as e:
    logger.warning(f"Model management APIs not loaded: {e}")

# Import and register system health routes
try:
    from facecv.api.routes import system_health
    app.include_router(system_health.router)
    logger.info("System health APIs loaded")
except ImportError as e:
    logger.warning(f"System health APIs not loaded: {e}")

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