"""FaceCV API 主程序"""

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from facecv.api.routes import face, health, stream, webhook, camera_stream
from facecv.api.routes import insightface_real as insightface
from facecv.config import get_settings
from facecv.database import get_default_database, test_database_availability
from facecv.core.webhook import webhook_manager

# 获取配置
settings = get_settings()

# 创建FastAPI应用
app = FastAPI(
    title="FaceCV API",
    description="专业的人脸识别API服务",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc"
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
app.include_router(health.router, tags=["health"])
# Legacy face routes removed - using InsightFace only
app.include_router(stream.router, prefix="/api/v1", tags=["stream"])
# DeepFace router removed - using InsightFace only
app.include_router(webhook.router, prefix="/api/v1", tags=["webhook"])
app.include_router(insightface.router, prefix="/api/v1/insightface", tags=["insightface"])
app.include_router(camera_stream.router, tags=["camera"])

# Import and register batch processing routes
from facecv.api.routes import batch_processing
app.include_router(batch_processing.router, tags=["batch"])

@app.on_event("startup")
async def startup_event():
    """启动事件"""
    logger.info("FaceCV API 服务启动")
    logger.info(f"文档地址: http://localhost:{settings.port}/docs")
    
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

@app.on_event("shutdown")
async def shutdown_event():
    """关闭事件"""
    logger.info("FaceCV API 服务关闭")
    
    # 停止 webhook manager
    webhook_manager.stop()
    logger.info("Webhook manager 已停止")

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=7002,  # Changed to port 7002
        reload=settings.debug
    )