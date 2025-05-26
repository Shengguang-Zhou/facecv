"""应用配置"""

from typing import List, Optional
from functools import lru_cache
from pydantic_settings import BaseSettings
import os

class Settings(BaseSettings):
    """应用配置类"""
    
    # API配置
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    cors_origins: List[str] = ["*"]
    
    # 模型配置
    model_backend: str = "insightface"  # insightface 或 deepface
    model_device: str = "cpu"  # cpu 或 cuda
    model_path: Optional[str] = None
    
    # 数据库配置
    db_type: str = "sqlite"  # mongodb, mysql, sqlite, chromadb
    db_connection_string: str = "sqlite:///facecv.db"
    
    # 性能配置
    batch_size: int = 32
    num_workers: int = 4
    max_faces_per_image: int = 10
    
    # 安全配置
    secret_key: str = "your-secret-key-here"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    
    # 文件上传配置
    max_upload_size: int = 10 * 1024 * 1024  # 10MB
    allowed_extensions: List[str] = [".jpg", ".jpeg", ".png", ".bmp"]
    
    # 日志配置
    log_level: str = "INFO"
    log_file: Optional[str] = "facecv.log"
    
    class Config:
        env_file = ".env"
        env_prefix = "FACECV_"
        extra = "allow"  # Allow extra fields from .env

@lru_cache()
def get_settings() -> Settings:
    """获取配置单例"""
    return Settings()