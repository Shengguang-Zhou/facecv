"""数据库配置管理"""

import os
from typing import Optional
from dataclasses import dataclass
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()


@dataclass
class DatabaseConfig:
    """数据库配置类"""
    
    # 数据库类型
    db_type: str = "sqlite"
    
    # MySQL配置
    mysql_host: str = "localhost"
    mysql_port: int = 3306
    mysql_user: str = "root"
    mysql_password: str = ""
    mysql_database: str = "facecv"
    
    # SQLite配置
    sqlite_path: str = "facecv.db"
    
    # ChromaDB配置
    chromadb_path: str = "./chromadb_data"
    chromadb_collection_name: str = "face_embeddings"
    
    # 连接池配置
    pool_size: int = 10
    max_overflow: int = 20
    pool_timeout: int = 30
    pool_recycle: int = 3600
    
    @classmethod
    def from_env(cls) -> 'DatabaseConfig':
        """从环境变量创建配置"""
        return cls(
            db_type=os.getenv("DB_TYPE", "sqlite"),
            mysql_host=os.getenv("MYSQL_HOST", "localhost"),
            mysql_port=int(os.getenv("MYSQL_PORT", "3306")),
            mysql_user=os.getenv("MYSQL_USER", "root"),
            mysql_password=os.getenv("MYSQL_PASSWORD", ""),
            mysql_database=os.getenv("MYSQL_DATABASE", "facecv"),
            sqlite_path=os.getenv("SQLITE_PATH", "facecv.db"),
            chromadb_path=os.getenv("CHROMADB_PATH", "./chromadb_data"),
            chromadb_collection_name=os.getenv("CHROMADB_COLLECTION_NAME", "face_embeddings"),
            pool_size=int(os.getenv("DB_POOL_SIZE", "10")),
            max_overflow=int(os.getenv("DB_MAX_OVERFLOW", "20")),
            pool_timeout=int(os.getenv("DB_POOL_TIMEOUT", "30")),
            pool_recycle=int(os.getenv("DB_POOL_RECYCLE", "3600"))
        )
    
    @property
    def mysql_url(self) -> str:
        """获取MySQL连接URL"""
        return f"mysql+pymysql://{self.mysql_user}:{self.mysql_password}@{self.mysql_host}:{self.mysql_port}/{self.mysql_database}"
    
    @property
    def async_mysql_url(self) -> str:
        """获取异步MySQL连接URL"""
        return f"mysql+aiomysql://{self.mysql_user}:{self.mysql_password}@{self.mysql_host}:{self.mysql_port}/{self.mysql_database}"
    
    @property
    def sqlite_url(self) -> str:
        """获取SQLite连接URL"""
        return f"sqlite:///{self.sqlite_path}"


# 全局配置实例
db_config = DatabaseConfig.from_env()