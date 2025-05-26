"""数据库配置管理 - 标准化路径和配置验证"""

import os
from typing import Optional, Literal
from dataclasses import dataclass
from dotenv import load_dotenv
from pathlib import Path

# 加载环境变量
load_dotenv()


@dataclass
class DatabaseConfig:
    """数据库配置类 - 包含路径标准化和验证"""
    
    # 数据库类型
    db_type: str = "sqlite"  # sqlite, mysql, chromadb
    
    # 标准化路径配置
    base_data_dir: str = "./data"
    db_dir: str = "./data/db"
    
    # MySQL配置
    mysql_host: str = ""  # 从环境变量加载
    mysql_port: int = 3306
    mysql_user: str = ""
    mysql_password: str = ""  # 从环境变量加载
    mysql_database: str = "facecv"
    mysql_charset: str = "utf8mb4"
    
    # SQLite配置 - 标准化路径
    sqlite_filename: str = "facecv.db"
    
    # ChromaDB配置 - 标准化路径
    chromadb_dirname: str = "chromadb_data"
    chromadb_collection_name: str = "face_embeddings"
    
    # 连接池配置
    pool_size: int = 10
    max_overflow: int = 20
    pool_timeout: int = 30
    pool_recycle: int = 3600
    
    # 连接超时配置
    connect_timeout: int = 30
    read_timeout: int = 60
    write_timeout: int = 60
    
    def __post_init__(self):
        """初始化后验证和标准化路径"""
        self._validate_config()
        self._ensure_directories()
    
    def _validate_config(self):
        """验证配置参数"""
        if self.db_type not in ["sqlite", "mysql", "chromadb"]:
            raise ValueError(f"不支持的数据库类型: {self.db_type}")
        
        if not (1 <= self.mysql_port <= 65535):
            raise ValueError(f"MySQL端口号无效: {self.mysql_port}")
        
        if self.db_type == "mysql":
            if not self.mysql_host:
                raise ValueError("MySQL主机不能为空 (设置 FACECV_MYSQL_HOST)")
            if not self.mysql_user:
                raise ValueError("MySQL用户名不能为空 (设置 FACECV_MYSQL_USER)")
            if not self.mysql_password:
                raise ValueError("MySQL密码不能为空 (设置 FACECV_MYSQL_PASSWORD)")
        
        if not (1 <= self.pool_size <= 100):
            raise ValueError(f"连接池大小无效: {self.pool_size}")
    
    def _ensure_directories(self):
        """确保所有必要目录存在"""
        directories = [
            self.base_data_dir,
            self.db_dir,
            self.get_chromadb_path()
        ]
        
        for directory in directories:
            if directory:
                Path(directory).mkdir(parents=True, exist_ok=True)
    
    def get_sqlite_path(self) -> str:
        """获取标准化的SQLite数据库路径"""
        if os.path.isabs(self.sqlite_filename):
            return self.sqlite_filename
        return os.path.join(self.db_dir, self.sqlite_filename)
    
    def get_chromadb_path(self) -> str:
        """获取标准化的ChromaDB路径"""
        if os.path.isabs(self.chromadb_dirname):
            return self.chromadb_dirname
        return os.path.join(self.base_data_dir, self.chromadb_dirname)
    
    @classmethod
    def from_env(cls) -> 'DatabaseConfig':
        """从环境变量创建配置（保留环境变量覆盖能力）"""
        import logging
        logger = logging.getLogger(__name__)
        
        facecv_db_type_set = "FACECV_DB_TYPE" in os.environ
        facecv_mysql_host_set = "FACECV_MYSQL_HOST" in os.environ
        facecv_mysql_user_set = "FACECV_MYSQL_USER" in os.environ
        facecv_mysql_password_set = "FACECV_MYSQL_PASSWORD" in os.environ
        
        db_type = os.getenv("FACECV_DB_TYPE") or os.getenv("DB_TYPE", "sqlite")
        if os.getenv("DB_TYPE"):
            logger.warning("DB_TYPE 环境变量已弃用，请使用 FACECV_DB_TYPE")
            
        mysql_host = os.getenv("FACECV_MYSQL_HOST") if facecv_mysql_host_set else os.getenv("MYSQL_HOST", "localhost")
        if os.getenv("MYSQL_HOST"):
            logger.warning("MYSQL_HOST 环境变量已弃用，请使用 FACECV_MYSQL_HOST")
            
        mysql_port = os.getenv("FACECV_MYSQL_PORT") or os.getenv("MYSQL_PORT", "3306")
        if os.getenv("MYSQL_PORT"):
            logger.warning("MYSQL_PORT 环境变量已弃用，请使用 FACECV_MYSQL_PORT")
            
        mysql_user = os.getenv("FACECV_MYSQL_USER") if facecv_mysql_user_set else os.getenv("MYSQL_USER", "")
        if os.getenv("MYSQL_USER"):
            logger.warning("MYSQL_USER 环境变量已弃用，请使用 FACECV_MYSQL_USER")
            
        mysql_password = os.getenv("FACECV_MYSQL_PASSWORD") if facecv_mysql_password_set else os.getenv("MYSQL_PASSWORD", "")
        if os.getenv("MYSQL_PASSWORD"):
            logger.warning("MYSQL_PASSWORD 环境变量已弃用，请使用 FACECV_MYSQL_PASSWORD")
            
        mysql_database = os.getenv("FACECV_MYSQL_DATABASE") or os.getenv("MYSQL_DATABASE", "facecv")
        if os.getenv("MYSQL_DATABASE"):
            logger.warning("MYSQL_DATABASE 环境变量已弃用，请使用 FACECV_MYSQL_DATABASE")
        
        return cls(
            db_type=db_type,
            base_data_dir=os.getenv("FACECV_DATA_DIR") or os.getenv("DATA_DIR", "./data"),
            db_dir=os.getenv("FACECV_DB_DIR") or os.getenv("DB_DIR", "./data/db"),
            # MySQL配置 - 使用安全的本地默认值
            mysql_host=mysql_host,
            mysql_port=int(mysql_port),
            mysql_user=mysql_user,
            mysql_password=mysql_password,
            mysql_database=mysql_database,
            mysql_charset=os.getenv("FACECV_MYSQL_CHARSET") or os.getenv("MYSQL_CHARSET", "utf8mb4"),
            # SQLite配置
            sqlite_filename=os.getenv("FACECV_SQLITE_FILENAME") or os.getenv("SQLITE_FILENAME", "facecv.db"),
            # ChromaDB配置
            chromadb_dirname=os.getenv("FACECV_CHROMADB_DIRNAME") or os.getenv("CHROMADB_DIRNAME", "chromadb_data"),
            chromadb_collection_name=os.getenv("FACECV_CHROMADB_COLLECTION_NAME") or os.getenv("CHROMADB_COLLECTION_NAME", "face_embeddings"),
            # 连接池配置
            pool_size=int(os.getenv("FACECV_DB_POOL_SIZE") or os.getenv("DB_POOL_SIZE", "10")),
            max_overflow=int(os.getenv("FACECV_DB_MAX_OVERFLOW") or os.getenv("DB_MAX_OVERFLOW", "20")),
            pool_timeout=int(os.getenv("FACECV_DB_POOL_TIMEOUT") or os.getenv("DB_POOL_TIMEOUT", "30")),
            pool_recycle=int(os.getenv("FACECV_DB_POOL_RECYCLE") or os.getenv("DB_POOL_RECYCLE", "3600")),
            # 超时配置
            connect_timeout=int(os.getenv("FACECV_DB_CONNECT_TIMEOUT") or os.getenv("DB_CONNECT_TIMEOUT", "30")),
            read_timeout=int(os.getenv("FACECV_DB_READ_TIMEOUT") or os.getenv("DB_READ_TIMEOUT", "60")),
            write_timeout=int(os.getenv("FACECV_DB_WRITE_TIMEOUT") or os.getenv("DB_WRITE_TIMEOUT", "60"))
        )
    
    @property
    def mysql_url(self) -> str:
        """获取MySQL连接URL"""
        return (f"mysql+pymysql://{self.mysql_user}:{self.mysql_password}@"
                f"{self.mysql_host}:{self.mysql_port}/{self.mysql_database}"
                f"?charset={self.mysql_charset}&autocommit=true")
    
    @property
    def async_mysql_url(self) -> str:
        """获取异步MySQL连接URL"""
        return (f"mysql+aiomysql://{self.mysql_user}:{self.mysql_password}@"
                f"{self.mysql_host}:{self.mysql_port}/{self.mysql_database}"
                f"?charset={self.mysql_charset}&autocommit=true")
    
    @property
    def sqlite_url(self) -> str:
        """获取SQLite连接URL"""
        sqlite_path = self.get_sqlite_path()
        return f"sqlite:///{sqlite_path}"
    
    def get_connection_url(self) -> str:
        """根据数据库类型获取连接URL"""
        if self.db_type == "mysql":
            return self.mysql_url
        elif self.db_type == "sqlite":
            return self.sqlite_url
        elif self.db_type == "chromadb":
            return self.get_chromadb_path()
        else:
            raise ValueError(f"不支持的数据库类型: {self.db_type}")
    
    def get_connection_params(self) -> dict:
        """获取连接参数"""
        base_params: dict[str, object] = {
            "pool_size": self.pool_size,
            "max_overflow": self.max_overflow,
            "pool_timeout": self.pool_timeout,
            "pool_recycle": self.pool_recycle
        }
        
        if self.db_type == "mysql":
            result = dict(base_params)
            result["connect_args"] = {
                "connect_timeout": self.connect_timeout,
                "read_timeout": self.read_timeout,
                "write_timeout": self.write_timeout,
                "charset": self.mysql_charset
            }
            return result
        
        return base_params


# 全局配置实例
db_config = DatabaseConfig.from_env()

# 配置常量
SUPPORTED_DB_TYPES = ["sqlite", "mysql", "chromadb"]
DEFAULT_DB_TYPE = "sqlite"
DEFAULT_SQLITE_FILENAME = "facecv.db"
# ChromaDB配置应从环境变量加载，不应硬编码


def get_standardized_db_config(db_type: Optional[str] = None) -> DatabaseConfig:
    """获取标准化的数据库配置"""
    config = DatabaseConfig.from_env()
    if db_type:
        config.db_type = db_type
    return config
