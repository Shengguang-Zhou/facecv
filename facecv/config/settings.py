"""应用配置"""

from typing import List, Optional, Dict, Any, Literal, Union
from functools import lru_cache
from pydantic_settings import BaseSettings
from pydantic import validator, Field
import os
import yaml
from pathlib import Path

MODEL_CONFIG_PATH = Path(__file__).parent / "model_config.yaml"

class Settings(BaseSettings):
    """应用配置类 - 包含验证和路径标准化"""
    
    # API配置
    host: str = Field(default="0.0.0.0", description="API服务器主机地址")
    port: int = Field(default=7000, ge=1, le=65535, description="API服务器端口")
    debug: bool = Field(default=False, description="调试模式")
    cors_origins: List[str] = Field(default=["*"], description="CORS允许的源")
    
    # 模型配置
    model_backend: Literal["insightface", "deepface"] = Field(default="insightface", description="模型后端")
    model_device: Literal["cpu", "cuda", "auto"] = Field(default="auto", description="模型设备")
    model_path: Optional[str] = Field(default=None, description="自定义模型路径")
    
    # InsightFace 模型配置
    insightface_model_pack: Literal["buffalo_l", "buffalo_m", "buffalo_s", "antelopev2"] = Field(
        default="buffalo_l", 
        description="InsightFace模型包选择 - buffalo_l(最佳精度), buffalo_m(平衡), buffalo_s(速度优先), antelopev2(高精度)"
    )
    insightface_det_size: List[int] = Field(default=[640, 640], description="人脸检测输入尺寸 [width, height]")
    insightface_det_thresh: float = Field(default=0.5, ge=0.1, le=1.0, description="人脸检测置信度阈值")
    insightface_similarity_thresh: float = Field(default=0.35, ge=0.1, le=1.0, description="人脸识别相似度阈值")
    insightface_enable_emotion: bool = Field(default=True, description="启用情感识别")
    insightface_enable_mask: bool = Field(default=True, description="启用口罩检测")
    insightface_prefer_gpu: bool = Field(default=True, description="优先使用GPU加速")
    
    # ArcFace 专用模型配置 (独立于buffalo模型的原生ArcFace)
    arcface_enabled: bool = Field(default=False, description="启用专用ArcFace模型 (替代buffalo)")
    arcface_backbone: Literal["resnet18", "resnet34", "resnet50", "resnet100", "mobilefacenet"] = Field(
        default="resnet50", 
        description="ArcFace骨干网络 - resnet50(平衡), resnet100(最佳精度), resnet18(快速), mobilefacenet(移动端)"
    )
    arcface_dataset: Literal["ms1mv2", "ms1mv3", "glint360k", "webface600k"] = Field(
        default="webface600k", 
        description="ArcFace训练数据集 - webface600k(推荐), ms1mv3(通用), glint360k(大规模), ms1mv2(经典)"
    )
    arcface_embedding_size: int = Field(default=512, ge=128, le=2048, description="ArcFace特征向量维度")
    arcface_margin: float = Field(default=0.5, ge=0.1, le=1.0, description="ArcFace角度边界参数")
    arcface_scale: float = Field(default=64.0, ge=16.0, le=128.0, description="ArcFace缩放参数")
    arcface_auto_download: bool = Field(default=True, description="自动下载ArcFace模型权重")
    arcface_weights_dir: str = Field(default="./weights/arcface", description="ArcFace权重存储目录")
    
    # DeepFace 模型配置  
    deepface_model_name: Literal["VGG-Face", "Facenet", "ArcFace", "Dlib", "SFace", "OpenFace"] = Field(
        default="ArcFace", 
        description="DeepFace识别模型 - ArcFace(推荐), VGG-Face(经典), Facenet(Google), Dlib(轻量), SFace(快速), OpenFace(开源)"
    )
    deepface_detector: Literal["opencv", "ssd", "dlib", "mtcnn", "retinaface", "mediapipe"] = Field(
        default="retinaface", 
        description="DeepFace检测器 - retinaface(最佳), mtcnn(平衡), opencv(快速), ssd(轻量), dlib(经典), mediapipe(实时)"
    )
    deepface_distance_metric: Literal["cosine", "euclidean", "euclidean_l2"] = Field(
        default="cosine",
        description="距离度量方式 - cosine(推荐), euclidean(欧氏), euclidean_l2(标准化欧氏)"
    )
    
    # 数据库配置
    db_type: Literal["sqlite", "mysql", "chromadb", "hybrid"] = Field(default="mysql", description="数据库类型")
    db_connection_string: Optional[str] = Field(default=None, description="数据库连接串")
    
    mysql_host: str = Field(default="localhost", description="MySQL主机地址")
    mysql_port: int = Field(default=3306, description="MySQL端口")
    mysql_user: str = Field(default="root", description="MySQL用户名")
    mysql_password: str = Field(default="", description="MySQL密码")
    mysql_database: str = Field(default="facecv", description="MySQL数据库名")
    
    # 标准化路径配置
    data_dir: str = Field(default="./data", description="数据目录")
    db_dir: str = Field(default="./data/db", description="数据库目录")
    log_dir: str = Field(default="./data/logs", description="日志目录")
    model_cache_dir: str = Field(default="./models", description="模型缓存目录")
    upload_dir: str = Field(default="./data/uploads", description="上传文件目录")
    
    # 性能配置
    batch_size: int = Field(default=32, ge=1, le=512, description="批处理大小")
    num_workers: int = Field(default=4, ge=1, le=32, description="工作进程数")
    max_faces_per_image: int = Field(default=10, ge=1, le=100, description="每张图片最大人脸数")
    
    # 安全配置
    secret_key: str = Field(default="facecv-default-secret-key-change-in-production", min_length=32, description="JWT密钥")
    algorithm: str = Field(default="HS256", description="JWT算法")
    access_token_expire_minutes: int = Field(default=30, ge=1, le=43200, description="访问令牌过期时间(分钟)")
    
    # 文件上传配置
    max_upload_size: int = Field(default=10 * 1024 * 1024, ge=1024, le=100 * 1024 * 1024, description="最大上传文件大小(字节)")
    allowed_extensions: List[str] = Field(default=[".jpg", ".jpeg", ".png", ".bmp", ".webp"], description="允许的文件扩展名")
    
    # 日志配置
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(default="INFO", description="日志级别")
    log_file: Optional[str] = Field(default="facecv.log", description="日志文件名")
    
    @validator('port')
    def validate_port(cls, v):
        """验证端口号"""
        if not (1 <= v <= 65535):
            raise ValueError('端口号必须在1-65535之间')
        return v
    
    @validator('cors_origins')
    def validate_cors_origins(cls, v):
        """验证CORS源"""
        if not v:
            raise ValueError('CORS源不能为空')
        return v
    
    @validator('secret_key')
    def validate_secret_key(cls, v):
        """验证密钥长度"""
        if len(v) < 32:
            raise ValueError('密钥长度至少32个字符')
        if v == "your-secret-key-here":
            raise ValueError('请更改默认密钥')
        return v
    
    @validator('db_connection_string')
    def validate_db_connection(cls, v, values):
        """验证数据库连接串格式"""
        if v is not None:
            db_type = values.get('db_type', 'sqlite')
            if db_type == 'sqlite' and not v.startswith('sqlite:///'):
                raise ValueError('SQLite连接串必须以sqlite:///开头')
            elif db_type == 'mysql' and not v.startswith(('mysql://', 'mysql+pymysql://')):
                raise ValueError('MySQL连接串必须以mysql://或mysql+pymysql://开头')
        return v
    
    @validator('allowed_extensions')
    def validate_extensions(cls, v):
        """验证文件扩展名格式"""
        for ext in v:
            if not ext.startswith('.'):
                raise ValueError(f'扩展名必须以.开头: {ext}')
            if not ext.lower() == ext:
                raise ValueError(f'扩展名必须小写: {ext}')
        return v
    
    @validator('data_dir', 'db_dir', 'log_dir', 'model_cache_dir', 'upload_dir', 'arcface_weights_dir')
    def validate_directories(cls, v):
        """验证目录路径"""
        if not v:
            raise ValueError('目录路径不能为空')
        # 确保路径使用正斜杠
        v = v.replace('\\', '/')
        return v
    
    def get_absolute_path(self, relative_path: str) -> str:
        """获取相对于数据目录的绝对路径"""
        if os.path.isabs(relative_path):
            return relative_path
        return os.path.join(self.data_dir, relative_path)
    
    def get_db_path(self) -> str:
        """获取数据库文件路径或连接串"""
        if self.db_connection_string is not None:
            if self.db_type == 'sqlite' and self.db_connection_string.startswith('sqlite:///'):
                db_file = self.db_connection_string[10:]  # 移除 'sqlite:///'
                if not os.path.isabs(db_file):
                    return os.path.join(self.db_dir, os.path.basename(db_file))
                return db_file
            return self.db_connection_string
        
        if self.db_type == 'sqlite':
            db_file = os.path.join(self.db_dir, 'facecv.db')
            return f"sqlite:///{db_file}"
        elif self.db_type == 'mysql':
            return f"mysql+pymysql://{self.mysql_user}:{self.mysql_password}@{self.mysql_host}:{self.mysql_port}/{self.mysql_database}"
        
        return f"sqlite:///{os.path.join(self.db_dir, 'facecv.db')}"
    
    def get_log_path(self) -> Optional[str]:
        """获取日志文件路径"""
        if not self.log_file:
            return None
        if os.path.isabs(self.log_file):
            return self.log_file
        return os.path.join(self.log_dir, self.log_file)
    
    def ensure_directories(self):
        """确保所有必要目录存在"""
        directories = [
            self.data_dir,
            self.db_dir,
            self.log_dir,
            self.model_cache_dir,
            self.upload_dir,
            self.arcface_weights_dir
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def get_model_cache_path(self, model_name: str) -> str:
        """获取模型缓存路径"""
        return os.path.join(self.model_cache_dir, model_name)
    
    def get_arcface_weights_path(self, backbone: Optional[str] = None, dataset: Optional[str] = None) -> str:
        """获取ArcFace权重路径"""
        backbone = backbone or self.arcface_backbone
        dataset = dataset or self.arcface_dataset
        return os.path.join(self.arcface_weights_dir, backbone, dataset)
    
    def get_arcface_model_path(self, backbone: Optional[str] = None, dataset: Optional[str] = None, filename: Optional[str] = None) -> str:
        """获取ArcFace模型文件完整路径"""
        base_path = self.get_arcface_weights_path(backbone, dataset)
        if filename:
            return os.path.join(base_path, filename)
        # 默认文件名格式: {backbone}_{dataset}.onnx
        backbone = backbone or self.arcface_backbone
        dataset = dataset or self.arcface_dataset
        default_filename = f"{backbone}_{dataset}.onnx"
        return os.path.join(base_path, default_filename)
    
    def is_production(self) -> bool:
        """判断是否为生产环境"""
        return not self.debug and os.getenv('FACECV_ENVIRONMENT', 'production') == 'production'
    
    class Config:
        env_file = ".env"
        env_prefix = "FACECV_"
        extra = "allow"  # Allow extra fields from .env
        validate_assignment = True  # 验证赋值操作
        use_enum_values = True  # 使用枚举值


@lru_cache()
def load_model_config(config_path: Optional[Union[str, Path]] = None, environment: str = "production") -> Dict[str, Any]:
    """Load model configuration from YAML file"""
    import logging
    logger = logging.getLogger(__name__)
    
    if config_path is None:
        config_path = MODEL_CONFIG_PATH
    else:
        if isinstance(config_path, str):
            config_path = Path(config_path)
    
    try:
        with config_path.open('r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # Apply environment-specific overrides
        if "environments" in config and environment in config["environments"]:
            env_config = config["environments"][environment]
            config = _merge_configs(config, env_config)
        
        return config
    except FileNotFoundError as e:
        logger.error(f"配置文件未找到: {config_path} - {e}")
        logger.info("使用默认模型配置")
        return _get_default_model_config()
    except yaml.YAMLError as e:
        logger.error(f"YAML解析错误: {e}")
        logger.warning(f"配置文件 {config_path} 格式无效，使用默认配置")
        return _get_default_model_config()
    except Exception as e:
        logger.error(f"加载模型配置时出错: {e}")
        logger.warning("使用默认模型配置")
        return _get_default_model_config()


def _merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge two configuration dictionaries"""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _merge_configs(result[key], value)
        else:
            result[key] = value
    return result


def _get_default_model_config() -> Dict[str, Any]:
    """Return default model configuration"""
    return {
        "insightface": {
            "model_pack": "buffalo_l",
            "detection": {"det_size": [640, 640], "det_thresh": 0.5},
            "recognition": {"similarity_threshold": 0.4},
            "performance": {"ctx_id": 0, "enable_gpu": True}
        },
        "deepface": {
            "model_name": "ArcFace",
            "backend": "opencv",
            "recognition": {"distance_metric": "cosine", "threshold": 0.4}
        }
    }


@lru_cache()
def get_settings() -> Settings:
    """获取配置单例"""
    return Settings()


@lru_cache()
def get_model_config(environment: Optional[str] = None) -> Dict[str, Any]:
    """Get model configuration singleton"""
    if environment is None:
        environment = os.getenv("FACECV_ENVIRONMENT", "production")
    return load_model_config(environment=environment)
