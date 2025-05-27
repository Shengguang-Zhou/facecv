"""运行时配置管理

提供可在运行时动态修改的配置，用于那些需要在API运行期间更改的设置。
与基于Pydantic的Settings类不同，RuntimeConfig允许在运行时修改值。
"""

from typing import Dict, Any, Optional, List
import threading
import logging
import os

from facecv.config.settings import get_settings
from facecv.utils.cuda_utils import get_cuda_version, get_cudnn_version, get_execution_providers

logger = logging.getLogger(__name__)


class RuntimeConfig:
    """运行时可修改的配置类
    
    用于存储那些在应用运行期间可能需要动态修改的配置项，
    例如模型选择、阈值调整等。
    
    与Settings类不同，此类的实例可以在运行时修改。
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(RuntimeConfig, cls).__new__(cls)
                cls._instance._config = {}
                cls._instance._initialize_defaults()
            return cls._instance
    
    def _initialize_defaults(self):
        """初始化默认运行时配置"""
        # Detect CUDA and set execution providers
        cuda_version = get_cuda_version()
        execution_providers = get_execution_providers()
        
        # Log CUDA detection results
        if cuda_version:
            logger.info(f"Detected CUDA version: {cuda_version[0]}.{cuda_version[1]}")
            logger.info(f"Using execution providers: {execution_providers}")
        else:
            logger.info("No CUDA detected, using CPU execution")
        
        self._config = {
            "insightface_model_pack": "buffalo_l",
            "insightface_det_size": [640, 640],
            "insightface_det_thresh": 0.5,
            "insightface_similarity_thresh": 0.35,
            
            "arcface_enabled": False,
            "arcface_backbone": "resnet50",
            "arcface_dataset": "webface600k",
            
            "max_faces_per_image": 10,
            "enable_emotion": True,
            "enable_mask": True,
            "prefer_gpu": True,
            
            # CUDA and execution provider settings
            "cuda_available": cuda_version is not None,
            "cuda_version": cuda_version,
            "execution_providers": execution_providers,
            "onnx_providers": execution_providers  # For ONNX Runtime
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """获取配置值"""
        return self._config.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """设置配置值
        
        当配置值更改时，会清除设置缓存以确保获取最新配置
        """
        self._config[key] = value
        self._clear_config_caches()
    
    def update(self, config_dict: Dict[str, Any]) -> None:
        """批量更新配置
        
        当配置值更改时，会清除设置缓存以确保获取最新配置
        """
        self._config.update(config_dict)
        self._clear_config_caches()
    
    def reset(self) -> None:
        """重置为默认配置
        
        重置后会清除设置缓存以确保获取最新配置
        """
        self._initialize_defaults()
        self._clear_config_caches()
    
    def _clear_config_caches(self) -> None:
        """清除所有配置相关的缓存"""
        try:
            get_settings.cache_clear()
            logger.debug("配置缓存已清除")
        except Exception as e:
            logger.warning(f"清除配置缓存时出错: {e}")
    
    def get_all(self) -> Dict[str, Any]:
        """获取所有配置"""
        return self._config.copy()


_runtime_config = None

def get_runtime_config() -> RuntimeConfig:
    """获取运行时配置单例"""
    global _runtime_config
    if _runtime_config is None:
        _runtime_config = RuntimeConfig()
    return _runtime_config
