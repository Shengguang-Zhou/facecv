"""配置管理模块 - 统一配置接口

提供统一的配置管理接口，包括：
1. 应用设置 - 从.env加载，使用FACECV_前缀
2. 模型配置 - 从YAML文件加载
3. 数据库配置 - 标准化路径和连接参数
4. 运行时配置 - 支持在应用运行期间动态修改

使用示例:
```python
from facecv.config import get_settings, load_model_config, get_db_config, get_runtime_config

settings = get_settings()
print(settings.host, settings.port)

model_config = load_model_config()
print(model_config["insightface"]["model_pack"])

db_config = get_db_config()
print(db_config.get_connection_url())

runtime_config = get_runtime_config()
runtime_config.set("insightface_model_pack", "buffalo_m")
print(runtime_config.get("insightface_model_pack"))
```
"""

from .database import DatabaseConfig, get_standardized_db_config
from .runtime_config import RuntimeConfig, get_runtime_config
from .settings import Settings, get_model_config, get_settings, load_model_config

get_db_config = get_standardized_db_config

__all__ = [
    "Settings",
    "get_settings",
    "load_model_config",
    "get_model_config",
    "DatabaseConfig",
    "get_db_config",
    "get_standardized_db_config",
    "RuntimeConfig",
    "get_runtime_config",
]
