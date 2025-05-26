# FaceCV - 专业人脸识别框架

FaceCV 是一个高性能、易扩展的人脸识别框架，专为智能监控场景设计。支持人脸识别考勤、陌生人检测、多人脸同时识别等功能。

## 特性

- 🚀 **高性能**: GPU加速，支持批量处理
- 🎯 **多模型支持**: 集成InsightFace和DeepFace
- 📊 **多数据库**: 支持MongoDB、MySQL、SQLite和向量数据库
- 🔄 **实时处理**: 支持视频流实时人脸识别
- 🛡️ **隐私保护**: 内置数据加密和访问控制
- 📈 **可扩展**: 模块化设计，易于扩展

## 安装

```bash
# 基础安装
pip install facecv

# GPU支持
pip install facecv[gpu]

# 开发环境
pip install facecv[dev]
```

## 快速开始

### 1. 人脸注册

```python
from facecv import FaceRecognizer

# 初始化识别器
recognizer = FaceRecognizer(model="insightface")

# 注册人脸
face_id = recognizer.register(
    image_path="path/to/face.jpg",
    name="张三",
    metadata={"department": "研发部", "employee_id": "E001"}
)
```

### 2. 人脸识别

```python
# 识别单张图片
result = recognizer.recognize(image_path="path/to/test.jpg")
print(f"识别结果: {result.name}, 相似度: {result.similarity}")

# 批量识别
results = recognizer.recognize_batch(image_paths=["img1.jpg", "img2.jpg"])
```

### 3. 视频流处理

```python
# 处理视频流
from facecv import VideoProcessor

processor = VideoProcessor(recognizer)
processor.process_stream(
    source="rtsp://camera_ip/stream",
    on_face_detected=lambda face: print(f"检测到: {face.name}")
)
```

## API 服务

### 启动服务

```bash
# 默认启动（端口7000）
python main.py

# 自定义端口
python main.py --port 7003

# 使用环境变量
FACECV_PORT=7003 python main.py
```

### API 端点

#### 健康检查
- `GET /health` - 基础健康检查
- `GET /api/v1/health/comprehensive` - 综合健康状态
- `GET /api/v1/health/database` - 数据库连接状态

#### InsightFace API
- `GET /api/v1/insightface/faces` - 列出所有人脸
- `GET /api/v1/insightface/faces/count` - 获取人脸总数
- `POST /api/v1/insightface/add_face` - 添加人脸
- `DELETE /api/v1/insightface/faces/{face_id}` - 删除人脸
- `POST /api/v1/insightface/recognize` - 识别人脸
- `POST /api/v1/insightface/verify` - 验证人脸
- `GET /api/v1/insightface/models/available` - 可用模型列表
- `POST /api/v1/insightface/models/select` - 切换模型

#### DeepFace API
- `POST /api/v1/deepface/faces` - 注册人脸
- `GET /api/v1/deepface/faces` - 列出人脸
- `POST /api/v1/deepface/recognize` - 识别人脸
- `POST /api/v1/deepface/verify` - 验证人脸

### API 文档
访问 `http://localhost:7003/docs` 查看交互式API文档

## 配置系统

FaceCV 使用三层配置架构，支持灵活的配置管理：

### 1. 环境变量配置（推荐）

所有配置项使用 `FACECV_` 前缀，通过 `.env` 文件或环境变量设置：

```bash
# 创建 .env 文件
cp .env.example .env

# 主要配置项
FACECV_DB_TYPE=mysql              # 数据库类型: sqlite, mysql, chromadb
FACECV_MYSQL_HOST=localhost       # MySQL主机
FACECV_MYSQL_USER=root           # MySQL用户
FACECV_MYSQL_PASSWORD=password   # MySQL密码
FACECV_MYSQL_DATABASE=facecv     # 数据库名

# 模型配置
FACECV_INSIGHTFACE_MODEL_PACK=buffalo_l  # 模型选择: buffalo_l/m/s, antelopev2
FACECV_INSIGHTFACE_PREFER_GPU=true      # GPU加速
FACECV_MODEL_OFFLOAD_TIMEOUT=300        # 模型自动卸载时间（秒）
```

### 2. 数据库配置

支持三种数据库，通过 `FACECV_DB_TYPE` 切换：

#### SQLite（默认，开发环境）
```bash
FACECV_DB_TYPE=sqlite
FACECV_SQLITE_FILENAME=./data/facecv.db
```

#### MySQL（生产环境）
```bash
FACECV_DB_TYPE=mysql
FACECV_MYSQL_HOST=your-mysql-host
FACECV_MYSQL_PORT=3306
FACECV_MYSQL_USER=your-user
FACECV_MYSQL_PASSWORD=your-password
FACECV_MYSQL_DATABASE=facecv
```

#### ChromaDB（向量数据库）
```bash
FACECV_DB_TYPE=chromadb
FACECV_CHROMADB_DIRNAME=./data/chromadb
FACECV_CHROMADB_COLLECTION_NAME=face_embeddings
```

### 3. 模型管理

#### 可用模型
- **buffalo_l**: 大模型，最高精度，适合生产环境（默认）
- **buffalo_m**: 中等模型，平衡精度和速度
- **buffalo_s**: 小模型，速度最快，适合边缘设备
- **antelopev2**: 研究级高精度模型

#### 运行时切换模型
```python
from facecv.config import get_runtime_config

# 动态切换模型
runtime_config = get_runtime_config()
runtime_config.set("insightface_model_pack", "buffalo_s")
```

### 4. 配置系统架构

```python
from facecv.config import get_settings, get_db_config, get_runtime_config

# 静态配置（从环境变量加载）
settings = get_settings()
print(f"服务器端口: {settings.port}")

# 数据库配置
db_config = get_db_config()
print(f"数据库类型: {db_config.db_type}")

# 运行时配置（可动态修改）
runtime_config = get_runtime_config()
runtime_config.set("insightface_model_pack", "buffalo_m")
```

## 架构

```
FaceCV/
├── facecv/
│   ├── core/          # 核心功能
│   ├── models/        # 模型实现
│   ├── api/           # REST API
│   ├── database/      # 数据库接口
│   └── utils/         # 工具函数
└── tests/             # 测试用例
```

## 部署指南

### 生产环境部署

1. **资源充足环境**
```bash
# 使用GPU加速，大模型
FACECV_DB_TYPE=mysql
FACECV_INSIGHTFACE_MODEL_PACK=buffalo_l
FACECV_INSIGHTFACE_PREFER_GPU=true
FACECV_MODEL_OFFLOAD_TIMEOUT=0  # 禁用自动卸载
```

2. **资源受限环境**
```bash
# CPU优化，小模型，自动卸载
FACECV_DB_TYPE=sqlite
FACECV_INSIGHTFACE_MODEL_PACK=buffalo_s
FACECV_INSIGHTFACE_PREFER_GPU=false
FACECV_MODEL_OFFLOAD_TIMEOUT=60  # 1分钟自动卸载
FACECV_API_MODE=insightface  # 只启用一个API
```

### Docker 部署
```bash
# 构建镜像
docker build -t facecv:latest .

# 运行容器
docker run -d \
  -p 7003:7003 \
  -v $(pwd)/data:/app/data \
  -e FACECV_DB_TYPE=mysql \
  -e FACECV_MYSQL_HOST=your-host \
  -e FACECV_MYSQL_PASSWORD=your-password \
  facecv:latest
```

### 迁移指南

从旧版本迁移到新配置系统：

1. **更新环境变量**：所有变量添加 `FACECV_` 前缀
   ```bash
   # 旧版本
   MYSQL_HOST=localhost
   # 新版本
   FACECV_MYSQL_HOST=localhost
   ```

2. **更新代码导入**
   ```python
   # 旧版本
   from facecv.config.database import db_config
   # 新版本
   from facecv.config import get_db_config
   db_config = get_db_config()
   ```

## 性能指标

- 人脸检测: < 20ms/张 (GPU) / < 50ms/张 (CPU)
- 特征提取: < 30ms/张 (GPU) / < 80ms/张 (CPU)
- 1:N识别(1万人): < 50ms
- API响应时间: < 100ms
- 内存占用: 500MB-2GB (根据模型大小)

## 故障排除

### 常见问题

1. **数据库连接失败**
   ```bash
   # 检查环境变量
   echo $FACECV_MYSQL_HOST
   # 验证MySQL连接
   mysql -h $FACECV_MYSQL_HOST -u $FACECV_MYSQL_USER -p
   ```

2. **模型加载失败**
   ```bash
   # 清除模型缓存
   rm -rf ~/.insightface/models/
   # 重新下载模型
   python -c "from insightface.app import FaceAnalysis; app = FaceAnalysis()"
   ```

3. **内存不足**
   ```bash
   # 使用小模型
   FACECV_INSIGHTFACE_MODEL_PACK=buffalo_s
   # 启用自动卸载
   FACECV_MODEL_OFFLOAD_TIMEOUT=60
   ```

4. **GPU不可用**
   ```bash
   # 检查CUDA
   python -c "import torch; print(torch.cuda.is_available())"
   # 强制使用CPU
   FACECV_INSIGHTFACE_PREFER_GPU=false
   ```

### 日志级别
```bash
# 调试模式
FACECV_LOG_LEVEL=DEBUG python main.py
```

## 贡献

欢迎提交Issue和Pull Request！

### 开发环境设置
```bash
# 克隆仓库
git clone https://github.com/yourusername/facecv.git
cd facecv

# 安装开发依赖
pip install -r requirements-dev.txt

# 运行测试
pytest tests/
```

## 许可证

MIT License