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

```bash
# 启动API服务
facecv serve --host 0.0.0.0 --port 8000

# 或使用Python
python -m facecv.api
```

### API 端点

- `POST /api/v1/faces/register` - 注册人脸
- `POST /api/v1/faces/recognize` - 识别人脸
- `POST /api/v1/faces/verify` - 验证人脸
- `GET /api/v1/faces/list` - 列出已注册人脸
- `DELETE /api/v1/faces/{face_id}` - 删除人脸

## 配置

创建 `config.yaml`:

```yaml
# 模型配置
model:
  backend: "insightface"  # 或 "deepface"
  device: "cuda"  # 或 "cpu"
  
# 数据库配置
database:
  type: "mongodb"  # 或 "mysql", "sqlite", "chromadb"
  connection_string: "mongodb://localhost:27017/facecv"
  
# API配置
api:
  cors_origins: ["*"]
  max_upload_size: 10485760  # 10MB
  
# 性能配置
performance:
  batch_size: 32
  num_workers: 4
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

## 性能指标

- 人脸检测: < 20ms/张
- 特征提取: < 30ms/张
- 1:N识别(1万人): < 50ms
- API响应时间: < 100ms

## 贡献

欢迎提交Issue和Pull Request！

## 许可证

MIT License