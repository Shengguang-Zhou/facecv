# FaceCV 迁移指南

## 项目概述

FaceCV 是从 EurekCV 单体框架中拆分出来的人脸识别专用框架，作为五个独立框架迁移计划的第一部分。

### 迁移背景
- **源项目**: EurekCV (`/home/a/PycharmProjects/EurekCV`)
- **目标项目**: FaceCV (`/home/a/PycharmProjects/facecv`)
- **迁移计划**: `/home/a/PycharmProjects/EurekCV/plan/PLAN.md`
- **待办清单**: `/home/a/PycharmProjects/EurekCV/plan/TODO.md`

## 已完成的迁移工作

### 1. 项目结构 ✅
```
FaceCV/
├── setup.py                    # 项目配置
├── requirements.txt            # 依赖管理
├── README.md                   # 项目说明
├── main.py                     # API 主程序
├── facecv/                     # 核心包
│   ├── __init__.py
│   ├── core/                   # 核心功能
│   │   ├── interface.py        # 统一接口
│   │   ├── video_stream.py     # 视频流处理
│   │   ├── processor.py        # 视频处理器
│   │   ├── attendance.py       # 考勤系统
│   │   └── stranger.py         # 陌生人检测
│   ├── models/                 # 模型实现
│   │   └── insightface/        # InsightFace 模型
│   │       ├── recognizer.py   # 识别器实现
│   │       └── recognizer_mock.py # 模拟实现
│   ├── database/               # 数据库层
│   │   ├── abstract_facedb.py  # 抽象基类
│   │   └── sqlite_facedb.py    # SQLite 实现
│   ├── api/                    # API 层
│   │   └── routes/             # 路由定义
│   │       ├── face.py         # 人脸相关 API
│   │       ├── stream.py       # 视频流 API
│   │       └── health.py       # 健康检查
│   ├── schemas/                # 数据模型
│   │   └── face.py             # Pydantic 模型
│   ├── config/                 # 配置管理
│   │   └── settings.py         # 应用配置
│   └── utils/                  # 工具函数
├── tests/                      # 测试用例
└── docs/                       # 文档
    └── MIGRATION_GUIDE.md      # 本文档
```

### 2. 核心代码迁移 ✅

#### InsightFace 模块迁移
| 源文件 (EurekCV) | 目标文件 (FaceCV) | 状态 |
|-----------------|------------------|------|
| `app/cv/faceRecognition/insightface_recognition/recognizer.py` | `facecv/models/insightface/recognizer.py` | ✅ 已迁移并重构 |
| `app/cv/faceRecognition/insightface_recognition/db/` | `facecv/database/` | ✅ 重新设计数据库抽象层 |
| `app/api/face_recognition/insightface_.py` | `facecv/api/routes/face.py` | ✅ 重写为新 API |
| `app/schema/face_recognition/insightface_schema.py` | `facecv/schemas/face.py` | ✅ 更新数据模型 |

#### DeepFace 模块（待迁移）
| 源文件 (EurekCV) | 目标文件 (FaceCV) | 状态 |
|-----------------|------------------|------|
| `app/cv/faceRecognition/deepface_recognition/base.py` | `facecv/models/deepface/base.py` | ❌ 待迁移 |
| `app/cv/faceRecognition/deepface_recognition/embedding.py` | `facecv/models/deepface/embedding.py` | ❌ 待迁移 |
| `app/cv/faceRecognition/deepface_recognition/recognition.py` | `facecv/models/deepface/recognition.py` | ❌ 待迁移 |
| `app/api/face_recognition/deepface_.py` | 集成到统一接口 | ❌ 待实现 |

### 3. API 实现状态 ✅

#### 人脸管理 API
- ✅ `POST /api/v1/faces/register` - 人脸注册
- ✅ `POST /api/v1/faces/recognize` - 人脸识别
- ✅ `POST /api/v1/faces/verify` - 人脸验证
- ✅ `GET /api/v1/faces` - 列出人脸
- ✅ `GET /api/v1/faces/count` - 人脸计数
- ✅ `DELETE /api/v1/faces/{face_id}` - 删除人脸
- ✅ `DELETE /api/v1/faces/by-name/{name}` - 按姓名删除

#### 视频流处理 API
- ✅ `POST /api/v1/stream/process` - 处理视频流
- ✅ `WS /api/v1/stream/ws` - WebSocket 实时流
- ✅ `GET /api/v1/stream/sources` - 列出视频源

### 4. 新增功能 ✅
- ✅ 统一的人脸识别接口 (`FaceRecognizer`)
- ✅ 视频流处理器 (`VideoStreamProcessor`)
- ✅ 异步 API 支持
- ✅ WebSocket 实时通信
- ✅ Swagger 文档自动生成

## 待完成工作

根据 `/home/a/PycharmProjects/EurekCV/plan/TODO.md` 第一阶段剩余任务：

### Day 2-3 剩余任务
- [ ] 迁移 DeepFace 核心代码
- [ ] 实现陌生人检测功能（框架已创建）
- [ ] 添加人脸属性分析（年龄、性别等）
- [ ] 实现考勤系统相关功能（框架已创建）
- [ ] 编写单元测试
- [ ] 性能优化（GPU 加速、批处理）
- [ ] 编写完整使用文档

## 后续迁移指引

### 1. 在原代码库中需要关注的部分

#### 人脸识别相关
- `app/cv/faceRecognition/` - 所有人脸识别实现
- `app/api/face_recognition/` - API 路由定义
- `app/schema/face_recognition/` - 数据模型定义
- `app/cv/face/landmark.py` - 人脸关键点检测（待迁移）

#### 通用工具
- `app/utils/cv/` - 图像/视频处理工具
- `app/streaming/` - 流处理相关代码
- `app/manager/live_stream_manager.py` - 实时流管理器

#### 配置和依赖
- `config/` - 配置文件
- `app/dependency/` - 依赖注入
- `router/router.py` - 主路由配置（注意第26-27行的人脸识别路由）

### 2. 重要文件追踪

需要持续关注的原始文件：
1. **模型权重**: `app/cv/faceRecognition/insightface_recognition/model-r100-ii/`
2. **数据库实现**: 
   - `app/cv/faceRecognition/insightface_recognition/db/mongo_facedb.py`
   - `app/cv/faceRecognition/insightface_recognition/db/mysql_facedb.py`
3. **API 文档**: `app/api/docs/insightface_api_doc.md`

### 3. 集成建议

当 FaceCV 开发完成后，需要：
1. 在 EurekCV 中移除相关代码
2. 添加 FaceCV 作为依赖
3. 更新路由配置以使用 FaceCV API
4. 迁移数据库和模型文件

### 4. 其他框架迁移参考

根据迁移计划，后续框架包括：
- **OpenSetCV** - 开放集检测框架（第4-5天）✅ 已完成
- **YoloTaskCV** - YOLO任务框架（第6-8天）
- **ActionCV** - 动作识别框架（第9-11天）
- **MLLMCV** - 多模态LLM框架（第12-14天）

## 开发规范

1. **代码风格**：遵循 PEP 8，使用中文注释
2. **测试驱动**：功能开发 → 测试 → 调试循环
3. **模块化设计**：保持文件不超过 700 行
4. **异步优先**：API 层使用异步，计算密集型任务保持同步
5. **文档完整**：每个 API 都需要完整的文档字符串

## 联系和支持

- 原项目路径: `/home/a/PycharmProjects/EurekCV`
- 迁移计划: `/home/a/PycharmProjects/EurekCV/plan/PLAN.md`
- 待办事项: `/home/a/PycharmProjects/EurekCV/plan/TODO.md`
- API 测试指南: `/home/a/PycharmProjects/facecv/API_TEST_GUIDE.md`

---

最后更新: 2025-05-26