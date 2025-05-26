# FaceCV API 使用指南

## 目录
1. [概述](#概述)
2. [部署配置](#部署配置)
3. [认证与安全](#认证与安全)
4. [核心面部识别 API](#核心面部识别-api)
5. [高级模型管理 API](#高级模型管理-api)
6. [系统健康监控 API](#系统健康监控-api)
7. [视频流处理 API](#视频流处理-api)
8. [Webhook 集成 API](#webhook-集成-api)
9. [Java Spring 集成示例](#java-spring-集成示例)
10. [常见错误与解决方案](#常见错误与解决方案)
11. [性能优化建议](#性能优化建议)

## 概述

FaceCV 是一个基于 FastAPI 的高性能面部识别服务，支持 InsightFace 和 DeepFace 双引擎，提供完整的面部识别、验证、分析和管理功能。

### 核心特性
- 🚀 高性能面部识别（支持 GPU 加速）
- 🔄 双引擎支持（InsightFace + DeepFace）
- 🎯 **NEW!** 智能模型选择 (buffalo_l/m/s/antelopev2)
- 📊 实时系统监控
- 🎯 智能模型管理
- 🔗 Webhook 事件通知
- 📹 视频流处理
- 🌐 多语言支持

### 🆕 最新更新 (2025年5月)
- ✅ **真实模型支持**: 移除所有Mock数据，使用真实InsightFace模型
- ✅ **🚀 ArcFace专用模型**: 支持独立ArcFace权重，优化识别精度
- ✅ **动态模型切换**: 支持运行时在ArcFace和Buffalo间切换，无需重启
- ✅ **智能模型发现**: 自动发现weights/arcface/、weights/insightface/和~/.insightface/中的模型
- ✅ **优化相似度阈值**: 降低阈值至0.35，提升识别准确性
- ✅ **GPU加速优化**: 自动检测并使用NVIDIA GPU (RTX 4070等)
- ✅ **生产就绪配置**: buffalo_l模型包，512维嵌入向量
- ✅ **增强API文档**: 中英文双语，详细示例和错误处理

### 基础信息
- **默认端口**: 7000/7003
- **API 版本**: v1
- **基础 URL**: `http://localhost:7000/api/v1` (推荐) 或 `http://localhost:7003/api/v1`
- **文档地址**: `http://localhost:7000/docs` 或 `http://localhost:7003/docs`

## 部署配置

### Docker 部署（推荐）
```bash
# 构建镜像
docker build -t facecv:latest .

# 运行容器
docker run -d \
  --name facecv \
  -p 7003:7003 \
  --env-file .env \
  facecv:latest
```

### Docker Compose 部署
```yaml
version: '3.8'
services:
  facecv:
    build: .
    ports:
      - "7003:7003"
    environment:
      - DATABASE_URL=mysql://root:password@mysql:3306/facecv
      - REDIS_URL=redis://redis:6379
    depends_on:
      - mysql
      - redis
```

### 环境变量配置
```env
# 数据库配置
DATABASE_URL=mysql://root:Zsg20010115_@eurekailab.mysql.rds.aliyuncs.com:3306/facecv
DB_TYPE=mysql

# 模型配置
INSIGHTFACE_MODEL=buffalo_l
DEEPFACE_MODEL=VGG-Face
USE_GPU=true

# ArcFace 专用配置 (NEW! 🚀)
FACECV_ARCFACE_ENABLED=false           # 启用ArcFace专用模型
FACECV_ARCFACE_BACKBONE=resnet50       # resnet50/mobilefacenet
FACECV_ARCFACE_DATASET=webface600k     # 训练数据集
FACECV_ARCFACE_EMBEDDING_SIZE=512      # 嵌入向量维度
FACECV_ARCFACE_WEIGHTS_DIR=./weights/arcface  # ArcFace权重目录

# 高级配置
FACECV_INSIGHTFACE_DET_SIZE=[640,640]  # 检测分辨率
FACECV_INSIGHTFACE_DET_THRESH=0.5      # 检测阈值
FACECV_INSIGHTFACE_SIMILARITY_THRESH=0.35  # 相似度阈值

# 系统配置
LOG_LEVEL=INFO
MAX_WORKERS=4
CACHE_SIZE=1000
```

### ArcFace 模型部署配置
```bash
# 1. 创建模型目录结构
mkdir -p weights/arcface/resnet50/ms1mv3
mkdir -p weights/arcface/mobilefacenet/ms1mv3

# 2. 下载ArcFace模型权重 (示例)
# ResNet50 模型 (生产推荐)
wget https://example.com/arcface_resnet50_ms1mv3.onnx \
     -O weights/arcface/resnet50/ms1mv3/arcface_resnet50_ms1mv3.onnx

# MobileFaceNet 模型 (移动端)  
wget https://example.com/arcface_mobilefacenet_ms1mv3.onnx \
     -O weights/arcface/mobilefacenet/ms1mv3/arcface_mobilefacenet_ms1mv3.onnx

# 3. 验证模型文件
ls -la weights/arcface/*/ms1mv3/*.onnx

# 4. 启动服务并启用ArcFace
FACECV_ARCFACE_ENABLED=true python main.py
```

## 认证与安全

### API Key 认证
```python
headers = {
    "Authorization": "Bearer YOUR_API_KEY",
    "Content-Type": "application/json"
}
```

### 速率限制
- 面部识别: 100 请求/分钟
- 模型管理: 10 请求/分钟
- 系统监控: 60 请求/分钟

## 核心面部识别 API

### 1. 面部检测与识别

#### POST /api/v1/face/detect
**功能**: 检测图像中的人脸并返回坐标信息

**Java Spring 示例**:
```java
@RestController
@RequestMapping("/face")
public class FaceController {
    
    @Autowired
    private RestTemplate restTemplate;
    
    @PostMapping("/detect")
    public ResponseEntity<?> detectFace(@RequestParam("file") MultipartFile file) {
        try {
            HttpHeaders headers = new HttpHeaders();
            headers.setContentType(MediaType.MULTIPART_FORM_DATA);
            headers.set("Authorization", "Bearer " + apiKey);
            
            MultiValueMap<String, Object> body = new LinkedMultiValueMap<>();
            body.add("file", new FileSystemResource(convertToFile(file)));
            
            HttpEntity<MultiValueMap<String, Object>> requestEntity = 
                new HttpEntity<>(body, headers);
            
            ResponseEntity<FaceDetectionResponse> response = restTemplate.postForEntity(
                "http://localhost:7003/api/v1/face/detect",
                requestEntity,
                FaceDetectionResponse.class
            );
            
            return ResponseEntity.ok(response.getBody());
            
        } catch (Exception e) {
            return ResponseEntity.status(500).body("检测失败: " + e.getMessage());
        }
    }
}

// 响应模型
public class FaceDetectionResponse {
    private boolean success;
    private List<FaceInfo> faces;
    private double processing_time;
    
    // getters and setters
}

public class FaceInfo {
    private List<Double> bbox;           // [x1, y1, x2, y2]
    private double confidence;
    private List<List<Double>> landmarks; // 5点关键点
    private String quality;              // "high", "medium", "low"
    
    // getters and setters
}
```

**输入参数**:
- `file`: 图像文件（支持 jpg, png, bmp）
- `min_confidence`: 最小置信度（默认 0.5）
- `return_landmarks`: 是否返回关键点（默认 true）

**输出示例**:
```json
{
  "success": true,
  "faces": [
    {
      "bbox": [100, 120, 200, 250],
      "confidence": 0.98,
      "landmarks": [[110, 140], [190, 140], [150, 170], [130, 200], [170, 200]],
      "quality": "high"
    }
  ],
  "processing_time": 0.045
}
```

**常见错误**:
- `400`: 未提供图像文件
- `413`: 图像文件过大（>10MB）
- `422`: 图像格式不支持
- `500`: 模型加载失败

#### POST /api/v1/face/recognize
**功能**: 识别图像中的人脸并返回身份信息

**Java Spring 示例**:
```java
@PostMapping("/recognize")
public ResponseEntity<?> recognizeFace(@RequestParam("file") MultipartFile file,
                                     @RequestParam(defaultValue = "0.6") double threshold) {
    try {
        MultiValueMap<String, Object> body = new LinkedMultiValueMap<>();
        body.add("file", new FileSystemResource(convertToFile(file)));
        body.add("threshold", threshold);
        
        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.MULTIPART_FORM_DATA);
        
        HttpEntity<MultiValueMap<String, Object>> request = new HttpEntity<>(body, headers);
        
        ResponseEntity<FaceRecognitionResponse> response = restTemplate.postForEntity(
            "http://localhost:7003/api/v1/face/recognize",
            request,
            FaceRecognitionResponse.class
        );
        
        return ResponseEntity.ok(response.getBody());
        
    } catch (HttpClientErrorException e) {
        if (e.getStatusCode() == HttpStatus.NOT_FOUND) {
            return ResponseEntity.ok().body("未找到匹配的人脸");
        }
        return ResponseEntity.status(e.getStatusCode()).body(e.getResponseBodyAsString());
    }
}

public class FaceRecognitionResponse {
    private boolean success;
    private List<RecognitionResult> results;
    private double processing_time;
}

public class RecognitionResult {
    private String name;
    private String person_id;
    private double similarity;
    private List<Double> bbox;
    private String group_name;
}
```

### 2. 人脸数据库管理

#### POST /api/v1/face/add
**功能**: 添加新的人脸到数据库

**Java Spring 示例**:
```java
@PostMapping("/add")
public ResponseEntity<?> addFace(@RequestParam("file") MultipartFile file,
                               @RequestParam("name") String name,
                               @RequestParam("person_id") String personId,
                               @RequestParam(defaultValue = "default") String groupName) {
    try {
        MultiValueMap<String, Object> body = new LinkedMultiValueMap<>();
        body.add("file", new FileSystemResource(convertToFile(file)));
        body.add("name", name);
        body.add("person_id", personId);
        body.add("group_name", groupName);
        
        ResponseEntity<AddFaceResponse> response = restTemplate.postForEntity(
            "http://localhost:7003/api/v1/face/add",
            new HttpEntity<>(body, getMultipartHeaders()),
            AddFaceResponse.class
        );
        
        return ResponseEntity.ok(response.getBody());
        
    } catch (HttpClientErrorException e) {
        return handleFaceApiError(e);
    }
}

public class AddFaceResponse {
    private boolean success;
    private String message;
    private String face_id;
    private FaceInfo face_info;
    private double processing_time;
}
```

#### GET /api/v1/face/list
**功能**: 获取人脸数据库列表

**Java Spring 示例**:
```java
@GetMapping("/list")
public ResponseEntity<?> listFaces(@RequestParam(defaultValue = "default") String groupName,
                                 @RequestParam(defaultValue = "0") int page,
                                 @RequestParam(defaultValue = "50") int size) {
    try {
        UriComponentsBuilder builder = UriComponentsBuilder
            .fromHttpUrl("http://localhost:7003/api/v1/face/list")
            .queryParam("group_name", groupName)
            .queryParam("page", page)
            .queryParam("size", size);
        
        ResponseEntity<FaceListResponse> response = restTemplate.getForEntity(
            builder.toUriString(),
            FaceListResponse.class
        );
        
        return ResponseEntity.ok(response.getBody());
        
    } catch (Exception e) {
        return ResponseEntity.status(500).body("获取列表失败: " + e.getMessage());
    }
}

public class FaceListResponse {
    private boolean success;
    private List<FaceRecord> faces;
    private int total;
    private int page;
    private int size;
}

public class FaceRecord {
    private String face_id;
    private String name;
    private String person_id;
    private String group_name;
    private String created_at;
    private String updated_at;
    private Map<String, Object> metadata;
}
```

### 3. 人脸验证

#### POST /api/v1/face/verify
**功能**: 验证两张图像是否为同一人

**Java Spring 示例**:
```java
@PostMapping("/verify")
public ResponseEntity<?> verifyFace(@RequestParam("file1") MultipartFile file1,
                                  @RequestParam("file2") MultipartFile file2,
                                  @RequestParam(defaultValue = "0.6") double threshold) {
    try {
        MultiValueMap<String, Object> body = new LinkedMultiValueMap<>();
        body.add("file1", new FileSystemResource(convertToFile(file1)));
        body.add("file2", new FileSystemResource(convertToFile(file2)));
        body.add("threshold", threshold);
        
        ResponseEntity<VerificationResponse> response = restTemplate.postForEntity(
            "http://localhost:7003/api/v1/face/verify",
            new HttpEntity<>(body, getMultipartHeaders()),
            VerificationResponse.class
        );
        
        return ResponseEntity.ok(response.getBody());
        
    } catch (Exception e) {
        return ResponseEntity.status(500).body("验证失败: " + e.getMessage());
    }
}

public class VerificationResponse {
    private boolean success;
    private boolean verified;
    private double similarity;
    private double threshold;
    private String confidence_level;  // "high", "medium", "low"
    private double processing_time;
}
```

## 高级模型管理 API

### 1. ArcFace 专用模型管理 (NEW! 🚀)

#### POST /api/v1/insightface/models/switch
**功能**: 在ArcFace专用模型和Buffalo模型间动态切换

**参数**:
- `enable_arcface` (form): 是否启用ArcFace专用模型 (true/false)
- `arcface_backbone` (form, 可选): ArcFace骨干网络 (resnet50/mobilefacenet)

**curl 示例**:
```bash
# 切换到ArcFace ResNet50模型
curl -X POST "http://localhost:7000/api/v1/insightface/models/switch" \
     -H "Content-Type: application/x-www-form-urlencoded" \
     -d "enable_arcface=true&arcface_backbone=resnet50"

# 切换到ArcFace MobileFaceNet模型  
curl -X POST "http://localhost:7000/api/v1/insightface/models/switch" \
     -H "Content-Type: application/x-www-form-urlencoded" \
     -d "enable_arcface=true&arcface_backbone=mobilefacenet"

# 切换回Buffalo模型
curl -X POST "http://localhost:7000/api/v1/insightface/models/switch" \
     -H "Content-Type: application/x-www-form-urlencoded" \
     -d "enable_arcface=false"
```

**响应示例**:
```json
{
  "success": true,
  "message": "Successfully switched to ArcFace model",
  "model_type": "ArcFace",
  "model_info": {
    "model_name": "buffalo_l_resnet50",
    "backbone": "resnet50", 
    "dataset": "webface600k",
    "embedding_size": 512,
    "initialized": true,
    "detection_enabled": true
  },
  "configuration": {
    "arcface_enabled": true,
    "backbone": "resnet50",
    "similarity_threshold": 0.35,
    "detection_threshold": 0.5
  }
}
```

**Java Spring 示例**:
```java
@PostMapping("/models/arcface/switch")
public ResponseEntity<?> switchToArcFace(@RequestParam boolean enableArcface,
                                        @RequestParam(defaultValue = "resnet50") String backbone) {
    try {
        MultiValueMap<String, String> body = new LinkedMultiValueMap<>();
        body.add("enable_arcface", String.valueOf(enableArcface));
        body.add("arcface_backbone", backbone);
        
        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.APPLICATION_FORM_URLENCODED);
        
        HttpEntity<MultiValueMap<String, String>> request = new HttpEntity<>(body, headers);
        
        ResponseEntity<ModelSwitchResponse> response = restTemplate.postForEntity(
            "http://localhost:7000/api/v1/insightface/models/switch",
            request,
            ModelSwitchResponse.class
        );
        
        if (response.getBody().isSuccess()) {
            log.info("Successfully switched to {} model: {}", 
                response.getBody().getModelType(), 
                response.getBody().getModelInfo().getModelName());
        }
        
        return ResponseEntity.ok(response.getBody());
        
    } catch (Exception e) {
        log.error("模型切换失败", e);
        return ResponseEntity.status(500).body("Model switch failed: " + e.getMessage());
    }
}

public class ModelSwitchResponse {
    private boolean success;
    private String message;
    private String modelType;  // "ArcFace" or "Buffalo"
    private ModelInfo modelInfo;
    private ModelConfiguration configuration;
    
    // getters and setters
}
```

#### GET /api/v1/insightface/models/available
**功能**: 获取可用的InsightFace和ArcFace模型列表及特性对比

**响应示例**:
```json
{
  "available_models": {
    "buffalo_l": {
      "name": "buffalo_l",
      "description": "Buffalo-L 大型模型包 - 最佳精度，生产环境推荐",
      "accuracy": "最高 (★★★★★)",
      "speed": "中等 (★★★☆☆)",
      "size": "大 (~1.5GB)",
      "recommended_use": "生产环境、高精度要求、服务器部署"
    },
    "buffalo_m": {
      "name": "buffalo_m", 
      "description": "Buffalo-M 中型模型包 - 精度与速度平衡",
      "accuracy": "高 (★★★★☆)",
      "speed": "快 (★★★★☆)",
      "recommended_use": "边缘设备、实时应用、平衡性能"
    }
  },
  "arcface_models": {
    "arcface_resnet50_ms1mv3": {
      "name": "arcface_resnet50_ms1mv3",
      "type": "ArcFace",
      "description": "ArcFace resnet50 - ms1mv3数据集",
      "backbone": "resnet50",
      "dataset": "ms1mv3", 
      "embedding_size": "512D",
      "accuracy": "极高 (★★★★★)",
      "speed": "中等 (★★★☆☆)",
      "recommended_use": "生产环境、高精度识别"
    },
    "buffalo_l_resnet50": {
      "name": "buffalo_l_resnet50",
      "type": "ArcFace", 
      "description": "ArcFace resnet50 - webface600k数据集",
      "backbone": "resnet50",
      "dataset": "webface600k",
      "embedding_size": "512D",
      "accuracy": "极高 (★★★★★)",
      "speed": "中等 (★★★☆☆)", 
      "recommended_use": "生产环境、高精度识别"
    },
    "buffalo_s_mobilefacenet": {
      "name": "buffalo_s_mobilefacenet",
      "type": "ArcFace",
      "description": "ArcFace mobilefacenet - webface600k数据集", 
      "backbone": "mobilefacenet",
      "dataset": "webface600k",
      "embedding_size": "256D",
      "accuracy": "高 (★★★★☆)",
      "speed": "快 (★★★★☆)",
      "recommended_use": "移动端、边缘计算"
    }
  },
  "current_model": "buffalo_l",
  "arcface_enabled": false,
  "recommendation": {
    "production": "buffalo_l 或 ArcFace ResNet50 - 生产环境首选",
    "edge_device": "buffalo_m 或 ArcFace MobileFaceNet - 边缘设备推荐", 
    "mobile": "buffalo_s 或 ArcFace MobileFaceNet - 移动端推荐",
    "research": "ArcFace 独立模型 - 研究和定制化需求"
  }
}
```

**模型路径说明**:
```
ArcFace模型自动发现路径 (按优先级):
1. weights/arcface/backbone/dataset/   # 独立ArcFace权重 (最高优先级)
2. weights/insightface/buffalo_*/      # 本地Buffalo包中的ArcFace模型
3. ~/.insightface/models/buffalo_*/    # 默认InsightFace模型目录 (fallback)

示例:
- weights/arcface/resnet50/ms1mv3/arcface_resnet50_ms1mv3.onnx
- weights/insightface/buffalo_l/w600k_r50.onnx  
- ~/.insightface/models/buffalo_l/w600k_r50.onnx
```

#### POST /api/v1/insightface/models/select
**功能**: 动态切换InsightFace模型 (运行时无需重启)

**参数**:
- `model` (query): 模型名称 - buffalo_l, buffalo_m, buffalo_s, antelopev2

**curl 示例**:
```bash
# 切换到高速模型 (buffalo_m)
curl -X POST "http://localhost:7000/api/v1/insightface/models/select?model=buffalo_m"

# 切换回最佳精度模型 (buffalo_l)  
curl -X POST "http://localhost:7000/api/v1/insightface/models/select?model=buffalo_l"
```

**响应示例**:
```json
{
  "success": true,
  "message": "Successfully switched from buffalo_l to buffalo_m",
  "previous_model": "buffalo_l",
  "current_model": "buffalo_m",
  "model_info": {
    "initialized": true,
    "model_pack": "buffalo_m",
    "available_models": {...},
    "insightface_available": true
  }
}
```

**Java Spring 示例**:
```java
@PostMapping("/models/select")
public ResponseEntity<?> selectInsightFaceModel(@RequestParam String model) {
    try {
        // 验证模型选择
        List<String> validModels = Arrays.asList("buffalo_l", "buffalo_m", "buffalo_s", "antelopev2");
        if (!validModels.contains(model)) {
            return ResponseEntity.badRequest().body("Invalid model: " + model);
        }
        
        String url = "http://localhost:7000/api/v1/insightface/models/select?model=" + model;
        ResponseEntity<ModelSelectionResponse> response = restTemplate.postForEntity(
            url, null, ModelSelectionResponse.class
        );
        
        if (response.getBody().isSuccess()) {
            log.info("Successfully switched to model: " + model);
            return ResponseEntity.ok(response.getBody());
        } else {
            return ResponseEntity.status(500).body("Model selection failed");
        }
        
    } catch (Exception e) {
        return ResponseEntity.status(500).body("模型切换失败: " + e.getMessage());
    }
}

public class ModelSelectionResponse {
    private boolean success;
    private String message;
    private String previousModel;
    private String currentModel;
    private Map<String, Object> modelInfo;
    
    // getters and setters
}
```

**模型选择指南**:
- **buffalo_l**: 生产环境首选，最高精度 (推荐)
- **buffalo_m**: 边缘设备，精度速度平衡
- **buffalo_s**: 移动端，速度优先
- **antelopev2**: 研究环境，极高精度

### 2. 模型状态管理

#### GET /api/v1/models/status
**功能**: 获取所有模型的状态信息

**Java Spring 示例**:
```java
@GetMapping("/models/status")
public ResponseEntity<?> getModelsStatus() {
    try {
        ResponseEntity<ModelStatusResponse> response = restTemplate.getForEntity(
            "http://localhost:7003/api/v1/models/status",
            ModelStatusResponse.class
        );
        
        return ResponseEntity.ok(response.getBody());
        
    } catch (Exception e) {
        return ResponseEntity.status(500).body("获取模型状态失败: " + e.getMessage());
    }
}

public class ModelStatusResponse {
    private Map<String, ModelStatus> models;
}

public class ModelStatus {
    private boolean loaded;
    private String status;        // "active", "loading", "error", "unloaded"
    private String provider;      // "CUDAExecutionProvider", "CPUExecutionProvider"
    private long memory_usage;    // MB
    private double load_time;     // seconds
    private String last_used;     // ISO datetime
    private String error_message;
}
```

#### POST /api/v1/models/load
**功能**: 加载指定模型

**Java Spring 示例**:
```java
@PostMapping("/models/load")
public ResponseEntity<?> loadModel(@RequestBody LoadModelRequest request) {
    try {
        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.APPLICATION_JSON);
        
        HttpEntity<LoadModelRequest> entity = new HttpEntity<>(request, headers);
        
        ResponseEntity<LoadModelResponse> response = restTemplate.postForEntity(
            "http://localhost:7003/api/v1/models/load",
            entity,
            LoadModelResponse.class
        );
        
        return ResponseEntity.ok(response.getBody());
        
    } catch (HttpClientErrorException e) {
        return handleModelError(e);
    }
}

public class LoadModelRequest {
    private String model_name;     // "buffalo_l", "buffalo_m", "buffalo_s"
    private List<String> providers; // ["CUDAExecutionProvider", "CPUExecutionProvider"]
    private boolean force_reload;
    
    // constructors, getters, setters
}

public class LoadModelResponse {
    private boolean success;
    private String message;
    private String model_name;
    private String provider;
    private double load_time;
    private long memory_usage;
}
```

### 2. 高级模型功能

#### GET /api/v1/models/advanced/available
**功能**: 获取可用的高级模型列表

**Java Spring 示例**:
```java
@GetMapping("/models/advanced/available")
public ResponseEntity<?> getAvailableModels() {
    try {
        ResponseEntity<AvailableModelsResponse> response = restTemplate.getForEntity(
            "http://localhost:7003/api/v1/models/advanced/available",
            AvailableModelsResponse.class
        );
        
        return ResponseEntity.ok(response.getBody());
        
    } catch (Exception e) {
        return ResponseEntity.status(500).body("获取模型列表失败: " + e.getMessage());
    }
}

public class AvailableModelsResponse {
    private boolean success;
    private List<ModelInfo> models;
}

public class ModelInfo {
    private String name;
    private String description;
    private List<String> use_cases;
    private Map<String, Object> performance;
    private Map<String, Object> requirements;
    private boolean downloaded;
    private String size;
    private String accuracy;
    private String speed;
}
```

#### POST /api/v1/models/advanced/recommendations
**功能**: 根据使用场景获取模型推荐

**Java Spring 示例**:
```java
@PostMapping("/models/advanced/recommendations")
public ResponseEntity<?> getModelRecommendations(@RequestBody RecommendationRequest request) {
    try {
        ResponseEntity<RecommendationResponse> response = restTemplate.postForEntity(
            "http://localhost:7003/api/v1/models/advanced/recommendations",
            request,
            RecommendationResponse.class
        );
        
        return ResponseEntity.ok(response.getBody());
        
    } catch (Exception e) {
        return ResponseEntity.status(500).body("获取推荐失败: " + e.getMessage());
    }
}

public class RecommendationRequest {
    private String use_case;      // "high_accuracy", "real_time", "mobile", "server"
    private boolean has_gpu;
    private int memory_limit_mb;
    private double latency_requirement; // ms
    
    // constructors, getters, setters
}

public class RecommendationResponse {
    private boolean success;
    private List<ModelRecommendation> recommendations;
}

public class ModelRecommendation {
    private String model_name;
    private double score;
    private String reason;
    private Map<String, Object> expected_performance;
}
```

## 系统健康监控 API

### 1. 综合健康检查

#### GET /api/v1/health/comprehensive
**功能**: 获取系统综合健康状态

**Java Spring 示例**:
```java
@Component
public class HealthMonitorService {
    
    @Autowired
    private RestTemplate restTemplate;
    
    @Scheduled(fixedRate = 30000) // 每30秒检查一次
    public void checkSystemHealth() {
        try {
            ResponseEntity<ComprehensiveHealthResponse> response = restTemplate.getForEntity(
                "http://localhost:7003/api/v1/health/comprehensive",
                ComprehensiveHealthResponse.class
            );
            
            ComprehensiveHealthResponse health = response.getBody();
            
            if (!health.isHealthy()) {
                // 发送告警
                alertService.sendAlert("系统健康检查失败", health.getIssues());
            }
            
            // 记录健康状态
            healthRepository.save(new HealthRecord(health));
            
        } catch (Exception e) {
            alertService.sendAlert("健康检查API异常", e.getMessage());
        }
    }
}

public class ComprehensiveHealthResponse {
    private boolean healthy;
    private String status;           // "healthy", "warning", "critical"
    private List<String> issues;
    private List<String> warnings;
    private List<String> recommendations;
    private SystemMetrics metrics;
    private String timestamp;
}

public class SystemMetrics {
    private CpuInfo cpu;
    private MemoryInfo memory;
    private DiskInfo disk;
    private GpuInfo gpu;
    private DatabaseInfo database;
    private ModelInfo models;
}
```

### 2. GPU 监控

#### GET /api/v1/health/gpu
**功能**: 获取 GPU 使用状态

**Java Spring 示例**:
```java
@GetMapping("/health/gpu")
public ResponseEntity<?> getGpuHealth() {
    try {
        ResponseEntity<GpuHealthResponse> response = restTemplate.getForEntity(
            "http://localhost:7003/api/v1/health/gpu",
            GpuHealthResponse.class
        );
        
        GpuHealthResponse gpuHealth = response.getBody();
        
        // 检查 GPU 使用率是否过高
        if (gpuHealth.getUtilization() > 90) {
            // 触发扩容或负载均衡
            scaleService.triggerGpuScale();
        }
        
        return ResponseEntity.ok(gpuHealth);
        
    } catch (Exception e) {
        return ResponseEntity.status(500).body("GPU监控失败: " + e.getMessage());
    }
}

public class GpuHealthResponse {
    private boolean available;
    private int gpu_count;
    private List<GpuDevice> devices;
    private double total_memory_gb;
    private double used_memory_gb;
    private double utilization;
    private double temperature;
    private String driver_version;
    private String cuda_version;
}

public class GpuDevice {
    private int id;
    private String name;
    private double memory_total;
    private double memory_used;
    private double utilization;
    private double temperature;
}
```

### 3. 性能监控

#### GET /api/v1/health/performance
**功能**: 获取系统性能指标

**Java Spring 示例**:
```java
@GetMapping("/health/performance")
public ResponseEntity<?> getPerformanceMetrics() {
    try {
        ResponseEntity<PerformanceResponse> response = restTemplate.getForEntity(
            "http://localhost:7003/api/v1/health/performance",
            PerformanceResponse.class
        );
        
        return ResponseEntity.ok(response.getBody());
        
    } catch (Exception e) {
        return ResponseEntity.status(500).body("性能监控失败: " + e.getMessage());
    }
}

public class PerformanceResponse {
    private RequestMetrics requests;
    private ModelMetrics models;
    private SystemMetrics system;
    private String timestamp;
}

public class RequestMetrics {
    private int total_requests;
    private int requests_per_minute;
    private double avg_response_time;
    private double p95_response_time;
    private double error_rate;
    private Map<String, Integer> endpoint_stats;
}

public class ModelMetrics {
    private Map<String, ModelPerformance> model_performance;
    private int total_inferences;
    private double avg_inference_time;
}
```

## 视频流处理 API

### 1. 实时流处理

#### POST /api/v1/stream/start
**功能**: 启动视频流处理

**Java Spring 示例**:
```java
@PostMapping("/stream/start")
public ResponseEntity<?> startStream(@RequestBody StreamRequest request) {
    try {
        ResponseEntity<StreamResponse> response = restTemplate.postForEntity(
            "http://localhost:7003/api/v1/stream/start",
            request,
            StreamResponse.class
        );
        
        StreamResponse streamInfo = response.getBody();
        
        // 保存流信息
        streamRepository.save(new StreamRecord(streamInfo.getStreamId(), request));
        
        return ResponseEntity.ok(streamInfo);
        
    } catch (Exception e) {
        return ResponseEntity.status(500).body("启动流处理失败: " + e.getMessage());
    }
}

public class StreamRequest {
    private String source_url;      // RTSP/HTTP 流地址或摄像头 ID
    private String stream_type;     // "rtsp", "webcam", "http"
    private int fps;               // 处理帧率
    private boolean save_frames;   // 是否保存帧
    private String webhook_url;    // 结果回调地址
    private StreamConfig config;
}

public class StreamConfig {
    private double recognition_threshold;
    private boolean enable_tracking;
    private int max_faces_per_frame;
    private String output_format;  // "json", "xml"
    private boolean enable_alerts;
}

public class StreamResponse {
    private boolean success;
    private String stream_id;
    private String status;
    private String websocket_url;  // WebSocket 连接地址
    private String message;
}
```

#### GET /api/v1/stream/status/{stream_id}
**功能**: 获取流处理状态

**Java Spring 示例**:
```java
@GetMapping("/stream/status/{streamId}")
public ResponseEntity<?> getStreamStatus(@PathVariable String streamId) {
    try {
        ResponseEntity<StreamStatusResponse> response = restTemplate.getForEntity(
            "http://localhost:7003/api/v1/stream/status/" + streamId,
            StreamStatusResponse.class
        );
        
        return ResponseEntity.ok(response.getBody());
        
    } catch (HttpClientErrorException e) {
        if (e.getStatusCode() == HttpStatus.NOT_FOUND) {
            return ResponseEntity.notFound().build();
        }
        return ResponseEntity.status(e.getStatusCode()).body(e.getResponseBodyAsString());
    }
}

public class StreamStatusResponse {
    private String stream_id;
    private String status;          // "running", "stopped", "error"
    private int frames_processed;
    private int faces_detected;
    private int faces_recognized;
    private double fps;
    private String start_time;
    private String last_frame_time;
    private List<String> recent_errors;
}
```

## Webhook 集成 API

### 1. Webhook 配置

#### POST /api/v1/webhook/register
**功能**: 注册 Webhook 端点

**Java Spring 示例**:
```java
@PostMapping("/webhook/register")
public ResponseEntity<?> registerWebhook(@RequestBody WebhookRequest request) {
    try {
        ResponseEntity<WebhookResponse> response = restTemplate.postForEntity(
            "http://localhost:7003/api/v1/webhook/register",
            request,
            WebhookResponse.class
        );
        
        return ResponseEntity.ok(response.getBody());
        
    } catch (Exception e) {
        return ResponseEntity.status(500).body("注册Webhook失败: " + e.getMessage());
    }
}

public class WebhookRequest {
    private String url;
    private List<String> events;    // ["face_detected", "face_recognized", "stream_started"]
    private String secret;          // 用于签名验证
    private Map<String, String> headers; // 自定义请求头
    private boolean active;
    private WebhookConfig config;
}

public class WebhookConfig {
    private int timeout_seconds;
    private int retry_attempts;
    private boolean verify_ssl;
    private String content_type;    // "application/json", "application/xml"
}
```

### 2. Webhook 事件处理

**接收 Webhook 事件的 Spring Controller**:
```java
@RestController
@RequestMapping("/webhook")
public class WebhookController {
    
    @PostMapping("/facecv")
    public ResponseEntity<?> handleFaceCVWebhook(@RequestBody WebhookEvent event,
                                               @RequestHeader("X-FaceCV-Signature") String signature,
                                               HttpServletRequest request) {
        try {
            // 验证签名
            if (!webhookService.verifySignature(request, signature)) {
                return ResponseEntity.status(401).body("签名验证失败");
            }
            
            // 处理不同类型的事件
            switch (event.getEventType()) {
                case "face_detected":
                    handleFaceDetected(event);
                    break;
                case "face_recognized":
                    handleFaceRecognized(event);
                    break;
                case "stream_started":
                    handleStreamStarted(event);
                    break;
                case "system_alert":
                    handleSystemAlert(event);
                    break;
                default:
                    logger.warn("未知事件类型: " + event.getEventType());
            }
            
            return ResponseEntity.ok().body("事件处理成功");
            
        } catch (Exception e) {
            logger.error("Webhook处理失败", e);
            return ResponseEntity.status(500).body("处理失败: " + e.getMessage());
        }
    }
    
    private void handleFaceRecognized(WebhookEvent event) {
        FaceRecognizedData data = (FaceRecognizedData) event.getData();
        
        // 记录识别结果
        AttendanceRecord record = new AttendanceRecord();
        record.setPersonId(data.getPersonId());
        record.setName(data.getName());
        record.setTimestamp(data.getTimestamp());
        record.setConfidence(data.getSimilarity());
        record.setSource(data.getSource());
        
        attendanceService.recordAttendance(record);
        
        // 发送实时通知
        notificationService.sendRealTimeNotification(
            "人脸识别", 
            String.format("识别到 %s，相似度: %.2f", data.getName(), data.getSimilarity())
        );
    }
}

public class WebhookEvent {
    private String event_type;
    private String event_id;
    private String timestamp;
    private Object data;
    private String source;
    
    // getters and setters
}

public class FaceRecognizedData {
    private String person_id;
    private String name;
    private double similarity;
    private List<Double> bbox;
    private String source;
    private String timestamp;
    private String image_url;
    
    // getters and setters
}
```

## Java Spring 集成示例

### 1. 完整的 Spring Boot 配置

#### 主配置类
```java
@SpringBootApplication
@EnableScheduling
@EnableConfigurationProperties({FaceCVProperties.class})
public class FaceCVIntegrationApplication {
    
    public static void main(String[] args) {
        SpringApplication.run(FaceCVIntegrationApplication.class, args);
    }
    
    @Bean
    public RestTemplate restTemplate() {
        RestTemplate restTemplate = new RestTemplate();
        
        // 配置超时
        HttpComponentsClientHttpRequestFactory factory = 
            new HttpComponentsClientHttpRequestFactory();
        factory.setConnectTimeout(5000);
        factory.setReadTimeout(30000);
        restTemplate.setRequestFactory(factory);
        
        // 配置错误处理
        restTemplate.setErrorHandler(new FaceCVErrorHandler());
        
        return restTemplate;
    }
    
    @Bean
    public FaceCVClient faceCVClient(RestTemplate restTemplate, FaceCVProperties properties) {
        return new FaceCVClient(restTemplate, properties);
    }
}
```

#### 配置属性类
```java
@ConfigurationProperties(prefix = "facecv")
@Data
public class FaceCVProperties {
    private String baseUrl = "http://localhost:7003/api/v1";
    private String apiKey;
    private int timeout = 30000;
    private int maxRetries = 3;
    private boolean enableHealthCheck = true;
    private int healthCheckInterval = 30; // seconds
    
    // Webhook 配置
    private Webhook webhook = new Webhook();
    
    @Data
    public static class Webhook {
        private String secret;
        private List<String> events = Arrays.asList("face_recognized", "system_alert");
        private boolean enabled = true;
    }
}
```

#### FaceCV 客户端封装
```java
@Component
@Slf4j
public class FaceCVClient {
    
    private final RestTemplate restTemplate;
    private final FaceCVProperties properties;
    
    public FaceCVClient(RestTemplate restTemplate, FaceCVProperties properties) {
        this.restTemplate = restTemplate;
        this.properties = properties;
    }
    
    /**
     * 人脸识别
     */
    public FaceRecognitionResponse recognizeFace(MultipartFile file, double threshold) {
        try {
            MultiValueMap<String, Object> body = new LinkedMultiValueMap<>();
            body.add("file", new FileSystemResource(convertToFile(file)));
            body.add("threshold", threshold);
            
            HttpHeaders headers = new HttpHeaders();
            headers.setContentType(MediaType.MULTIPART_FORM_DATA);
            headers.setBearerAuth(properties.getApiKey());
            
            HttpEntity<MultiValueMap<String, Object>> request = new HttpEntity<>(body, headers);
            
            ResponseEntity<FaceRecognitionResponse> response = restTemplate.exchange(
                properties.getBaseUrl() + "/face/recognize",
                HttpMethod.POST,
                request,
                FaceRecognitionResponse.class
            );
            
            return response.getBody();
            
        } catch (Exception e) {
            log.error("人脸识别失败", e);
            throw new FaceCVException("人脸识别失败: " + e.getMessage(), e);
        }
    }
    
    /**
     * 批量人脸识别
     */
    @Async
    public CompletableFuture<List<FaceRecognitionResponse>> recognizeFacesBatch(
            List<MultipartFile> files, double threshold) {
        
        List<CompletableFuture<FaceRecognitionResponse>> futures = files.stream()
            .map(file -> CompletableFuture.supplyAsync(() -> recognizeFace(file, threshold)))
            .collect(Collectors.toList());
        
        return CompletableFuture.allOf(futures.toArray(new CompletableFuture[0]))
            .thenApply(v -> futures.stream()
                .map(CompletableFuture::join)
                .collect(Collectors.toList()));
    }
    
    /**
     * 系统健康检查
     */
    public boolean isHealthy() {
        try {
            ResponseEntity<ComprehensiveHealthResponse> response = restTemplate.getForEntity(
                properties.getBaseUrl() + "/health/comprehensive",
                ComprehensiveHealthResponse.class
            );
            
            return response.getBody() != null && response.getBody().isHealthy();
            
        } catch (Exception e) {
            log.warn("健康检查失败", e);
            return false;
        }
    }
    
    private File convertToFile(MultipartFile multipartFile) throws IOException {
        File tempFile = File.createTempFile("upload", multipartFile.getOriginalFilename());
        multipartFile.transferTo(tempFile);
        return tempFile;
    }
}
```

### 2. 考勤系统集成示例

```java
@Service
@Transactional
@Slf4j
public class AttendanceService {
    
    @Autowired
    private FaceCVClient faceCVClient;
    
    @Autowired
    private AttendanceRepository attendanceRepository;
    
    @Autowired
    private EmployeeRepository employeeRepository;
    
    /**
     * 处理考勤打卡
     */
    public AttendanceResult processAttendance(MultipartFile faceImage, String deviceId) {
        try {
            // 1. 人脸识别
            FaceRecognitionResponse recognition = faceCVClient.recognizeFace(faceImage, 0.7);
            
            if (!recognition.isSuccess() || recognition.getResults().isEmpty()) {
                return AttendanceResult.failed("未识别到有效人脸");
            }
            
            RecognitionResult result = recognition.getResults().get(0);
            
            // 2. 查找员工信息
            Employee employee = employeeRepository.findByPersonId(result.getPersonId())
                .orElseThrow(() -> new EmployeeNotFoundException("未找到员工信息"));
            
            // 3. 检查重复打卡
            LocalDateTime now = LocalDateTime.now();
            LocalDateTime startOfDay = now.toLocalDate().atStartOfDay();
            
            Optional<AttendanceRecord> existingRecord = attendanceRepository
                .findByEmployeeIdAndTimestampBetween(
                    employee.getId(), 
                    startOfDay, 
                    startOfDay.plusDays(1)
                );
            
            // 4. 创建考勤记录
            AttendanceRecord record = new AttendanceRecord();
            record.setEmployeeId(employee.getId());
            record.setEmployeeName(employee.getName());
            record.setTimestamp(now);
            record.setConfidence(result.getSimilarity());
            record.setDeviceId(deviceId);
            record.setImagePath(saveAttendanceImage(faceImage, employee.getId()));
            
            if (existingRecord.isPresent()) {
                record.setType(AttendanceType.CHECK_OUT);
                record.setCheckInId(existingRecord.get().getId());
            } else {
                record.setType(AttendanceType.CHECK_IN);
            }
            
            attendanceRepository.save(record);
            
            // 5. 发送通知
            notificationService.sendAttendanceNotification(record);
            
            return AttendanceResult.success(record);
            
        } catch (Exception e) {
            log.error("考勤处理失败", e);
            return AttendanceResult.failed("考勤处理失败: " + e.getMessage());
        }
    }
    
    /**
     * 获取考勤统计
     */
    public AttendanceStatistics getAttendanceStatistics(String employeeId, 
                                                       LocalDate startDate, 
                                                       LocalDate endDate) {
        List<AttendanceRecord> records = attendanceRepository
            .findByEmployeeIdAndDateRange(employeeId, startDate, endDate);
        
        return AttendanceStatistics.builder()
            .totalDays(records.size())
            .presentDays((int) records.stream().filter(r -> r.getType() == AttendanceType.CHECK_IN).count())
            .lateDays((int) records.stream().filter(this::isLate).count())
            .earlyLeaveDays((int) records.stream().filter(this::isEarlyLeave).count())
            .averageWorkingHours(calculateAverageWorkingHours(records))
            .build();
    }
}
```

### 3. 安全访问控制集成

```java
@RestController
@RequestMapping("/api/security")
@PreAuthorize("hasRole('SECURITY')")
public class SecurityController {
    
    @Autowired
    private FaceCVClient faceCVClient;
    
    @Autowired
    private SecurityEventService securityEventService;
    
    /**
     * 访客识别
     */
    @PostMapping("/visitor/identify")
    public ResponseEntity<?> identifyVisitor(@RequestParam MultipartFile image,
                                           @RequestParam String location) {
        try {
            FaceRecognitionResponse recognition = faceCVClient.recognizeFace(image, 0.6);
            
            if (recognition.isSuccess() && !recognition.getResults().isEmpty()) {
                // 已知人员
                RecognitionResult result = recognition.getResults().get(0);
                securityEventService.logKnownPersonEntry(result, location);
                
                return ResponseEntity.ok(SecurityResponse.knownPerson(result));
            } else {
                // 陌生人告警
                String alertId = securityEventService.createStrangerAlert(image, location);
                
                return ResponseEntity.ok(SecurityResponse.strangerAlert(alertId));
            }
            
        } catch (Exception e) {
            return ResponseEntity.status(500).body("识别失败: " + e.getMessage());
        }
    }
    
    /**
     * 实时监控流
     */
    @PostMapping("/monitor/start")
    public ResponseEntity<?> startMonitoring(@RequestBody MonitorRequest request) {
        try {
            StreamRequest streamRequest = StreamRequest.builder()
                .sourceUrl(request.getCameraUrl())
                .streamType("rtsp")
                .fps(request.getFps())
                .webhookUrl(getSecurityWebhookUrl())
                .config(StreamConfig.builder()
                    .recognitionThreshold(0.7)
                    .enableTracking(true)
                    .enableAlerts(true)
                    .build())
                .build();
            
            StreamResponse response = faceCVClient.startStream(streamRequest);
            
            // 记录监控会话
            securityEventService.createMonitorSession(response.getStreamId(), request);
            
            return ResponseEntity.ok(response);
            
        } catch (Exception e) {
            return ResponseEntity.status(500).body("启动监控失败: " + e.getMessage());
        }
    }
}
```

## 常见错误与解决方案

### 1. API 错误代码

| 错误代码 | 说明 | 解决方案 |
|---------|------|---------|
| 400 | 请求参数错误 | 检查请求参数格式和必填字段 |
| 401 | 认证失败 | 检查 API Key 或认证头 |
| 403 | 权限不足 | 检查用户权限或 API 访问限制 |
| 404 | 资源不存在 | 检查 API 路径或资源 ID |
| 413 | 文件过大 | 压缩图像或检查文件大小限制 |
| 422 | 数据验证失败 | 检查数据格式和字段约束 |
| 429 | 请求频率超限 | 实施请求限流或增加延迟 |
| 500 | 服务器内部错误 | 检查服务器日志和系统状态 |
| 503 | 服务不可用 | 检查服务状态或进行故障转移 |

### 2. 常见问题及解决方案

#### 问题 1: 人脸识别准确率低
**原因**:
- 图像质量差（模糊、光线不足）
- 人脸角度不正
- 训练数据不足

**解决方案**:
```java
// 图像质量检查
public boolean checkImageQuality(MultipartFile image) {
    try {
        // 检查图像分辨率
        BufferedImage img = ImageIO.read(image.getInputStream());
        if (img.getWidth() < 300 || img.getHeight() < 300) {
            throw new IllegalArgumentException("图像分辨率过低，最小 300x300");
        }
        
        // 检查文件大小
        if (image.getSize() > 10 * 1024 * 1024) {
            throw new IllegalArgumentException("图像文件过大，最大 10MB");
        }
        
        return true;
    } catch (IOException e) {
        return false;
    }
}

// 动态调整识别阈值
public FaceRecognitionResponse recognizeWithAdaptiveThreshold(MultipartFile image) {
    double[] thresholds = {0.8, 0.7, 0.6, 0.5};
    
    for (double threshold : thresholds) {
        FaceRecognitionResponse response = faceCVClient.recognizeFace(image, threshold);
        if (response.isSuccess() && !response.getResults().isEmpty()) {
            return response;
        }
    }
    
    return FaceRecognitionResponse.noMatch();
}
```

#### 问题 2: 性能问题
**原因**:
- GPU 内存不足
- 模型加载慢
- 并发请求过多

**解决方案**:
```java
// 连接池配置
@Configuration
public class HttpClientConfig {
    
    @Bean
    public HttpComponentsClientHttpRequestFactory httpRequestFactory() {
        HttpComponentsClientHttpRequestFactory factory = 
            new HttpComponentsClientHttpRequestFactory();
        
        // 连接池配置
        PoolingHttpClientConnectionManager connectionManager = 
            new PoolingHttpClientConnectionManager();
        connectionManager.setMaxTotal(100);
        connectionManager.setDefaultMaxPerRoute(20);
        
        CloseableHttpClient httpClient = HttpClients.custom()
            .setConnectionManager(connectionManager)
            .build();
        
        factory.setHttpClient(httpClient);
        factory.setConnectTimeout(5000);
        factory.setReadTimeout(30000);
        
        return factory;
    }
}

// 异步处理
@Service
public class AsyncFaceService {
    
    @Async
    public CompletableFuture<FaceRecognitionResponse> recognizeAsync(MultipartFile image) {
        return CompletableFuture.supplyAsync(() -> {
            return faceCVClient.recognizeFace(image, 0.7);
        });
    }
    
    @Async
    public void processAttendanceBatch(List<AttendanceRequest> requests) {
        requests.parallelStream().forEach(this::processAttendance);
    }
}
```

#### 问题 3: 内存泄漏
**原因**:
- 临时文件未清理
- 大图像未及时释放
- 长时间运行的流未正确关闭

**解决方案**:
```java
// 资源管理
@Component
public class ResourceManager {
    
    private final ScheduledExecutorService cleanupExecutor = 
        Executors.newSingleThreadScheduledExecutor();
    
    @PostConstruct
    public void init() {
        // 定期清理临时文件
        cleanupExecutor.scheduleAtFixedRate(this::cleanupTempFiles, 1, 1, TimeUnit.HOURS);
    }
    
    public void cleanupTempFiles() {
        try {
            Path tempDir = Paths.get(System.getProperty("java.io.tmpdir"));
            Files.walk(tempDir)
                .filter(path -> path.toString().contains("upload"))
                .filter(path -> {
                    try {
                        return Files.getLastModifiedTime(path)
                            .toInstant()
                            .isBefore(Instant.now().minus(1, ChronoUnit.HOURS));
                    } catch (IOException e) {
                        return false;
                    }
                })
                .forEach(path -> {
                    try {
                        Files.deleteIfExists(path);
                    } catch (IOException e) {
                        log.warn("清理临时文件失败: " + path, e);
                    }
                });
        } catch (Exception e) {
            log.error("清理临时文件异常", e);
        }
    }
}

// 使用 try-with-resources
public FaceRecognitionResponse processImageSafely(MultipartFile file) {
    File tempFile = null;
    try {
        tempFile = convertToFile(file);
        return faceCVClient.recognizeFace(file, 0.7);
    } finally {
        if (tempFile != null && tempFile.exists()) {
            tempFile.delete();
        }
    }
}
```

## 性能优化建议

### 1. 硬件优化

#### GPU 配置
```yaml
# Docker Compose GPU 配置
version: '3.8'
services:
  facecv:
    build: .
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    environment:
      - NVIDIA_VISIBLE_DEVICES=0
      - CUDA_VISIBLE_DEVICES=0
```

#### 内存优化
```java
// JVM 参数优化
java -Xms2g -Xmx8g \
     -XX:+UseG1GC \
     -XX:MaxGCPauseMillis=200 \
     -XX:+UnlockExperimentalVMOptions \
     -XX:+UseZGC \
     -jar facecv-client.jar
```

### 2. 应用层优化

#### 缓存策略
```java
@Service
public class CacheService {
    
    @Cacheable(value = "faceRecognition", key = "#imageHash")
    public FaceRecognitionResponse getCachedRecognition(String imageHash) {
        // 缓存识别结果
        return faceCVClient.recognizeFace(getImageByHash(imageHash), 0.7);
    }
    
    @CacheEvict(value = "faceRecognition", allEntries = true)
    @Scheduled(fixedRate = 3600000) // 1小时清理一次
    public void clearCache() {
        log.info("清理人脸识别缓存");
    }
}
```

#### 连接复用
```java
@Configuration
public class RestTemplateConfig {
    
    @Bean
    public RestTemplate restTemplate() {
        // 配置连接池
        PoolingHttpClientConnectionManager connectionManager = 
            new PoolingHttpClientConnectionManager();
        connectionManager.setMaxTotal(200);
        connectionManager.setDefaultMaxPerRoute(50);
        connectionManager.setValidateAfterInactivity(30000);
        
        // 配置请求重试
        HttpRequestRetryHandler retryHandler = new DefaultHttpRequestRetryHandler(3, true);
        
        CloseableHttpClient httpClient = HttpClients.custom()
            .setConnectionManager(connectionManager)
            .setRetryHandler(retryHandler)
            .build();
        
        HttpComponentsClientHttpRequestFactory factory = 
            new HttpComponentsClientHttpRequestFactory(httpClient);
        factory.setConnectTimeout(5000);
        factory.setReadTimeout(30000);
        
        return new RestTemplate(factory);
    }
}
```

### 3. 监控和告警

#### 性能监控
```java
@Component
public class PerformanceMonitor {
    
    private final MeterRegistry meterRegistry;
    
    @EventListener
    public void handleFaceRecognition(FaceRecognitionEvent event) {
        // 记录识别耗时
        Timer.Sample sample = Timer.start(meterRegistry);
        sample.stop(Timer.builder("face.recognition.duration")
            .description("Face recognition processing time")
            .tag("success", String.valueOf(event.isSuccess()))
            .register(meterRegistry));
        
        // 记录识别准确率
        Gauge.builder("face.recognition.accuracy")
            .description("Face recognition accuracy")
            .register(meterRegistry, () -> event.getAccuracy());
    }
    
    @Scheduled(fixedRate = 60000)
    public void checkSystemHealth() {
        ComprehensiveHealthResponse health = faceCVClient.getSystemHealth();
        
        if (!health.isHealthy()) {
            alertService.sendAlert("FaceCV系统异常", health.getIssues());
        }
        
        // 记录系统指标
        meterRegistry.gauge("system.cpu.usage", health.getMetrics().getCpu().getUsage());
        meterRegistry.gauge("system.memory.usage", health.getMetrics().getMemory().getUsage());
        meterRegistry.gauge("system.gpu.usage", health.getMetrics().getGpu().getUtilization());
    }
}
```

## 🧪 测试验证

### 1. ArcFace模型测试 (NEW! 🚀)

#### 功能测试脚本
```bash
#!/bin/bash
# test_arcface_integration.sh - ArcFace集成测试

BASE_URL="http://localhost:7000/api/v1/insightface"
TEST_IMAGE="/home/a/PycharmProjects/EurekCV/dataset/faces/trump1.jpeg"

echo "🚀 ArcFace 集成测试开始..."

# 1. 检查系统健康
echo "1. 检查系统健康..."
curl -s "${BASE_URL}/health" | jq '.'

# 2. 查看可用模型
echo "2. 查看可用模型..."
curl -s "${BASE_URL}/models/available" | jq '.arcface_models'

# 3. 切换到ArcFace ResNet50
echo "3. 切换到ArcFace ResNet50..."
curl -X POST "${BASE_URL}/models/switch" \
     -H "Content-Type: application/x-www-form-urlencoded" \
     -d "enable_arcface=true&arcface_backbone=resnet50" | jq '.'

# 4. 测试人脸检测
echo "4. 测试人脸检测..."
curl -X POST "${BASE_URL}/detect" \
     -F "file=@${TEST_IMAGE}" | jq '.'

# 5. 测试人脸注册
echo "5. 测试人脸注册..."
curl -X POST "${BASE_URL}/register" \
     -F "file=@${TEST_IMAGE}" \
     -F "name=TestUser" \
     -F "metadata={\"description\":\"ArcFace测试用户\"}" | jq '.'

# 6. 测试人脸验证
echo "6. 测试人脸验证..."
curl -X POST "${BASE_URL}/verify" \
     -F "file1=@${TEST_IMAGE}" \
     -F "file2=@${TEST_IMAGE}" | jq '.'

# 7. 切换到MobileFaceNet
echo "7. 切换到ArcFace MobileFaceNet..."
curl -X POST "${BASE_URL}/models/switch" \
     -H "Content-Type: application/x-www-form-urlencoded" \
     -d "enable_arcface=true&arcface_backbone=mobilefacenet" | jq '.'

# 8. 测试性能对比
echo "8. 测试MobileFaceNet性能..."
curl -X POST "${BASE_URL}/detect" \
     -F "file=@${TEST_IMAGE}" | jq '.[] | {bbox, confidence}'

# 9. 切换回Buffalo模型
echo "9. 切换回Buffalo模型..."
curl -X POST "${BASE_URL}/models/switch" \
     -H "Content-Type: application/x-www-form-urlencoded" \
     -d "enable_arcface=false" | jq '.'

echo "✅ ArcFace 集成测试完成!"
```

#### Java Spring测试示例
```java
@TestMethodOrder(OrderAnnotation.class)
@SpringBootTest
public class ArcFaceIntegrationTest {
    
    @Autowired
    private FaceCVClient faceCVClient;
    
    @Test
    @Order(1)
    public void testSwitchToArcFace() {
        ModelSwitchResponse response = faceCVClient.switchToArcFace(true, "resnet50");
        
        assertThat(response.isSuccess()).isTrue();
        assertThat(response.getModelType()).isEqualTo("ArcFace");
        assertThat(response.getModelInfo().getBackbone()).isEqualTo("resnet50");
        assertThat(response.getModelInfo().getEmbeddingSize()).isEqualTo(512);
    }
    
    @Test
    @Order(2)
    public void testArcFaceDetection() throws IOException {
        MultipartFile testImage = createTestImage("trump1.jpeg");
        
        FaceDetectionResponse response = faceCVClient.detectFaces(testImage, 0.5);
        
        assertThat(response.isSuccess()).isTrue();
        assertThat(response.getFaces()).isNotEmpty();
        assertThat(response.getFaces().get(0).getConfidence()).isGreaterThan(0.8);
    }
    
    @Test
    @Order(3)
    public void testArcFaceRegistration() throws IOException {
        MultipartFile testImage = createTestImage("trump1.jpeg");
        
        FaceRegisterResponse response = faceCVClient.registerFace(
            testImage, "TestUser", "test_001", "default");
        
        assertThat(response.isSuccess()).isTrue();
        assertThat(response.getPersonName()).isEqualTo("TestUser");
        assertThat(response.getFaceId()).isNotNull();
    }
    
    @Test
    @Order(4)
    public void testArcFaceVerification() throws IOException {
        MultipartFile testImage1 = createTestImage("trump1.jpeg");
        MultipartFile testImage2 = createTestImage("trump2.jpeg");
        
        VerificationResponse response = faceCVClient.verifyFaces(
            testImage1, testImage2, 0.4);
        
        assertThat(response.isSuccess()).isTrue();
        assertThat(response.isVerified()).isTrue();
        assertThat(response.getSimilarity()).isGreaterThan(0.4);
    }
    
    @Test
    @Order(5)
    public void testModelPerformanceComparison() throws IOException {
        MultipartFile testImage = createTestImage("trump1.jpeg");
        
        // Test ArcFace ResNet50
        faceCVClient.switchToArcFace(true, "resnet50");
        long start1 = System.currentTimeMillis();
        FaceDetectionResponse arcfaceResponse = faceCVClient.detectFaces(testImage, 0.5);
        long arcfaceTime = System.currentTimeMillis() - start1;
        
        // Test ArcFace MobileFaceNet  
        faceCVClient.switchToArcFace(true, "mobilefacenet");
        long start2 = System.currentTimeMillis();
        FaceDetectionResponse mobileResponse = faceCVClient.detectFaces(testImage, 0.5);
        long mobileTime = System.currentTimeMillis() - start2;
        
        // Test Buffalo
        faceCVClient.switchToArcFace(false, null);
        long start3 = System.currentTimeMillis();
        FaceDetectionResponse buffaloResponse = faceCVClient.detectFaces(testImage, 0.5);
        long buffaloTime = System.currentTimeMillis() - start3;
        
        System.out.println("性能对比:");
        System.out.println("ArcFace ResNet50: " + arcfaceTime + "ms");
        System.out.println("ArcFace MobileFaceNet: " + mobileTime + "ms");
        System.out.println("Buffalo: " + buffaloTime + "ms");
        
        // All should detect faces successfully
        assertThat(arcfaceResponse.getFaces()).isNotEmpty();
        assertThat(mobileResponse.getFaces()).isNotEmpty();
        assertThat(buffaloResponse.getFaces()).isNotEmpty();
    }
    
    @Test
    @Order(6)
    public void testModelInfoEndpoint() {
        AvailableModelsResponse response = faceCVClient.getAvailableModels();
        
        assertThat(response.getAvailableModels()).isNotEmpty();
        assertThat(response.getArcfaceModels()).isNotEmpty();
        
        // Verify ArcFace models are listed
        assertThat(response.getArcfaceModels()).containsKey("buffalo_l_resnet50");
        assertThat(response.getArcfaceModels()).containsKey("buffalo_s_mobilefacenet");
        
        ArcFaceModelInfo resnet50Model = response.getArcfaceModels().get("buffalo_l_resnet50");
        assertThat(resnet50Model.getBackbone()).isEqualTo("resnet50");
        assertThat(resnet50Model.getEmbeddingSize()).isEqualTo("512D");
    }
}
```

### 2. 性能基准测试

#### 延迟测试
```bash
# 测试不同模型的推理延迟
echo "模型性能基准测试..."

MODELS=("buffalo_l" "arcface_resnet50" "arcface_mobilefacenet")
for model in "${MODELS[@]}"; do
    echo "Testing $model..."
    
    # 切换模型
    if [[ $model == "buffalo_l" ]]; then
        curl -X POST "${BASE_URL}/models/switch" \
             -H "Content-Type: application/x-www-form-urlencoded" \
             -d "enable_arcface=false"
    elif [[ $model == "arcface_resnet50" ]]; then
        curl -X POST "${BASE_URL}/models/switch" \
             -H "Content-Type: application/x-www-form-urlencoded" \
             -d "enable_arcface=true&arcface_backbone=resnet50"
    else
        curl -X POST "${BASE_URL}/models/switch" \
             -H "Content-Type: application/x-www-form-urlencoded" \
             -d "enable_arcface=true&arcface_backbone=mobilefacenet"
    fi
    
    # 预热
    for i in {1..3}; do
        curl -s -X POST "${BASE_URL}/detect" -F "file=@${TEST_IMAGE}" > /dev/null
    done
    
    # 性能测试
    echo "开始 $model 性能测试..."
    total_time=0
    for i in {1..10}; do
        start_time=$(date +%s%3N)
        curl -s -X POST "${BASE_URL}/detect" -F "file=@${TEST_IMAGE}" > /dev/null
        end_time=$(date +%s%3N)
        time_diff=$((end_time - start_time))
        total_time=$((total_time + time_diff))
        echo "  请求 $i: ${time_diff}ms"
    done
    
    avg_time=$((total_time / 10))
    echo "$model 平均延迟: ${avg_time}ms"
    echo "---"
done
```

### 3. 压力测试

#### 并发测试脚本
```python
import asyncio
import aiohttp
import time
from typing import List, Dict

async def test_concurrent_requests():
    """并发请求压力测试"""
    
    async def single_request(session: aiohttp.ClientSession, image_path: str) -> Dict:
        """单个请求"""
        start_time = time.time()
        
        with open(image_path, 'rb') as f:
            data = aiohttp.FormData()
            data.add_field('file', f, filename='test.jpg')
            
            async with session.post(
                'http://localhost:7000/api/v1/insightface/detect',
                data=data
            ) as response:
                result = await response.json()
                end_time = time.time()
                
                return {
                    'success': response.status == 200,
                    'latency': (end_time - start_time) * 1000,
                    'faces_detected': len(result) if isinstance(result, list) else 0
                }
    
    # 测试配置
    image_path = '/home/a/PycharmProjects/EurekCV/dataset/faces/trump1.jpeg'
    concurrent_levels = [1, 5, 10, 20, 50]
    
    for concurrency in concurrent_levels:
        print(f"\n🧪 测试并发级别: {concurrency}")
        
        async with aiohttp.ClientSession() as session:
            # 执行并发请求
            tasks = [single_request(session, image_path) for _ in range(concurrency)]
            start_time = time.time()
            results = await asyncio.gather(*tasks)
            total_time = time.time() - start_time
            
            # 统计结果
            successful_requests = sum(1 for r in results if r['success'])
            avg_latency = sum(r['latency'] for r in results) / len(results)
            max_latency = max(r['latency'] for r in results)
            min_latency = min(r['latency'] for r in results)
            
            print(f"  总请求: {len(results)}")
            print(f"  成功请求: {successful_requests}")
            print(f"  成功率: {successful_requests/len(results)*100:.1f}%")
            print(f"  总耗时: {total_time:.2f}s")
            print(f"  QPS: {len(results)/total_time:.1f}")
            print(f"  平均延迟: {avg_latency:.1f}ms")
            print(f"  最大延迟: {max_latency:.1f}ms")
            print(f"  最小延迟: {min_latency:.1f}ms")

if __name__ == "__main__":
    asyncio.run(test_concurrent_requests())
```

### 4. 集成测试报告

#### 预期测试结果
```
✅ ArcFace 集成测试结果 (示例)

模型切换测试:
- Buffalo → ArcFace ResNet50: ✅ 成功 (1.2s)
- ArcFace ResNet50 → MobileFaceNet: ✅ 成功 (0.8s)  
- ArcFace → Buffalo: ✅ 成功 (1.0s)

性能对比:
- ArcFace ResNet50: 平均 45ms, 精度 99.8%
- ArcFace MobileFaceNet: 平均 15ms, 精度 98.9%
- Buffalo L: 平均 38ms, 精度 99.6%

并发测试 (10并发):
- 成功率: 100%
- QPS: 22.5
- 平均延迟: 44ms
- P95延迟: 78ms

功能验证:
- 人脸检测: ✅ 置信度 > 0.8
- 人脸注册: ✅ 生成face_id
- 人脸验证: ✅ 相似度 > 0.47 (同一人)
- 模型信息: ✅ 返回完整元数据
```

---
