# FaceCV API 测试指南

## API 服务状态

✅ **已实现的功能：**
1. 图片人脸注册
2. 图片人脸识别
3. 图片人脸验证
4. 人脸管理（列表、查询、删除）
5. 视频流处理（摄像头/RTSP）
6. WebSocket 实时流处理

## 测试 API 端点

### 1. 健康检查
```bash
curl http://localhost:8000/health
```

### 2. 查看 API 文档
浏览器访问: http://localhost:8000/docs

### 3. 人脸注册
```bash
# 注册人脸（需要真实的人脸图片）
curl -X POST http://localhost:8000/api/v1/faces/register \
  -F "name=张三" \
  -F "file=@/path/to/face.jpg" \
  -F "department=技术部" \
  -F "employee_id=E001"
```

### 4. 人脸识别
```bash
# 识别图片中的人脸
curl -X POST http://localhost:8000/api/v1/faces/recognize \
  -F "file=@/path/to/test.jpg" \
  -F "threshold=0.6"
```

### 5. 人脸验证
```bash
# 验证两张人脸是否为同一人
curl -X POST http://localhost:8000/api/v1/faces/verify \
  -F "file1=@/path/to/face1.jpg" \
  -F "file2=@/path/to/face2.jpg" \
  -F "threshold=0.6"
```

### 6. 人脸列表
```bash
# 列出所有人脸
curl http://localhost:8000/api/v1/faces

# 按姓名查询
curl "http://localhost:8000/api/v1/faces?name=张三"
```

### 7. 删除人脸
```bash
# 按ID删除
curl -X DELETE http://localhost:8000/api/v1/faces/{face_id}

# 按姓名删除所有
curl -X DELETE http://localhost:8000/api/v1/faces/by-name/张三
```

### 8. 获取人脸数量
```bash
curl http://localhost:8000/api/v1/faces/count
```

## 视频流处理

### 1. 查看支持的视频源
```bash
curl http://localhost:8000/api/v1/stream/sources
```

### 2. 处理视频流（摄像头）
```bash
# 处理默认摄像头10秒
curl -X POST http://localhost:8000/api/v1/stream/process \
  -H "Content-Type: application/json" \
  -d '{
    "source": "0",
    "duration": 10,
    "skip_frames": 2
  }'
```

### 3. 处理RTSP流
```bash
# 处理网络摄像头
curl -X POST http://localhost:8000/api/v1/stream/process \
  -H "Content-Type: application/json" \
  -d '{
    "source": "rtsp://192.168.1.100:554/stream",
    "duration": 30,
    "skip_frames": 3
  }'
```

### 4. WebSocket 实时处理
```javascript
// JavaScript 示例
const ws = new WebSocket('ws://localhost:8000/api/v1/stream/ws');

ws.onopen = () => {
    // 开始处理
    ws.send(JSON.stringify({
        action: 'start',
        source: '0',
        skip_frames: 2
    }));
};

ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    if (data.type === 'recognition') {
        console.log('识别结果:', data.faces);
    }
};

// 停止处理
ws.send(JSON.stringify({ action: 'stop' }));
```

## Python 客户端示例

```python
import requests
import json

# API基础URL
API_BASE = "http://localhost:8000/api/v1"

# 注册人脸
def register_face(image_path, name, metadata=None):
    files = {'file': open(image_path, 'rb')}
    data = {'name': name}
    if metadata:
        data.update(metadata)
    
    response = requests.post(f"{API_BASE}/faces/register", files=files, data=data)
    return response.json()

# 识别人脸
def recognize_face(image_path, threshold=0.6):
    files = {'file': open(image_path, 'rb')}
    params = {'threshold': threshold}
    
    response = requests.post(f"{API_BASE}/faces/recognize", files=files, params=params)
    return response.json()

# 处理视频流
def process_video_stream(source, duration=None):
    data = {
        'source': source,
        'duration': duration,
        'skip_frames': 2
    }
    
    response = requests.post(f"{API_BASE}/stream/process", json=data)
    return response.json()

# 使用示例
if __name__ == "__main__":
    # 注册人脸
    face_ids = register_face("path/to/face.jpg", "张三", {"department": "技术部"})
    print(f"注册成功: {face_ids}")
    
    # 识别人脸
    results = recognize_face("path/to/test.jpg")
    for face in results:
        print(f"识别到: {face['recognized_name']} (相似度: {face['similarity_score']})")
    
    # 处理摄像头
    stream_result = process_video_stream("0", duration=10)
    print(f"处理完成: 检测到 {stream_result['total_detections']} 个人脸")
```

## 注意事项

1. **图片要求**：
   - 支持格式：JPG, JPEG, PNG, BMP
   - 最大文件大小：10MB
   - 建议分辨率：640x480 以上

2. **视频流支持**：
   - 本地摄像头：使用索引 "0", "1" 等
   - RTSP流：需要提供完整的RTSP URL
   - 视频文件：支持 MP4, AVI, MOV, MKV

3. **性能优化**：
   - skip_frames：跳帧处理可以提高性能
   - 批量处理：一次识别多个人脸
   - GPU加速：如果有CUDA支持会自动启用

4. **当前限制**：
   - 使用模拟识别器（需要安装真实的InsightFace）
   - WebSocket仅支持单个连接
   - 视频预览仅在本地有效

## 故障排查

1. **无法打开摄像头**：
   - 检查摄像头权限
   - 确认摄像头未被其他程序占用
   - 尝试使用不同的索引

2. **RTSP连接失败**：
   - 检查网络连接
   - 验证RTSP URL格式
   - 确认用户名密码正确

3. **识别率低**：
   - 调整相似度阈值
   - 确保图片质量良好
   - 增加同一人的多张注册照片