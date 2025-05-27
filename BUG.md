Please use /home/a/PycharmProjects/EurekCV/dataset/faces for testing, as well as this database

```
# db info
FACECV_MYSQL_HOST=eurekailab.mysql.rds.aliyuncs.com
FACECV_MYSQL_PORT=3306
FACECV_MYSQL_USER=root
FACECV_MYSQL_PASSWORD=Zsg20010115_
FACECV_MYSQL_DATABASE=facecv
FACECV_DB_POOL_SIZE=10
FACECV_DB_POOL_RECYCLE=3600
```

# DeepFace
1. GET  /api/v1/deepface/faces/  获取人脸列表

```
curl -X 'GET' \
  'http://127.0.0.1:7003/api/v1/deepface/faces/' \
  -H 'accept: application/json'
```
Request URL
```http://127.0.0.1:7003/api/v1/deepface/faces/```
Server response
Code	Details
```
500 Undocumented
Error: Internal Server Error

Response body
Download
{
  "detail": "服务器内部错误: Descriptors cannot be created directly.\nIf this call came from a _pb2.py file, your generated code is out of date and must be regenerated with protoc >= 3.19.0.\nIf you cannot immediately regenerate your protos, some other possible workarounds are:\n 1. Downgrade the protobuf package to 3.20.x or lower.\n 2. Set PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python (but this will use pure-Python parsing and will be much slower).\n\nMore information: https://developers.google.com/protocol-buffers/docs/news/2022-05-06#python-updates"
}
```

2. POST  /api/v1/deepface/faces/ 注册人脸
```
curl -X 'POST' \
  'http://127.0.0.1:7003/api/v1/deepface/faces/' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'name=string' \
  -F 'file=@2.jpg;type=image/jpeg' \
  -F 'metadata=string'
```

Error:
```
{
  "detail": "服务器内部错误: Descriptors cannot be created directly.\nIf this call came from a _pb2.py file, your generated code is out of date and must be regenerated with protoc >= 3.19.0.\nIf you cannot immediately regenerate your protos, some other possible workarounds are:\n 1. Downgrade the protobuf package to 3.20.x or lower.\n 2. Set PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python (but this will use pure-Python parsing and will be much slower).\n\nMore information: https://developers.google.com/protocol-buffers/docs/news/2022-05-06#python-updates"
}
```
3. ALL DEEPFACE RELATED HAVING SIMILAR PROBLEM

4. POST /api/v1/stream/process 处理视频流
Camera cannot be closed after start, expecting to be disconnected

5. POST /api/v1/camera/connect
Camera and SCRFD not closed after disconnect
Also why we have two camera different from 4?

6. Is that better combining /api/v1/camera/connect + /api/v1/camera/connect + /api/v1/camera/status in one api by webrtc(fastrtc repository, search it online)? Can we sending status to user realtime?

7. /api/v1/health/gpu and /api/v1/health/comprehensive having problem, not consistent with local gpu status, check todo in that file