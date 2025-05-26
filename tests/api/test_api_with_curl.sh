#!/bin/bash

# FaceCV API 测试脚本
# 使用 curl 测试所有 API 端点

API_BASE="http://localhost:8000/api/v1"
TEST_IMAGE1="/tmp/test_face1.jpg"
TEST_IMAGE2="/tmp/test_face2.jpg"

# 创建测试图片
echo "创建测试图片..."
python3 -c "
import numpy as np
from PIL import Image
# 创建第一张测试图片
img1 = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
Image.fromarray(img1).save('$TEST_IMAGE1')
# 创建第二张测试图片
img2 = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
Image.fromarray(img2).save('$TEST_IMAGE2')
print('测试图片创建成功')
"

echo -e "\n=== FaceCV API 测试 ===\n"

# 1. 健康检查
echo "1. 测试健康检查"
curl -X GET http://localhost:8000/health
echo -e "\n"

# 2. 获取人脸数量
echo "2. 获取人脸数量"
curl -X GET $API_BASE/faces/count
echo -e "\n"

# 3. 注册人脸
echo "3. 注册人脸 - 张三"
FACE_ID1=$(curl -X POST $API_BASE/faces/register \
  -F "name=张三" \
  -F "file=@$TEST_IMAGE1" \
  -F "department=技术部" \
  -F "employee_id=E001" \
  -s | python3 -c "import sys, json; print(json.load(sys.stdin)[0])")
echo "注册成功，Face ID: $FACE_ID1"

echo -e "\n注册人脸 - 李四"
FACE_ID2=$(curl -X POST $API_BASE/faces/register \
  -F "name=李四" \
  -F "file=@$TEST_IMAGE2" \
  -F "department=市场部" \
  -F "employee_id=E002" \
  -s | python3 -c "import sys, json; print(json.load(sys.stdin)[0])")
echo "注册成功，Face ID: $FACE_ID2"

# 4. 列出所有人脸
echo -e "\n4. 列出所有人脸"
curl -X GET $API_BASE/faces | python3 -m json.tool

# 5. 按姓名查询人脸
echo -e "\n5. 按姓名查询人脸 - 张三"
curl -X GET "$API_BASE/faces?name=张三" | python3 -m json.tool

# 6. 识别人脸
echo -e "\n6. 识别人脸"
curl -X POST $API_BASE/faces/recognize \
  -F "file=@$TEST_IMAGE1" \
  -F "threshold=0.5" | python3 -m json.tool

# 7. 验证两张人脸
echo -e "\n7. 验证两张人脸是否为同一人"
curl -X POST $API_BASE/faces/verify \
  -F "file1=@$TEST_IMAGE1" \
  -F "file2=@$TEST_IMAGE2" \
  -F "threshold=0.6" | python3 -m json.tool

# 8. 删除人脸 (按 ID)
echo -e "\n8. 删除人脸 - ID: $FACE_ID1"
curl -X DELETE $API_BASE/faces/$FACE_ID1
echo -e "\n"

# 9. 删除人脸 (按姓名)
echo "9. 删除所有李四的人脸"
curl -X DELETE $API_BASE/faces/by-name/李四
echo -e "\n"

# 10. 最终检查
echo "10. 最终人脸数量"
curl -X GET $API_BASE/faces/count
echo -e "\n"

# 清理测试文件
rm -f $TEST_IMAGE1 $TEST_IMAGE2

echo -e "\n=== 测试完成 ==="