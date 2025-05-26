#!/bin/bash

# FaceCV API 实际测试
API_BASE="http://localhost:8000/api/v1"

echo "=== FaceCV API 功能测试 ==="
echo ""

# 1. 健康检查
echo "1. 健康检查"
curl -s http://localhost:8000/health | python3 -m json.tool
echo ""

# 2. 获取当前人脸数量
echo "2. 当前人脸数量"
curl -s $API_BASE/faces/count
echo -e "\n"

# 3. 查看 API 文档
echo "3. API 文档地址: http://localhost:8000/docs"
echo ""

# 4. 测试错误处理
echo "4. 测试错误处理 - 删除不存在的人脸"
curl -s -X DELETE $API_BASE/faces/non-existent-id
echo -e "\n"

echo "=== 测试完成 ==="
echo ""
echo "说明："
echo "- 由于没有真实的人脸图片，无法测试注册和识别功能"
echo "- 要测试完整功能，请使用包含人脸的 JPG/PNG 图片"
echo "- 使用方法："
echo "  curl -X POST $API_BASE/faces/register -F 'name=姓名' -F 'file=@/path/to/face.jpg'"
echo "  curl -X POST $API_BASE/faces/recognize -F 'file=@/path/to/face.jpg'"