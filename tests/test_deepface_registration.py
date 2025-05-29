#!/usr/bin/env python3
"""测试 DeepFace 注册功能"""

import requests
import os
import sys

# 设置测试图片路径
test_image = "test_face.jpg"

if not os.path.exists(test_image):
    print(f"测试图片不存在: {test_image}")
    sys.exit(1)

# API 端点
base_url = "http://localhost:7003/api/v1/deepface"

# 测试注册
print("测试 DeepFace 注册...")
with open(test_image, 'rb') as f:
    files = {'file': ('test.jpg', f, 'image/jpeg')}
    data = {
        'name': '金洁芝',
        'metadata': '{"department": "测试部", "employee_id": "DF001"}'
    }
    
    response = requests.post(f"{base_url}/faces/", files=files, data=data)
    
    print(f"状态码: {response.status_code}")
    print(f"响应: {response.text}")
    
    if response.status_code == 200:
        print("✅ 注册成功!")
        
        # 测试获取人脸列表
        print("\n测试获取人脸列表...")
        list_response = requests.get(f"{base_url}/faces/")
        print(f"状态码: {list_response.status_code}")
        print(f"响应: {list_response.json()}")
    else:
        print("❌ 注册失败!")