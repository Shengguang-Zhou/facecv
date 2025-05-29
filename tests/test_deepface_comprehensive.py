#!/usr/bin/env python3
"""
全面测试 DeepFace API 功能
"""

import requests
import time

base_url = "http://localhost:7003/api/v1/deepface"

print("="*60)
print("DeepFace API 全面测试")
print("="*60)

# 1. 健康检查
print("\n1. 健康检查...")
try:
    response = requests.get(f"{base_url}/health")
    print(f"   状态码: {response.status_code}")
    print(f"   响应: {response.json()}")
except Exception as e:
    print(f"   错误: {e}")

# 2. 注册人脸
print("\n2. 注册人脸...")
try:
    with open('../test_face.jpg', 'rb') as f:
        files = {'file': ('test.jpg', f, 'image/jpeg')}
        data = {
            'name': '测试用户1',
            'metadata': '{"department": "技术部", "employee_id": "DF001"}'
        }
        response = requests.post(f"{base_url}/faces/", files=files, data=data)
        print(f"   状态码: {response.status_code}")
        print(f"   响应: {response.json()}")
except Exception as e:
    print(f"   错误: {e}")

# 3. 获取人脸列表
print("\n3. 获取人脸列表...")
try:
    response = requests.get(f"{base_url}/faces/")
    print(f"   状态码: {response.status_code}")
    data = response.json()
    print(f"   总数: {data.get('total', 0)}")
    if data.get('faces'):
        for face in data['faces']:
            print(f"   - {face['person_name']} (ID: {face['face_id']})")
except Exception as e:
    print(f"   错误: {e}")

# 4. 人脸识别
print("\n4. 人脸识别...")
try:
    with open('../test_face.jpg', 'rb') as f:
        files = {'file': ('test.jpg', f, 'image/jpeg')}
        data = {'threshold': '0.6'}
        response = requests.post(f"{base_url}/recognition", files=files, data=data)
        print(f"   状态码: {response.status_code}")
        result = response.json()
        if result.get('faces'):
            for face in result['faces']:
                print(f"   - 识别为: {face['person_name']} (置信度: {face['confidence']})")
        else:
            print("   - 未识别到任何人脸")
except Exception as e:
    print(f"   错误: {e}")

# 5. 人脸验证（使用同一张图片模拟）
print("\n5. 人脸验证...")
try:
    with open('../test_face.jpg', 'rb') as f1:
        with open('../test_face.jpg', 'rb') as f2:
            files = {
                'file1': ('test1.jpg', f1, 'image/jpeg'),
                'file2': ('test2.jpg', f2, 'image/jpeg')
            }
            data = {
                'threshold': '0.6',
                'model_name': 'ArcFace'
            }
            response = requests.post(f"{base_url}/verify/", files=files, data=data)
            print(f"   状态码: {response.status_code}")
            if response.status_code == 200:
                result = response.json()
                print(f"   验证结果: {'是同一人' if result['verified'] else '不是同一人'}")
                print(f"   置信度: {result.get('confidence', 'N/A')}")
except Exception as e:
    print(f"   错误: {e}")

# 6. 人脸属性分析
print("\n6. 人脸属性分析...")
try:
    with open('../test_face.jpg', 'rb') as f:
        files = {'file': ('test.jpg', f, 'image/jpeg')}
        data = {
            'actions': 'emotion,age,gender,race',
            'detector_backend': 'mtcnn'
        }
        response = requests.post(f"{base_url}/analyze/", files=files, data=data)
        print(f"   状态码: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"   检测到 {result.get('total_faces', 0)} 张人脸")
            # Mock 实现可能返回简单结果
except Exception as e:
    print(f"   错误: {e}")

# 7. 按姓名查询
print("\n7. 按姓名查询人脸...")
try:
    response = requests.get(f"{base_url}/faces/name/测试用户1")
    print(f"   状态码: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"   找到 {data.get('total', 0)} 条记录")
except Exception as e:
    print(f"   错误: {e}")

print("\n" + "="*60)
print("测试完成！")
print("="*60)