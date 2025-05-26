"""测试 FaceCV API 端点"""

import requests
import numpy as np
from PIL import Image
import io
import time
import subprocess
import sys
import os

API_BASE = "http://localhost:8000/api/v1"

def create_test_image(name: str = "test.jpg") -> bytes:
    """创建测试图片"""
    # 创建一个随机图片
    img_array = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    img = Image.fromarray(img_array)
    
    # 保存到字节流
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='JPEG')
    img_bytes.seek(0)
    
    return img_bytes.getvalue()

def test_api_endpoints():
    """测试所有 API 端点"""
    print("=== FaceCV API 测试 ===\n")
    
    # 1. 测试健康检查
    print("1. 测试健康检查...")
    try:
        resp = requests.get("http://localhost:8000/health")
        assert resp.status_code == 200
        print(f"✓ 健康检查: {resp.json()['status']}")
    except Exception as e:
        print(f"✗ 健康检查失败: {e}")
        return
    
    # 2. 测试人脸计数
    print("\n2. 测试人脸计数...")
    try:
        resp = requests.get(f"{API_BASE}/faces/count")
        assert resp.status_code == 200
        count = resp.json()['total']
        print(f"✓ 当前人脸数量: {count}")
    except Exception as e:
        print(f"✗ 人脸计数失败: {e}")
    
    # 3. 测试人脸注册
    print("\n3. 测试人脸注册...")
    try:
        # 准备测试数据
        files = {'file': ('test1.jpg', create_test_image(), 'image/jpeg')}
        data = {
            'name': '测试用户1',
            'department': '技术部',
            'employee_id': 'TEST001'
        }
        
        resp = requests.post(f"{API_BASE}/faces/register", files=files, data=data)
        assert resp.status_code == 200
        face_ids = resp.json()
        print(f"✓ 注册成功，Face IDs: {face_ids}")
        
        # 注册第二个人
        files = {'file': ('test2.jpg', create_test_image(), 'image/jpeg')}
        data = {'name': '测试用户2', 'department': '市场部'}
        
        resp = requests.post(f"{API_BASE}/faces/register", files=files, data=data)
        assert resp.status_code == 200
        print(f"✓ 注册第二个用户成功")
        
    except Exception as e:
        print(f"✗ 人脸注册失败: {e}")
        if 'resp' in locals():
            print(f"  响应: {resp.text}")
    
    # 4. 测试人脸列表
    print("\n4. 测试人脸列表...")
    try:
        resp = requests.get(f"{API_BASE}/faces")
        assert resp.status_code == 200
        faces = resp.json()
        print(f"✓ 获取人脸列表成功，共 {len(faces)} 个人脸")
        for face in faces[:3]:  # 显示前3个
            print(f"  - {face['name']} (ID: {face['id']})")
    except Exception as e:
        print(f"✗ 获取人脸列表失败: {e}")
    
    # 5. 测试人脸识别
    print("\n5. 测试人脸识别...")
    try:
        files = {'file': ('test_recognize.jpg', create_test_image(), 'image/jpeg')}
        params = {'threshold': 0.5}
        
        resp = requests.post(f"{API_BASE}/faces/recognize", files=files, params=params)
        assert resp.status_code == 200
        results = resp.json()
        print(f"✓ 识别成功，识别到 {len(results)} 个人脸")
        if results:
            result = results[0]
            print(f"  - 姓名: {result['recognized_name']}")
            print(f"  - 相似度: {result['similarity_score']:.3f}")
    except Exception as e:
        print(f"✗ 人脸识别失败: {e}")
    
    # 6. 测试人脸验证
    print("\n6. 测试人脸验证...")
    try:
        files = {
            'file1': ('verify1.jpg', create_test_image(), 'image/jpeg'),
            'file2': ('verify2.jpg', create_test_image(), 'image/jpeg')
        }
        params = {'threshold': 0.6}
        
        resp = requests.post(f"{API_BASE}/faces/verify", files=files, params=params)
        assert resp.status_code == 200
        result = resp.json()
        print(f"✓ 验证成功")
        print(f"  - 是否同一人: {'是' if result['is_same_person'] else '否'}")
        print(f"  - 相似度: {result['similarity_score']:.3f}")
        print(f"  - 阈值: {result['threshold']}")
    except Exception as e:
        print(f"✗ 人脸验证失败: {e}")
    
    # 7. 测试删除功能
    print("\n7. 测试删除功能...")
    try:
        # 先获取一个人脸 ID
        resp = requests.get(f"{API_BASE}/faces", params={'name': '测试用户1'})
        if resp.status_code == 200 and resp.json():
            face_id = resp.json()[0]['id']
            
            # 删除该人脸
            resp = requests.delete(f"{API_BASE}/faces/{face_id}")
            assert resp.status_code == 200
            print(f"✓ 成功删除人脸 ID: {face_id}")
        
        # 按姓名删除
        resp = requests.delete(f"{API_BASE}/faces/by-name/测试用户2")
        if resp.status_code == 200:
            result = resp.json()
            print(f"✓ 成功删除 {result['name']} 的所有人脸")
    except Exception as e:
        print(f"✗ 删除功能测试失败: {e}")
    
    # 8. 最终检查
    print("\n8. 最终检查...")
    try:
        resp = requests.get(f"{API_BASE}/faces/count")
        assert resp.status_code == 200
        final_count = resp.json()['total']
        print(f"✓ 最终人脸数量: {final_count}")
    except Exception as e:
        print(f"✗ 最终检查失败: {e}")
    
    print("\n=== API 测试完成 ===")

def main():
    # 启动 API 服务
    print("启动 FaceCV API 服务...")
    process = subprocess.Popen(
        [sys.executable, "/home/a/PycharmProjects/facecv/main.py"],
        env={**os.environ, "PYTHONPATH": "/home/a/PycharmProjects/facecv"}
    )
    
    # 等待服务启动
    print("等待服务启动...")
    time.sleep(3)
    
    try:
        # 运行测试
        test_api_endpoints()
    finally:
        # 关闭服务
        print("\n关闭 API 服务...")
        process.terminate()
        process.wait()

if __name__ == "__main__":
    main()