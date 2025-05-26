"""测试API启动"""

import requests
import time
import subprocess
import sys
import os

# 启动API服务
print("启动 FaceCV API 服务...")
process = subprocess.Popen(
    [sys.executable, "/home/a/PycharmProjects/facecv/main.py"],
    env={**os.environ, "PYTHONPATH": "/home/a/PycharmProjects/facecv"}
)

# 等待服务启动
time.sleep(3)

try:
    # 测试健康检查端点
    response = requests.get("http://localhost:8000/health")
    print(f"健康检查响应: {response.json()}")
    
    # 测试根路径
    response = requests.get("http://localhost:8000/")
    print(f"根路径响应: {response.json()}")
    
    print("\nAPI 服务启动成功！")
    print("文档地址: http://localhost:8000/docs")
    
except Exception as e:
    print(f"测试失败: {e}")
finally:
    # 关闭服务
    process.terminate()
    process.wait()