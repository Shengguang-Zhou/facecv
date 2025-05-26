"""
Comprehensive DeepFace API Test Suite

Main test runner for all DeepFace API endpoints with organized test execution.
"""

import pytest
import sys
import os
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Import all test classes
from tests.api.deepface.test_face_management import TestFaceManagement
from tests.api.deepface.test_recognition_verification import TestRecognitionVerification
from tests.api.deepface.test_video_stream import TestVideoStream
from tests.api.deepface.test_health import TestDeepFaceHealth


class TestDeepFaceAPISuite:
    """DeepFace API完整测试套件"""
    
    def test_health_check_first(self, api_client):
        """首先测试健康检查"""
        from tests.conftest import DEEPFACE_BASE_URL, TIMEOUT
        
        response = api_client.get(f"{DEEPFACE_BASE_URL}/health", timeout=TIMEOUT)
        assert response.status_code == 200
        result = response.json()
        assert result["service"] == "DeepFace"
        
    def test_face_management_basic(self, api_client):
        """测试基本人脸管理功能"""
        from tests.conftest import DEEPFACE_BASE_URL, TIMEOUT
        
        # 测试获取人脸列表
        response = api_client.get(f"{DEEPFACE_BASE_URL}/faces/", timeout=TIMEOUT)
        assert response.status_code == 200
        result = response.json()
        assert "faces" in result
        assert "total" in result
        
    def test_recognition_basic(self, api_client):
        """测试基本识别功能"""
        from tests.conftest import DEEPFACE_BASE_URL, TIMEOUT
        from PIL import Image
        import io
        
        # 创建测试图像
        image = Image.new('RGB', (100, 100), color='white')
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG')
        image_data = buffer.getvalue()
        
        files = {"file": ("test.jpg", image_data, "image/jpeg")}
        data = {"threshold": 0.6}
        
        response = api_client.post(
            f"{DEEPFACE_BASE_URL}/recognition",
            files=files,
            data=data,
            timeout=TIMEOUT
        )
        
        assert response.status_code == 200
        result = response.json()
        assert "faces" in result
        
    def test_comprehensive_workflow(self, api_client):
        """测试综合工作流"""
        from tests.conftest import DEEPFACE_BASE_URL, TIMEOUT
        
        # 简单的工作流测试
        health_response = api_client.get(f"{DEEPFACE_BASE_URL}/health", timeout=TIMEOUT)
        assert health_response.status_code == 200
        
        faces_response = api_client.get(f"{DEEPFACE_BASE_URL}/faces/", timeout=TIMEOUT)
        assert faces_response.status_code == 200


def run_all_deepface_tests():
    """运行所有DeepFace API测试"""
    
    print("=" * 60)
    print("🚀 Starting DeepFace API Test Suite")
    print("=" * 60)
    
    # 定义测试模块
    test_modules = [
        "tests/api/deepface/test_health.py",
        "tests/api/deepface/test_face_management.py", 
        "tests/api/deepface/test_recognition_verification.py",
        "tests/api/deepface/test_video_stream.py"
    ]
    
    # 构建pytest参数
    pytest_args = [
        "-v",  # 详细输出
        "-s",  # 不捕获输出
        "--tb=short",  # 简短的错误追踪
        "--maxfail=10",  # 最多失败10个测试后停止
        "-x",  # 遇到第一个失败就停止
        "--durations=10",  # 显示最慢的10个测试
    ]
    
    # 添加测试文件
    pytest_args.extend(test_modules)
    
    print(f"Running pytest with args: {' '.join(pytest_args)}")
    print("-" * 60)
    
    # 运行测试
    exit_code = pytest.main(pytest_args)
    
    print("-" * 60)
    if exit_code == 0:
        print("✅ All DeepFace API tests passed!")
    else:
        print(f"❌ Some tests failed. Exit code: {exit_code}")
    print("=" * 60)
    
    return exit_code


def run_quick_smoke_tests():
    """运行快速冒烟测试"""
    
    print("🔥 Running Quick Smoke Tests for DeepFace API")
    print("-" * 50)
    
    pytest_args = [
        "-v",
        "-k", "health or test_register_face_success or test_list_faces_empty",  # 只运行关键测试
        "--tb=short",
        "tests/api/deepface/"
    ]
    
    exit_code = pytest.main(pytest_args)
    
    if exit_code == 0:
        print("✅ Smoke tests passed!")
    else:
        print("❌ Smoke tests failed!")
    
    return exit_code


def run_stress_tests():
    """运行压力测试"""
    
    print("💪 Running Stress Tests for DeepFace API")
    print("-" * 50)
    
    pytest_args = [
        "-v",
        "-k", "concurrent or performance or large",  # 运行压力相关测试
        "--tb=short",
        "tests/api/deepface/"
    ]
    
    exit_code = pytest.main(pytest_args)
    
    if exit_code == 0:
        print("✅ Stress tests passed!")
    else:
        print("❌ Some stress tests failed!")
    
    return exit_code


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="DeepFace API Test Runner")
    parser.add_argument(
        "--mode", 
        choices=["all", "smoke", "stress", "quick"], 
        default="all",
        help="Test mode to run"
    )
    parser.add_argument(
        "--api-url",
        default="http://localhost:7003",
        help="API base URL"
    )
    
    args = parser.parse_args()
    
    # 设置环境变量
    os.environ["API_BASE_URL"] = args.api_url
    
    print(f"🎯 Target API: {args.api_url}")
    print(f"📋 Test Mode: {args.mode}")
    print()
    
    # 根据模式运行测试
    if args.mode == "smoke" or args.mode == "quick":
        exit_code = run_quick_smoke_tests()
    elif args.mode == "stress":
        exit_code = run_stress_tests()
    else:  # all
        exit_code = run_all_deepface_tests()
    
    sys.exit(exit_code)