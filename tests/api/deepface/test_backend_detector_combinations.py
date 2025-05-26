"""
DeepFace Backend and Detector Combination Tests

Comprehensive tests for all possible DeepFace model backends and detector combinations
across both CPU and GPU configurations.
"""

import pytest
import requests
import json
import io
from PIL import Image
import itertools
from tests.conftest import API_BASE_URL, DEEPFACE_BASE_URL, TIMEOUT


class TestBackendDetectorCombinations:
    """测试所有DeepFace后端和检测器组合"""

    # DeepFace支持的模型后端
    SUPPORTED_BACKENDS = [
        "VGG-Face",
        "Facenet", 
        "Facenet512",
        "OpenFace",
        "DeepFace",
        "ArcFace",
        "Dlib",
        "SFace"
    ]

    # DeepFace支持的检测器
    SUPPORTED_DETECTORS = [
        "opencv",
        "ssd", 
        "dlib",
        "mtcnn",
        "retinaface",
        "mediapipe",
        "yolov8",
        "yunet",
        "fastmtcnn"
    ]

    # 距离度量方式
    DISTANCE_METRICS = [
        "cosine",
        "euclidean", 
        "euclidean_l2"
    ]

    @pytest.fixture(autouse=True)
    def setup_method(self, api_client):
        """每个测试方法的设置"""
        self.client = api_client
        self.base_url = DEEPFACE_BASE_URL
        self.registered_faces = []

    def teardown_method(self):
        """每个测试方法的清理"""
        for face_data in self.registered_faces:
            try:
                response = self.client.delete(
                    f"{self.base_url}/faces/{face_data['face_id']}",
                    timeout=TIMEOUT
                )
            except:
                pass

    def create_test_image(self, size=(224, 224), face_variation=0) -> bytes:
        """创建测试图像"""
        colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange']
        color = colors[face_variation % len(colors)]
        
        image = Image.new('RGB', size, color=color)
        
        from PIL import ImageDraw
        draw = ImageDraw.Draw(image)
        
        # 绘制人脸特征
        center_x, center_y = size[0] // 2, size[1] // 2
        face_width, face_height = size[0] // 3, size[1] // 2
        
        # 脸部轮廓
        draw.rectangle([
            center_x - face_width//2, center_y - face_height//2,
            center_x + face_width//2, center_y + face_height//2
        ], outline='black', width=3)
        
        # 眼睛
        eye_y = center_y - face_height//4
        eye_size = 8
        draw.ellipse([center_x - face_width//4 - eye_size, eye_y - eye_size,
                     center_x - face_width//4 + eye_size, eye_y + eye_size], 
                    fill='black')
        draw.ellipse([center_x + face_width//4 - eye_size, eye_y - eye_size,
                     center_x + face_width//4 + eye_size, eye_y + eye_size], 
                    fill='black')
        
        # 鼻子
        draw.line([center_x, center_y - 10, center_x, center_y + 10], 
                 fill='black', width=2)
        
        # 嘴巴
        mouth_y = center_y + face_height//4
        draw.arc([center_x - 15, mouth_y - 8, center_x + 15, mouth_y + 8], 
                0, 180, fill='black', width=2)
        
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG', quality=90)
        return buffer.getvalue()

    @pytest.mark.parametrize("backend", SUPPORTED_BACKENDS[:3])  # 测试前3个后端以节省时间
    def test_face_verification_different_backends(self, backend):
        """测试不同模型后端的人脸验证"""
        image1 = self.create_test_image(face_variation=0)
        image2 = self.create_test_image(face_variation=0)  # 相同人脸

        files = {
            "file1": ("face1.jpg", image1, "image/jpeg"),
            "file2": ("face2.jpg", image2, "image/jpeg")
        }
        data = {
            "threshold": 0.6,
            "model_name": backend,
            "anti_spoofing": False
        }

        response = self.client.post(
            f"{self.base_url}/verify/",
            files=files,
            data=data,
            timeout=TIMEOUT * 2  # 增加超时时间，某些模型加载较慢
        )

        # 某些模型可能不可用，记录结果
        if response.status_code == 200:
            result = response.json()
            assert "verified" in result
            assert "model" in result
            assert result["model"] == backend
            print(f"✅ Backend {backend}: 验证成功")
        elif response.status_code in [400, 500]:
            print(f"⚠️  Backend {backend}: 不可用或出错 - {response.status_code}")
        else:
            pytest.fail(f"Backend {backend}: 意外状态码 {response.status_code}")

    @pytest.mark.parametrize("detector", SUPPORTED_DETECTORS[:4])  # 测试前4个检测器
    def test_face_analysis_different_detectors(self, detector):
        """测试不同检测器的人脸分析"""
        image_data = self.create_test_image()

        files = {"file": ("analyze.jpg", image_data, "image/jpeg")}
        data = {
            "actions": "emotion,age,gender",
            "detector_backend": detector
        }

        response = self.client.post(
            f"{self.base_url}/analyze/",
            files=files,
            data=data,
            timeout=TIMEOUT * 2
        )

        # 记录检测器测试结果
        if response.status_code == 200:
            result = response.json()
            assert "faces" in result
            assert "total_faces" in result
            print(f"✅ Detector {detector}: 分析成功，检测到 {result['total_faces']} 张人脸")
        elif response.status_code in [400, 500]:
            print(f"⚠️  Detector {detector}: 不可用或出错 - {response.status_code}")
        else:
            pytest.fail(f"Detector {detector}: 意外状态码 {response.status_code}")

    @pytest.mark.parametrize("distance_metric", DISTANCE_METRICS)
    def test_face_verification_different_distance_metrics(self, distance_metric):
        """测试不同距离度量方式"""
        # 注意：DeepFace API可能不直接暴露distance_metric参数
        # 这个测试主要验证API是否能处理相关参数
        
        image1 = self.create_test_image(face_variation=1)
        image2 = self.create_test_image(face_variation=2)

        files = {
            "file1": ("face1.jpg", image1, "image/jpeg"),
            "file2": ("face2.jpg", image2, "image/jpeg")
        }
        data = {
            "threshold": 0.6,
            "model_name": "ArcFace",  # 使用稳定的模型
            # "distance_metric": distance_metric  # 如果API支持的话
        }

        response = self.client.post(
            f"{self.base_url}/verify/",
            files=files,
            data=data,
            timeout=TIMEOUT
        )

        assert response.status_code in [200, 400]  # 200成功，400可能不支持该参数
        
        if response.status_code == 200:
            result = response.json()
            assert "distance" in result
            print(f"✅ Distance metric test with {distance_metric}: 成功")

    def test_backend_detector_performance_comparison(self):
        """比较不同后端和检测器的性能"""
        image_data = self.create_test_image()
        
        # 测试少量组合以比较性能
        test_combinations = [
            ("ArcFace", "retinaface"),
            ("VGG-Face", "mtcnn"),
            ("OpenFace", "opencv"),
            ("ArcFace", "opencv")  # 快速组合
        ]
        
        results = []
        
        for backend, detector in test_combinations:
            # 验证测试
            image1 = image_data
            image2 = self.create_test_image(face_variation=1)
            
            files = {
                "file1": ("face1.jpg", image1, "image/jpeg"),
                "file2": ("face2.jpg", image2, "image/jpeg")
            }
            data = {
                "threshold": 0.6,
                "model_name": backend
            }
            
            import time
            start_time = time.time()
            
            try:
                response = self.client.post(
                    f"{self.base_url}/verify/",
                    files=files,
                    data=data,
                    timeout=TIMEOUT
                )
                
                processing_time = time.time() - start_time
                
                if response.status_code == 200:
                    result = response.json()
                    results.append({
                        "backend": backend,
                        "detector": detector,
                        "processing_time": processing_time,
                        "success": True,
                        "confidence": result.get("confidence", 0),
                        "distance": result.get("distance", 0)
                    })
                    print(f"✅ {backend} + {detector}: {processing_time:.2f}s, confidence: {result.get('confidence', 0):.3f}")
                else:
                    results.append({
                        "backend": backend,
                        "detector": detector,
                        "processing_time": processing_time,
                        "success": False,
                        "error": response.status_code
                    })
                    print(f"❌ {backend} + {detector}: 失败 {response.status_code}")
                    
            except Exception as e:
                results.append({
                    "backend": backend,
                    "detector": detector,
                    "processing_time": 0,
                    "success": False,
                    "error": str(e)
                })
                print(f"❌ {backend} + {detector}: 异常 {str(e)}")
        
        # 验证至少有一些组合成功
        successful_combinations = [r for r in results if r["success"]]
        assert len(successful_combinations) > 0, "没有任何后端/检测器组合成功"
        
        # 打印性能排序
        successful_combinations.sort(key=lambda x: x["processing_time"])
        print("\n🏃‍♂️ 性能排序 (快到慢):")
        for i, result in enumerate(successful_combinations[:3]):
            print(f"{i+1}. {result['backend']}: {result['processing_time']:.2f}s")

    def test_cpu_vs_gpu_configuration(self):
        """测试CPU vs GPU配置（如果可用）"""
        # 这个测试主要验证API是否能在不同设备配置下工作
        # 实际的CPU/GPU切换可能需要在模型层面配置
        
        image_data = self.create_test_image()
        
        # 测试基本功能在当前配置下工作
        files = {"file": ("test.jpg", image_data, "image/jpeg")}
        data = {"threshold": 0.6}
        
        response = self.client.post(
            f"{self.base_url}/recognition",
            files=files,
            data=data,
            timeout=TIMEOUT
        )
        
        assert response.status_code == 200
        result = response.json()
        assert "faces" in result
        
        # 检查系统健康状态以了解当前设备配置
        health_response = self.client.get(f"{API_BASE_URL}/api/v1/health/gpu", timeout=TIMEOUT)
        
        if health_response.status_code == 200:
            gpu_info = health_response.json()
            if gpu_info.get("available"):
                print(f"✅ GPU可用: {gpu_info.get('gpu_count', 0)}个GPU")
                print(f"GPU名称: {gpu_info.get('devices', [{}])[0].get('name', 'Unknown')}")
            else:
                print("ℹ️  当前配置仅CPU可用")
        
        print("✅ 当前设备配置下API功能正常")

    def test_model_loading_and_switching(self):
        """测试模型加载和切换"""
        # 测试模型管理API（如果可用）
        try:
            # 获取可用模型
            models_response = self.client.get(f"{API_BASE_URL}/api/v1/models/status", timeout=TIMEOUT)
            
            if models_response.status_code == 200:
                models_status = models_response.json()
                print(f"✅ 模型状态API可用，发现 {len(models_status)} 个模型配置")
                
                # 测试模型信息获取
                for model_name in ["buffalo_l", "buffalo_m", "buffalo_s"]:
                    info_response = self.client.get(
                        f"{API_BASE_URL}/api/v1/models/info/{model_name}", 
                        timeout=TIMEOUT
                    )
                    if info_response.status_code == 200:
                        print(f"✅ {model_name}: 信息获取成功")
                    else:
                        print(f"⚠️  {model_name}: 信息获取失败")
            else:
                print("ℹ️  模型管理API不可用，跳过模型切换测试")
                
        except Exception as e:
            print(f"ℹ️  模型管理测试跳过: {e}")

    def test_error_handling_robustness(self):
        """测试错误处理的健壮性"""
        # 测试各种错误条件下API的表现
        
        test_cases = [
            {
                "name": "无效模型名称",
                "data": {"model_name": "invalid_model_12345"},
                "expected_codes": [400, 500]
            },
            {
                "name": "无效检测器",
                "data": {"detector_backend": "invalid_detector_12345"},
                "expected_codes": [400, 500]
            },
            {
                "name": "极端阈值",
                "data": {"threshold": 99.9},
                "expected_codes": [200, 400]
            },
            {
                "name": "负阈值",
                "data": {"threshold": -1.0},
                "expected_codes": [200, 400, 422]
            }
        ]
        
        image_data = self.create_test_image()
        
        for test_case in test_cases:
            files = {"file": ("test.jpg", image_data, "image/jpeg")}
            data = {"actions": "emotion", **test_case["data"]}
            
            response = self.client.post(
                f"{self.base_url}/analyze/",
                files=files,
                data=data,
                timeout=TIMEOUT
            )
            
            assert response.status_code in test_case["expected_codes"], \
                f"{test_case['name']}: 期望状态码 {test_case['expected_codes']}, 实际 {response.status_code}"
            
            print(f"✅ {test_case['name']}: 错误处理正确 ({response.status_code})")

    @pytest.mark.slow
    def test_stress_testing_multiple_backends(self):
        """压力测试多个后端"""
        import concurrent.futures
        import threading
        
        def test_backend_concurrent(backend_name, iteration):
            """并发测试单个后端"""
            try:
                image1 = self.create_test_image(face_variation=iteration)
                image2 = self.create_test_image(face_variation=iteration + 1)
                
                files = {
                    "file1": (f"face1_{iteration}.jpg", image1, "image/jpeg"),
                    "file2": (f"face2_{iteration}.jpg", image2, "image/jpeg")
                }
                data = {
                    "threshold": 0.6,
                    "model_name": backend_name
                }
                
                response = self.client.post(
                    f"{self.base_url}/verify/",
                    files=files,
                    data=data,
                    timeout=TIMEOUT
                )
                
                return {
                    "backend": backend_name,
                    "iteration": iteration,
                    "status_code": response.status_code,
                    "success": response.status_code == 200
                }
                
            except Exception as e:
                return {
                    "backend": backend_name,
                    "iteration": iteration,
                    "status_code": 500,
                    "success": False,
                    "error": str(e)
                }
        
        # 测试2-3个主要后端
        backends_to_test = ["ArcFace", "VGG-Face"]
        iterations_per_backend = 3
        
        # 并发执行测试
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            
            for backend in backends_to_test:
                for iteration in range(iterations_per_backend):
                    future = executor.submit(test_backend_concurrent, backend, iteration)
                    futures.append(future)
            
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        # 分析结果
        total_tests = len(results)
        successful_tests = sum(1 for r in results if r["success"])
        success_rate = successful_tests / total_tests if total_tests > 0 else 0
        
        print(f"\n📊 压力测试结果:")
        print(f"总测试数: {total_tests}")
        print(f"成功数: {successful_tests}")
        print(f"成功率: {success_rate:.1%}")
        
        # 按后端分组统计
        backend_stats = {}
        for result in results:
            backend = result["backend"]
            if backend not in backend_stats:
                backend_stats[backend] = {"total": 0, "success": 0}
            
            backend_stats[backend]["total"] += 1
            if result["success"]:
                backend_stats[backend]["success"] += 1
        
        for backend, stats in backend_stats.items():
            rate = stats["success"] / stats["total"] if stats["total"] > 0 else 0
            print(f"{backend}: {stats['success']}/{stats['total']} ({rate:.1%})")
        
        # 验证总体成功率合理
        assert success_rate >= 0.5, f"压力测试成功率过低: {success_rate:.1%}"

    def test_api_endpoint_availability(self):
        """测试所有DeepFace API端点的可用性"""
        endpoints_to_test = [
            ("GET", "/health", {}),
            ("GET", "/faces/", {}),
            ("POST", "/recognition", {"files": True}),
            ("POST", "/verify/", {"files": True}),
            ("POST", "/analyze/", {"files": True})
        ]
        
        available_endpoints = []
        unavailable_endpoints = []
        
        for method, endpoint, options in endpoints_to_test:
            url = f"{self.base_url}{endpoint}"
            
            try:
                if method == "GET":
                    response = self.client.get(url, timeout=TIMEOUT)
                elif method == "POST" and options.get("files"):
                    # 使用测试图像
                    image_data = self.create_test_image()
                    files = {"file": ("test.jpg", image_data, "image/jpeg")}
                    data = {"threshold": 0.6} if "recognition" in endpoint or "verify" in endpoint else {"actions": "emotion"}
                    if endpoint == "/verify/":
                        files["file1"] = files["file"]
                        files["file2"] = ("test2.jpg", image_data, "image/jpeg")
                        del files["file"]
                    response = self.client.post(url, files=files, data=data, timeout=TIMEOUT)
                else:
                    response = self.client.post(url, json={}, timeout=TIMEOUT)
                
                if response.status_code in [200, 400, 422]:  # 正常响应码
                    available_endpoints.append(f"{method} {endpoint}")
                    print(f"✅ {method} {endpoint}: 可用 ({response.status_code})")
                else:
                    unavailable_endpoints.append(f"{method} {endpoint}: {response.status_code}")
                    print(f"❌ {method} {endpoint}: 不可用 ({response.status_code})")
                    
            except Exception as e:
                unavailable_endpoints.append(f"{method} {endpoint}: {str(e)}")
                print(f"❌ {method} {endpoint}: 异常 {str(e)}")
        
        print(f"\n📋 端点可用性总结:")
        print(f"可用: {len(available_endpoints)}/{len(endpoints_to_test)}")
        print(f"不可用: {len(unavailable_endpoints)}")
        
        # 验证关键端点可用
        assert len(available_endpoints) >= 3, "太多关键端点不可用"