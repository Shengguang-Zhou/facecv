"""
DeepFace Health Check API Tests

Tests for DeepFace service health monitoring endpoint.
"""

import pytest
import requests
from tests.conftest import API_BASE_URL, DEEPFACE_BASE_URL, TIMEOUT


class TestDeepFaceHealth:
    """测试DeepFace健康检查API"""

    @pytest.fixture(autouse=True)
    def setup_method(self, api_client):
        """每个测试方法的设置"""
        self.client = api_client
        self.base_url = DEEPFACE_BASE_URL

    def test_health_check_basic(self):
        """测试基本健康检查"""
        response = self.client.get(f"{self.base_url}/health", timeout=TIMEOUT)
        
        # 健康检查应该总是返回响应
        assert response.status_code in [200, 503]
        
        result = response.json()
        
        # 验证响应结构
        assert "status" in result
        assert "service" in result
        assert result["service"] == "DeepFace"
        
        if response.status_code == 200:
            # 健康状态
            assert result["status"] == "healthy"
            assert "version" in result
        else:
            # 不健康状态
            assert result["status"] == "unhealthy"
            assert "error" in result

    def test_health_check_response_time(self):
        """测试健康检查响应时间"""
        import time
        
        start_time = time.time()
        response = self.client.get(f"{self.base_url}/health", timeout=TIMEOUT)
        end_time = time.time()
        
        response_time = end_time - start_time
        
        # 健康检查应该快速响应（5秒内）
        assert response_time < 5.0
        
        # 验证响应
        assert response.status_code in [200, 503]

    def test_health_check_concurrent_requests(self):
        """测试并发健康检查请求"""
        import concurrent.futures
        
        def make_health_request():
            try:
                response = self.client.get(f"{self.base_url}/health", timeout=TIMEOUT)
                return response.status_code, response.json()
            except Exception as e:
                return 500, {"error": str(e)}
        
        # 发起多个并发健康检查请求
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(make_health_request) for _ in range(5)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        # 验证所有请求都得到了响应
        assert len(results) == 5
        
        for status_code, result in results:
            assert status_code in [200, 503]
            assert "status" in result
            assert "service" in result

    def test_health_check_repeated_calls(self):
        """测试重复健康检查调用"""
        results = []
        
        # 进行多次健康检查
        for i in range(3):
            response = self.client.get(f"{self.base_url}/health", timeout=TIMEOUT)
            results.append((response.status_code, response.json()))
            
            # 短暂等待
            import time
            time.sleep(0.5)
        
        # 验证所有调用都成功
        for status_code, result in results:
            assert status_code in [200, 503]
            assert result["service"] == "DeepFace"
            
        # 健康状态应该保持一致（除非在测试期间发生变化）
        statuses = [result["status"] for _, result in results]
        # 大多数状态应该相同
        from collections import Counter
        status_counts = Counter(statuses)
        most_common_status, count = status_counts.most_common(1)[0]
        assert count >= len(results) // 2  # 至少一半的请求有相同状态

    def test_health_check_during_load(self):
        """测试负载期间的健康检查"""
        # 先发起一些可能产生负载的请求
        
        # 1. 尝试获取人脸列表
        try:
            list_response = self.client.get(f"{self.base_url}/faces/", timeout=5)
        except:
            pass
        
        # 2. 尝试进行一个可能失败的识别请求
        try:
            # 创建一个简单的测试图像
            from PIL import Image
            import io
            image = Image.new('RGB', (100, 100), color='red')
            buffer = io.BytesIO()
            image.save(buffer, format='JPEG')
            image_data = buffer.getvalue()
            
            files = {"file": ("test.jpg", image_data, "image/jpeg")}
            data = {"threshold": 0.6}
            
            recognition_response = self.client.post(
                f"{self.base_url}/recognition",
                files=files,
                data=data,
                timeout=5
            )
        except:
            pass
        
        # 3. 在可能的负载期间检查健康状态
        health_response = self.client.get(f"{self.base_url}/health", timeout=TIMEOUT)
        
        # 健康检查应该仍然响应
        assert health_response.status_code in [200, 503]
        result = health_response.json()
        assert "status" in result
        assert result["service"] == "DeepFace"

    def test_health_check_with_invalid_method(self):
        """测试使用无效HTTP方法的健康检查"""
        # 健康检查端点应该只支持GET方法
        
        # 测试POST方法
        post_response = self.client.post(f"{self.base_url}/health", timeout=TIMEOUT)
        assert post_response.status_code == 405  # Method Not Allowed
        
        # 测试PUT方法
        put_response = self.client.put(f"{self.base_url}/health", timeout=TIMEOUT)
        assert put_response.status_code == 405
        
        # 测试DELETE方法
        delete_response = self.client.delete(f"{self.base_url}/health", timeout=TIMEOUT)
        assert delete_response.status_code == 405

    def test_health_check_with_parameters(self):
        """测试带参数的健康检查请求"""
        # 健康检查端点应该忽略查询参数
        
        params = {
            "test_param": "test_value",
            "another_param": 123
        }
        
        response = self.client.get(
            f"{self.base_url}/health",
            params=params,
            timeout=TIMEOUT
        )
        
        # 应该正常响应，忽略参数
        assert response.status_code in [200, 503]
        result = response.json()
        assert result["service"] == "DeepFace"

    def test_health_check_headers(self):
        """测试健康检查响应头"""
        response = self.client.get(f"{self.base_url}/health", timeout=TIMEOUT)
        
        # 验证响应头
        assert "content-type" in response.headers
        assert "application/json" in response.headers["content-type"]
        
        # 可能包含其他标准头
        # assert "server" in response.headers  # 可选
        # assert "date" in response.headers    # 可选

    def test_health_check_error_conditions(self):
        """测试健康检查的错误条件"""
        # 这个测试主要验证当底层服务有问题时健康检查的行为
        
        response = self.client.get(f"{self.base_url}/health", timeout=TIMEOUT)
        
        if response.status_code == 503:
            # 服务不健康的情况
            result = response.json()
            assert result["status"] == "unhealthy"
            assert "error" in result
            assert isinstance(result["error"], str)
            assert len(result["error"]) > 0
        
        # 无论如何都应该有响应
        assert response.status_code in [200, 503]

    def test_health_check_json_format(self):
        """测试健康检查响应的JSON格式"""
        response = self.client.get(f"{self.base_url}/health", timeout=TIMEOUT)
        
        # 验证返回的是有效JSON
        try:
            result = response.json()
        except ValueError:
            pytest.fail("Health check response is not valid JSON")
        
        # 验证JSON结构
        assert isinstance(result, dict)
        
        # 必需字段
        required_fields = ["status", "service"]
        for field in required_fields:
            assert field in result
            assert isinstance(result[field], str)
        
        # 条件字段
        if "version" in result:
            assert isinstance(result["version"], str)
        
        if "error" in result:
            assert isinstance(result["error"], str)

    def test_health_check_status_values(self):
        """测试健康检查状态值的有效性"""
        response = self.client.get(f"{self.base_url}/health", timeout=TIMEOUT)
        result = response.json()
        
        # 验证状态值是预期的值
        valid_statuses = ["healthy", "unhealthy"]
        assert result["status"] in valid_statuses
        
        # 验证服务名称
        assert result["service"] == "DeepFace"
        
        # 根据状态验证其他字段
        if result["status"] == "healthy":
            # 健康状态应该有版本信息
            assert "version" in result
            # 不应该有错误信息
            assert "error" not in result or result["error"] is None
        
        elif result["status"] == "unhealthy":
            # 不健康状态应该有错误信息
            assert "error" in result
            assert result["error"] is not None
            assert len(result["error"]) > 0

    def test_health_check_timeout_behavior(self):
        """测试健康检查的超时行为"""
        # 使用较短的超时时间测试
        short_timeout = 2.0
        
        try:
            response = self.client.get(f"{self.base_url}/health", timeout=short_timeout)
            
            # 如果在超时时间内响应，验证响应
            assert response.status_code in [200, 503]
            result = response.json()
            assert "status" in result
            
        except requests.exceptions.Timeout:
            # 如果超时，这表明健康检查可能有性能问题
            pytest.fail(f"Health check timed out after {short_timeout} seconds")

    def test_health_check_integration_with_other_endpoints(self):
        """测试健康检查与其他端点的集成"""
        # 1. 首先检查健康状态
        health_response = self.client.get(f"{self.base_url}/health", timeout=TIMEOUT)
        health_result = health_response.json()
        
        # 2. 如果服务健康，其他端点应该也能响应
        if health_result["status"] == "healthy":
            # 尝试获取人脸列表
            faces_response = self.client.get(f"{self.base_url}/faces/", timeout=TIMEOUT)
            assert faces_response.status_code in [200, 400, 500]  # 应该有响应
            
        # 3. 再次检查健康状态
        health_response_2 = self.client.get(f"{self.base_url}/health", timeout=TIMEOUT)
        health_result_2 = health_response_2.json()
        
        # 健康状态应该保持一致（短时间内）
        assert health_result_2["service"] == "DeepFace"

    def test_health_check_service_identification(self):
        """测试健康检查的服务标识"""
        response = self.client.get(f"{self.base_url}/health", timeout=TIMEOUT)
        result = response.json()
        
        # 验证这确实是DeepFace服务的健康检查
        assert result["service"] == "DeepFace"
        
        # 与其他服务的健康检查进行对比
        # 尝试调用InsightFace健康检查进行对比
        try:
            insightface_health_response = self.client.get(
                f"{API_BASE_URL}/api/v1/insightface/health",
                timeout=5
            )
            if insightface_health_response.status_code in [200, 503]:
                insightface_result = insightface_health_response.json()
                # 应该有不同的服务标识
                assert insightface_result.get("service") != result["service"]
        except:
            # 如果InsightFace服务不可用，跳过比较
            pass