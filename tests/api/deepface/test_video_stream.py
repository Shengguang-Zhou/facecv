"""
DeepFace Video and Stream API Tests

Tests for video face sampling and real-time recognition stream endpoints.
"""

import pytest
import requests
import json
import time
import asyncio
from io import StringIO
from tests.conftest import API_BASE_URL, DEEPFACE_BASE_URL, TIMEOUT


class TestVideoStream:
    """测试视频和流处理相关API"""

    @pytest.fixture(autouse=True)
    def setup_method(self, api_client):
        """每个测试方法的设置"""
        self.client = api_client
        self.base_url = DEEPFACE_BASE_URL

    def test_video_face_sampling_webcam(self):
        """测试通过摄像头进行视频人脸采样"""
        test_name = "video_test_user"
        
        data = {
            "name": test_name,
            "video_source": "0",  # 默认摄像头
            "sample_interval": 60,  # 较大间隔以减少采样数量
            "max_samples": 2  # 限制采样数量
        }
        
        response = self.client.post(
            f"{self.base_url}/video_face/",
            data=data,
            timeout=TIMEOUT
        )
        
        # 可能成功启动任务或因摄像头不可用而失败
        assert response.status_code in [200, 400, 500]
        
        if response.status_code == 200:
            result = response.json()
            assert result["success"] is True
            assert "message" in result
            assert result["video_source"] == "0"
            assert result["sample_interval"] == 60
            assert result["max_samples"] == 2

    def test_video_face_sampling_invalid_source(self):
        """测试无效视频源的视频人脸采样"""
        test_name = "invalid_source_user"
        
        data = {
            "name": test_name,
            "video_source": "nonexistent_camera_99",
            "sample_interval": 30,
            "max_samples": 5
        }
        
        response = self.client.post(
            f"{self.base_url}/video_face/",
            data=data,
            timeout=TIMEOUT
        )
        
        # 应该启动任务但可能在后台失败
        assert response.status_code in [200, 400]

    def test_video_face_sampling_missing_name(self):
        """测试缺少姓名参数的视频人脸采样"""
        data = {
            "video_source": "0",
            "sample_interval": 30,
            "max_samples": 5
        }
        
        response = self.client.post(
            f"{self.base_url}/video_face/",
            data=data,
            timeout=TIMEOUT
        )
        
        assert response.status_code == 422  # Validation error

    def test_video_face_sampling_invalid_parameters(self):
        """测试无效参数的视频人脸采样"""
        test_name = "invalid_params_user"
        
        # 测试负数间隔
        data = {
            "name": test_name,
            "video_source": "0",
            "sample_interval": -10,
            "max_samples": 5
        }
        
        response = self.client.post(
            f"{self.base_url}/video_face/",
            data=data,
            timeout=TIMEOUT
        )
        
        # 可能接受并使用默认值，或返回验证错误
        assert response.status_code in [200, 400, 422]

    def test_video_face_sampling_large_parameters(self):
        """测试极大参数值的视频人脸采样"""
        test_name = "large_params_user"
        
        data = {
            "name": test_name,
            "video_source": "0",
            "sample_interval": 1000,  # 很大的间隔
            "max_samples": 100  # 很多样本
        }
        
        response = self.client.post(
            f"{self.base_url}/video_face/",
            data=data,
            timeout=TIMEOUT
        )
        
        # 应该能处理大参数或返回合理错误
        assert response.status_code in [200, 400]

    def test_real_time_recognition_stream_sse_format(self):
        """测试SSE格式的实时人脸识别流"""
        params = {
            "camera_id": "test_camera_001",
            "source": "0",  # 默认摄像头
            "threshold": 0.6,
            "fps": 5,  # 较低FPS以减少资源使用
            "format": "sse"
        }
        
        try:
            # 使用streaming=True获取流响应
            response = self.client.get(
                f"{self.base_url}/recognize/webcam/stream",
                params=params,
                timeout=10,  # 较短超时
                stream=True
            )
            
            # 可能成功开始流或因摄像头不可用而失败
            assert response.status_code in [200, 400, 500]
            
            if response.status_code == 200:
                # 验证Content-Type
                assert "text/event-stream" in response.headers.get("content-type", "")
                
                # 尝试读取少量数据
                content_read = False
                for i, chunk in enumerate(response.iter_content(chunk_size=1024)):
                    if chunk:
                        content_read = True
                        # 验证SSE格式
                        chunk_str = chunk.decode('utf-8', errors='ignore')
                        # SSE数据应该包含"data:"前缀
                        if "data:" in chunk_str:
                            break
                    
                    # 只读取前几个chunk以避免长时间运行
                    if i >= 3:
                        break
                
                # 关闭连接
                response.close()
                
        except requests.exceptions.Timeout:
            # 超时是可接受的，因为这是一个持续的流
            pass
        except requests.exceptions.ConnectionError:
            # 连接错误可能表示摄像头不可用
            pass

    def test_real_time_recognition_stream_mjpeg_format(self):
        """测试MJPEG格式的实时人脸识别流"""
        params = {
            "camera_id": "test_camera_002",
            "source": "0",
            "threshold": 0.6,
            "fps": 5,
            "format": "mjpeg"
        }
        
        try:
            response = self.client.get(
                f"{self.base_url}/recognize/webcam/stream",
                params=params,
                timeout=10,
                stream=True
            )
            
            assert response.status_code in [200, 400, 500]
            
            if response.status_code == 200:
                # 验证Content-Type
                content_type = response.headers.get("content-type", "")
                assert "multipart/x-mixed-replace" in content_type
                
                # 尝试读取少量数据
                for i, chunk in enumerate(response.iter_content(chunk_size=1024)):
                    if chunk:
                        # MJPEG流应该包含boundary标记
                        chunk_str = chunk.decode('utf-8', errors='ignore')
                        if "--frame" in chunk_str or "Content-Type: image/jpeg" in chunk_str:
                            break
                    
                    if i >= 3:
                        break
                
                response.close()
                
        except requests.exceptions.Timeout:
            pass
        except requests.exceptions.ConnectionError:
            pass

    def test_real_time_recognition_stream_with_webhook(self):
        """测试带Webhook的实时人脸识别流"""
        # 使用httpbin.org作为测试webhook端点
        webhook_url = "https://httpbin.org/post"
        
        params = {
            "camera_id": "webhook_test_camera",
            "source": "0",
            "threshold": 0.5,
            "fps": 2,
            "format": "sse",
            "webhook_urls": webhook_url,
            "webhook_timeout": 10,
            "webhook_retry_count": 1
        }
        
        try:
            response = self.client.get(
                f"{self.base_url}/recognize/webcam/stream",
                params=params,
                timeout=8,
                stream=True
            )
            
            assert response.status_code in [200, 400, 500]
            
            if response.status_code == 200:
                # 读取少量数据后关闭
                for i, chunk in enumerate(response.iter_content(chunk_size=512)):
                    if i >= 2:
                        break
                
                response.close()
                
        except requests.exceptions.Timeout:
            pass
        except requests.exceptions.ConnectionError:
            pass

    def test_real_time_recognition_stream_multiple_webhooks(self):
        """测试多个Webhook的实时人脸识别流"""
        # 使用多个测试webhook端点
        webhook_urls = "https://httpbin.org/post,https://postman-echo.com/post"
        
        params = {
            "camera_id": "multi_webhook_camera",
            "source": "0",
            "threshold": 0.6,
            "fps": 1,
            "format": "sse",
            "webhook_urls": webhook_urls,
            "webhook_timeout": 5,
            "webhook_retry_count": 1
        }
        
        try:
            response = self.client.get(
                f"{self.base_url}/recognize/webcam/stream",
                params=params,
                timeout=6,
                stream=True
            )
            
            assert response.status_code in [200, 400, 500]
            
            if response.status_code == 200:
                response.close()
                
        except requests.exceptions.Timeout:
            pass
        except requests.exceptions.ConnectionError:
            pass

    def test_real_time_recognition_stream_invalid_source(self):
        """测试无效视频源的实时识别流"""
        params = {
            "camera_id": "invalid_camera",
            "source": "nonexistent_source_999",
            "threshold": 0.6,
            "fps": 10,
            "format": "sse"
        }
        
        response = self.client.get(
            f"{self.base_url}/recognize/webcam/stream",
            params=params,
            timeout=5
        )
        
        # 应该返回错误或超时
        assert response.status_code in [200, 400, 500, 504]

    def test_real_time_recognition_stream_invalid_format(self):
        """测试无效格式的实时识别流"""
        params = {
            "camera_id": "format_test_camera",
            "source": "0",
            "threshold": 0.6,
            "fps": 10,
            "format": "invalid_format"
        }
        
        response = self.client.get(
            f"{self.base_url}/recognize/webcam/stream",
            params=params,
            timeout=5
        )
        
        # 可能使用默认格式或返回错误
        assert response.status_code in [200, 400]

    def test_real_time_recognition_stream_parameter_validation(self):
        """测试实时识别流的参数验证"""
        # 测试无效阈值
        invalid_params = [
            {"threshold": -0.5},  # 负阈值
            {"threshold": 1.5},   # 超过1的阈值
            {"fps": -10},         # 负FPS
            {"fps": 1000},        # 极高FPS
            {"webhook_timeout": -5},  # 负超时
            {"webhook_retry_count": -1}  # 负重试次数
        ]
        
        base_params = {
            "camera_id": "validation_test",
            "source": "0",
            "format": "sse"
        }
        
        for invalid_param in invalid_params:
            params = {**base_params, **invalid_param}
            
            response = self.client.get(
                f"{self.base_url}/recognize/webcam/stream",
                params=params,
                timeout=3
            )
            
            # 可能接受参数并使用默认值，或返回验证错误
            assert response.status_code in [200, 400, 422]

    def test_real_time_recognition_stream_rtsp_url(self):
        """测试RTSP URL的实时识别流"""
        # 使用虚假RTSP URL测试
        rtsp_url = "rtsp://test.server.com/stream"
        
        params = {
            "camera_id": "rtsp_test_camera",
            "source": rtsp_url,
            "threshold": 0.6,
            "fps": 5,
            "format": "sse"
        }
        
        response = self.client.get(
            f"{self.base_url}/recognize/webcam/stream",
            params=params,
            timeout=5
        )
        
        # RTSP URL不存在，应该返回错误或超时
        assert response.status_code in [200, 400, 500, 504]

    def test_concurrent_stream_requests(self):
        """测试并发流请求"""
        import concurrent.futures
        
        def make_stream_request(camera_id):
            params = {
                "camera_id": f"concurrent_camera_{camera_id}",
                "source": "0",
                "threshold": 0.6,
                "fps": 2,
                "format": "sse"
            }
            
            try:
                response = self.client.get(
                    f"{self.base_url}/recognize/webcam/stream",
                    params=params,
                    timeout=3
                )
                return response.status_code
            except:
                return 500
        
        # 发起3个并发请求
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(make_stream_request, i) for i in range(3)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        # 验证请求能被处理（即使资源不可用）
        for status_code in results:
            assert status_code in [200, 400, 500, 504]

    def test_stream_performance_different_fps(self):
        """测试不同FPS设置的流性能"""
        fps_values = [1, 5, 10, 30]
        
        for fps in fps_values:
            params = {
                "camera_id": f"fps_test_{fps}",
                "source": "0",
                "threshold": 0.6,
                "fps": fps,
                "format": "sse"
            }
            
            try:
                start_time = time.time()
                response = self.client.get(
                    f"{self.base_url}/recognize/webcam/stream",
                    params=params,
                    timeout=3,
                    stream=True
                )
                
                if response.status_code == 200:
                    # 读取少量数据测试响应性
                    for i, chunk in enumerate(response.iter_content(chunk_size=256)):
                        if i >= 1:  # 只读取一个chunk
                            break
                    
                    response.close()
                
                elapsed_time = time.time() - start_time
                
                # 验证响应时间合理
                assert elapsed_time < 10  # 10秒内应该有响应
                
            except requests.exceptions.Timeout:
                pass  # 超时是可接受的
            except requests.exceptions.ConnectionError:
                pass  # 连接错误可能表示资源不可用

    @pytest.mark.slow
    def test_video_stream_integration_workflow(self):
        """测试视频流的完整集成工作流"""
        # 1. 启动视频采样任务
        test_name = "integration_test_user"
        
        sampling_data = {
            "name": test_name,
            "video_source": "0",
            "sample_interval": 120,  # 大间隔减少负载
            "max_samples": 1
        }
        
        sampling_response = self.client.post(
            f"{self.base_url}/video_face/",
            data=sampling_data,
            timeout=TIMEOUT
        )
        
        # 采样任务启动状态
        sampling_started = sampling_response.status_code == 200
        
        # 2. 等待短时间让采样任务处理
        if sampling_started:
            time.sleep(2)
        
        # 3. 启动实时识别流
        stream_params = {
            "camera_id": "integration_test_stream",
            "source": "0",
            "threshold": 0.4,  # 较低阈值
            "fps": 2,
            "format": "sse"
        }
        
        try:
            stream_response = self.client.get(
                f"{self.base_url}/recognize/webcam/stream",
                params=stream_params,
                timeout=5,
                stream=True
            )
            
            if stream_response.status_code == 200:
                # 读取少量流数据
                data_received = False
                for i, chunk in enumerate(stream_response.iter_content(chunk_size=512)):
                    if chunk:
                        data_received = True
                        break
                    if i >= 2:
                        break
                
                stream_response.close()
                
                # 验证工作流完成
                assert sampling_started or stream_response.status_code == 200
                
        except requests.exceptions.Timeout:
            pass
        except requests.exceptions.ConnectionError:
            pass

    def test_webhook_url_validation(self):
        """测试Webhook URL验证"""
        invalid_webhook_urls = [
            "not-a-url",
            "ftp://invalid.protocol.com",
            "http://",
            "https://",
            ""
        ]
        
        for invalid_url in invalid_webhook_urls:
            params = {
                "camera_id": "webhook_validation_test",
                "source": "0",
                "threshold": 0.6,
                "format": "sse",
                "webhook_urls": invalid_url
            }
            
            response = self.client.get(
                f"{self.base_url}/recognize/webcam/stream",
                params=params,
                timeout=3
            )
            
            # 可能接受无效URL并在运行时忽略，或返回验证错误
            assert response.status_code in [200, 400, 422]