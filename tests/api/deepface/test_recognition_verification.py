"""
DeepFace Recognition and Verification API Tests

Tests for face recognition, verification, and analysis endpoints.
"""

import pytest
import requests
import json
import io
from PIL import Image
import numpy as np
from tests.conftest import API_BASE_URL, DEEPFACE_BASE_URL, TIMEOUT


class TestRecognitionVerification:
    """测试人脸识别和验证相关API"""

    @pytest.fixture(autouse=True)
    def setup_method(self, api_client):
        """每个测试方法的设置"""
        self.client = api_client
        self.base_url = DEEPFACE_BASE_URL
        self.registered_faces = []  # 跟踪注册的人脸用于清理

    def teardown_method(self):
        """每个测试方法的清理"""
        # 清理测试中注册的人脸
        for face_data in self.registered_faces:
            try:
                response = self.client.delete(
                    f"{self.base_url}/faces/{face_data['face_id']}",
                    timeout=TIMEOUT
                )
            except:
                pass  # 忽略清理错误

    def create_test_image(self, color='blue', size=(300, 300), face_variation=0) -> bytes:
        """创建测试图像，可以生成稍微不同的变体"""
        image = Image.new('RGB', size, color=color)
        
        from PIL import ImageDraw
        draw = ImageDraw.Draw(image)
        
        # 添加变化以模拟不同的人脸
        center_x, center_y = size[0] // 2, size[1] // 2
        face_width, face_height = size[0] // 3, size[1] // 2
        
        # 根据variation调整位置
        offset_x = face_variation * 5
        offset_y = face_variation * 3
        
        # 脸部边框
        draw.rectangle([
            center_x - face_width//2 + offset_x, center_y - face_height//2 + offset_y,
            center_x + face_width//2 + offset_x, center_y + face_height//2 + offset_y
        ], outline='black', width=2)
        
        # 眼睛
        eye_y = center_y - face_height//4 + offset_y
        eye_size = 8 + face_variation
        eye_offset = face_width//4 + face_variation
        
        draw.ellipse([center_x - eye_offset - eye_size + offset_x, eye_y - eye_size,
                     center_x - eye_offset + eye_size + offset_x, eye_y + eye_size], 
                    fill='black')
        draw.ellipse([center_x + eye_offset - eye_size + offset_x, eye_y - eye_size,
                     center_x + eye_offset + eye_size + offset_x, eye_y + eye_size], 
                    fill='black')
        
        # 鼻子
        nose_y = center_y + offset_y
        draw.line([center_x + offset_x, nose_y - 10, center_x + offset_x, nose_y + 10], 
                 fill='black', width=2)
        
        # 嘴巴
        mouth_y = center_y + face_height//4 + offset_y
        mouth_width = 15 + face_variation * 2
        draw.arc([center_x - mouth_width + offset_x, mouth_y - 10, 
                 center_x + mouth_width + offset_x, mouth_y + 10], 
                0, 180, fill='black', width=2)
        
        # 转换为字节
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG', quality=85)
        return buffer.getvalue()

    def register_test_face(self, name, image_data=None, metadata=None):
        """注册测试人脸并返回结果"""
        if image_data is None:
            image_data = self.create_test_image()
        
        files = {"file": ("test.jpg", image_data, "image/jpeg")}
        data = {"name": name}
        
        if metadata:
            data["metadata"] = json.dumps(metadata)
        
        response = self.client.post(
            f"{self.base_url}/faces/",
            files=files,
            data=data,
            timeout=TIMEOUT
        )
        
        if response.status_code == 200:
            result = response.json()
            # 记录用于清理
            if "face_id" in result:
                self.registered_faces.append({
                    "face_id": result["face_id"],
                    "name": name
                })
            return result
        return None

    def test_face_recognition_with_registered_face(self):
        """测试识别已注册的人脸"""
        # 先注册一个人脸
        test_name = "recognition_test_user"
        register_image = self.create_test_image(color='blue')
        
        register_result = self.register_test_face(test_name, register_image)
        
        if register_result and register_result["success"]:
            # 使用相似的图像进行识别
            recognition_image = self.create_test_image(color='blue', face_variation=1)
            
            files = {"file": ("recognize.jpg", recognition_image, "image/jpeg")}
            data = {
                "threshold": 0.5,  # 较低的阈值以增加匹配可能性
                "return_all_candidates": True
            }
            
            response = self.client.post(
                f"{self.base_url}/recognition",
                files=files,
                data=data,
                timeout=TIMEOUT
            )
            
            assert response.status_code == 200
            result = response.json()
            
            # 验证响应结构
            assert "faces" in result
            assert "total_faces" in result
            assert "processing_time" in result
            assert isinstance(result["faces"], list)
            
            # 如果识别成功，验证人脸数据结构
            for face in result["faces"]:
                assert "person_name" in face
                assert "confidence" in face
                assert "bbox" in face
                assert isinstance(face["confidence"], (int, float))
                assert isinstance(face["bbox"], list)
                assert len(face["bbox"]) == 4  # [x, y, width, height]

    def test_face_recognition_no_faces(self):
        """测试识别空背景图像"""
        # 创建没有明显人脸特征的图像
        blank_image = Image.new('RGB', (300, 300), color='white')
        buffer = io.BytesIO()
        blank_image.save(buffer, format='JPEG')
        image_data = buffer.getvalue()
        
        files = {"file": ("blank.jpg", image_data, "image/jpeg")}
        data = {"threshold": 0.6}
        
        response = self.client.post(
            f"{self.base_url}/recognition",
            files=files,
            data=data,
            timeout=TIMEOUT
        )
        
        assert response.status_code == 200
        result = response.json()
        
        # 应该返回空结果或检测不到人脸
        assert "faces" in result
        assert "total_faces" in result
        # 可能为0或包含检测错误

    def test_face_recognition_multiple_thresholds(self):
        """测试不同阈值的人脸识别"""
        # 注册测试人脸
        test_name = "threshold_test_user"
        register_image = self.create_test_image(color='green')
        
        register_result = self.register_test_face(test_name, register_image)
        
        if register_result and register_result["success"]:
            recognition_image = self.create_test_image(color='green', face_variation=2)
            
            thresholds = [0.3, 0.5, 0.7, 0.9]
            
            for threshold in thresholds:
                files = {"file": ("test.jpg", recognition_image, "image/jpeg")}
                data = {"threshold": threshold}
                
                response = self.client.post(
                    f"{self.base_url}/recognition",
                    files=files,
                    data=data,
                    timeout=TIMEOUT
                )
                
                assert response.status_code == 200
                result = response.json()
                
                # 验证基本响应结构
                assert "faces" in result
                assert "total_faces" in result

    def test_face_recognition_invalid_image(self):
        """测试识别无效图像"""
        invalid_data = b"not an image"
        
        files = {"file": ("invalid.txt", invalid_data, "text/plain")}
        data = {"threshold": 0.6}
        
        response = self.client.post(
            f"{self.base_url}/recognition",
            files=files,
            data=data,
            timeout=TIMEOUT
        )
        
        assert response.status_code in [400, 500]

    def test_face_verification_same_person(self):
        """测试验证同一人的两张图片"""
        # 创建两张相似的图像
        image1 = self.create_test_image(color='red', face_variation=0)
        image2 = self.create_test_image(color='red', face_variation=1)
        
        files = {
            "file1": ("face1.jpg", image1, "image/jpeg"),
            "file2": ("face2.jpg", image2, "image/jpeg")
        }
        data = {
            "threshold": 0.5,
            "model_name": "ArcFace",
            "anti_spoofing": False
        }
        
        response = self.client.post(
            f"{self.base_url}/verify/",
            files=files,
            data=data,
            timeout=TIMEOUT
        )
        
        assert response.status_code == 200
        result = response.json()
        
        # 验证响应结构
        assert "verified" in result
        assert "confidence" in result
        assert "distance" in result
        assert "threshold" in result
        assert "model" in result
        
        assert isinstance(result["verified"], bool)
        assert isinstance(result["confidence"], (int, float))
        assert isinstance(result["distance"], (int, float))
        assert result["model"] == "ArcFace"

    def test_face_verification_different_persons(self):
        """测试验证不同人的两张图片"""
        # 创建明显不同的图像
        image1 = self.create_test_image(color='blue', face_variation=0)
        image2 = self.create_test_image(color='yellow', face_variation=5)
        
        files = {
            "file1": ("face1.jpg", image1, "image/jpeg"),
            "file2": ("face2.jpg", image2, "image/jpeg")
        }
        data = {
            "threshold": 0.6,
            "model_name": "VGG-Face"
        }
        
        response = self.client.post(
            f"{self.base_url}/verify/",
            files=files,
            data=data,
            timeout=TIMEOUT
        )
        
        assert response.status_code == 200
        result = response.json()
        
        # 验证响应结构
        assert "verified" in result
        assert "confidence" in result
        assert "distance" in result
        assert result["model"] == "VGG-Face"

    def test_face_verification_different_models(self):
        """测试不同模型的人脸验证"""
        image1 = self.create_test_image(color='purple')
        image2 = self.create_test_image(color='purple', face_variation=1)
        
        models = ["ArcFace", "VGG-Face", "OpenFace", "DeepFace"]
        
        for model in models:
            files = {
                "file1": ("face1.jpg", image1, "image/jpeg"),
                "file2": ("face2.jpg", image2, "image/jpeg")
            }
            data = {
                "threshold": 0.6,
                "model_name": model
            }
            
            response = self.client.post(
                f"{self.base_url}/verify/",
                files=files,
                data=data,
                timeout=TIMEOUT
            )
            
            # 某些模型可能不可用，允许合理的错误响应
            assert response.status_code in [200, 400, 500]
            
            if response.status_code == 200:
                result = response.json()
                assert result["model"] == model

    def test_face_verification_missing_file(self):
        """测试验证缺少文件的情况"""
        image1 = self.create_test_image()
        
        # 只提供一个文件
        files = {"file1": ("face1.jpg", image1, "image/jpeg")}
        data = {"threshold": 0.6}
        
        response = self.client.post(
            f"{self.base_url}/verify/",
            files=files,
            data=data,
            timeout=TIMEOUT
        )
        
        assert response.status_code == 422  # Validation error

    def test_face_verification_invalid_threshold(self):
        """测试无效阈值的人脸验证"""
        image1 = self.create_test_image()
        image2 = self.create_test_image(face_variation=1)
        
        files = {
            "file1": ("face1.jpg", image1, "image/jpeg"),
            "file2": ("face2.jpg", image2, "image/jpeg")
        }
        
        # 测试无效阈值
        invalid_thresholds = [-0.5, 1.5, "invalid"]
        
        for threshold in invalid_thresholds:
            data = {"threshold": threshold}
            
            response = self.client.post(
                f"{self.base_url}/verify/",
                files=files,
                data=data,
                timeout=TIMEOUT
            )
            
            # 应该返回验证错误或使用默认值
            assert response.status_code in [200, 400, 422]

    def test_face_analysis_all_attributes(self):
        """测试分析所有人脸属性"""
        image_data = self.create_test_image(color='orange')
        
        files = {"file": ("analyze.jpg", image_data, "image/jpeg")}
        data = {
            "actions": "emotion,age,gender,race",
            "detector_backend": "mtcnn"
        }
        
        response = self.client.post(
            f"{self.base_url}/analyze/",
            files=files,
            data=data,
            timeout=TIMEOUT
        )
        
        assert response.status_code == 200
        result = response.json()
        
        # 验证响应结构
        assert "faces" in result
        assert "total_faces" in result
        assert isinstance(result["faces"], list)
        
        # 如果检测到人脸，验证分析结果结构
        for face in result["faces"]:
            # 根据请求的actions验证相应字段
            if "emotion" in data["actions"]:
                # emotion字段可能存在
                pass
            if "age" in data["actions"]:
                # age字段可能存在
                pass
            if "gender" in data["actions"]:
                # gender字段可能存在
                pass
            if "race" in data["actions"]:
                # race字段可能存在
                pass

    def test_face_analysis_single_attribute(self):
        """测试分析单个人脸属性"""
        image_data = self.create_test_image()
        
        single_actions = ["emotion", "age", "gender", "race"]
        
        for action in single_actions:
            files = {"file": ("analyze.jpg", image_data, "image/jpeg")}
            data = {
                "actions": action,
                "detector_backend": "opencv"
            }
            
            response = self.client.post(
                f"{self.base_url}/analyze/",
                files=files,
                data=data,
                timeout=TIMEOUT
            )
            
            assert response.status_code == 200
            result = response.json()
            
            assert "faces" in result
            assert "total_faces" in result

    def test_face_analysis_different_detectors(self):
        """测试不同检测器的人脸分析"""
        image_data = self.create_test_image()
        
        detectors = ["mtcnn", "opencv", "ssd", "dlib", "retinaface"]
        
        for detector in detectors:
            files = {"file": ("analyze.jpg", image_data, "image/jpeg")}
            data = {
                "actions": "emotion,age",
                "detector_backend": detector
            }
            
            response = self.client.post(
                f"{self.base_url}/analyze/",
                files=files,
                data=data,
                timeout=TIMEOUT
            )
            
            # 某些检测器可能不可用
            assert response.status_code in [200, 400, 500]
            
            if response.status_code == 200:
                result = response.json()
                assert "faces" in result

    def test_face_analysis_invalid_action(self):
        """测试无效的分析动作"""
        image_data = self.create_test_image()
        
        files = {"file": ("analyze.jpg", image_data, "image/jpeg")}
        data = {
            "actions": "invalid_action,unknown_attribute",
            "detector_backend": "mtcnn"
        }
        
        response = self.client.post(
            f"{self.base_url}/analyze/",
            files=files,
            data=data,
            timeout=TIMEOUT
        )
        
        # 应该返回错误或忽略无效动作
        assert response.status_code in [200, 400]

    def test_recognition_verification_workflow(self):
        """测试识别和验证的完整工作流"""
        # 1. 注册用户
        test_name = "workflow_user"
        register_image = self.create_test_image(color='cyan')
        
        register_result = self.register_test_face(test_name, register_image)
        assert register_result is not None
        
        # 2. 识别相同用户
        recognition_image = self.create_test_image(color='cyan', face_variation=1)
        
        files = {"file": ("recognize.jpg", recognition_image, "image/jpeg")}
        data = {"threshold": 0.4}  # 较低阈值
        
        recognition_response = self.client.post(
            f"{self.base_url}/recognition",
            files=files,
            data=data,
            timeout=TIMEOUT
        )
        
        assert recognition_response.status_code == 200
        recognition_result = recognition_response.json()
        
        # 3. 验证两张相似图片
        verification_image = self.create_test_image(color='cyan', face_variation=2)
        
        verify_files = {
            "file1": ("face1.jpg", register_image, "image/jpeg"),
            "file2": ("face2.jpg", verification_image, "image/jpeg")
        }
        verify_data = {"threshold": 0.5}
        
        verification_response = self.client.post(
            f"{self.base_url}/verify/",
            files=verify_files,
            data=verify_data,
            timeout=TIMEOUT
        )
        
        assert verification_response.status_code == 200
        verification_result = verification_response.json()
        
        # 4. 分析人脸属性
        analyze_files = {"file": ("analyze.jpg", register_image, "image/jpeg")}
        analyze_data = {"actions": "emotion,age,gender"}
        
        analysis_response = self.client.post(
            f"{self.base_url}/analyze/",
            files=analyze_files,
            data=analyze_data,
            timeout=TIMEOUT
        )
        
        assert analysis_response.status_code == 200
        analysis_result = analysis_response.json()
        
        # 验证所有响应都有预期结构
        assert "faces" in recognition_result
        assert "verified" in verification_result
        assert "faces" in analysis_result

    def test_batch_recognition(self):
        """测试批量识别（模拟多张图片）"""
        # 注册多个用户
        users = ["batch_user_1", "batch_user_2", "batch_user_3"]
        
        for i, user in enumerate(users):
            image = self.create_test_image(color=f'color_{i}', face_variation=i)
            self.register_test_face(user, image)
        
        # 对每个用户进行识别测试
        for i, user in enumerate(users):
            test_image = self.create_test_image(color=f'color_{i}', face_variation=i+1)
            
            files = {"file": ("test.jpg", test_image, "image/jpeg")}
            data = {"threshold": 0.3}  # 较低阈值
            
            response = self.client.post(
                f"{self.base_url}/recognition",
                files=files,
                data=data,
                timeout=TIMEOUT
            )
            
            assert response.status_code == 200
            result = response.json()
            assert "faces" in result

    def test_performance_with_various_image_sizes(self):
        """测试不同图像尺寸的性能"""
        sizes = [(150, 150), (300, 300), (600, 600), (1200, 900)]
        
        for size in sizes:
            image_data = self.create_test_image(size=size)
            
            files = {"file": ("test.jpg", image_data, "image/jpeg")}
            data = {"actions": "emotion"}
            
            response = self.client.post(
                f"{self.base_url}/analyze/",
                files=files,
                data=data,
                timeout=TIMEOUT * 2  # 大图像需要更多时间
            )
            
            # 验证能处理不同尺寸
            assert response.status_code in [200, 400, 413]  # 413: Payload Too Large
            
            if response.status_code == 200:
                result = response.json()
                assert "processing_time" in result or "faces" in result