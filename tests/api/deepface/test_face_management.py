"""
DeepFace Face Management API Tests

Tests for face registration, listing, updating, and deletion endpoints.
"""

import pytest
import requests
import json
import base64
import io
from pathlib import Path
from PIL import Image
import numpy as np
from tests.conftest import API_BASE_URL, DEEPFACE_BASE_URL, TIMEOUT


class TestFaceManagement:
    """测试人脸管理相关API"""

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

    def create_test_image(self, color='blue', size=(300, 300)) -> bytes:
        """创建测试图像"""
        # 创建简单的测试图像
        image = Image.new('RGB', size, color=color)
        
        # 添加简单的"人脸"特征
        from PIL import ImageDraw
        draw = ImageDraw.Draw(image)
        
        # 绘制脸部轮廓
        center_x, center_y = size[0] // 2, size[1] // 2
        face_width, face_height = size[0] // 3, size[1] // 2
        
        # 脸部边框
        draw.rectangle([
            center_x - face_width//2, center_y - face_height//2,
            center_x + face_width//2, center_y + face_height//2
        ], outline='black', width=2)
        
        # 眼睛
        eye_y = center_y - face_height//4
        eye_size = 10
        draw.ellipse([center_x - face_width//4 - eye_size, eye_y - eye_size,
                     center_x - face_width//4 + eye_size, eye_y + eye_size], 
                    fill='black')
        draw.ellipse([center_x + face_width//4 - eye_size, eye_y - eye_size,
                     center_x + face_width//4 + eye_size, eye_y + eye_size], 
                    fill='black')
        
        # 鼻子
        nose_y = center_y
        draw.line([center_x, nose_y - 10, center_x, nose_y + 10], 
                 fill='black', width=2)
        
        # 嘴巴
        mouth_y = center_y + face_height//4
        draw.arc([center_x - 20, mouth_y - 10, center_x + 20, mouth_y + 10], 
                0, 180, fill='black', width=2)
        
        # 转换为字节
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG', quality=85)
        return buffer.getvalue()

    def test_register_face_success(self):
        """测试成功注册人脸"""
        image_data = self.create_test_image()
        test_name = "test_user_register"
        metadata = {
            "department": "技术部",
            "employee_id": "E001",
            "position": "工程师"
        }
        
        files = {"file": ("test.jpg", image_data, "image/jpeg")}
        data = {
            "name": test_name,
            "metadata": json.dumps(metadata)
        }
        
        response = self.client.post(
            f"{self.base_url}/faces/",
            files=files,
            data=data,
            timeout=TIMEOUT
        )
        
        assert response.status_code == 200
        result = response.json()
        
        # 验证响应结构
        assert result["success"] is True
        assert result["person_name"] == test_name
        assert "message" in result
        
        # 记录用于清理
        if "face_id" in result:
            self.registered_faces.append({
                "face_id": result["face_id"],
                "name": test_name
            })

    def test_register_face_without_metadata(self):
        """测试不带元数据注册人脸"""
        image_data = self.create_test_image()
        test_name = "test_user_no_metadata"
        
        files = {"file": ("test.jpg", image_data, "image/jpeg")}
        data = {"name": test_name}
        
        response = self.client.post(
            f"{self.base_url}/faces/",
            files=files,
            data=data,
            timeout=TIMEOUT
        )
        
        assert response.status_code == 200
        result = response.json()
        assert result["success"] is True
        assert result["person_name"] == test_name

    def test_register_face_invalid_image(self):
        """测试注册无效图像"""
        invalid_data = b"not an image"
        test_name = "test_invalid_image"
        
        files = {"file": ("test.txt", invalid_data, "text/plain")}
        data = {"name": test_name}
        
        response = self.client.post(
            f"{self.base_url}/faces/",
            files=files,
            data=data,
            timeout=TIMEOUT
        )
        
        # 应该返回错误
        assert response.status_code in [400, 500]

    def test_register_face_missing_name(self):
        """测试缺少姓名参数"""
        image_data = self.create_test_image()
        
        files = {"file": ("test.jpg", image_data, "image/jpeg")}
        data = {}  # 缺少name参数
        
        response = self.client.post(
            f"{self.base_url}/faces/",
            files=files,
            data=data,
            timeout=TIMEOUT
        )
        
        assert response.status_code == 422  # Validation error

    def test_register_face_invalid_metadata(self):
        """测试无效的元数据格式"""
        image_data = self.create_test_image()
        test_name = "test_invalid_metadata"
        
        files = {"file": ("test.jpg", image_data, "image/jpeg")}
        data = {
            "name": test_name,
            "metadata": "invalid json"  # 无效JSON
        }
        
        response = self.client.post(
            f"{self.base_url}/faces/",
            files=files,
            data=data,
            timeout=TIMEOUT
        )
        
        # 应该仍然成功，但忽略无效元数据
        assert response.status_code == 200

    def test_list_faces_empty(self):
        """测试获取空人脸列表"""
        response = self.client.get(f"{self.base_url}/faces/", timeout=TIMEOUT)
        
        assert response.status_code == 200
        result = response.json()
        
        # 验证响应结构
        assert "faces" in result
        assert "total" in result
        assert isinstance(result["faces"], list)
        assert isinstance(result["total"], int)

    def test_list_faces_with_data(self):
        """测试获取包含数据的人脸列表"""
        # 先注册一个人脸
        image_data = self.create_test_image()
        test_name = "test_list_user"
        
        files = {"file": ("test.jpg", image_data, "image/jpeg")}
        data = {"name": test_name}
        
        register_response = self.client.post(
            f"{self.base_url}/faces/",
            files=files,
            data=data,
            timeout=TIMEOUT
        )
        
        if register_response.status_code == 200:
            # 获取人脸列表
            list_response = self.client.get(f"{self.base_url}/faces/", timeout=TIMEOUT)
            
            assert list_response.status_code == 200
            result = list_response.json()
            
            assert result["total"] >= 1
            
            # 验证人脸数据结构
            for face in result["faces"]:
                assert "face_id" in face
                assert "person_name" in face
                # created_at 和 metadata 可能为空

    def test_get_face_by_name_existing(self):
        """测试按姓名查询存在的人脸"""
        # 先注册一个人脸
        image_data = self.create_test_image()
        test_name = "test_query_user"
        
        files = {"file": ("test.jpg", image_data, "image/jpeg")}
        data = {"name": test_name}
        
        register_response = self.client.post(
            f"{self.base_url}/faces/",
            files=files,
            data=data,
            timeout=TIMEOUT
        )
        
        if register_response.status_code == 200:
            # 按姓名查询
            query_response = self.client.get(
                f"{self.base_url}/faces/name/{test_name}",
                timeout=TIMEOUT
            )
            
            assert query_response.status_code == 200
            result = query_response.json()
            
            assert "faces" in result
            assert "total" in result
            assert result["total"] >= 1
            
            # 验证返回的人脸信息
            found_face = False
            for face in result["faces"]:
                if face["person_name"] == test_name:
                    found_face = True
                    assert "face_id" in face
                    break
            
            assert found_face, f"未找到姓名为 {test_name} 的人脸"

    def test_get_face_by_name_nonexistent(self):
        """测试按姓名查询不存在的人脸"""
        nonexistent_name = "nonexistent_user_12345"
        
        response = self.client.get(
            f"{self.base_url}/faces/name/{nonexistent_name}",
            timeout=TIMEOUT
        )
        
        assert response.status_code == 404

    def test_delete_face_existing(self):
        """测试删除存在的人脸"""
        # 先注册一个人脸
        image_data = self.create_test_image()
        test_name = "test_delete_user"
        
        files = {"file": ("test.jpg", image_data, "image/jpeg")}
        data = {"name": test_name}
        
        register_response = self.client.post(
            f"{self.base_url}/faces/",
            files=files,
            data=data,
            timeout=TIMEOUT
        )
        
        if register_response.status_code == 200:
            # 获取人脸列表以获取face_id
            list_response = self.client.get(f"{self.base_url}/faces/", timeout=TIMEOUT)
            
            if list_response.status_code == 200:
                faces = list_response.json()["faces"]
                test_face = None
                
                for face in faces:
                    if face["person_name"] == test_name:
                        test_face = face
                        break
                
                if test_face:
                    # 删除人脸
                    delete_response = self.client.delete(
                        f"{self.base_url}/faces/{test_face['face_id']}",
                        timeout=TIMEOUT
                    )
                    
                    assert delete_response.status_code == 200
                    result = delete_response.json()
                    
                    assert result["success"] is True
                    assert "message" in result

    def test_delete_face_nonexistent(self):
        """测试删除不存在的人脸"""
        fake_face_id = "nonexistent_face_id_12345"
        
        response = self.client.delete(
            f"{self.base_url}/faces/{fake_face_id}",
            timeout=TIMEOUT
        )
        
        assert response.status_code == 404

    def test_update_face_name(self):
        """测试更新人脸姓名"""
        # 先注册一个人脸
        image_data = self.create_test_image()
        original_name = "test_original_name"
        new_name = "test_updated_name"
        
        files = {"file": ("test.jpg", image_data, "image/jpeg")}
        data = {"name": original_name}
        
        register_response = self.client.post(
            f"{self.base_url}/faces/",
            files=files,
            data=data,
            timeout=TIMEOUT
        )
        
        if register_response.status_code == 200:
            # 获取face_id
            list_response = self.client.get(f"{self.base_url}/faces/", timeout=TIMEOUT)
            
            if list_response.status_code == 200:
                faces = list_response.json()["faces"]
                test_face = None
                
                for face in faces:
                    if face["person_name"] == original_name:
                        test_face = face
                        break
                
                if test_face:
                    # 更新姓名
                    update_data = {"name": new_name}
                    
                    update_response = self.client.put(
                        f"{self.base_url}/faces/{test_face['face_id']}",
                        data=update_data,
                        timeout=TIMEOUT
                    )
                    
                    assert update_response.status_code == 200
                    result = update_response.json()
                    
                    assert result["success"] is True

    def test_update_face_image(self):
        """测试更新人脸图片"""
        # 先注册一个人脸
        original_image = self.create_test_image(color='blue')
        test_name = "test_update_image"
        
        files = {"file": ("test.jpg", original_image, "image/jpeg")}
        data = {"name": test_name}
        
        register_response = self.client.post(
            f"{self.base_url}/faces/",
            files=files,
            data=data,
            timeout=TIMEOUT
        )
        
        if register_response.status_code == 200:
            # 获取face_id
            list_response = self.client.get(f"{self.base_url}/faces/", timeout=TIMEOUT)
            
            if list_response.status_code == 200:
                faces = list_response.json()["faces"]
                test_face = None
                
                for face in faces:
                    if face["person_name"] == test_name:
                        test_face = face
                        break
                
                if test_face:
                    # 更新图片
                    new_image = self.create_test_image(color='red')
                    update_files = {"file": ("new_test.jpg", new_image, "image/jpeg")}
                    
                    update_response = self.client.put(
                        f"{self.base_url}/faces/{test_face['face_id']}",
                        files=update_files,
                        timeout=TIMEOUT
                    )
                    
                    assert update_response.status_code == 200
                    result = update_response.json()
                    
                    assert result["success"] is True

    def test_update_face_nonexistent(self):
        """测试更新不存在的人脸"""
        fake_face_id = "nonexistent_face_id_12345"
        update_data = {"name": "new_name"}
        
        response = self.client.put(
            f"{self.base_url}/faces/{fake_face_id}",
            data=update_data,
            timeout=TIMEOUT
        )
        
        assert response.status_code == 404

    def test_face_management_workflow(self):
        """测试完整的人脸管理工作流"""
        test_name = "workflow_test_user"
        updated_name = "workflow_updated_user"
        
        # 1. 注册人脸
        image_data = self.create_test_image()
        metadata = {"department": "测试部", "role": "测试员"}
        
        files = {"file": ("test.jpg", image_data, "image/jpeg")}
        data = {
            "name": test_name,
            "metadata": json.dumps(metadata)
        }
        
        register_response = self.client.post(
            f"{self.base_url}/faces/",
            files=files,
            data=data,
            timeout=TIMEOUT
        )
        
        assert register_response.status_code == 200
        
        # 2. 查询人脸列表
        list_response = self.client.get(f"{self.base_url}/faces/", timeout=TIMEOUT)
        assert list_response.status_code == 200
        
        faces = list_response.json()["faces"]
        test_face = None
        for face in faces:
            if face["person_name"] == test_name:
                test_face = face
                break
        
        assert test_face is not None, "注册的人脸未在列表中找到"
        
        # 3. 按姓名查询
        name_query_response = self.client.get(
            f"{self.base_url}/faces/name/{test_name}",
            timeout=TIMEOUT
        )
        assert name_query_response.status_code == 200
        
        # 4. 更新人脸姓名
        update_data = {"name": updated_name}
        update_response = self.client.put(
            f"{self.base_url}/faces/{test_face['face_id']}",
            data=update_data,
            timeout=TIMEOUT
        )
        assert update_response.status_code == 200
        
        # 5. 验证更新
        updated_query_response = self.client.get(
            f"{self.base_url}/faces/name/{updated_name}",
            timeout=TIMEOUT
        )
        assert updated_query_response.status_code == 200
        
        # 6. 删除人脸
        delete_response = self.client.delete(
            f"{self.base_url}/faces/{test_face['face_id']}",
            timeout=TIMEOUT
        )
        assert delete_response.status_code == 200
        
        # 7. 验证删除
        final_query_response = self.client.get(
            f"{self.base_url}/faces/name/{updated_name}",
            timeout=TIMEOUT
        )
        assert final_query_response.status_code == 404

    def test_concurrent_face_registration(self):
        """测试并发人脸注册"""
        import concurrent.futures
        import threading
        
        def register_face(index):
            image_data = self.create_test_image(color=f'color_{index}')
            test_name = f"concurrent_user_{index}"
            
            files = {"file": ("test.jpg", image_data, "image/jpeg")}
            data = {"name": test_name}
            
            response = self.client.post(
                f"{self.base_url}/faces/",
                files=files,
                data=data,
                timeout=TIMEOUT
            )
            return response.status_code, test_name
        
        # 并发注册多个人脸
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(register_face, i) for i in range(3)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        # 验证结果
        successful_registrations = [r for r in results if r[0] == 200]
        assert len(successful_registrations) >= 1, "至少应有一个并发注册成功"

    def test_large_image_registration(self):
        """测试大尺寸图像注册"""
        # 创建大尺寸图像 (但不要太大以避免超时)
        large_image = self.create_test_image(size=(1920, 1080))
        test_name = "large_image_user"
        
        files = {"file": ("large_test.jpg", large_image, "image/jpeg")}
        data = {"name": test_name}
        
        response = self.client.post(
            f"{self.base_url}/faces/",
            files=files,
            data=data,
            timeout=TIMEOUT * 2  # 增加超时时间
        )
        
        # 应该能处理大图像或返回适当的错误
        assert response.status_code in [200, 400, 413, 500]

    def test_special_characters_in_name(self):
        """测试姓名中的特殊字符"""
        image_data = self.create_test_image()
        special_names = [
            "张三-李四",
            "test user with spaces",
            "用户@公司.com",
            "test_user_123",
            "José María",
            "测试用户"
        ]
        
        for name in special_names:
            files = {"file": ("test.jpg", image_data, "image/jpeg")}
            data = {"name": name}
            
            response = self.client.post(
                f"{self.base_url}/faces/",
                files=files,
                data=data,
                timeout=TIMEOUT
            )
            
            # 应该能处理特殊字符或返回适当的错误
            assert response.status_code in [200, 400]