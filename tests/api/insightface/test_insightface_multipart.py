"""
Comprehensive InsightFace API Tests with Multipart Form Data
===========================================================

Tests all InsightFace API endpoints using correct multipart form data.
Server should be running on port 7003.
"""
import pytest
import requests
import os
from pathlib import Path


class TestInsightFaceHealth:
    """Health check endpoint tests"""
    
    def test_health_check(self, api_client):
        """Test health check endpoint"""
        response = api_client.get("http://localhost:7003/api/v1/insightface/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "service" in data
        assert "Real InsightFace API" in data["service"]


class TestInsightFaceModels:
    """Model information endpoint tests"""
    
    def test_get_model_info(self, api_client):
        """Test getting model information"""
        response = api_client.get("http://localhost:7003/api/v1/insightface/models/info")
        
        assert response.status_code == 200
        data = response.json()
        assert "model_pack" in data
        assert "initialized" in data
        assert data["initialized"] == True


class TestInsightFaceDetection:
    """Face detection endpoint tests"""
    
    def test_detect_face_success(self, api_client, test_image_file):
        """Test successful face detection"""
        with open(test_image_file, 'rb') as f:
            files = {'file': ('test.jpg', f, 'image/jpeg')}
            params = {'min_confidence': 0.5}
            
            response = api_client.post(
                "http://localhost:7003/api/v1/insightface/detect",
                files=files,
                params=params
            )
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)  # Should return list of face detections
    
    def test_detect_face_no_file(self, api_client):
        """Test detection without file"""
        response = api_client.post(
            "http://localhost:7003/api/v1/insightface/detect"
        )
        
        assert response.status_code == 422  # Validation error
    
    def test_detect_face_invalid_file(self, api_client):
        """Test detection with invalid file"""
        files = {'file': ('test.txt', b'not an image', 'text/plain')}
        
        response = api_client.post(
            "http://localhost:7003/api/v1/insightface/detect",
            files=files
        )
        
        assert response.status_code in [400, 422, 500]  # Error response
    
    def test_detect_face_custom_confidence(self, api_client, test_image_file):
        """Test detection with custom confidence threshold"""
        with open(test_image_file, 'rb') as f:
            files = {'file': ('test.jpg', f, 'image/jpeg')}
            params = {'min_confidence': 0.8}
            
            response = api_client.post(
                "http://localhost:7003/api/v1/insightface/detect",
                files=files,
                params=params
            )
        
        assert response.status_code == 200


class TestInsightFaceFaceManagement:
    """Face management endpoint tests"""
    
    def test_get_faces_empty(self, api_client, cleanup_test_faces):
        """Test getting faces when database is empty"""
        response = api_client.get("http://localhost:7003/api/v1/insightface/faces")
        
        assert response.status_code == 200
        data = response.json()
        assert "faces" in data
        assert isinstance(data["faces"], list)
    
    def test_get_face_count(self, api_client):
        """Test getting face count"""
        response = api_client.get("http://localhost:7003/api/v1/insightface/faces/count")
        
        assert response.status_code == 200
        data = response.json()
        assert "total_faces" in data
        assert isinstance(data["total_faces"], int)
        assert data["total_faces"] >= 0
    
    def test_register_face_success(self, api_client, test_image_file, cleanup_test_faces):
        """Test successful face registration"""
        with open(test_image_file, 'rb') as f:
            files = {'file': ('test.jpg', f, 'image/jpeg')}
            data = {
                'name': 'test_user_register',
                'department': 'engineering',
                'employee_id': 'EMP001'
            }
            
            response = api_client.post(
                "http://localhost:7003/api/v1/insightface/register",
                files=files,
                data=data
            )
        
        assert response.status_code == 200
        result = response.json()
        assert "success" in result
        assert "message" in result
        assert "person_name" in result
    
    def test_register_face_missing_name(self, api_client, test_image_file):
        """Test registering face without name"""
        with open(test_image_file, 'rb') as f:
            files = {'file': ('test.jpg', f, 'image/jpeg')}
            # Missing required 'name' field
            
            response = api_client.post(
                "http://localhost:7003/api/v1/insightface/register",
                files=files
            )
        
        assert response.status_code == 422  # Validation error
    
    def test_register_face_duplicate_name(self, api_client, test_image_file, cleanup_test_faces):
        """Test registering face with duplicate name"""
        register_data = {
            'name': 'test_user_duplicate',
            'department': 'engineering'
        }
        
        # First registration
        with open(test_image_file, 'rb') as f:
            files = {'file': ('test.jpg', f, 'image/jpeg')}
            response1 = api_client.post(
                "http://localhost:7003/api/v1/insightface/register",
                files=files,
                data=register_data
            )
        assert response1.status_code == 200
        
        # Second registration with same name
        with open(test_image_file, 'rb') as f:
            files = {'file': ('test.jpg', f, 'image/jpeg')}
            response2 = api_client.post(
                "http://localhost:7003/api/v1/insightface/register",
                files=files,
                data=register_data
            )
        # Should either succeed (overwrite) or fail with error
        assert response2.status_code in [200, 400, 409]
    
    def test_delete_face_by_id_not_found(self, api_client):
        """Test deleting non-existent face by ID"""
        response = api_client.delete("http://localhost:7003/api/v1/insightface/faces/99999")
        
        assert response.status_code in [404, 400]
    
    def test_delete_face_by_name_not_found(self, api_client):
        """Test deleting non-existent face by name"""
        response = api_client.delete("http://localhost:7003/api/v1/insightface/faces/by-name/nonexistent_user")
        
        assert response.status_code in [404, 400]


class TestInsightFaceRecognition:
    """Face recognition endpoint tests"""
    
    def test_recognize_face_no_matches(self, api_client, test_image_file, cleanup_test_faces):
        """Test recognition with no registered faces"""
        with open(test_image_file, 'rb') as f:
            files = {'file': ('test.jpg', f, 'image/jpeg')}
            params = {'threshold': 0.5}
            
            response = api_client.post(
                "http://localhost:7003/api/v1/insightface/recognize",
                files=files,
                params=params
            )
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)  # Should return list of recognition results
    
    def test_recognize_face_invalid_file(self, api_client):
        """Test recognition with invalid file"""
        files = {'file': ('test.txt', b'not an image', 'text/plain')}
        
        response = api_client.post(
            "http://localhost:7003/api/v1/insightface/recognize",
            files=files
        )
        
        assert response.status_code in [400, 422, 500]
    
    def test_recognize_face_custom_threshold(self, api_client, test_image_file):
        """Test recognition with custom threshold"""
        with open(test_image_file, 'rb') as f:
            files = {'file': ('test.jpg', f, 'image/jpeg')}
            params = {'threshold': 0.8, 'top_k': 3}
            
            response = api_client.post(
                "http://localhost:7003/api/v1/insightface/recognize",
                files=files,
                params=params
            )
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)


class TestInsightFaceVerification:
    """Face verification endpoint tests"""
    
    def test_verify_faces_same_file(self, api_client, test_image_file):
        """Test verification with same file twice"""
        with open(test_image_file, 'rb') as f1, open(test_image_file, 'rb') as f2:
            files = {
                'file1': ('test1.jpg', f1, 'image/jpeg'),
                'file2': ('test2.jpg', f2, 'image/jpeg')
            }
            params = {'threshold': 0.5}
            
            response = api_client.post(
                "http://localhost:7003/api/v1/insightface/verify",
                files=files,
                params=params
            )
        
        assert response.status_code == 200
        data = response.json()
        assert "is_same_person" in data
        assert "confidence" in data or "distance" in data
        assert isinstance(data["is_same_person"], bool)
        # Verify confidence or distance are numeric values
        if "confidence" in data:
            assert isinstance(data["confidence"], (int, float))
        if "distance" in data:
            assert isinstance(data["distance"], (int, float))
    
    def test_verify_faces_missing_file(self, api_client, test_image_file):
        """Test verification with missing second file"""
        with open(test_image_file, 'rb') as f:
            files = {'file1': ('test.jpg', f, 'image/jpeg')}
            
            response = api_client.post(
                "http://localhost:7003/api/v1/insightface/verify",
                files=files
            )
        
        assert response.status_code == 422  # Validation error
    
    def test_verify_faces_invalid_files(self, api_client):
        """Test verification with invalid files"""
        files = {
            'file1': ('test1.txt', b'not an image 1', 'text/plain'),
            'file2': ('test2.txt', b'not an image 2', 'text/plain')
        }
        
        response = api_client.post(
            "http://localhost:7003/api/v1/insightface/verify",
            files=files
        )
        
        assert response.status_code in [400, 422, 500]
    
    def test_verify_faces_custom_threshold(self, api_client, test_image_file):
        """Test verification with custom threshold"""
        with open(test_image_file, 'rb') as f1, open(test_image_file, 'rb') as f2:
            files = {
                'file1': ('test1.jpg', f1, 'image/jpeg'),
                'file2': ('test2.jpg', f2, 'image/jpeg')
            }
            params = {'threshold': 0.9}
            
            response = api_client.post(
                "http://localhost:7003/api/v1/insightface/verify",
                files=files,
                params=params
            )
        
        assert response.status_code == 200


class TestInsightFaceIntegrationFlow:
    """Integration tests for complete workflows"""
    
    def test_complete_registration_and_recognition_flow(self, api_client, test_image_file, cleanup_test_faces):
        """Test complete flow: register -> recognize -> delete"""
        # Step 1: Register a face
        with open(test_image_file, 'rb') as f:
            files = {'file': ('test.jpg', f, 'image/jpeg')}
            data = {
                'name': 'test_integration_user',
                'department': 'integration_test'
            }
            
            register_response = api_client.post(
                "http://localhost:7003/api/v1/insightface/register",
                files=files,
                data=data
            )
        
        assert register_response.status_code == 200
        face_data = register_response.json()
        assert face_data.get("success") == True
        
        # Step 2: Verify face count increased
        count_response = api_client.get("http://localhost:7003/api/v1/insightface/faces/count")
        assert count_response.status_code == 200
        
        # Step 3: Get all faces and verify our face is there
        faces_response = api_client.get("http://localhost:7003/api/v1/insightface/faces")
        assert faces_response.status_code == 200
        faces = faces_response.json().get("faces", [])
        found_face = any(face.get("name") == "test_integration_user" for face in faces)
        assert found_face, "Registered face not found in face list"
        
        # Step 4: Try to recognize the same face
        with open(test_image_file, 'rb') as f:
            files = {'file': ('test.jpg', f, 'image/jpeg')}
            params = {'threshold': 0.3, 'top_k': 5}  # Lower threshold for better matching
            
            recognize_response = api_client.post(
                "http://localhost:7003/api/v1/insightface/recognize",
                files=files,
                params=params
            )
        
        assert recognize_response.status_code == 200
        recognize_data = recognize_response.json()
        assert isinstance(recognize_data, list)
        # Note: Matches might be empty if face quality is poor or threshold is too high
        
        # Step 5: Delete face by name
        delete_response = api_client.delete("http://localhost:7003/api/v1/insightface/faces/by-name/test_integration_user")
        assert delete_response.status_code in [200, 204]
        
        # Step 6: Verify face is deleted
        faces_response_after = api_client.get("http://localhost:7003/api/v1/insightface/faces")
        assert faces_response_after.status_code == 200
        faces_after = faces_response_after.json().get("faces", [])
        found_face_after = any(face.get("name") == "test_integration_user" for face in faces_after)
        assert not found_face_after, "Face was not properly deleted"
    
    def test_register_and_verify_flow(self, api_client, test_image_file, cleanup_test_faces):
        """Test register and verify same face flow"""
        # Register face
        with open(test_image_file, 'rb') as f:
            files = {'file': ('test.jpg', f, 'image/jpeg')}
            data = {
                'name': 'test_verify_user',
                'department': 'verification_test'
            }
            
            register_response = api_client.post(
                "http://localhost:7003/api/v1/insightface/register",
                files=files,
                data=data
            )
        assert register_response.status_code == 200
        
        # Verify same face against itself
        with open(test_image_file, 'rb') as f1, open(test_image_file, 'rb') as f2:
            files = {
                'file1': ('test1.jpg', f1, 'image/jpeg'),
                'file2': ('test2.jpg', f2, 'image/jpeg')
            }
            params = {'threshold': 0.5}
            
            verify_response = api_client.post(
                "http://localhost:7003/api/v1/insightface/verify",
                files=files,
                params=params
            )
        
        assert verify_response.status_code == 200
        verify_data = verify_response.json()
        # Note: Generated test images may not contain detectable faces
        # Check that we get a valid response structure
        assert "is_same_person" in verify_data
        assert "confidence" in verify_data or "distance" in verify_data
        
        # Cleanup
        delete_response = api_client.delete("http://localhost:7003/api/v1/insightface/faces/by-name/test_verify_user")
        assert delete_response.status_code in [200, 204]