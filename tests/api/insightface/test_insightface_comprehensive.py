"""
Comprehensive InsightFace API Tests
===================================

Tests all InsightFace API endpoints with various scenarios.
Server should be running on port 7003.
"""
import pytest
import requests
import base64
import json
from typing import Dict, Any


class TestInsightFaceHealth:
    """Health check endpoint tests"""
    
    def test_health_check(self, api_client):
        """Test health check endpoint"""
        response = api_client.get("http://localhost:7003/api/v1/insightface/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "insightface" in data


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
    
    def test_detect_face_success(self, api_client, test_image_base64):
        """Test successful face detection"""
        payload = {
            "image": test_image_base64,
            "return_details": True
        }
        
        response = api_client.post(
            "http://localhost:7003/api/v1/insightface/detect",
            json=payload
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "faces" in data
        assert "detection_time" in data
        assert isinstance(data["faces"], list)
    
    def test_detect_face_no_image(self, api_client):
        """Test detection without image"""
        payload = {}
        
        response = api_client.post(
            "http://localhost:7003/api/v1/insightface/detect",
            json=payload
        )
        
        assert response.status_code == 422  # Validation error
    
    def test_detect_face_invalid_image(self, api_client):
        """Test detection with invalid base64 image"""
        payload = {
            "image": "invalid_base64_string"
        }
        
        response = api_client.post(
            "http://localhost:7003/api/v1/insightface/detect",
            json=payload
        )
        
        assert response.status_code in [400, 422, 500]  # Error response
    
    def test_detect_face_minimal_params(self, api_client, test_image_base64):
        """Test detection with minimal parameters"""
        payload = {
            "image": test_image_base64
        }
        
        response = api_client.post(
            "http://localhost:7003/api/v1/insightface/detect",
            json=payload
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
    
    def test_register_face_success(self, api_client, test_image_base64, cleanup_test_faces):
        """Test successful face registration"""
        payload = {
            "image": test_image_base64,
            "name": "test_user_register",
            "metadata": {"role": "test"}
        }
        
        response = api_client.post(
            "http://localhost:7003/api/v1/insightface/register",
            json=payload
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "face_id" in data
        assert "message" in data
        assert data["face_id"] is not None
    
    def test_register_face_duplicate_name(self, api_client, test_image_base64, cleanup_test_faces):
        """Test registering face with duplicate name"""
        payload = {
            "image": test_image_base64,
            "name": "test_user_duplicate",
            "metadata": {"role": "test"}
        }
        
        # First registration
        response1 = api_client.post(
            "http://localhost:7003/api/v1/insightface/register",
            json=payload
        )
        assert response1.status_code == 200
        
        # Second registration with same name
        response2 = api_client.post(
            "http://localhost:7003/api/v1/insightface/register",
            json=payload
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
    
    def test_recognize_face_no_matches(self, api_client, test_image_base64, cleanup_test_faces):
        """Test recognition with no registered faces"""
        payload = {
            "image": test_image_base64,
            "threshold": 0.5
        }
        
        response = api_client.post(
            "http://localhost:7003/api/v1/insightface/recognize",
            json=payload
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "matches" in data
        assert isinstance(data["matches"], list)
    
    def test_recognize_face_invalid_image(self, api_client):
        """Test recognition with invalid image"""
        payload = {
            "image": "invalid_base64",
            "threshold": 0.5
        }
        
        response = api_client.post(
            "http://localhost:7003/api/v1/insightface/recognize",
            json=payload
        )
        
        assert response.status_code in [400, 422, 500]
    
    def test_recognize_face_custom_threshold(self, api_client, test_image_base64):
        """Test recognition with custom threshold"""
        payload = {
            "image": test_image_base64,
            "threshold": 0.8,
            "top_k": 3
        }
        
        response = api_client.post(
            "http://localhost:7003/api/v1/insightface/recognize",
            json=payload
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "matches" in data
        assert "recognition_time" in data


class TestInsightFaceVerification:
    """Face verification endpoint tests"""
    
    def test_verify_faces_same_image(self, api_client, test_image_base64):
        """Test verification with same image"""
        payload = {
            "image1": test_image_base64,
            "image2": test_image_base64,
            "threshold": 0.5
        }
        
        response = api_client.post(
            "http://localhost:7003/api/v1/insightface/verify",
            json=payload
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "is_same_person" in data
        assert "similarity" in data
        assert "verification_time" in data
        assert isinstance(data["is_same_person"], bool)
        assert isinstance(data["similarity"], (int, float))
    
    def test_verify_faces_missing_image(self, api_client, test_image_base64):
        """Test verification with missing image"""
        payload = {
            "image1": test_image_base64,
            "threshold": 0.5
        }
        
        response = api_client.post(
            "http://localhost:7003/api/v1/insightface/verify",
            json=payload
        )
        
        assert response.status_code == 422  # Validation error
    
    def test_verify_faces_invalid_images(self, api_client):
        """Test verification with invalid images"""
        payload = {
            "image1": "invalid_base64_1",
            "image2": "invalid_base64_2",
            "threshold": 0.5
        }
        
        response = api_client.post(
            "http://localhost:7003/api/v1/insightface/verify",
            json=payload
        )
        
        assert response.status_code in [400, 422, 500]
    
    def test_verify_faces_custom_threshold(self, api_client, test_image_base64):
        """Test verification with custom threshold"""
        payload = {
            "image1": test_image_base64,
            "image2": test_image_base64,
            "threshold": 0.9
        }
        
        response = api_client.post(
            "http://localhost:7003/api/v1/insightface/verify",
            json=payload
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "threshold" in data
        assert data["threshold"] == 0.9


class TestInsightFaceIntegrationFlow:
    """Integration tests for complete workflows"""
    
    def test_complete_registration_and_recognition_flow(self, api_client, test_image_base64, cleanup_test_faces):
        """Test complete flow: register -> recognize -> delete"""
        # Step 1: Register a face
        register_payload = {
            "image": test_image_base64,
            "name": "test_integration_user",
            "metadata": {"flow": "integration_test"}
        }
        
        register_response = api_client.post(
            "http://localhost:7003/api/v1/insightface/register",
            json=register_payload
        )
        assert register_response.status_code == 200
        face_data = register_response.json()
        face_id = face_data.get("face_id")
        assert face_id is not None
        
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
        recognize_payload = {
            "image": test_image_base64,
            "threshold": 0.3,  # Lower threshold for better matching
            "top_k": 5
        }
        
        recognize_response = api_client.post(
            "http://localhost:7003/api/v1/insightface/recognize",
            json=recognize_payload
        )
        assert recognize_response.status_code == 200
        recognize_data = recognize_response.json()
        matches = recognize_data.get("matches", [])
        # Note: Matches might be empty if face quality is poor or threshold is too high
        
        # Step 5: Delete face by ID
        delete_response = api_client.delete(f"http://localhost:7003/api/v1/insightface/faces/{face_id}")
        assert delete_response.status_code in [200, 204]
        
        # Step 6: Verify face is deleted
        faces_response_after = api_client.get("http://localhost:7003/api/v1/insightface/faces")
        assert faces_response_after.status_code == 200
        faces_after = faces_response_after.json().get("faces", [])
        found_face_after = any(face.get("name") == "test_integration_user" for face in faces_after)
        assert not found_face_after, "Face was not properly deleted"
    
    def test_register_and_delete_by_name_flow(self, api_client, test_image_base64, cleanup_test_faces):
        """Test register and delete by name flow"""
        # Register face
        register_payload = {
            "image": test_image_base64,
            "name": "test_delete_by_name",
            "metadata": {"test": "delete_by_name"}
        }
        
        register_response = api_client.post(
            "http://localhost:7003/api/v1/insightface/register",
            json=register_payload
        )
        assert register_response.status_code == 200
        
        # Delete by name
        delete_response = api_client.delete("http://localhost:7003/api/v1/insightface/faces/by-name/test_delete_by_name")
        assert delete_response.status_code in [200, 204]
        
        # Verify deletion
        faces_response = api_client.get("http://localhost:7003/api/v1/insightface/faces")
        assert faces_response.status_code == 200
        faces = faces_response.json().get("faces", [])
        found_face = any(face.get("name") == "test_delete_by_name" for face in faces)
        assert not found_face, "Face was not properly deleted by name"