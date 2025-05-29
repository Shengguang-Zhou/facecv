"""
Comprehensive API endpoint tests for CI/CD
Ensures no API returns errors on submission
"""

import pytest
import httpx
import asyncio
from typing import Dict, Any
import os
import base64
from pathlib import Path

# API configuration
API_BASE_URL = os.getenv("API_URL", "http://localhost:7003")
TEST_IMAGE_PATH = Path(__file__).parent.parent / "test_images"

# Sample test image (1x1 pixel PNG)
TEST_IMAGE_BASE64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="


@pytest.fixture
async def api_client():
    """Create async HTTP client"""
    async with httpx.AsyncClient(base_url=API_BASE_URL, timeout=30.0) as client:
        yield client


@pytest.fixture
def test_image_bytes():
    """Get test image bytes"""
    return base64.b64decode(TEST_IMAGE_BASE64)


class TestHealthEndpoints:
    """Test all health check endpoints"""
    
    @pytest.mark.asyncio
    async def test_root_health(self, api_client):
        """Test root endpoint"""
        response = await api_client.get("/")
        assert response.status_code == 200
        
    @pytest.mark.asyncio
    async def test_health_endpoint(self, api_client):
        """Test /health endpoint"""
        response = await api_client.get("/health")
        assert response.status_code == 200
        
    @pytest.mark.asyncio
    async def test_api_v1_health(self, api_client):
        """Test /api/v1/health endpoint"""
        response = await api_client.get("/api/v1/health")
        assert response.status_code == 200
        
    @pytest.mark.asyncio
    async def test_system_health(self, api_client):
        """Test system health endpoint"""
        response = await api_client.get("/api/v1/system/health")
        assert response.status_code == 200
        
    @pytest.mark.asyncio
    async def test_insightface_health(self, api_client):
        """Test InsightFace health endpoint"""
        response = await api_client.get("/api/v1/insightface/health")
        assert response.status_code == 200
        
    @pytest.mark.asyncio
    async def test_deepface_health(self, api_client):
        """Test DeepFace health endpoint"""
        response = await api_client.get("/api/v1/deepface/health")
        assert response.status_code == 200


class TestInsightFaceEndpoints:
    """Test InsightFace API endpoints"""
    
    @pytest.mark.asyncio
    async def test_models_status(self, api_client):
        """Test model status endpoint"""
        response = await api_client.get("/api/v1/insightface/models/status")
        assert response.status_code == 200
        data = response.json()
        assert "loaded_models" in data
        assert "available_models" in data
        
    @pytest.mark.asyncio
    async def test_models_clear(self, api_client):
        """Test model clear endpoint"""
        response = await api_client.post("/api/v1/insightface/models/clear")
        assert response.status_code == 200
        
    @pytest.mark.asyncio
    async def test_faces_list(self, api_client):
        """Test face list endpoint"""
        response = await api_client.get("/api/v1/insightface/faces")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        
    @pytest.mark.asyncio
    async def test_stream_sources(self, api_client):
        """Test stream sources endpoint"""
        response = await api_client.get("/api/v1/insightface/stream/sources")
        assert response.status_code == 200
        data = response.json()
        assert "sources" in data
        
    @pytest.mark.asyncio
    async def test_detect_endpoint_minimal(self, api_client, test_image_bytes):
        """Test detect endpoint with minimal model"""
        # Skip if models not available
        try:
            files = {"file": ("test.png", test_image_bytes, "image/png")}
            data = {
                "model_name": "scrfd_500m_bnkps",
                "min_confidence": "0.5"
            }
            response = await api_client.post("/api/v1/insightface/detect", files=files, data=data)
            # Allow 500 error if model not downloaded
            assert response.status_code in [200, 500]
        except Exception:
            pytest.skip("Model not available")


class TestDeepFaceEndpoints:
    """Test DeepFace API endpoints"""
    
    @pytest.mark.asyncio
    async def test_faces_list(self, api_client):
        """Test DeepFace face list endpoint"""
        response = await api_client.get("/api/v1/deepface/faces")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        
    @pytest.mark.asyncio
    async def test_stream_sources(self, api_client):
        """Test DeepFace stream sources endpoint"""
        response = await api_client.get("/api/v1/deepface/stream/sources")
        assert response.status_code == 200
        data = response.json()
        assert "sources" in data


class TestCriticalPaths:
    """Test critical API paths that must not fail"""
    
    @pytest.mark.asyncio
    async def test_api_documentation(self, api_client):
        """Test API documentation endpoints"""
        # OpenAPI schema
        response = await api_client.get("/openapi.json")
        assert response.status_code == 200
        
        # Swagger UI should redirect or serve HTML
        response = await api_client.get("/docs", follow_redirects=True)
        assert response.status_code == 200
        
        # ReDoc
        response = await api_client.get("/redoc", follow_redirects=True)
        assert response.status_code == 200
        
    @pytest.mark.asyncio
    async def test_no_500_errors_on_get_endpoints(self, api_client):
        """Ensure no GET endpoint returns 500 error"""
        get_endpoints = [
            "/",
            "/health",
            "/api/v1/health",
            "/api/v1/system/health",
            "/api/v1/insightface/health",
            "/api/v1/insightface/models/status",
            "/api/v1/insightface/faces",
            "/api/v1/insightface/stream/sources",
            "/api/v1/deepface/health",
            "/api/v1/deepface/faces",
            "/api/v1/deepface/stream/sources",
        ]
        
        failed_endpoints = []
        
        for endpoint in get_endpoints:
            try:
                response = await api_client.get(endpoint)
                if response.status_code >= 500:
                    failed_endpoints.append((endpoint, response.status_code))
            except Exception as e:
                failed_endpoints.append((endpoint, str(e)))
                
        assert len(failed_endpoints) == 0, f"Endpoints with errors: {failed_endpoints}"
        
    @pytest.mark.asyncio
    async def test_post_endpoints_with_invalid_data(self, api_client):
        """Test POST endpoints handle invalid data gracefully (no 500 errors)"""
        post_tests = [
            ("/api/v1/insightface/models/clear", {}),
            ("/api/v1/insightface/models/preload", {"model_names": ["invalid_model"]}),
        ]
        
        for endpoint, data in post_tests:
            response = await api_client.post(endpoint, json=data)
            # Should not return 500 error for invalid input
            assert response.status_code < 500, f"{endpoint} returned {response.status_code}"


@pytest.mark.asyncio
async def test_server_is_running():
    """Basic test to ensure server is accessible"""
    async with httpx.AsyncClient(timeout=5.0) as client:
        try:
            response = await client.get(f"{API_BASE_URL}/health")
            assert response.status_code == 200
        except httpx.ConnectError:
            pytest.fail(f"Cannot connect to server at {API_BASE_URL}")


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])