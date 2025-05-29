#!/usr/bin/env python3
"""
API Health Check Script
Verifies all API endpoints are responding correctly
"""

import httpx
import asyncio
import sys
import json
from typing import Dict, List, Tuple
import time

API_BASE_URL = "http://localhost:7003"
TIMEOUT = 30.0

class APIHealthChecker:
    def __init__(self, base_url: str = API_BASE_URL):
        self.base_url = base_url
        self.client = httpx.AsyncClient(base_url=base_url, timeout=TIMEOUT)
        self.results = []
        
    async def __aenter__(self):
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.aclose()
        
    async def check_endpoint(self, method: str, path: str, **kwargs) -> Tuple[bool, str]:
        """Check a single endpoint"""
        try:
            response = await self.client.request(method, path, **kwargs)
            if response.status_code < 400:
                return True, f"OK ({response.status_code})"
            else:
                return False, f"Error {response.status_code}: {response.text[:100]}"
        except Exception as e:
            return False, f"Exception: {str(e)}"
            
    async def check_health_endpoints(self):
        """Check all health endpoints"""
        print("\n=== Checking Health Endpoints ===")
        
        health_endpoints = [
            ("GET", "/"),
            ("GET", "/health"),
            ("GET", "/api/v1/health"),
            ("GET", "/api/v1/system/health"),
            ("GET", "/api/v1/insightface/health"),
            ("GET", "/api/v1/deepface/health"),
        ]
        
        for method, path in health_endpoints:
            success, message = await self.check_endpoint(method, path)
            self.results.append((path, success, message))
            status = "✓" if success else "✗"
            print(f"{status} {method} {path}: {message}")
            
    async def check_model_endpoints(self):
        """Check model management endpoints"""
        print("\n=== Checking Model Endpoints ===")
        
        model_endpoints = [
            ("GET", "/api/v1/insightface/models/status"),
            ("POST", "/api/v1/insightface/models/clear"),
        ]
        
        for method, path in model_endpoints:
            success, message = await self.check_endpoint(method, path)
            self.results.append((path, success, message))
            status = "✓" if success else "✗"
            print(f"{status} {method} {path}: {message}")
            
    async def check_face_management_endpoints(self):
        """Check face management endpoints"""
        print("\n=== Checking Face Management Endpoints ===")
        
        face_endpoints = [
            ("GET", "/api/v1/insightface/faces"),
            ("GET", "/api/v1/deepface/faces"),
        ]
        
        for method, path in face_endpoints:
            success, message = await self.check_endpoint(method, path)
            self.results.append((path, success, message))
            status = "✓" if success else "✗"
            print(f"{status} {method} {path}: {message}")
            
    async def check_stream_endpoints(self):
        """Check streaming endpoints"""
        print("\n=== Checking Stream Endpoints ===")
        
        stream_endpoints = [
            ("GET", "/api/v1/insightface/stream/sources"),
            ("GET", "/api/v1/deepface/stream/sources"),
        ]
        
        for method, path in stream_endpoints:
            success, message = await self.check_endpoint(method, path)
            self.results.append((path, success, message))
            status = "✓" if success else "✗"
            print(f"{status} {method} {path}: {message}")
            
    async def wait_for_server(self, max_retries: int = 30):
        """Wait for server to be ready"""
        print(f"Waiting for server at {self.base_url}...")
        
        for i in range(max_retries):
            try:
                response = await self.client.get("/health")
                if response.status_code == 200:
                    print("Server is ready!")
                    return True
            except:
                pass
            
            print(f"Retry {i+1}/{max_retries}...")
            await asyncio.sleep(2)
            
        print("Server failed to start!")
        return False
        
    async def run_all_checks(self):
        """Run all health checks"""
        print(f"\n{'='*50}")
        print(f"API Health Check - {self.base_url}")
        print(f"{'='*50}")
        
        # Wait for server
        if not await self.wait_for_server():
            return False
            
        # Run all checks
        await self.check_health_endpoints()
        await self.check_model_endpoints()
        await self.check_face_management_endpoints()
        await self.check_stream_endpoints()
        
        # Summary
        print(f"\n{'='*50}")
        print("SUMMARY")
        print(f"{'='*50}")
        
        total = len(self.results)
        passed = sum(1 for _, success, _ in self.results if success)
        failed = total - passed
        
        print(f"Total endpoints checked: {total}")
        print(f"Passed: {passed}")
        print(f"Failed: {failed}")
        
        if failed > 0:
            print("\nFailed endpoints:")
            for path, success, message in self.results:
                if not success:
                    print(f"  - {path}: {message}")
                    
        return failed == 0


async def main():
    """Main function"""
    # Allow custom base URL
    base_url = sys.argv[1] if len(sys.argv) > 1 else API_BASE_URL
    
    async with APIHealthChecker(base_url) as checker:
        success = await checker.run_all_checks()
        
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())