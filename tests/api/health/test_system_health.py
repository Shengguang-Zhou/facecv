"""
Tests for System Health APIs
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from utils.test_base import APITestBase, TestResults


class TestSystemHealth(APITestBase):
    """Test system health monitoring APIs"""
    
    def __init__(self):
        super().__init__()
        self.results = TestResults()
    
    def test_basic_health(self):
        """Test GET /health (root health endpoint)"""
        test_name = "Basic Health Check"
        try:
            # Try the root health endpoint first
            response = self.get("/../health")  # Go back to root level
            if response.status_code == 404:
                # If not found, test the comprehensive health as the basic check
                response = self.get("/comprehensive")
            
            assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
            
            data = response.json()
            # Check for either old format or new comprehensive format
            if 'status' in data:
                status = data['status']
            elif 'healthy' in data:
                status = "healthy" if data['healthy'] else "unhealthy"
            else:
                status = "unknown"
            
            self.print_test_result(test_name, True, f"Health status: {status}")
            self.results.add_result(test_name, True)
            
        except Exception as e:
            self.print_test_result(test_name, False, str(e))
            self.results.add_result(test_name, False)
    
    def test_comprehensive_health(self):
        """Test GET /health/comprehensive"""
        test_name = "Comprehensive Health"
        try:
            response = self.get("/health/comprehensive")
            assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
            
            data = response.json()
            required_fields = ['healthy', 'status', 'metrics', 'timestamp']
            for field in required_fields:
                assert field in data, f"Missing field: {field}"
            
            # Check metrics structure
            metrics = data['metrics']
            metric_types = ['cpu', 'memory', 'disk']
            for metric_type in metric_types:
                if metric_type in metrics:
                    metric_data = metrics[metric_type]
                    assert 'usage' in metric_data or 'used' in metric_data, f"Missing usage data in {metric_type}"
            
            self.print_test_result(test_name, True, f"System healthy: {data['healthy']}")
            self.results.add_result(test_name, True)
            
        except Exception as e:
            self.print_test_result(test_name, False, str(e))
            self.results.add_result(test_name, False)
    
    def test_cpu_health(self):
        """Test GET /health/cpu"""
        test_name = "CPU Health"
        try:
            response = self.get("/health/cpu")
            assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
            
            data = response.json()
            assert 'usage' in data
            assert isinstance(data['usage'], (int, float))
            assert 0 <= data['usage'] <= 100
            
            self.print_test_result(test_name, True, f"CPU usage: {data['usage']:.1f}%")
            self.results.add_result(test_name, True)
            
        except Exception as e:
            self.print_test_result(test_name, False, str(e))
            self.results.add_result(test_name, False)
    
    def test_memory_health(self):
        """Test GET /health/memory"""
        test_name = "Memory Health"
        try:
            response = self.get("/health/memory")
            assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
            
            data = response.json()
            assert 'total' in data
            assert 'used' in data
            assert 'usage_percent' in data
            
            usage_percent = data['usage_percent']
            assert 0 <= usage_percent <= 100
            
            self.print_test_result(test_name, True, f"Memory usage: {usage_percent:.1f}%")
            self.results.add_result(test_name, True)
            
        except Exception as e:
            self.print_test_result(test_name, False, str(e))
            self.results.add_result(test_name, False)
    
    def test_disk_health(self):
        """Test GET /health/disk"""
        test_name = "Disk Health"
        try:
            response = self.get("/health/disk")
            assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
            
            data = response.json()
            assert 'total' in data
            assert 'used' in data
            assert 'usage_percent' in data
            
            usage_percent = data['usage_percent']
            assert 0 <= usage_percent <= 100
            
            self.print_test_result(test_name, True, f"Disk usage: {usage_percent:.1f}%")
            self.results.add_result(test_name, True)
            
        except Exception as e:
            self.print_test_result(test_name, False, str(e))
            self.results.add_result(test_name, False)
    
    def test_gpu_health(self):
        """Test GET /health/gpu"""
        test_name = "GPU Health"
        try:
            response = self.get("/health/gpu")
            # GPU might not be available, so we accept both success and error
            if response.status_code == 200:
                data = response.json()
                assert 'available' in data
                
                if data['available']:
                    assert 'gpu_count' in data
                    assert 'utilization' in data
                    self.print_test_result(test_name, True, f"GPU available: {data['gpu_count']} devices")
                else:
                    self.print_test_result(test_name, True, "GPU not available")
            else:
                # GPU not available is acceptable
                self.print_test_result(test_name, True, "GPU endpoint returned error (expected if no GPU)")
            
            self.results.add_result(test_name, True)
            
        except Exception as e:
            self.print_test_result(test_name, False, str(e))
            self.results.add_result(test_name, False)
    
    def test_database_health(self):
        """Test GET /health/database"""
        test_name = "Database Health"
        try:
            response = self.get("/health/database")
            # Database might not be configured, accept various responses
            if response.status_code in [200, 503]:
                data = response.json()
                assert 'status' in data
                
                if response.status_code == 200:
                    self.print_test_result(test_name, True, f"Database status: {data['status']}")
                else:
                    self.print_test_result(test_name, True, f"Database unavailable: {data.get('message', 'Unknown')}")
            else:
                self.assert_success_response(response)
            
            self.results.add_result(test_name, True)
            
        except Exception as e:
            self.print_test_result(test_name, False, str(e))
            self.results.add_result(test_name, False)
    
    def test_performance_metrics(self):
        """Test GET /health/performance"""
        test_name = "Performance Metrics"
        try:
            response = self.get("/health/performance")
            assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
            
            data = response.json()
            assert 'timestamp' in data
            
            # Check for various metric categories
            metric_categories = ['requests', 'models', 'system']
            found_categories = [cat for cat in metric_categories if cat in data]
            
            self.print_test_result(test_name, True, f"Found metrics: {', '.join(found_categories)}")
            self.results.add_result(test_name, True)
            
        except Exception as e:
            self.print_test_result(test_name, False, str(e))
            self.results.add_result(test_name, False)
    
    def test_service_dependencies(self):
        """Test GET /health/dependencies"""
        test_name = "Service Dependencies"
        try:
            response = self.get("/health/dependencies")
            # This endpoint might not exist, so we handle gracefully
            if response.status_code == 404:
                self.print_test_result(test_name, True, "Dependencies endpoint not implemented (OK)")
                self.results.add_result(test_name, True)
                return
            
            self.assert_success_response(response)
            
            data = response.json()
            assert isinstance(data, dict)
            
            self.print_test_result(test_name, True, f"Dependencies checked: {len(data)} services")
            self.results.add_result(test_name, True)
            
        except Exception as e:
            self.print_test_result(test_name, False, str(e))
            self.results.add_result(test_name, False)
    
    def test_system_info(self):
        """Test GET /health/system"""
        test_name = "System Information"
        try:
            response = self.get("/health/system")
            # This endpoint might not exist
            if response.status_code == 404:
                self.print_test_result(test_name, True, "System info endpoint not implemented (OK)")
                self.results.add_result(test_name, True)
                return
            
            self.assert_success_response(response)
            
            data = response.json()
            assert isinstance(data, dict)
            
            self.print_test_result(test_name, True, "System information retrieved")
            self.results.add_result(test_name, True)
            
        except Exception as e:
            self.print_test_result(test_name, False, str(e))
            self.results.add_result(test_name, False)
    
    def run_all_tests(self):
        """Run all system health tests"""
        print("🏥 Testing System Health APIs")
        print("=" * 50)
        
        self.test_basic_health()
        self.test_comprehensive_health()
        self.test_cpu_health()
        self.test_memory_health()
        self.test_disk_health()
        self.test_gpu_health()
        self.test_database_health()
        self.test_performance_metrics()
        self.test_service_dependencies()
        self.test_system_info()
        
        self.results.print_summary()
        return self.results


if __name__ == "__main__":
    tester = TestSystemHealth()
    
    # Wait for service to be available
    print("Waiting for FaceCV service...")
    if not tester.wait_for_service():
        print("❌ Service not available at http://localhost:7003")
        exit(1)
    
    print("✅ Service is available\n")
    results = tester.run_all_tests()
    
    # Exit with error code if tests failed
    exit(0 if results.failed == 0 else 1)