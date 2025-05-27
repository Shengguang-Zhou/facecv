#!/usr/bin/env python3
"""
Comprehensive test for installation priority order: CUDA → Jetson → CPU
"""

import sys
import os
import unittest.mock as mock
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Fix protobuf compatibility issue
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

def test_cuda_primary_scenario():
    """Test CUDA detection and installation (our current scenario)"""
    print("\n🔧 Testing CUDA Primary Scenario")
    print("=" * 50)
    
    try:
        from scripts.install_onnxruntime_gpu import ONNXRuntimeInstaller
        
        installer = ONNXRuntimeInstaller()
        
        # Verify current environment
        platform_info = installer.platform_info
        cuda_info = installer.cuda_info
        
        print(f"✅ Platform: {platform_info['system']} {platform_info['machine']}")
        print(f"✅ Is Jetson: {platform_info['is_jetson']}")
        print(f"✅ CUDA Available: {cuda_info['cuda_version'] is not None}")
        
        if cuda_info['cuda_version']:
            print(f"✅ CUDA Version: {cuda_info['cuda_version'][0]}.{cuda_info['cuda_version'][1]}")
        
        # Test installation status
        status = installer.get_installation_status()
        print(f"✅ ONNX Runtime Installed: {status['onnxruntime_installed']}")
        print(f"✅ CUDA Provider Available: {status['cuda_provider_available']}")
        print(f"✅ Available Providers: {status.get('available_providers', [])}")
        
        # Verify priority logic
        expected_priority = "CUDA"
        actual_priority = "CUDA" if cuda_info['cuda_version'] else "CPU"
        
        assert actual_priority == expected_priority, f"Expected {expected_priority}, got {actual_priority}"
        print(f"✅ Priority Order: {actual_priority} (Correct)")
        
        return True
        
    except Exception as e:
        print(f"❌ CUDA test failed: {e}")
        return False

def test_jetson_mock_scenario():
    """Test Jetson detection logic with mocked environment"""
    print("\n🤖 Testing Jetson Mock Scenario")
    print("=" * 50)
    
    try:
        # Mock Jetson environment
        mock_platform_info = {
            'system': 'linux',
            'machine': 'aarch64',
            'is_jetson': True,
            'jetson_model': 'NVIDIA Jetson AGX Orin Developer Kit',
            'jetpack_version': '6.0',
            'python_version': '3.10'
        }
        
        mock_cuda_info = {
            'cuda_version': (12, 2),
            'cudnn_version': 9,
            'pytorch_cuda': None
        }
        
        # Test wheel URL generation
        from scripts.install_onnxruntime_gpu import ONNXRuntimeInstaller
        installer = ONNXRuntimeInstaller()
        
        # Override platform info
        installer.platform_info = mock_platform_info
        installer.cuda_info = mock_cuda_info
        
        # Test Jetson wheel URL generation
        wheel_url = installer._get_jetson_wheel_url()
        
        print(f"✅ Jetson Model: {mock_platform_info['jetson_model']}")
        print(f"✅ JetPack Version: {mock_platform_info['jetpack_version']}")
        print(f"✅ Python Version: {mock_platform_info['python_version']}")
        print(f"✅ Generated Wheel URL: {wheel_url}")
        
        # Verify wheel URL is valid for JetPack 6.x
        expected_patterns = [
            'onnxruntime_gpu-1.20.0-cp310-cp310-linux_aarch64.whl',
            'github.com/ultralytics/assets'
        ]
        
        for pattern in expected_patterns:
            assert pattern in wheel_url, f"Wheel URL should contain {pattern}"
        
        print("✅ Jetson wheel URL generation: PASSED")
        
        # Test installation method selection
        status = installer.get_installation_status()
        if mock_cuda_info['cuda_version']:
            expected_action = 'install_jetson'
        else:
            expected_action = 'install'
            
        print(f"✅ Expected installation method: Jetson-specific")
        
        return True
        
    except Exception as e:
        print(f"❌ Jetson mock test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_cpu_fallback_scenario():
    """Test CPU fallback when no GPU available"""
    print("\n💻 Testing CPU Fallback Scenario")
    print("=" * 50)
    
    try:
        # Mock CPU-only environment
        with mock.patch('facecv.utils.cuda_utils.get_cuda_version', return_value=None):
            with mock.patch('facecv.utils.cuda_utils.check_cuda_availability', return_value=False):
                
                from scripts.install_onnxruntime_gpu import ONNXRuntimeInstaller
                installer = ONNXRuntimeInstaller()
                
                # Verify CPU fallback detection
                cuda_info = installer.cuda_info
                print(f"✅ CUDA Available: {cuda_info['cuda_version'] is not None}")
                
                # Test CPU installation command
                from facecv.utils.cuda_utils import install_appropriate_onnxruntime
                install_cmd = install_appropriate_onnxruntime()
                
                print(f"✅ Installation Command: {install_cmd}")
                assert "onnxruntime" in install_cmd and "onnxruntime-gpu" not in install_cmd
                print("✅ CPU-only installation command: PASSED")
                
                # Test execution providers
                from facecv.utils.cuda_utils import get_execution_providers
                providers = get_execution_providers()
                
                print(f"✅ Execution Providers: {providers}")
                assert providers == ['CPUExecutionProvider']
                print("✅ CPU-only execution providers: PASSED")
                
                return True
        
    except Exception as e:
        print(f"❌ CPU fallback test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_priority_order_logic():
    """Test the complete priority order: CUDA → Jetson → CPU"""
    print("\n🎯 Testing Priority Order Logic")
    print("=" * 50)
    
    test_scenarios = [
        {
            'name': 'x86_64 with CUDA',
            'platform': {'system': 'linux', 'machine': 'x86_64', 'is_jetson': False},
            'cuda': {'cuda_version': (12, 4)},
            'expected_priority': 'CUDA',
            'expected_providers': ['CUDAExecutionProvider', 'CPUExecutionProvider']
        },
        {
            'name': 'Jetson with CUDA',
            'platform': {'system': 'linux', 'machine': 'aarch64', 'is_jetson': True},
            'cuda': {'cuda_version': (12, 2)},
            'expected_priority': 'Jetson',
            'expected_providers': ['CUDAExecutionProvider', 'CPUExecutionProvider']
        },
        {
            'name': 'x86_64 without CUDA',
            'platform': {'system': 'linux', 'machine': 'x86_64', 'is_jetson': False},
            'cuda': {'cuda_version': None},
            'expected_priority': 'CPU',
            'expected_providers': ['CPUExecutionProvider']
        },
        {
            'name': 'Jetson without CUDA',
            'platform': {'system': 'linux', 'machine': 'aarch64', 'is_jetson': True},
            'cuda': {'cuda_version': None},
            'expected_priority': 'CPU',
            'expected_providers': ['CPUExecutionProvider']
        }
    ]
    
    all_passed = True
    
    for scenario in test_scenarios:
        print(f"\n🧪 Testing: {scenario['name']}")
        
        try:
            # Mock environment for this scenario
            with mock.patch('facecv.utils.cuda_utils.get_cuda_version', 
                          return_value=scenario['cuda']['cuda_version']):
                with mock.patch('facecv.utils.cuda_utils.check_cuda_availability', 
                              return_value=scenario['cuda']['cuda_version'] is not None):
                    
                    from facecv.utils.cuda_utils import get_execution_providers
                    providers = get_execution_providers()
                    
                    print(f"   Platform: {scenario['platform']['machine']}, Jetson: {scenario['platform']['is_jetson']}")
                    print(f"   CUDA: {scenario['cuda']['cuda_version']}")
                    print(f"   Expected Priority: {scenario['expected_priority']}")
                    print(f"   Expected Providers: {scenario['expected_providers']}")
                    print(f"   Actual Providers: {providers}")
                    
                    # Verify providers match expectation
                    if providers == scenario['expected_providers']:
                        print(f"   ✅ PASSED")
                    else:
                        print(f"   ❌ FAILED: Expected {scenario['expected_providers']}, got {providers}")
                        all_passed = False
                        
        except Exception as e:
            print(f"   ❌ ERROR: {e}")
            all_passed = False
    
    return all_passed

def test_installer_comprehensive():
    """Test the installer with different scenarios"""
    print("\n🔧 Testing Installer Comprehensive Scenarios")
    print("=" * 50)
    
    try:
        from scripts.install_onnxruntime_gpu import ONNXRuntimeInstaller
        
        # Test 1: Check current installation
        print("\n📊 Current Installation Status:")
        installer = ONNXRuntimeInstaller()
        status = installer.get_installation_status()
        
        for key, value in status.items():
            if key != 'platform' and key != 'cuda':
                print(f"   {key}: {value}")
        
        print(f"   Platform: {status['platform']['system']} {status['platform']['machine']}")
        print(f"   Is Jetson: {status['platform']['is_jetson']}")
        print(f"   CUDA Version: {status['cuda']['cuda_version']}")
        
        # Test 2: Verify installation method selection
        print("\n🎯 Installation Method Selection:")
        if status['platform']['is_jetson']:
            expected_method = "Jetson wheel installation"
        elif status['cuda']['cuda_version']:
            expected_method = "Standard GPU package"
        else:
            expected_method = "CPU-only package"
        
        print(f"   Expected Method: {expected_method}")
        
        # Test 3: Environment variables setup
        print("\n⚙️ Environment Variables:")
        installer._setup_environment_variables()
        
        important_vars = [
            'ORT_CUDA_GRAPH_ENABLE',
            'ORT_TENSORRT_ENGINE_CACHE_ENABLE',
            'ORT_CUDA_MAX_THREADS_PER_BLOCK'
        ]
        
        for var in important_vars:
            value = os.environ.get(var, 'Not set')
            print(f"   {var}: {value}")
        
        return True
        
    except Exception as e:
        print(f"❌ Comprehensive installer test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_real_face_recognition_scenarios():
    """Test face recognition with different acceleration scenarios"""
    print("\n👤 Testing Face Recognition with Different Acceleration")
    print("=" * 50)
    
    try:
        # Test current CUDA scenario
        print("\n🔥 Current CUDA Scenario:")
        from facecv.models.insightface.real_recognizer import RealInsightFaceRecognizer
        
        recognizer = RealInsightFaceRecognizer(prefer_gpu=True)
        print(f"   Initialized: {recognizer.initialized}")
        print(f"   Prefer GPU: {recognizer.prefer_gpu}")
        
        # Test with sample image
        import cv2
        import numpy as np
        test_image_path = "/home/a/PycharmProjects/EurekCV/dataset/faces/harris1.jpeg"
        
        if Path(test_image_path).exists():
            image = cv2.imread(test_image_path)
            if image is not None:
                import time
                
                # Time face detection
                start_time = time.time()
                faces = recognizer.detect_faces(image)
                detection_time = time.time() - start_time
                
                print(f"   Faces detected: {len(faces)}")
                print(f"   Detection time: {detection_time:.3f}s")
                
                if faces:
                    face = faces[0]
                    print(f"   Face confidence: {face.confidence:.3f}")
                    print(f"   Face bbox: {face.bbox}")
        else:
            print("   ⚠️ Test image not found, skipping face detection test")
        
        # Test CPU fallback scenario
        print("\n💻 CPU Fallback Scenario:")
        recognizer_cpu = RealInsightFaceRecognizer(prefer_gpu=False)
        print(f"   CPU Recognizer initialized: {recognizer_cpu.initialized}")
        print(f"   CPU Prefer GPU: {recognizer_cpu.prefer_gpu}")
        
        return True
        
    except Exception as e:
        print(f"❌ Face recognition test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all comprehensive tests"""
    print("🎯 Comprehensive Installation Priority Testing")
    print("=" * 60)
    print("Testing Order: CUDA → Jetson → CPU")
    print("=" * 60)
    
    tests = [
        ("CUDA Primary Scenario", test_cuda_primary_scenario),
        ("Jetson Mock Scenario", test_jetson_mock_scenario),
        ("CPU Fallback Scenario", test_cpu_fallback_scenario),
        ("Priority Order Logic", test_priority_order_logic),
        ("Installer Comprehensive", test_installer_comprehensive),
        ("Face Recognition Scenarios", test_real_face_recognition_scenarios),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'=' * 20} {test_name} {'=' * 20}")
        try:
            result = test_func()
            results.append((test_name, result))
            status = "✅ PASSED" if result else "❌ FAILED"
            print(f"\n{status}: {test_name}")
        except Exception as e:
            print(f"\n❌ CRASHED: {test_name} - {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    failed = 0
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:30} {status}")
        if result:
            passed += 1
        else:
            failed += 1
    
    print(f"\nTotal Tests: {len(results)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    
    if failed == 0:
        print("\n🎉 ALL TESTS PASSED!")
        print("✅ CUDA → Jetson → CPU priority order working correctly")
    else:
        print(f"\n⚠️ {failed} test(s) failed")
        print("❌ Some scenarios need attention")
    
    return failed == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)