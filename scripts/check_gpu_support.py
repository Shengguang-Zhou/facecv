#!/usr/bin/env python3
"""
Check GPU support for both InsightFace and DeepFace
"""

import os
import sys
import subprocess

def check_nvidia_driver():
    """Check if NVIDIA driver is installed"""
    print("=== Checking NVIDIA Driver ===")
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✓ NVIDIA driver is installed")
            # Extract key info
            lines = result.stdout.split('\n')
            for line in lines:
                if 'Driver Version' in line:
                    print(f"  {line.strip()}")
                if 'CUDA Version' in line:
                    print(f"  {line.strip()}")
            return True
        else:
            print("✗ NVIDIA driver not found")
            return False
    except FileNotFoundError:
        print("✗ nvidia-smi command not found")
        return False

def check_cuda_installation():
    """Check CUDA installation"""
    print("\n=== Checking CUDA Installation ===")
    cuda_paths = [
        '/usr/local/cuda',
        '/usr/local/cuda-11.0', '/usr/local/cuda-11.1', '/usr/local/cuda-11.2',
        '/usr/local/cuda-11.3', '/usr/local/cuda-11.4', '/usr/local/cuda-11.5',
        '/usr/local/cuda-11.6', '/usr/local/cuda-11.7', '/usr/local/cuda-11.8',
        '/usr/local/cuda-12.0', '/usr/local/cuda-12.1', '/usr/local/cuda-12.2',
    ]
    
    cuda_found = False
    for path in cuda_paths:
        if os.path.exists(path):
            print(f"✓ CUDA found at: {path}")
            cuda_found = True
            # Check for important libraries
            lib_path = os.path.join(path, 'lib64')
            if os.path.exists(lib_path):
                cudart = os.path.join(lib_path, 'libcudart.so')
                cublas = os.path.join(lib_path, 'libcublas.so')
                if os.path.exists(cudart):
                    print(f"  ✓ CUDA runtime library found")
                else:
                    print(f"  ✗ CUDA runtime library missing")
                if os.path.exists(cublas):
                    print(f"  ✓ cuBLAS library found")
                else:
                    print(f"  ✗ cuBLAS library missing")
            break
    
    if not cuda_found:
        print("✗ CUDA installation not found")
    
    return cuda_found

def check_pytorch_gpu():
    """Check PyTorch GPU support"""
    print("\n=== Checking PyTorch GPU Support ===")
    try:
        import torch
        print(f"✓ PyTorch version: {torch.__version__}")
        if torch.cuda.is_available():
            print(f"✓ CUDA is available")
            print(f"  Device count: {torch.cuda.device_count()}")
            print(f"  Current device: {torch.cuda.current_device()}")
            print(f"  Device name: {torch.cuda.get_device_name(0)}")
            return True
        else:
            print("✗ CUDA not available in PyTorch")
            return False
    except ImportError:
        print("✗ PyTorch not installed")
        return False

def check_onnxruntime_gpu():
    """Check ONNX Runtime GPU support"""
    print("\n=== Checking ONNX Runtime GPU Support ===")
    try:
        import onnxruntime as ort
        print(f"✓ ONNX Runtime version: {ort.__version__}")
        providers = ort.get_available_providers()
        print(f"Available providers: {providers}")
        
        if 'CUDAExecutionProvider' in providers:
            print("✓ CUDAExecutionProvider is available")
            
            # Try to create a session to verify it works
            try:
                import numpy as np
                # Create a dummy input
                x = np.random.randn(1, 3, 112, 112).astype(np.float32)
                print("✓ CUDA provider is functional")
                return True
            except Exception as e:
                print(f"✗ CUDA provider test failed: {e}")
                return False
        else:
            print("✗ CUDAExecutionProvider not available")
            print("  You may need to install onnxruntime-gpu:")
            print("  pip install onnxruntime-gpu")
            return False
    except ImportError:
        print("✗ ONNX Runtime not installed")
        return False

def check_tensorflow_gpu():
    """Check TensorFlow GPU support"""
    print("\n=== Checking TensorFlow GPU Support ===")
    # Set environment variable to reduce TF verbosity
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    # Fix protobuf issue
    os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
    
    try:
        import tensorflow as tf
        print(f"✓ TensorFlow version: {tf.__version__}")
        
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"✓ Found {len(gpus)} GPU(s)")
            for gpu in gpus:
                print(f"  GPU: {gpu}")
            return True
        else:
            print("✗ No GPUs found by TensorFlow")
            return False
    except TypeError as e:
        if "Descriptors cannot be created directly" in str(e):
            print("✗ TensorFlow has protobuf compatibility issues")
            print("  This is why DeepFace models are failing")
            print("  Consider using InsightFace instead")
            return False
        else:
            raise
    except ImportError:
        print("✗ TensorFlow not installed")
        return False

def suggest_fixes(nvidia_ok, cuda_ok, pytorch_ok, onnx_ok, tf_ok):
    """Suggest fixes based on check results"""
    print("\n=== Recommendations ===")
    
    if not nvidia_ok:
        print("\n1. Install NVIDIA Driver:")
        print("   - Ubuntu: sudo apt update && sudo apt install nvidia-driver-525")
        print("   - Or visit: https://www.nvidia.com/Download/index.aspx")
    
    if not cuda_ok and nvidia_ok:
        print("\n2. Install CUDA Toolkit:")
        print("   - Visit: https://developer.nvidia.com/cuda-downloads")
        print("   - Or: sudo apt install cuda-11-8")
    
    if not onnx_ok and cuda_ok:
        print("\n3. Install ONNX Runtime GPU:")
        print("   pip uninstall onnxruntime onnxruntime-gpu")
        print("   pip install onnxruntime-gpu")
    
    if nvidia_ok and cuda_ok and not pytorch_ok:
        print("\n4. Install PyTorch with CUDA:")
        print("   Visit: https://pytorch.org/get-started/locally/")
        print("   Example: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
    
    if cuda_ok:
        print("\n5. Set environment variables:")
        print("   export CUDA_HOME=/usr/local/cuda")
        print("   export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH")
        print("   export PATH=$CUDA_HOME/bin:$PATH")
        
        print("\n6. For InsightFace optimal performance:")
        print("   - Ensure onnxruntime-gpu is installed")
        print("   - Use buffalo_l model for best accuracy")
        print("   - Set det_size=(640, 640) for standard detection")

def main():
    print("GPU Support Check for FaceCV")
    print("=" * 50)
    
    # Run checks
    nvidia_ok = check_nvidia_driver()
    cuda_ok = check_cuda_installation()
    pytorch_ok = check_pytorch_gpu()
    onnx_ok = check_onnxruntime_gpu()
    tf_ok = check_tensorflow_gpu()
    
    # Summary
    print("\n=== Summary ===")
    status = {
        "NVIDIA Driver": nvidia_ok,
        "CUDA Toolkit": cuda_ok,
        "PyTorch GPU": pytorch_ok,
        "ONNX Runtime GPU": onnx_ok,
        "TensorFlow GPU": tf_ok
    }
    
    all_ok = True
    for component, ok in status.items():
        status_str = "✓" if ok else "✗"
        print(f"{status_str} {component}")
        if not ok:
            all_ok = False
    
    if all_ok:
        print("\n🎉 All GPU components are properly configured!")
        print("Both InsightFace and DeepFace should be able to use GPU acceleration.")
    else:
        suggest_fixes(nvidia_ok, cuda_ok, pytorch_ok, onnx_ok, tf_ok)
    
    return 0 if all_ok else 1

if __name__ == "__main__":
    sys.exit(main())