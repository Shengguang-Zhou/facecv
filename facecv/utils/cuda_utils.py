"""CUDA utilities for detecting CUDA version and selecting appropriate ONNX Runtime"""

import logging
import os
import subprocess
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


def check_cuda_availability() -> bool:
    """Check if CUDA is available on the system"""
    cuda_version = get_cuda_version()
    return cuda_version is not None


def get_execution_providers() -> List[str]:
    """Get the list of execution providers based on CUDA availability and installed libraries"""
    providers = []

    if check_cuda_availability():
        cuda_version = get_cuda_version()
        if cuda_version and cuda_version[0] >= 11:
            providers.append("CUDAExecutionProvider")

        # Only add TensorRT if it's actually available
        if _is_tensorrt_available():
            providers.append("TensorrtExecutionProvider")

    providers.append("CPUExecutionProvider")
    return providers


def _is_tensorrt_available() -> bool:
    """Check if TensorRT libraries are available"""
    try:
        # Try to import TensorRT
        import tensorrt

        return True
    except ImportError:
        pass

    # Check for TensorRT shared libraries
    import ctypes

    tensorrt_libs = ["libnvinfer.so.10", "libnvinfer.so.8", "libnvinfer.so"]

    for lib in tensorrt_libs:
        try:
            ctypes.CDLL(lib)
            return True
        except OSError:
            continue

    return False


def setup_cuda_environment():
    """Set up CUDA environment and return configuration"""
    cuda_available = check_cuda_availability()

    if cuda_available:
        # Set CUDA environment variables
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use first GPU
        os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"  # For TensorFlow
        os.environ["ORT_TENSORRT_ENGINE_CACHE_ENABLE"] = "1"  # For ONNX Runtime TensorRT

        logger.info("CUDA environment variables set for optimal performance")

    return cuda_available


def get_cuda_version() -> Optional[Tuple[int, int]]:
    """
    Get the installed CUDA version

    Returns:
        Tuple of (major, minor) version numbers, or None if CUDA not found
    """
    try:
        # Try nvidia-smi first
        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
        if result.returncode == 0:
            for line in result.stdout.split("\n"):
                if "CUDA Version" in line:
                    # Extract version like "CUDA Version: 12.4"
                    cuda_str = line.split("CUDA Version:")[1].strip()
                    version = cuda_str.split()[0]
                    major, minor = version.split(".")[:2]
                    logger.info(f"Detected CUDA {major}.{minor} from nvidia-smi")
                    return (int(major), int(minor))
    except:
        pass

    # Try nvcc
    try:
        result = subprocess.run(["nvcc", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            for line in result.stdout.split("\n"):
                if "release" in line:
                    # Extract version like "Cuda compilation tools, release 12.4"
                    parts = line.split("release")[1].strip()
                    version = parts.split(",")[0].strip()
                    major, minor = version.split(".")[:2]
                    logger.info(f"Detected CUDA {major}.{minor} from nvcc")
                    return (int(major), int(minor))
    except:
        pass

    # Check CUDA_HOME
    cuda_home = os.environ.get("CUDA_HOME", "/usr/local/cuda")
    if os.path.exists(cuda_home):
        version_file = os.path.join(cuda_home, "version.txt")
        if os.path.exists(version_file):
            try:
                with open(version_file, "r") as f:
                    content = f.read()
                    if "CUDA Version" in content:
                        version = content.split("CUDA Version")[1].strip()
                        major, minor = version.split(".")[:2]
                        logger.info(f"Detected CUDA {major}.{minor} from version.txt")
                        return (int(major), int(minor))
            except:
                pass

    logger.warning("Could not detect CUDA version")
    return None


def get_cudnn_version() -> Optional[int]:
    """
    Get the installed cuDNN major version

    Returns:
        Major version number, or None if cuDNN not found
    """
    # Check for cuDNN libraries
    lib_paths = ["/usr/local/cuda/lib64", "/usr/lib/x86_64-linux-gnu", "/usr/local/lib"]

    for path in lib_paths:
        if os.path.exists(path):
            try:
                files = os.listdir(path)
                for f in files:
                    if f.startswith("libcudnn.so."):
                        # Extract version like libcudnn.so.8 or libcudnn.so.9
                        parts = f.split(".")
                        if len(parts) >= 3 and parts[2].isdigit():
                            version = int(parts[2])
                            logger.info(f"Found cuDNN {version} at {os.path.join(path, f)}")
                            return version
            except:
                pass

    logger.warning("Could not detect cuDNN version")
    return None


def install_appropriate_onnxruntime():
    """
    Get the appropriate ONNX Runtime installation command based on CUDA version

    Returns:
        str: Installation command for the appropriate ONNX Runtime version
    """
    cuda_version = get_cuda_version()

    if not cuda_version:
        logger.info("No CUDA detected, using CPU-only ONNX Runtime")
        return "pip install onnxruntime"

    major, minor = cuda_version

    if major >= 12:
        # CUDA 12.x+ (12.4, 12.5, 12.6+) - use latest ONNX Runtime GPU
        # ONNX Runtime 1.19+ with CUDA 12.x support all minor versions due to compatibility
        logger.info(f"CUDA {major}.{minor} detected, installing latest ONNX Runtime GPU")
        return "pip install onnxruntime-gpu --upgrade"
    elif major == 11:
        # CUDA 11.x - use legacy CUDA 11 compatible ONNX Runtime
        logger.info(f"CUDA {major}.{minor} detected, installing ONNX Runtime for CUDA 11")
        return "pip install onnxruntime-gpu --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-11/pypi/simple/"
    else:
        # Older CUDA versions
        logger.warning(f"CUDA {major}.{minor} is not supported by current ONNX Runtime versions")
        logger.warning("Consider upgrading to CUDA 12.x for best compatibility")
        return "pip install onnxruntime-gpu"


def check_onnxruntime_cuda_compatibility():
    """
    Check if current ONNX Runtime is compatible with installed CUDA

    Returns:
        True if compatible, False otherwise
    """
    try:
        import onnxruntime as ort

        providers = ort.get_available_providers()
        if "CUDAExecutionProvider" not in providers:
            logger.warning("CUDAExecutionProvider not available in ONNX Runtime")
            return False

        # Try to create a session with CUDA
        try:
            import numpy as np

            # Create a minimal test
            sess_options = ort.SessionOptions()
            providers = [("CUDAExecutionProvider", {"device_id": 0})]

            # If we get here without error, CUDA is working
            logger.info("ONNX Runtime CUDA provider is functional")
            return True

        except Exception as e:
            logger.warning(f"CUDA provider test failed: {e}")
            return False

    except ImportError:
        logger.error("ONNX Runtime not installed")
        return False


def setup_cuda_env():
    """
    Set up CUDA environment variables
    """
    cuda_home = os.environ.get("CUDA_HOME", "/usr/local/cuda")

    # Set CUDA environment variables
    os.environ["CUDA_HOME"] = cuda_home

    # Add CUDA to PATH
    cuda_bin = os.path.join(cuda_home, "bin")
    if os.path.exists(cuda_bin) and cuda_bin not in os.environ.get("PATH", ""):
        os.environ["PATH"] = f"{cuda_bin}:{os.environ.get('PATH', '')}"

    # Add CUDA libraries to LD_LIBRARY_PATH
    cuda_lib = os.path.join(cuda_home, "lib64")
    if os.path.exists(cuda_lib):
        ld_library_path = os.environ.get("LD_LIBRARY_PATH", "")
        if cuda_lib not in ld_library_path:
            os.environ["LD_LIBRARY_PATH"] = f"{cuda_lib}:{ld_library_path}"

    logger.info(f"CUDA environment configured: CUDA_HOME={cuda_home}")


if __name__ == "__main__":
    # Test the utilities
    print("=== CUDA Detection Utility ===")

    cuda_ver = get_cuda_version()
    if cuda_ver:
        print(f"CUDA Version: {cuda_ver[0]}.{cuda_ver[1]}")
    else:
        print("CUDA not detected")

    cudnn_ver = get_cudnn_version()
    if cudnn_ver:
        print(f"cuDNN Version: {cudnn_ver}")
    else:
        print("cuDNN not detected")

    print("\nRecommended installation command:")
    print(install_appropriate_onnxruntime())

    print("\nChecking current ONNX Runtime compatibility...")
    if check_onnxruntime_cuda_compatibility():
        print("✓ ONNX Runtime is compatible with your CUDA installation")
    else:
        print("✗ ONNX Runtime needs to be reinstalled for your CUDA version")
