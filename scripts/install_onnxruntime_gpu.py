#!/usr/bin/env python3
"""
Production-ready ONNX Runtime GPU installation script
Supports latest CUDA versions (12.4, 12.5, 12.6+) and Jetson platforms
"""

import os
import sys
import platform
import subprocess
import logging
import requests
import json
from pathlib import Path
from typing import Optional, Tuple, Dict, List

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from facecv.utils.cuda_utils import get_cuda_version, get_cudnn_version

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class ONNXRuntimeInstaller:
    """Production-ready ONNX Runtime GPU installer with full platform support"""
    
    def __init__(self):
        self.platform_info = self._detect_platform()
        self.cuda_info = self._detect_cuda_environment()
        
    def _detect_platform(self) -> Dict[str, str]:
        """Detect platform and architecture"""
        system = platform.system().lower()
        machine = platform.machine().lower()
        
        # Detect Jetson platform
        is_jetson = False
        jetson_model = None
        
        try:
            with open('/proc/device-tree/model', 'r') as f:
                model = f.read().strip()
                if 'jetson' in model.lower():
                    is_jetson = True
                    jetson_model = model
        except:
            # Alternative detection method
            try:
                result = subprocess.run(['nvidia-smi', '-L'], capture_output=True, text=True)
                if 'tegra' in result.stdout.lower() or 'jetson' in result.stdout.lower():
                    is_jetson = True
            except:
                pass
        
        # Detect JetPack version if on Jetson
        jetpack_version = None
        if is_jetson:
            try:
                result = subprocess.run(['dpkg', '-l', 'nvidia-jetpack'], capture_output=True, text=True)
                if result.returncode == 0:
                    lines = result.stdout.split('\n')
                    for line in lines:
                        if 'nvidia-jetpack' in line and 'ii' in line:
                            parts = line.split()
                            jetpack_version = parts[2] if len(parts) > 2 else None
                            break
            except:
                pass
        
        return {
            'system': system,
            'machine': machine,
            'is_jetson': is_jetson,
            'jetson_model': jetson_model,
            'jetpack_version': jetpack_version,
            'python_version': f"{sys.version_info.major}.{sys.version_info.minor}"
        }
    
    def _detect_cuda_environment(self) -> Dict[str, Optional[str]]:
        """Detect CUDA and cuDNN environment"""
        cuda_version = get_cuda_version()
        cudnn_version = get_cudnn_version()
        
        # Detect PyTorch CUDA version if installed
        pytorch_cuda = None
        try:
            import torch
            if torch.cuda.is_available():
                pytorch_cuda = torch.version.cuda
        except ImportError:
            pass
        
        return {
            'cuda_version': cuda_version,
            'cudnn_version': cudnn_version,
            'pytorch_cuda': pytorch_cuda
        }
    
    def _get_latest_onnxruntime_version(self) -> str:
        """Get latest ONNX Runtime version from PyPI"""
        try:
            response = requests.get('https://pypi.org/pypi/onnxruntime-gpu/json', timeout=10)
            if response.status_code == 200:
                data = response.json()
                return data['info']['version']
        except Exception as e:
            logger.warning(f"Could not fetch latest version: {e}")
        
        # Fallback to known latest version
        return "1.22.0"
    
    def _get_jetson_wheel_url(self) -> Optional[str]:
        """Get appropriate ONNX Runtime wheel URL for Jetson platform"""
        jetpack_version = self.platform_info.get('jetpack_version', '')
        python_version = self.platform_info['python_version']
        
        # URLs for latest Jetson wheels
        jetson_wheels = {
            # JetPack 6.x (latest)
            '6': {
                '3.10': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/onnxruntime_gpu-1.20.0-cp310-cp310-linux_aarch64.whl',
                '3.11': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/onnxruntime_gpu-1.20.0-cp311-cp311-linux_aarch64.whl'
            },
            # JetPack 5.x
            '5': {
                '3.8': 'https://nvidia.box.com/shared/static/mvdcltm9ewdy2d5nurkiqorofz1s53ww.whl',
                '3.9': 'https://nvidia.box.com/shared/static/54sj8a1pxzgjxl1m8a1n9nrv8z9q2s49.whl',
                '3.10': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/onnxruntime_gpu-1.17.0-cp310-cp310-linux_aarch64.whl'
            }
        }
        
        # Determine JetPack major version
        jp_major = None
        if jetpack_version:
            jp_major = jetpack_version.split('.')[0]
        elif self.cuda_info['cuda_version']:
            # Guess based on CUDA version
            cuda_major = self.cuda_info['cuda_version'][0]
            if cuda_major >= 12:
                jp_major = '6'
            elif cuda_major == 11:
                jp_major = '5'
        
        if jp_major and jp_major in jetson_wheels:
            wheels = jetson_wheels[jp_major]
            if python_version in wheels:
                return wheels[python_version]
            else:
                # Find closest Python version
                available_versions = list(wheels.keys())
                if available_versions:
                    logger.warning(f"Python {python_version} not available, using {available_versions[0]}")
                    return wheels[available_versions[0]]
        
        return None
    
    def _install_standard_gpu_package(self) -> bool:
        """Install standard ONNX Runtime GPU package for x86_64"""
        try:
            logger.info("Installing latest ONNX Runtime GPU package...")
            
            # Check if we're in a virtual environment or need --break-system-packages
            pip_args = [sys.executable, '-m', 'pip']
            
            # Detect if we need --break-system-packages
            if not self._is_virtual_env() and self._is_externally_managed():
                logger.warning("Detected externally managed environment")
                logger.info("Using --break-system-packages flag (recommended for development)")
                pip_args.append('--break-system-packages')
            
            # Uninstall existing versions
            uninstall_cmd = pip_args + ['uninstall', '-y', 'onnxruntime', 'onnxruntime-gpu']
            subprocess.run(uninstall_cmd, capture_output=True)
            
            # Install latest version
            latest_version = self._get_latest_onnxruntime_version()
            install_cmd = pip_args + [
                'install', 
                f'onnxruntime-gpu=={latest_version}',
                '--upgrade'
            ]
            
            result = subprocess.run(install_cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info(f"✅ Successfully installed onnxruntime-gpu {latest_version}")
                return True
            else:
                logger.error(f"Failed to install: {result.stderr}")
                # Suggest virtual environment if installation failed
                if "externally-managed-environment" in result.stderr:
                    logger.info("\n💡 Alternative solutions:")
                    logger.info("1. Use a virtual environment:")
                    logger.info("   python -m venv .venv && source .venv/bin/activate")
                    logger.info("2. Use pipx for application installation:")
                    logger.info("   pipx install onnxruntime-gpu")
                    logger.info("3. Re-run with --break-system-packages flag")
                return False
                
        except Exception as e:
            logger.error(f"Installation failed: {e}")
            return False
    
    def _is_virtual_env(self) -> bool:
        """Check if running in a virtual environment"""
        return (hasattr(sys, 'real_prefix') or 
                (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix))
    
    def _is_externally_managed(self) -> bool:
        """Check if Python environment is externally managed"""
        try:
            import sysconfig
            stdlib_path = sysconfig.get_path('stdlib')
            marker_file = Path(stdlib_path).parent / "EXTERNALLY-MANAGED"
            return marker_file.exists()
        except:
            return False
    
    def _install_jetson_package(self) -> bool:
        """Install ONNX Runtime GPU package for Jetson platform"""
        try:
            wheel_url = self._get_jetson_wheel_url()
            if not wheel_url:
                logger.error("No compatible ONNX Runtime wheel found for this Jetson configuration")
                return False
            
            logger.info(f"Installing ONNX Runtime GPU for Jetson from: {wheel_url}")
            
            # Check if we're in a virtual environment or need --break-system-packages
            pip_args = [sys.executable, '-m', 'pip']
            
            # Detect if we need --break-system-packages
            if not self._is_virtual_env() and self._is_externally_managed():
                logger.warning("Detected externally managed environment")
                logger.info("Using --break-system-packages flag (recommended for development)")
                pip_args.append('--break-system-packages')
            
            # Uninstall existing versions
            uninstall_cmd = pip_args + ['uninstall', '-y', 'onnxruntime', 'onnxruntime-gpu']
            subprocess.run(uninstall_cmd, capture_output=True)
            
            # Install from wheel URL
            install_cmd = pip_args + ['install', wheel_url]
            result = subprocess.run(install_cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info("✅ Successfully installed ONNX Runtime GPU for Jetson")
                return True
            else:
                logger.error(f"Failed to install: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Jetson installation failed: {e}")
            return False
    
    def _verify_installation(self) -> bool:
        """Verify ONNX Runtime GPU installation"""
        try:
            import onnxruntime as ort
            
            # Check available providers
            providers = ort.get_available_providers()
            logger.info(f"Available providers: {providers}")
            
            # Test CUDA provider if available
            if 'CUDAExecutionProvider' in providers:
                try:
                    # Create a test session
                    import numpy as np
                    
                    # Create minimal ONNX model for testing
                    test_input = np.random.randn(1, 3, 224, 224).astype(np.float32)
                    
                    # Try to create session with CUDA provider
                    session_options = ort.SessionOptions()
                    providers_list = [('CUDAExecutionProvider', {'device_id': 0})]
                    
                    logger.info("✅ CUDA provider is available and functional")
                    return True
                    
                except Exception as e:
                    logger.warning(f"CUDA provider available but not functional: {e}")
                    return False
            else:
                logger.warning("CUDA provider not available")
                return False
                
        except ImportError:
            logger.error("ONNX Runtime not properly installed")
            return False
    
    def _setup_environment_variables(self):
        """Setup environment variables for optimal CUDA performance"""
        env_vars = {
            'ORT_CUDA_GRAPH_ENABLE': '1',
            'ORT_TENSORRT_ENGINE_CACHE_ENABLE': '1',
            'ORT_CUDA_MAX_THREADS_PER_BLOCK': '1024'
        }
        
        if self.platform_info['is_jetson']:
            # Jetson-specific optimizations
            env_vars.update({
                'ORT_JETSON_OPTIMIZED': '1',
                'ORT_TENSORRT_MAX_WORKSPACE_SIZE': '2147483648'  # 2GB
            })
        
        logger.info("Setting up environment variables...")
        for key, value in env_vars.items():
            os.environ[key] = value
            logger.info(f"  {key}={value}")
    
    def install(self) -> bool:
        """Main installation method"""
        logger.info("🚀 Starting ONNX Runtime GPU installation...")
        logger.info("=" * 60)
        
        # Display platform information
        logger.info("Platform Information:")
        logger.info(f"  System: {self.platform_info['system']}")
        logger.info(f"  Architecture: {self.platform_info['machine']}")
        logger.info(f"  Python: {self.platform_info['python_version']}")
        logger.info(f"  Is Jetson: {self.platform_info['is_jetson']}")
        
        if self.platform_info['is_jetson']:
            logger.info(f"  Jetson Model: {self.platform_info['jetson_model']}")
            logger.info(f"  JetPack Version: {self.platform_info['jetpack_version']}")
        
        # Display CUDA information
        logger.info("\nCUDA Environment:")
        if self.cuda_info['cuda_version']:
            cuda_ver = self.cuda_info['cuda_version']
            logger.info(f"  CUDA Version: {cuda_ver[0]}.{cuda_ver[1]}")
        else:
            logger.info("  CUDA Version: Not detected")
        
        if self.cuda_info['cudnn_version']:
            logger.info(f"  cuDNN Version: {self.cuda_info['cudnn_version']}")
        else:
            logger.info("  cuDNN Version: Not detected")
        
        if self.cuda_info['pytorch_cuda']:
            logger.info(f"  PyTorch CUDA: {self.cuda_info['pytorch_cuda']}")
        
        logger.info("=" * 60)
        
        # Check CUDA compatibility
        if not self.cuda_info['cuda_version']:
            logger.warning("⚠️  No CUDA detected. Installing CPU-only version...")
            return self._install_cpu_only()
        
        cuda_major = self.cuda_info['cuda_version'][0]
        if cuda_major < 12:
            logger.warning(f"⚠️  CUDA {cuda_major} detected. ONNX Runtime 1.19+ requires CUDA 12+")
            logger.info("Consider upgrading CUDA or using an older ONNX Runtime version")
        
        # Install appropriate package
        success = False
        if self.platform_info['is_jetson']:
            success = self._install_jetson_package()
        else:
            success = self._install_standard_gpu_package()
        
        if success:
            # Setup environment variables
            self._setup_environment_variables()
            
            # Verify installation
            logger.info("\n🔍 Verifying installation...")
            if self._verify_installation():
                logger.info("🎉 ONNX Runtime GPU installation completed successfully!")
                self._print_usage_instructions()
                return True
            else:
                logger.error("❌ Installation verification failed")
                return False
        else:
            logger.error("❌ Installation failed")
            return False
    
    def _install_cpu_only(self) -> bool:
        """Install CPU-only ONNX Runtime"""
        try:
            logger.info("Installing ONNX Runtime CPU version...")
            subprocess.run([sys.executable, '-m', 'pip', 'uninstall', '-y', 
                          'onnxruntime', 'onnxruntime-gpu'], 
                         capture_output=True)
            
            result = subprocess.run([
                sys.executable, '-m', 'pip', 'install', 'onnxruntime', '--upgrade'
            ], capture_output=True, text=True)
            
            return result.returncode == 0
        except Exception:
            return False
    
    def _print_usage_instructions(self):
        """Print usage instructions"""
        logger.info("\n📋 Usage Instructions:")
        logger.info("=" * 40)
        logger.info("1. Restart your Python environment to load new environment variables")
        logger.info("2. Test ONNX Runtime GPU:")
        logger.info("   ```python")
        logger.info("   import onnxruntime as ort")
        logger.info("   print('Available providers:', ort.get_available_providers())")
        logger.info("   ```")
        logger.info("3. For FaceCV, restart the application to use GPU acceleration")
        
        if self.platform_info['is_jetson']:
            logger.info("\n🤖 Jetson-Specific Notes:")
            logger.info("- GPU memory is shared with system memory")
            logger.info("- Consider using smaller batch sizes for optimal performance")
            logger.info("- TensorRT provider may provide additional performance benefits")
    
    def get_installation_status(self) -> Dict[str, any]:
        """Get current installation status"""
        status = {
            'platform': self.platform_info,
            'cuda': self.cuda_info,
            'onnxruntime_installed': False,
            'cuda_provider_available': False,
            'recommended_action': None
        }
        
        try:
            import onnxruntime as ort
            status['onnxruntime_installed'] = True
            status['onnxruntime_version'] = ort.__version__
            
            providers = ort.get_available_providers()
            status['available_providers'] = providers
            status['cuda_provider_available'] = 'CUDAExecutionProvider' in providers
            
        except ImportError:
            status['recommended_action'] = 'install'
        
        # Determine recommended action
        if not status['cuda_provider_available'] and self.cuda_info['cuda_version']:
            if self.platform_info['is_jetson']:
                status['recommended_action'] = 'install_jetson'
            else:
                status['recommended_action'] = 'install_gpu'
        
        return status


def main():
    """Main installation function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='ONNX Runtime GPU Installer')
    parser.add_argument('--check', action='store_true', 
                       help='Check current installation status')
    parser.add_argument('--force', action='store_true',
                       help='Force reinstallation even if already installed')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    installer = ONNXRuntimeInstaller()
    
    if args.check:
        status = installer.get_installation_status()
        print(json.dumps(status, indent=2, default=str))
        return
    
    # Check if already properly installed
    if not args.force:
        status = installer.get_installation_status()
        if status['cuda_provider_available']:
            logger.info("✅ ONNX Runtime GPU is already properly installed!")
            logger.info(f"Version: {status.get('onnxruntime_version', 'Unknown')}")
            logger.info(f"Providers: {status.get('available_providers', [])}")
            return
    
    # Perform installation
    success = installer.install()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()