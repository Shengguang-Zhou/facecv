#!/usr/bin/env python3
"""
Setup script to install the correct ONNX Runtime for detected CUDA version
"""

import sys
import os
import subprocess

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from facecv.utils.cuda_utils import (
    get_cuda_version, 
    check_cuda_availability,
    install_appropriate_onnxruntime,
    check_onnxruntime_cuda_compatibility
)


def main():
    print("=== CUDA ONNX Runtime Setup ===")
    print()
    
    # Check CUDA availability
    if not check_cuda_availability():
        print("❌ No CUDA detected on this system")
        print("✓ CPU-only ONNX Runtime will be used")
        print()
        print("To install CPU-only ONNX Runtime:")
        print("  pip install onnxruntime")
        return
    
    # Get CUDA version
    cuda_version = get_cuda_version()
    print(f"✓ CUDA {cuda_version[0]}.{cuda_version[1]} detected")
    
    # Check current ONNX Runtime compatibility
    print("\nChecking current ONNX Runtime installation...")
    if check_onnxruntime_cuda_compatibility():
        print("✓ Current ONNX Runtime is already compatible with your CUDA installation")
        return
    
    # Get installation command
    install_cmd = install_appropriate_onnxruntime()
    
    print("\n❌ Current ONNX Runtime is not compatible with your CUDA version")
    print(f"\nTo install the correct ONNX Runtime for CUDA {cuda_version[0]}.{cuda_version[1]}, run:")
    print(f"\n  {install_cmd}")
    
    # Ask if user wants to install now
    print("\nWould you like to install it now? (y/n): ", end="")
    response = input().strip().lower()
    
    if response == 'y':
        print("\nInstalling ONNX Runtime...")
        try:
            # First uninstall existing versions
            subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "onnxruntime", "onnxruntime-gpu"], 
                         capture_output=True)
            
            # Install the appropriate version
            result = subprocess.run(install_cmd.split(), capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✓ ONNX Runtime installed successfully!")
                
                # Verify installation
                if check_onnxruntime_cuda_compatibility():
                    print("✓ CUDA support verified!")
                else:
                    print("⚠️  Installation completed but CUDA support could not be verified")
                    print("   This might be due to missing cuDNN libraries")
            else:
                print("❌ Installation failed!")
                print(f"Error: {result.stderr}")
        except Exception as e:
            print(f"❌ Installation error: {e}")
    else:
        print("\nPlease run the installation command manually when ready.")


if __name__ == "__main__":
    main()