#!/usr/bin/env python3
"""
Legacy setup script - redirects to comprehensive installer
This script is maintained for backward compatibility
"""

import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def main():
    print("=== CUDA ONNX Runtime Setup ===")
    print()
    print("⚠️  This script has been superseded by a more comprehensive installer.")
    print("Please use the new installer for the best experience:")
    print()
    print("  python scripts/install_onnxruntime_gpu.py")
    print()
    print("The new installer supports:")
    print("  ✅ Latest CUDA versions (12.4, 12.5, 12.6+)")
    print("  ✅ NVIDIA Jetson platforms (Nano, Xavier, AGX)")
    print("  ✅ Automatic platform detection")
    print("  ✅ Production-ready installation")
    print("  ✅ Comprehensive verification")
    print()
    
    response = input("Would you like to run the new installer now? (y/n): ").strip().lower()
    
    if response == 'y':
        print("\nStarting comprehensive installer...")
        try:
            import subprocess
            script_path = os.path.join(os.path.dirname(__file__), 'install_onnxruntime_gpu.py')
            result = subprocess.run([sys.executable, script_path], 
                                  capture_output=False, text=True)
            sys.exit(result.returncode)
        except Exception as e:
            print(f"❌ Error running installer: {e}")
            print(f"Please run manually: python {script_path}")
            sys.exit(1)
    else:
        print("\nYou can run the installer manually at any time:")
        print("  python scripts/install_onnxruntime_gpu.py")
        print()
        print("For status checking:")
        print("  python scripts/install_onnxruntime_gpu.py --check")


if __name__ == "__main__":
    main()