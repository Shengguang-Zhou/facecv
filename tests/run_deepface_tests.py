#!/usr/bin/env python3
"""
DeepFace API Test Runner

Quick and easy way to run DeepFace API tests with different configurations.
"""

import os
import sys
import subprocess
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def run_tests(test_type="smoke", verbose=True, port=7003):
    """Run DeepFace API tests"""
    
    # Set environment variables
    os.environ["API_BASE_URL"] = f"http://localhost:{port}"
    
    # Base pytest command - use full path to pytest
    pytest_cmd = [
        "/home/a/.local/bin/pytest",
        "-v" if verbose else "-q",
        "--tb=short",
        "--disable-warnings",
        f"--durations=5"
    ]
    
    # Test selection based on type
    if test_type == "smoke":
        # Quick smoke tests
        pytest_cmd.extend([
            "-k", "health or register_face_success or list_faces_empty",
            "tests/api/deepface/"
        ])
        print(f"🔥 Running DeepFace Smoke Tests (Port {port})")
        
    elif test_type == "core":
        # Core functionality tests
        pytest_cmd.extend([
            "-k", "health or register or list or recognition or verification",
            "tests/api/deepface/test_health.py",
            "tests/api/deepface/test_face_management.py",
            "tests/api/deepface/test_recognition_verification.py"
        ])
        print(f"🧪 Running DeepFace Core Tests (Port {port})")
        
    elif test_type == "all":
        # All tests except slow ones
        pytest_cmd.extend([
            "-m", "not slow",
            "tests/api/deepface/"
        ])
        print(f"🚀 Running All DeepFace Tests (Port {port})")
        
    elif test_type == "fast":
        # Only fast tests
        pytest_cmd.extend([
            "--maxfail=5",
            "-x",  # Stop on first failure
            "-k", "health or list_faces_empty or recognition_no_faces",
            "tests/api/deepface/"
        ])
        print(f"⚡ Running Fast DeepFace Tests (Port {port})")
        
    else:
        print(f"❌ Unknown test type: {test_type}")
        return 1
    
    print(f"📋 Command: {' '.join(pytest_cmd)}")
    print("-" * 60)
    
    # Run tests
    try:
        result = subprocess.run(pytest_cmd, cwd=project_root)
        
        print("-" * 60)
        if result.returncode == 0:
            print("✅ All tests passed!")
        else:
            print(f"❌ Some tests failed. Exit code: {result.returncode}")
        
        return result.returncode
        
    except KeyboardInterrupt:
        print("\n⚠️  Tests interrupted by user")
        return 130
    except Exception as e:
        print(f"❌ Error running tests: {e}")
        return 1


def check_api_status(port=7003):
    """Check if API is running"""
    try:
        import requests
        response = requests.get(f"http://localhost:{port}/api/v1/deepface/health", timeout=5)
        if response.status_code == 200:
            result = response.json()
            print(f"✅ DeepFace API is running on port {port}")
            print(f"   Status: {result.get('status', 'unknown')}")
            print(f"   Service: {result.get('service', 'unknown')}")
            return True
        else:
            print(f"⚠️  DeepFace API responded with status {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ DeepFace API not accessible on port {port}: {e}")
        return False


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="DeepFace API Test Runner")
    parser.add_argument(
        "test_type",
        choices=["smoke", "core", "all", "fast"],
        default="smoke",
        nargs="?",
        help="Type of tests to run (default: smoke)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7003,
        help="API port number (default: 7003)"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Run in quiet mode"
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only check API status, don't run tests"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🧪 DeepFace API Test Runner")
    print("=" * 60)
    
    # Check API status first
    if not check_api_status(args.port):
        if args.check_only:
            return 1
        print("⚠️  Continuing with tests anyway...")
    
    if args.check_only:
        return 0
    
    print()
    
    # Run tests
    exit_code = run_tests(
        test_type=args.test_type,
        verbose=not args.quiet,
        port=args.port
    )
    
    print("=" * 60)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())