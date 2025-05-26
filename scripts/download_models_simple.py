#!/usr/bin/env python3
"""
Simple Model Downloader for FaceCV
==================================

Downloads InsightFace models that contain ArcFace implementations.
The buffalo models already include ResNet50 ArcFace models!
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def download_insightface_models():
    """Download InsightFace models (which include ArcFace)"""
    print("🚀 Downloading InsightFace Models (includes ArcFace)")
    print("=" * 50)
    
    try:
        import insightface
        from insightface.app import FaceAnalysis
        
        # This will automatically download models if they don't exist
        print("📥 Downloading buffalo_l model package...")
        app_l = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
        app_l.prepare(ctx_id=-1, det_size=(640, 640))
        print("✅ buffalo_l downloaded successfully!")
        
        print("📥 Downloading buffalo_s model package...")
        app_s = FaceAnalysis(name='buffalo_s', providers=['CPUExecutionProvider'])
        app_s.prepare(ctx_id=-1, det_size=(640, 640))
        print("✅ buffalo_s downloaded successfully!")
        
        print("📥 Downloading buffalo_m model package...")
        app_m = FaceAnalysis(name='buffalo_m', providers=['CPUExecutionProvider'])
        app_m.prepare(ctx_id=-1, det_size=(640, 640))
        print("✅ buffalo_m downloaded successfully!")
        
        print("\n🎉 All InsightFace models downloaded!")
        print("📋 Available ArcFace models:")
        print("   - buffalo_l/w600k_r50.onnx (ResNet50, WebFace600K)")
        print("   - buffalo_m/w600k_r50.onnx (ResNet50, WebFace600K)")  
        print("   - buffalo_s/w600k_mbf.onnx (MobileFaceNet, WebFace600K)")
        
        return True
        
    except Exception as e:
        print(f"❌ Error downloading models: {e}")
        return False


def list_available_models():
    """List what models are available"""
    models_dir = Path.home() / ".insightface" / "models"
    
    if not models_dir.exists():
        print("⚠️  No InsightFace models found. Run download first.")
        return
    
    print("📋 Available InsightFace Models:")
    for model_pack in models_dir.iterdir():
        if model_pack.is_dir():
            print(f"\n📦 {model_pack.name}:")
            for model_file in model_pack.iterdir():
                if model_file.suffix == '.onnx':
                    print(f"   - {model_file.name}")


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Download InsightFace models")
    parser.add_argument("--download", action="store_true", help="Download all models")
    parser.add_argument("--list", action="store_true", help="List available models")
    
    args = parser.parse_args()
    
    if args.download:
        download_insightface_models()
    elif args.list:
        list_available_models()
    else:
        print("🤖 InsightFace Model Downloader")
        print("Usage:")
        print("  --download   Download all InsightFace models")
        print("  --list       List available models")


if __name__ == "__main__":
    main()