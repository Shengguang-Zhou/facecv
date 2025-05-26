#!/usr/bin/env python3
"""
Model Deployment Script for FaceCV
Copies DeepFace and InsightFace models to user-specific directories during installation.
"""

import os
import shutil
import sys
import getpass
from pathlib import Path


def get_user_home(username=None):
    """Get user home directory"""
    if username:
        return Path(f"/home/{username}")
    return Path.home()


def ensure_directory(path):
    """Create directory if it doesn't exist"""
    path.mkdir(parents=True, exist_ok=True)
    print(f"✓ Created directory: {path}")


def copy_file_with_progress(src, dst):
    """Copy file with progress indication"""
    try:
        shutil.copy2(src, dst)
        print(f"✓ Copied: {src.name}")
        return True
    except Exception as e:
        print(f"✗ Failed to copy {src.name}: {e}")
        return False


def copy_directory_with_progress(src, dst):
    """Copy directory with progress indication"""
    try:
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        print(f"✓ Copied directory: {src.name}")
        return True
    except Exception as e:
        print(f"✗ Failed to copy directory {src.name}: {e}")
        return False


def deploy_deepface_models(source_dir, target_dir):
    """Deploy DeepFace models to user directory"""
    print("\n🔧 Deploying DeepFace models...")
    
    # DeepFace models are in weights/deepface/weights/
    deepface_source = source_dir / "deepface" / "weights"
    
    if not deepface_source.exists():
        print(f"✗ DeepFace source directory not found: {deepface_source}")
        return False
    
    ensure_directory(target_dir)
    
    success_count = 0
    total_count = 0
    
    for model_file in deepface_source.glob("*.h5"):
        total_count += 1
        if copy_file_with_progress(model_file, target_dir / model_file.name):
            success_count += 1
    
    print(f"📊 DeepFace deployment: {success_count}/{total_count} models copied")
    return success_count == total_count


def deploy_insightface_models(source_dir, target_dir):
    """Deploy InsightFace models to user directory"""
    print("\n🔧 Deploying InsightFace models...")
    
    insightface_source = source_dir / "insightface"
    
    if not insightface_source.exists():
        print(f"✗ InsightFace source directory not found: {insightface_source}")
        return False
    
    ensure_directory(target_dir)
    
    success_count = 0
    total_count = 0
    
    # Copy model directories (buffalo_l, buffalo_s)
    for model_dir in insightface_source.iterdir():
        if model_dir.is_dir() and not model_dir.name.endswith('.zip'):
            total_count += 1
            target_model_dir = target_dir / model_dir.name
            if copy_directory_with_progress(model_dir, target_model_dir):
                success_count += 1
    
    print(f"📊 InsightFace deployment: {success_count}/{total_count} model packages copied")
    return success_count == total_count


def deploy_arcface_models(source_dir, target_dir):
    """Deploy ArcFace models to user directory"""
    print("\n🔧 Deploying ArcFace models...")
    
    arcface_source = source_dir / "arcface"
    
    if not arcface_source.exists():
        print(f"✓ ArcFace source directory not found: {arcface_source} (optional)")
        return True
    
    ensure_directory(target_dir)
    
    success_count = 0
    total_count = 0
    
    # Copy all backbone/dataset combinations
    for backbone_dir in arcface_source.iterdir():
        if backbone_dir.is_dir():
            target_backbone_dir = target_dir / backbone_dir.name
            for dataset_dir in backbone_dir.iterdir():
                if dataset_dir.is_dir():
                    total_count += 1
                    target_dataset_dir = target_backbone_dir / dataset_dir.name
                    if copy_directory_with_progress(dataset_dir, target_dataset_dir):
                        success_count += 1
    
    print(f"📊 ArcFace deployment: {success_count}/{total_count} model packages copied")
    return success_count == total_count


def main():
    """Main deployment function"""
    print("🚀 FaceCV Model Deployment Script")
    print("=" * 50)
    
    # Get username from command line or current user
    username = sys.argv[1] if len(sys.argv) > 1 else getpass.getuser()
    print(f"👤 Target user: {username}")
    
    # Set up paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    weights_dir = project_root / "weights"
    
    user_home = get_user_home(username)
    deepface_target = user_home / ".deepface" / "weights"
    insightface_target = user_home / ".insightface" / "models"
    arcface_target = user_home / ".arcface" / "models"
    
    print(f"📁 Source directory: {weights_dir}")
    print(f"📁 DeepFace target: {deepface_target}")
    print(f"📁 InsightFace target: {insightface_target}")
    print(f"📁 ArcFace target: {arcface_target}")
    
    # Check if source directory exists
    if not weights_dir.exists():
        print(f"✗ Source weights directory not found: {weights_dir}")
        sys.exit(1)
    
    # Deploy models
    deepface_success = deploy_deepface_models(weights_dir, deepface_target)
    insightface_success = deploy_insightface_models(weights_dir, insightface_target)
    arcface_success = deploy_arcface_models(weights_dir, arcface_target)
    
    # Summary
    print("\n" + "=" * 50)
    if deepface_success and insightface_success and arcface_success:
        print("✅ Model deployment completed successfully!")
        print(f"📦 DeepFace models: {deepface_target}")
        print(f"📦 InsightFace models: {insightface_target}")
        print(f"📦 ArcFace models: {arcface_target}")
    else:
        print("❌ Model deployment completed with errors!")
        if not deepface_success:
            print("   - DeepFace deployment failed")
        if not insightface_success:
            print("   - InsightFace deployment failed")
        if not arcface_success:
            print("   - ArcFace deployment failed")
        sys.exit(1)


if __name__ == "__main__":
    main()