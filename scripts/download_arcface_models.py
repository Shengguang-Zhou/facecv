#!/usr/bin/env python3
"""
ArcFace Model Downloader for FaceCV
===================================

Downloads and organizes ArcFace models to ./weights/arcface directory.
Supports selective downloading, progress tracking, and checksum verification.
"""

import os
import sys
import requests
import zipfile
import hashlib
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
import yaml

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from facecv.config.settings import get_settings
from facecv.config import load_model_config


class ArcFaceModelDownloader:
    """ArcFace model downloader with progress tracking and verification"""
    
    def __init__(self, weights_dir: Optional[str] = None):
        self.settings = get_settings()
        self.model_config = load_model_config()
        self.weights_dir = Path(weights_dir or self.settings.arcface_weights_dir)
        self.weights_dir.mkdir(parents=True, exist_ok=True)
        
        # Get ArcFace config
        self.arcface_config = self.model_config.get("arcface", {})
        self.models = self.arcface_config.get("models", {})
        self.download_config = self.arcface_config.get("downloads", {})
        
    def list_available_models(self) -> List[str]:
        """List all available ArcFace models"""
        return list(self.models.keys())
    
    def get_model_info(self, model_name: str) -> Dict:
        """Get detailed model information"""
        return self.models.get(model_name, {})
    
    def get_model_path(self, model_name: str) -> Path:
        """Get expected model file path"""
        model_info = self.get_model_info(model_name)
        if not model_info:
            raise ValueError(f"Unknown model: {model_name}")
            
        backbone = model_info["backbone"]
        dataset = model_info["dataset"]
        filename = model_info.get("filename", "model.onnx")
        
        return self.weights_dir / backbone / dataset / filename
    
    def is_model_downloaded(self, model_name: str) -> bool:
        """Check if model is already downloaded"""
        model_path = self.get_model_path(model_name)
        return model_path.exists()
    
    def download_file(self, url: str, destination: Path, expected_checksum: Optional[str] = None) -> bool:
        """Download file with progress bar and checksum verification"""
        try:
            # Create directory
            destination.parent.mkdir(parents=True, exist_ok=True)
            
            # Download with progress bar
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(destination, 'wb') as f, tqdm(
                desc=destination.name,
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
            
            # Verify checksum if provided
            if expected_checksum:
                if not self.verify_checksum(destination, expected_checksum):
                    destination.unlink()  # Remove corrupted file
                    return False
            
            print(f"✅ Downloaded: {destination}")
            return True
            
        except Exception as e:
            print(f"❌ Download failed: {e}")
            if destination.exists():
                destination.unlink()
            return False
    
    def verify_checksum(self, file_path: Path, expected_checksum: str) -> bool:
        """Verify file checksum"""
        if not expected_checksum.startswith("sha256:"):
            print(f"⚠️  Unknown checksum format: {expected_checksum}")
            return True  # Skip verification for unknown formats
        
        expected_hash = expected_checksum[7:]  # Remove "sha256:" prefix
        
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        
        actual_hash = sha256_hash.hexdigest()
        if actual_hash == expected_hash:
            print(f"✅ Checksum verified: {file_path.name}")
            return True
        else:
            print(f"❌ Checksum mismatch: {file_path.name}")
            print(f"   Expected: {expected_hash}")
            print(f"   Actual:   {actual_hash}")
            return False
    
    def extract_zip(self, zip_path: Path, extract_to: Path, target_filename: str = "model.onnx") -> bool:
        """Extract ONNX model from zip file"""
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                # Find the ONNX file in the zip
                onnx_files = [f for f in zip_ref.namelist() if f.endswith('.onnx')]
                
                if not onnx_files:
                    print(f"❌ No ONNX file found in {zip_path}")
                    return False
                
                # Extract the first ONNX file
                onnx_file = onnx_files[0]
                extract_to.mkdir(parents=True, exist_ok=True)
                
                with zip_ref.open(onnx_file) as source, open(extract_to / target_filename, 'wb') as target:
                    target.write(source.read())
                
                print(f"✅ Extracted: {target_filename}")
                return True
                
        except Exception as e:
            print(f"❌ Extraction failed: {e}")
            return False
    
    def download_model(self, model_name: str, force: bool = False) -> bool:
        """Download a specific ArcFace model"""
        model_info = self.get_model_info(model_name)
        if not model_info:
            print(f"❌ Unknown model: {model_name}")
            return False
        
        model_path = self.get_model_path(model_name)
        
        if model_path.exists() and not force:
            print(f"✅ Model already exists: {model_path}")
            return True
        
        print(f"📥 Downloading {model_name}...")
        print(f"   Backbone: {model_info['backbone']}")
        print(f"   Dataset: {model_info['dataset']}")
        print(f"   Performance: LFW {model_info['performance']['lfw']}%")
        
        url = model_info.get("url")
        if not url:
            print(f"❌ No download URL for model: {model_name}")
            return False
        
        # Download to temporary file
        with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as tmp_file:
            tmp_path = Path(tmp_file.name)
        
        try:
            # Download zip file
            expected_checksum = model_info.get("checksum")
            if not self.download_file(url, tmp_path, expected_checksum):
                return False
            
            # Extract ONNX model
            target_filename = model_info.get("filename", "model.onnx")
            if not self.extract_zip(tmp_path, model_path.parent, target_filename):
                return False
            
            print(f"🎉 Successfully downloaded: {model_name}")
            return True
            
        finally:
            # Cleanup temporary file
            if tmp_path.exists():
                tmp_path.unlink()
    
    def download_models(self, model_names: List[str], force: bool = False) -> Tuple[int, int]:
        """Download multiple models"""
        success_count = 0
        total_count = len(model_names)
        
        print(f"🚀 Starting download of {total_count} models...")
        print(f"📁 Download directory: {self.weights_dir}")
        
        for i, model_name in enumerate(model_names, 1):
            print(f"\n[{i}/{total_count}] Processing {model_name}")
            if self.download_model(model_name, force):
                success_count += 1
        
        print(f"\n🏁 Download complete: {success_count}/{total_count} models")
        return success_count, total_count
    
    def download_recommended_models(self, use_case: str = "production") -> bool:
        """Download models recommended for specific use case"""
        recommendations = self.arcface_config.get("recommendations", {})
        
        if use_case == "all":
            # Download all recommended models
            models_to_download = list(recommendations.values())
        else:
            recommended_model = recommendations.get(use_case)
            if not recommended_model:
                print(f"❌ No recommendation for use case: {use_case}")
                print(f"Available use cases: {list(recommendations.keys())}")
                return False
            models_to_download = [recommended_model]
        
        print(f"📋 Downloading recommended models for: {use_case}")
        success_count, total_count = self.download_models(models_to_download)
        return success_count == total_count
    
    def list_downloaded_models(self) -> List[str]:
        """List all downloaded models"""
        downloaded = []
        for model_name in self.models.keys():
            if self.is_model_downloaded(model_name):
                downloaded.append(model_name)
        return downloaded
    
    def get_download_status(self) -> Dict[str, bool]:
        """Get download status for all models"""
        status = {}
        for model_name in self.models.keys():
            status[model_name] = self.is_model_downloaded(model_name)
        return status
    
    def print_model_info(self, model_name: str = None):
        """Print detailed model information"""
        if model_name:
            models_to_show = {model_name: self.models[model_name]} if model_name in self.models else {}
        else:
            models_to_show = self.models
        
        for name, info in models_to_show.items():
            status = "✅ Downloaded" if self.is_model_downloaded(name) else "⬜ Not downloaded"
            perf = info.get("performance", {})
            
            print(f"\n📊 {name} ({status})")
            print(f"   Backbone: {info['backbone']}")
            print(f"   Dataset: {info['dataset']}")
            print(f"   Embedding Size: {info['embedding_size']}")
            print(f"   Performance: LFW {perf.get('lfw', 'N/A')}%, Speed {perf.get('speed_ms', 'N/A')}ms")
            print(f"   Use Case: {info.get('recommended_use', 'General')}")


def main():
    """Main CLI interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Download ArcFace models for FaceCV")
    parser.add_argument("--list", action="store_true", help="List available models")
    parser.add_argument("--status", action="store_true", help="Show download status")
    parser.add_argument("--download", nargs="+", help="Download specific models")
    parser.add_argument("--recommended", choices=["production", "research", "edge", "mobile", "diversity", "all"],
                       help="Download recommended models for use case")
    parser.add_argument("--force", action="store_true", help="Force re-download existing models")
    parser.add_argument("--info", help="Show detailed info for specific model")
    parser.add_argument("--weights-dir", help="Custom weights directory")
    
    args = parser.parse_args()
    
    downloader = ArcFaceModelDownloader(args.weights_dir)
    
    if args.list:
        print("📋 Available ArcFace Models:")
        for model in downloader.list_available_models():
            status = "✅" if downloader.is_model_downloaded(model) else "⬜"
            print(f"  {status} {model}")
    
    elif args.status:
        print("📊 Download Status:")
        downloader.print_model_info()
    
    elif args.info:
        downloader.print_model_info(args.info)
    
    elif args.download:
        downloader.download_models(args.download, args.force)
    
    elif args.recommended:
        downloader.download_recommended_models(args.recommended)
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()