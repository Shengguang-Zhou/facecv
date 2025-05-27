#!/usr/bin/env python3
"""
Download all DeepFace models to ensure they're available before use
This helps avoid download issues and protobuf errors during runtime
"""

import os
import sys
import logging
from pathlib import Path
import shutil

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set protobuf environment variable before importing
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# DeepFace models and their details
DEEPFACE_MODELS = {
    "VGG-Face": {
        "url": "https://github.com/serengil/deepface_models/releases/download/v1.0/vgg_face_weights.h5",
        "file": "vgg_face_weights.h5"
    },
    "OpenFace": {
        "url": "https://github.com/serengil/deepface_models/releases/download/v1.0/openface_weights.h5",
        "file": "openface_weights.h5"
    },
    "Facenet": {
        "url": "https://github.com/serengil/deepface_models/releases/download/v1.0/facenet_weights.h5",
        "file": "facenet_weights.h5"
    },
    "Facenet512": {
        "url": "https://github.com/serengil/deepface_models/releases/download/v1.0/facenet512_weights.h5",
        "file": "facenet512_weights.h5"
    },
    "DeepFace": {
        "url": "https://github.com/serengil/deepface_models/releases/download/v1.0/deepface_weights.h5",
        "file": "deepface_weights.h5"
    },
    "DeepID": {
        "url": "https://github.com/serengil/deepface_models/releases/download/v1.0/deepid_keras_weights.h5",
        "file": "deepid_keras_weights.h5"
    },
    "Dlib": {
        "url": "http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2",
        "file": "dlib_face_recognition_resnet_model_v1.dat",
        "compressed": True
    },
    "ArcFace": {
        "url": "https://github.com/serengil/deepface_models/releases/download/v1.0/arcface_weights.h5",
        "file": "arcface_weights.h5"
    },
    "SFace": {
        "url": "https://github.com/serengil/deepface_models/releases/download/v1.0/sf_model.pth",
        "file": "sf_model.pth"
    }
}

# Detector models
DETECTOR_MODELS = {
    "retinaface": {
        "url": "https://github.com/serengil/deepface_models/releases/download/v1.0/retinaface.h5",
        "file": "retinaface.h5"
    },
    "mtcnn": {
        "weights": [
            {
                "url": "https://github.com/serengil/deepface_models/releases/download/v1.0/mtcnn_weights.npy",
                "file": "mtcnn_weights.npy"
            }
        ]
    },
    "dlib": {
        "shape_predictor": {
            "url": "http://dlib.net/files/shape_predictor_5_face_landmarks.dat.bz2",
            "file": "shape_predictor_5_face_landmarks.dat",
            "compressed": True
        },
        "face_detector": {
            "url": "http://dlib.net/files/mmod_human_face_detector.dat.bz2",
            "file": "mmod_human_face_detector.dat",
            "compressed": True
        }
    },
    "ssd": {
        "url": "https://github.com/serengil/deepface_models/releases/download/v1.0/deploy.prototxt",
        "file": "deploy.prototxt",
        "model_url": "https://github.com/serengil/deepface_models/releases/download/v1.0/res10_300x300_ssd_iter_140000.caffemodel",
        "model_file": "res10_300x300_ssd_iter_140000.caffemodel"
    }
}


def download_file(url, destination):
    """Download a file from URL to destination"""
    import urllib.request
    import urllib.error
    
    logger.info(f"Downloading {url} to {destination}")
    
    try:
        # Create parent directory if it doesn't exist
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        
        # Download the file
        urllib.request.urlretrieve(url, destination)
        logger.info(f"Successfully downloaded {os.path.basename(destination)}")
        
        # Handle compressed files
        if destination.endswith('.bz2'):
            import bz2
            logger.info(f"Decompressing {destination}")
            
            with bz2.open(destination, 'rb') as src:
                decompressed_path = destination[:-4]  # Remove .bz2
                with open(decompressed_path, 'wb') as dst:
                    dst.write(src.read())
            
            # Remove compressed file
            os.remove(destination)
            logger.info(f"Decompressed to {decompressed_path}")
            
        return True
    except Exception as e:
        logger.error(f"Failed to download {url}: {e}")
        return False


def ensure_deepface_models():
    """Download all DeepFace models if they don't exist"""
    
    # Get DeepFace home directory
    home = str(Path.home())
    deepface_home = os.path.join(home, ".deepface")
    weights_dir = os.path.join(deepface_home, "weights")
    
    logger.info(f"DeepFace home directory: {deepface_home}")
    logger.info(f"Weights directory: {weights_dir}")
    
    # Create directories if they don't exist
    os.makedirs(weights_dir, exist_ok=True)
    
    # Download face recognition models
    logger.info("\n=== Downloading Face Recognition Models ===")
    for model_name, model_info in DEEPFACE_MODELS.items():
        model_path = os.path.join(weights_dir, model_info["file"])
        
        if os.path.exists(model_path):
            logger.info(f"✓ {model_name} already exists at {model_path}")
        else:
            logger.info(f"⬇ Downloading {model_name}...")
            if model_info.get("compressed"):
                download_file(model_info["url"], model_path + ".bz2")
            else:
                download_file(model_info["url"], model_path)
    
    # Download detector models
    logger.info("\n=== Downloading Face Detector Models ===")
    
    # RetinaFace
    retinaface_path = os.path.join(weights_dir, DETECTOR_MODELS["retinaface"]["file"])
    if not os.path.exists(retinaface_path):
        download_file(DETECTOR_MODELS["retinaface"]["url"], retinaface_path)
    else:
        logger.info(f"✓ RetinaFace already exists")
    
    # MTCNN
    for weight_info in DETECTOR_MODELS["mtcnn"]["weights"]:
        mtcnn_path = os.path.join(weights_dir, weight_info["file"])
        if not os.path.exists(mtcnn_path):
            download_file(weight_info["url"], mtcnn_path)
        else:
            logger.info(f"✓ MTCNN weights already exist")
    
    # Dlib
    for key, dlib_info in DETECTOR_MODELS["dlib"].items():
        dlib_path = os.path.join(weights_dir, dlib_info["file"])
        if not os.path.exists(dlib_path):
            if dlib_info.get("compressed"):
                download_file(dlib_info["url"], dlib_path + ".bz2")
            else:
                download_file(dlib_info["url"], dlib_path)
        else:
            logger.info(f"✓ Dlib {key} already exists")
    
    # SSD
    ssd_proto_path = os.path.join(weights_dir, DETECTOR_MODELS["ssd"]["file"])
    ssd_model_path = os.path.join(weights_dir, DETECTOR_MODELS["ssd"]["model_file"])
    
    if not os.path.exists(ssd_proto_path):
        download_file(DETECTOR_MODELS["ssd"]["url"], ssd_proto_path)
    else:
        logger.info(f"✓ SSD prototxt already exists")
        
    if not os.path.exists(ssd_model_path):
        download_file(DETECTOR_MODELS["ssd"]["model_url"], ssd_model_path)
    else:
        logger.info(f"✓ SSD model already exists")
    
    logger.info("\n=== Model Download Complete ===")
    
    # Also copy to local weights directory if it exists
    local_weights = os.path.join(os.path.dirname(os.path.dirname(__file__)), "weights")
    if os.path.exists(local_weights):
        logger.info(f"\nCopying models to local weights directory: {local_weights}")
        for model_name, model_info in DEEPFACE_MODELS.items():
            src = os.path.join(weights_dir, model_info["file"])
            dst = os.path.join(local_weights, model_info["file"])
            if os.path.exists(src) and not os.path.exists(dst):
                shutil.copy2(src, dst)
                logger.info(f"Copied {model_info['file']} to local weights")


def test_model_loading():
    """Test loading models with DeepFace to ensure they work"""
    try:
        from deepface import DeepFace
        
        logger.info("\n=== Testing Model Loading ===")
        
        # Test a few models
        test_models = ["VGG-Face", "OpenFace", "Dlib"]
        
        for model in test_models:
            try:
                logger.info(f"Testing {model}...")
                DeepFace.build_model(model)
                logger.info(f"✓ {model} loaded successfully")
            except Exception as e:
                logger.error(f"✗ Failed to load {model}: {e}")
                
    except ImportError:
        logger.warning("DeepFace not installed, skipping model loading test")


if __name__ == "__main__":
    logger.info("DeepFace Model Downloader")
    logger.info("=" * 50)
    
    # Download all models
    ensure_deepface_models()
    
    # Test loading
    test_model_loading()
    
    logger.info("\nDone! All models should now be available.")
    logger.info("Models are stored in: ~/.deepface/weights/")