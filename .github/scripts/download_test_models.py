#!/usr/bin/env python3
"""Download minimal models for CI testing"""

import os
import sys
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_minimal_models():
    """Download only buffalo_s model for CI testing"""
    try:
        # Set minimal environment
        os.environ['INSIGHTFACE_MODEL'] = 'buffalo_s'
        
        # Import after setting environment
        import insightface
        from insightface.app import FaceAnalysis
        
        # Create models directory
        models_dir = Path.home() / '.insightface' / 'models'
        models_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Downloading buffalo_s model for testing...")
        
        # Initialize app to trigger download
        app = FaceAnalysis(
            name='buffalo_s',
            providers=['CPUExecutionProvider']
        )
        app.prepare(ctx_id=0, det_size=(640, 640))
        
        logger.info("Model download completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Failed to download models: {e}")
        return False

if __name__ == "__main__":
    success = download_minimal_models()
    sys.exit(0 if success else 1)