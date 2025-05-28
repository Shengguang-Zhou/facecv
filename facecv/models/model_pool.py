"""Model Pool for Face Recognition

This module provides a centralized way to manage face recognition models.
It handles model loading, caching, and integration with the database system.
"""

import logging
import os
from typing import Dict, Any, Optional, Union, List, Tuple

from facecv.config.database import DatabaseConfig
from facecv.database.factory import create_face_database
from facecv.database.abstract_facedb import AbstractFaceDB

from facecv.models.insightface.onnx_recognizer import ONNXFaceRecognizer
from facecv.models.insightface.real_recognizer import RealInsightFaceRecognizer
from facecv.models.insightface.arcface_recognizer import ArcFaceRecognizer

logger = logging.getLogger(__name__)

_model_cache = {}

def get_model_recognizer(
    model_name: str = 'buffalo_l',
    mode: str = 'detect',
    face_db: Optional[AbstractFaceDB] = None,
    det_size: Tuple[int, int] = (640, 640),
    det_thresh: float = 0.5,
    similarity_threshold: float = 0.4,
    **kwargs
) -> Union[ONNXFaceRecognizer, RealInsightFaceRecognizer, ArcFaceRecognizer]:
    """
    Get a face recognizer model from the pool.
    
    Args:
        model_name: Name of the model to use ('buffalo_l', 'buffalo_s', 'buffalo_m', 'antelopev2')
        mode: Mode of operation ('detect', 'recognize', 'verify', 'register')
        face_db: Face database instance (if None, will create based on environment)
        det_size: Detection size (width, height)
        det_thresh: Detection threshold
        similarity_threshold: Similarity threshold for face matching
        **kwargs: Additional parameters for the recognizer
        
    Returns:
        Face recognizer instance
    """
    global _model_cache
    
    cache_key = f"{model_name}_{mode}_{det_size}_{det_thresh}"
    
    if cache_key in _model_cache:
        logger.info(f"Using cached model: {cache_key}")
        recognizer = _model_cache[cache_key]
        
        if face_db is not None:
            recognizer.face_db = face_db
            
        return recognizer
    
    if face_db is None:
        try:
            db_config = DatabaseConfig.from_env()
            face_db = create_face_database(db_config.db_type)
            logger.info(f"Created face database of type: {db_config.db_type}")
        except Exception as e:
            logger.error(f"Error creating face database: {e}")
            face_db = None
    
    if model_name.startswith('onnx_'):
        model_path = kwargs.get('model_path', None)
        if model_path is None:
            model_path = os.path.expanduser(f"~/.insightface/models/{model_name}.onnx")
            if not os.path.exists(model_path):
                logger.warning(f"ONNX model not found at {model_path}")
                model_path = None
        
        recognizer = ONNXFaceRecognizer(
            face_db=face_db,
            model_path=model_path,
            det_size=det_size,
            det_thresh=det_thresh,
            similarity_threshold=similarity_threshold,
            **kwargs
        )
        
    elif model_name.startswith('arcface_'):
        recognizer = ArcFaceRecognizer(
            face_db=face_db,
            model_name=model_name,
            det_size=det_size,
            det_thresh=det_thresh,
            similarity_threshold=similarity_threshold,
            **kwargs
        )
        
    else:
        prefer_gpu = kwargs.get('prefer_gpu', True)
        ctx_id = 0 if prefer_gpu else -1
        
        recognizer = RealInsightFaceRecognizer(
            face_db=face_db,
            model_pack=model_name,
            det_size=det_size,
            det_thresh=det_thresh,
            similarity_threshold=similarity_threshold,
            prefer_gpu=prefer_gpu,
            **kwargs
        )
    
    _model_cache[cache_key] = recognizer
    logger.info(f"Created and cached new model: {cache_key}")
    
    return recognizer

def clear_model_cache():
    """Clear the model cache to free up memory."""
    global _model_cache
    _model_cache.clear()
    logger.info("Model cache cleared")

def get_available_models() -> List[Dict[str, Any]]:
    """
    Get a list of available models.
    
    Returns:
        List of dictionaries with model information
    """
    models = [
        {
            'name': 'buffalo_l',
            'description': 'Large InsightFace model with ResNet50 backbone',
            'type': 'insightface',
            'size': 'large',
            'performance': 'high',
            'speed': 'medium'
        },
        {
            'name': 'buffalo_m',
            'description': 'Medium InsightFace model with ResNet34 backbone',
            'type': 'insightface',
            'size': 'medium',
            'performance': 'medium',
            'speed': 'medium'
        },
        {
            'name': 'buffalo_s',
            'description': 'Small InsightFace model with MobileFaceNet backbone',
            'type': 'insightface',
            'size': 'small',
            'performance': 'medium',
            'speed': 'fast'
        },
        {
            'name': 'arcface_resnet50',
            'description': 'ArcFace model with ResNet50 backbone',
            'type': 'arcface',
            'size': 'large',
            'performance': 'high',
            'speed': 'medium'
        },
        {
            'name': 'arcface_mobilefacenet',
            'description': 'ArcFace model with MobileFaceNet backbone',
            'type': 'arcface',
            'size': 'small',
            'performance': 'medium',
            'speed': 'fast'
        },
        {
            'name': 'onnx_insightface',
            'description': 'ONNX version of InsightFace model',
            'type': 'onnx',
            'size': 'medium',
            'performance': 'medium',
            'speed': 'fast'
        }
    ]
    
    return models

def get_model_info(model_name: str) -> Optional[Dict[str, Any]]:
    """
    Get information about a specific model.
    
    Args:
        model_name: Name of the model
        
    Returns:
        Dictionary with model information, or None if model not found
    """
    models = get_available_models()
    
    for model in models:
        if model['name'] == model_name:
            return model
    
    return None
