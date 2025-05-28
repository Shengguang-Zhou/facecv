"""DeepFace人脸检测模块"""

import os
import logging
import numpy as np
from typing import Dict, Any, Optional, Union, List
import asyncio

from deepface import DeepFace

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def convert_numpy_types(obj):
    """将numpy类型转换为Python原生类型，以便JSON序列化"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return convert_numpy_types(obj.tolist())
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    else:
        return obj

async def face_detection(
    img_path: Union[str, np.ndarray],
    detector_backend: str = "retinaface",
    enforce_detection: bool = True,
    align: bool = True,
    use_cuda: bool = True
) -> List[Dict[str, Any]]:
    """
    异步检测图片中的人脸
    
    Args:
        img_path: 图片路径或numpy数组
        detector_backend: 人脸检测器后端
        enforce_detection: 是否强制检测人脸
        align: 是否对齐人脸
        use_cuda: 是否使用CUDA加速
        
    Returns:
        检测到的人脸列表，每个元素是一个包含人脸区域和置信度的字典
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = "0" if use_cuda else "-1"
    
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None, 
        lambda: _face_detection_sync(
            img_path, 
            detector_backend, 
            enforce_detection, 
            align
        )
    )
    return result

def _face_detection_sync(
    img_path: Union[str, np.ndarray],
    detector_backend: str = "retinaface",
    enforce_detection: bool = True,
    align: bool = True
) -> List[Dict[str, Any]]:
    """
    同步检测图片中的人脸
    
    Args:
        img_path: 图片路径或numpy数组
        detector_backend: 人脸检测器后端
        enforce_detection: 是否强制检测人脸
        align: 是否对齐人脸
        
    Returns:
        检测到的人脸列表，每个元素是一个包含人脸区域和置信度的字典
    """
    try:
        faces = DeepFace.extract_faces(
            img_path=img_path,
            detector_backend=detector_backend,
            enforce_detection=enforce_detection,
            align=align
        )
        
        converted_faces = convert_numpy_types(faces)
        if not isinstance(converted_faces, list):
            converted_faces = [converted_faces]
        return converted_faces
        
    except Exception as e:
        logger.error(f"人脸检测失败: {e}")
        return []
