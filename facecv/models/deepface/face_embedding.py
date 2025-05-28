"""DeepFace人脸特征提取模块"""

import os
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Union, Tuple
import asyncio

from deepface import DeepFace

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def face_embedding(
    image_path: Union[str, np.ndarray],
    model_name: str = "Facenet512",
    detector_backend: str = "retinaface",
    enforce_detection: bool = True,
    align: bool = True,
    normalization: str = "base",
    use_cuda: bool = True
) -> Optional[List[Dict[str, Any]]]:
    """
    异步提取人脸特征向量
    
    Args:
        image_path: 图片路径或numpy数组
        model_name: 使用的DeepFace模型名称
        detector_backend: 人脸检测器后端
        enforce_detection: 是否强制检测人脸
        align: 是否对齐人脸
        normalization: 归一化方法
        use_cuda: 是否使用CUDA加速
        
    Returns:
        包含人脸特征向量的字典列表，每个字典包含embedding键
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = "0" if use_cuda else "-1"
    
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None, 
        lambda: _face_embedding_sync(
            image_path, 
            model_name, 
            detector_backend, 
            enforce_detection, 
            align, 
            normalization
        )
    )
    return result

def _face_embedding_sync(
    image_path: Union[str, np.ndarray],
    model_name: str = "Facenet512",
    detector_backend: str = "retinaface",
    enforce_detection: bool = True,
    align: bool = True,
    normalization: str = "base"
) -> Optional[List[Dict[str, Any]]]:
    """
    同步提取人脸特征向量
    
    Args:
        image_path: 图片路径或numpy数组
        model_name: 使用的DeepFace模型名称
        detector_backend: 人脸检测器后端
        enforce_detection: 是否强制检测人脸
        align: 是否对齐人脸
        normalization: 归一化方法
        
    Returns:
        包含人脸特征向量的字典列表，每个字典包含embedding键
    """
    try:
        result = DeepFace.represent(
            img_path=image_path,
            model_name=model_name,
            detector_backend=detector_backend,
            enforce_detection=enforce_detection,
            align=align,
            normalization=normalization
        )
        
        if not isinstance(result, list):
            result = [result]
            
        return result
        
    except Exception as e:
        logger.error(f"提取人脸特征失败: {e}")
        return None
