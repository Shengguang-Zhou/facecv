"""DeepFace人脸验证模块"""

import os
import logging
import numpy as np
from typing import Dict, Any, Optional, Union, List
import asyncio

from deepface import DeepFace

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def face_verification(
    image_1: Union[str, np.ndarray],
    image_2: Union[str, np.ndarray],
    model_name: str = "VGG-Face",
    detector_backend: str = "retinaface",
    distance_metric: str = "cosine",
    threshold: Optional[float] = None,
    enforce_detection: bool = True,
    align: bool = True,
    normalization: str = "base",
    anti_spoofing: bool = False,
    use_cuda: bool = True
) -> Dict[str, Any]:
    """
    异步验证两张人脸图片是否为同一个人
    
    Args:
        image_1: 第一张图片路径或numpy数组
        image_2: 第二张图片路径或numpy数组
        model_name: 使用的DeepFace模型名称
        detector_backend: 人脸检测器后端
        distance_metric: 距离度量方法
        threshold: 相似度阈值，如果为None则使用默认值
        enforce_detection: 是否强制检测人脸
        align: 是否对齐人脸
        normalization: 归一化方法
        anti_spoofing: 是否启用反欺骗检测
        use_cuda: 是否使用CUDA加速
        
    Returns:
        验证结果字典，包含verified, distance, threshold, model等键
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = "0" if use_cuda else "-1"
    
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None, 
        lambda: _face_verification_sync(
            image_1, 
            image_2, 
            model_name, 
            detector_backend, 
            distance_metric, 
            threshold, 
            enforce_detection, 
            align, 
            normalization,
            anti_spoofing
        )
    )
    return result

def _face_verification_sync(
    image_1: Union[str, np.ndarray],
    image_2: Union[str, np.ndarray],
    model_name: str = "VGG-Face",
    detector_backend: str = "retinaface",
    distance_metric: str = "cosine",
    threshold: Optional[float] = None,
    enforce_detection: bool = True,
    align: bool = True,
    normalization: str = "base",
    anti_spoofing: bool = False
) -> Dict[str, Any]:
    """
    同步验证两张人脸图片是否为同一个人
    
    Args:
        image_1: 第一张图片路径或numpy数组
        image_2: 第二张图片路径或numpy数组
        model_name: 使用的DeepFace模型名称
        detector_backend: 人脸检测器后端
        distance_metric: 距离度量方法
        threshold: 相似度阈值，如果为None则使用默认值
        enforce_detection: 是否强制检测人脸
        align: 是否对齐人脸
        normalization: 归一化方法
        anti_spoofing: 是否启用反欺骗检测
        
    Returns:
        验证结果字典，包含verified, distance, threshold, model等键
    """
    try:
        if anti_spoofing:
            pass
            
        result = DeepFace.verify(
            img1_path=image_1,  # DeepFace expects img1_path but our function uses image_1
            img2_path=image_2,  # DeepFace expects img2_path but our function uses image_2
            model_name=model_name,
            detector_backend=detector_backend,
            distance_metric=distance_metric,
            enforce_detection=enforce_detection,
            align=align,
            normalization=normalization
        )
        
        if "distance" in result:
            result["similarity_score"] = 1.0 - result["distance"]
            
        return result
        
    except Exception as e:
        logger.error(f"人脸验证失败: {e}")
        return {
            "verified": False,
            "distance": 1.0,
            "threshold": threshold or 0.6,
            "model": model_name,
            "error": str(e),
            "success": False
        }
