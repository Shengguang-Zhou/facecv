"""DeepFace人脸分析模块"""

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

async def face_analysis(
    img_path: Union[str, np.ndarray],
    actions: Optional[List[str]] = None,
    detector_backend: str = "retinaface",
    enforce_detection: bool = True,
    align: bool = True,
    silent: bool = False,
    use_cuda: bool = True
) -> List[Dict[str, Any]]:
    """
    异步分析人脸属性，包括年龄、性别、情绪和种族
    
    Args:
        img_path: 图片路径或numpy数组
        actions: 要执行的分析操作列表，可包含'age', 'gender', 'emotion', 'race'
        detector_backend: 人脸检测器后端
        enforce_detection: 是否强制检测人脸
        align: 是否对齐人脸
        silent: 是否静默模式（不输出日志）
        use_cuda: 是否使用CUDA加速
        
    Returns:
        分析结果列表，每个元素是一个包含分析结果的字典
    """
    if actions is None:
        actions = ['age', 'gender', 'emotion', 'race']
    
    os.environ["CUDA_VISIBLE_DEVICES"] = "0" if use_cuda else "-1"
    
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None, 
        lambda: _face_analysis_sync(
            img_path, 
            actions, 
            detector_backend, 
            enforce_detection, 
            align, 
            silent
        )
    )
    return result

def _face_analysis_sync(
    img_path: Union[str, np.ndarray],
    actions: List[str],
    detector_backend: str = "retinaface",
    enforce_detection: bool = True,
    align: bool = True,
    silent: bool = False
) -> List[Dict[str, Any]]:
    """
    同步分析人脸属性，包括年龄、性别、情绪和种族
    
    Args:
        img_path: 图片路径或numpy数组
        actions: 要执行的分析操作列表，可包含'age', 'gender', 'emotion', 'race'
        detector_backend: 人脸检测器后端
        enforce_detection: 是否强制检测人脸
        align: 是否对齐人脸
        silent: 是否静默模式（不输出日志）
        
    Returns:
        分析结果列表，每个元素是一个包含分析结果的字典
    """
    try:
        result = DeepFace.analyze(
            img_path=img_path,
            actions=actions,
            detector_backend=detector_backend,
            enforce_detection=enforce_detection,
            align=align,
            silent=silent
        )
        
        if not isinstance(result, list):
            result = [result]
            
        converted_result = convert_numpy_types(result)
        if not isinstance(converted_result, list):
            converted_result = [converted_result]
        return converted_result
        
    except Exception as e:
        logger.error(f"人脸分析失败: {e}")
        return []
