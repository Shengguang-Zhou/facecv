"""DeepFace人脸识别器实现"""

import os
import uuid
import logging
import asyncio
import numpy as np
from typing import List, Dict, Any, Optional, Union, Tuple
from PIL import Image
import cv2
from deepface import DeepFace

from facecv.database.factory import create_face_database, FaceDBFactory
from facecv.database.abstract_facedb import AbstractFaceDB
from facecv.config import get_settings
from facecv.schemas.face import RecognitionResult

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DeepFaceRecognizer:
    """DeepFace人脸识别器实现"""
    
    def __init__(self, 
                 face_db: Optional[AbstractFaceDB] = None,
                 model_name: str = "VGG-Face",
                 detector_backend: str = "retinaface",
                 distance_metric: str = "cosine",
                 similarity_threshold: float = 0.6,
                 use_cuda: bool = True):
        """
        初始化DeepFace人脸识别器
        
        Args:
            face_db: 人脸数据库实例，如果为None则使用默认数据库
            model_name: 使用的DeepFace模型名称
            detector_backend: 人脸检测器后端
            distance_metric: 距离度量方法
            similarity_threshold: 相似度阈值
            use_cuda: 是否使用CUDA加速
        """
        self.settings = get_settings()
        
        self.face_db = face_db or FaceDBFactory.create_database()
        logger.info(f"使用数据库: {self.face_db.__class__.__name__}")
        
        self.model_name = model_name
        self.detector_backend = detector_backend
        self.distance_metric = distance_metric
        self.similarity_threshold = similarity_threshold
        self.use_cuda = use_cuda
        
        os.environ["CUDA_VISIBLE_DEVICES"] = "0" if use_cuda else "-1"
        
        logger.info(f"初始化DeepFaceRecognizer: model={model_name}, detector={detector_backend}, "
                   f"metric={distance_metric}, threshold={similarity_threshold}, cuda={use_cuda}")
    
    async def register_face_async(self, name: str, image: Union[str, np.ndarray], metadata: Optional[Dict] = None) -> Optional[str]:
        """
        异步注册人脸
        
        Args:
            name: 人员姓名
            image: 图片路径或numpy数组
            metadata: 额外的元数据
            
        Returns:
            注册的人脸ID，失败时返回None
        """
        loop = asyncio.get_event_loop()
        face_id = await loop.run_in_executor(None, lambda: self._register_face(name, image, metadata))
        return face_id
    
    def _register_face(self, name: str, image: Union[str, np.ndarray], metadata: Optional[Dict] = None) -> Optional[str]:
        """
        同步注册人脸
        
        Args:
            name: 人员姓名
            image: 图片路径或numpy数组
            metadata: 额外的元数据
            
        Returns:
            注册的人脸ID，失败时返回None
        """
        try:
            embedding = self._extract_embedding(image)
            
            if embedding is None:
                logger.error(f"无法从图片中提取人脸特征")
                return None
            
            face_id = self.face_db.add_face(name, embedding, metadata or {})
            logger.info(f"成功注册人脸: {name}, ID: {face_id}")
            
            return face_id
            
        except Exception as e:
            logger.error(f"注册人脸失败: {e}")
            return None
    
    def _extract_embedding(self, image: Union[str, np.ndarray]) -> Optional[np.ndarray]:
        """
        提取人脸特征向量
        
        Args:
            image: 图片路径或numpy数组
            
        Returns:
            人脸特征向量
        """
        try:
            embedding_obj = DeepFace.represent(
                img_path=image,
                model_name=self.model_name,
                detector_backend=self.detector_backend,
                enforce_detection=True,
                align=True
            )
            
            if isinstance(embedding_obj, list) and len(embedding_obj) > 0:
                embedding = np.array(embedding_obj[0]["embedding"])
                return embedding
            
            return None
            
        except Exception as e:
            logger.error(f"提取人脸特征失败: {e}")
            return None
    
    async def recognize_face_async(self, image: Union[str, np.ndarray], threshold: Optional[float] = None) -> List[RecognitionResult]:
        """
        异步识别人脸
        
        Args:
            image: 图片路径或numpy数组
            threshold: 相似度阈值，如果为None则使用默认值
            
        Returns:
            识别结果列表
        """
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(None, lambda: self._recognize_face(image, threshold))
        return results
    
    def _recognize_face(self, image: Union[str, np.ndarray], threshold: Optional[float] = None) -> List[RecognitionResult]:
        """
        同步识别人脸
        
        Args:
            image: 图片路径或numpy数组
            threshold: 相似度阈值，如果为None则使用默认值
            
        Returns:
            识别结果列表
        """
        if threshold is None:
            threshold = self.similarity_threshold
            
        try:
            all_faces = self.face_db.get_all_faces()
            
            if not all_faces or len(all_faces) == 0:
                logger.warning("数据库中没有注册的人脸")
                return []
                
            logger.info(f"数据库中有 {len(all_faces)} 个注册人脸")
            
            faces = DeepFace.extract_faces(
                img_path=image,
                detector_backend=self.detector_backend,
                enforce_detection=True,
                align=True
            )
            
            if not faces or len(faces) == 0:
                logger.warning("未检测到人脸")
                return []
                
            logger.info(f"检测到 {len(faces)} 个人脸")
            
            results = []
            
            for face_idx, face_info in enumerate(faces):
                face_img = face_info["face"]
                region = face_info["facial_area"]
                
                # 提取人脸特征向量
                embedding_obj = DeepFace.represent(
                    img_path=face_img,
                    model_name=self.model_name,
                    detector_backend=self.detector_backend,
                    enforce_detection=False,  # 已经检测过了
                    align=False  # 已经对齐过了
                )
                
                if isinstance(embedding_obj, list) and len(embedding_obj) > 0:
                    query_embedding = np.array(embedding_obj[0]["embedding"])
                    
                    matches = self.face_db.query_faces_by_embedding(query_embedding, top_k=10)
                    
                    if matches and len(matches) > 0:
                        best_match = matches[0]
                        distance = best_match.get("distance", 0.5)
                        confidence = 1.0 - distance
                        
                        if confidence >= threshold:
                            result = RecognitionResult(
                                person_name=best_match["name"],
                                confidence=confidence,
                                bbox=[region["x"], region["y"], region["w"], region["h"]],
                                face_id=best_match["id"],
                                candidates=[{
                                    "face_id": m["id"],
                                    "name": m["name"],
                                    "confidence": 1.0 - m.get("distance", 0.5)
                                } for m in matches]
                            )
                            
                            logger.info(f"人脸 {face_idx+1} 匹配成功: {best_match['name']}, 置信度: {confidence:.4f}")
                            results.append(result)
                        else:
                            logger.info(f"人脸 {face_idx+1} 匹配置信度 {confidence:.4f} 低于阈值 {threshold}")
                            result = RecognitionResult(
                                person_name="Unknown",
                                confidence=confidence,
                                bbox=[region["x"], region["y"], region["w"], region["h"]],
                                face_id=None,
                                candidates=[{
                                    "face_id": m["id"],
                                    "name": m["name"],
                                    "confidence": 1.0 - m.get("distance", 0.5)
                                } for m in matches]
                            )
                            results.append(result)
                    else:
                        logger.info(f"人脸 {face_idx+1} 没有匹配结果")
                        result = RecognitionResult(
                            person_name="Unknown",
                            confidence=0.0,
                            bbox=[region["x"], region["y"], region["w"], region["h"]],
                            face_id=None,
                            candidates=[]
                        )
                        results.append(result)
                else:
                    logger.warning(f"无法提取人脸 {face_idx+1} 的特征向量")
            
            return results
            
        except Exception as e:
            logger.error(f"识别人脸失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return []
    
    async def list_faces_async(self) -> List[Dict]:
        """
        异步获取所有注册的人脸
        
        Returns:
            人脸信息列表
        """
        loop = asyncio.get_event_loop()
        try:
            faces = await loop.run_in_executor(None, self.face_db.get_all_faces)
            if faces is None:
                logger.warning("数据库返回的人脸列表为None")
                return []
            return faces
        except Exception as e:
            logger.error(f"获取人脸列表失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return []
    
    async def get_face_by_id_async(self, face_id: str) -> Optional[Dict]:
        """
        异步获取指定ID的人脸
        
        Args:
            face_id: 人脸ID
            
        Returns:
            人脸信息
        """
        loop = asyncio.get_event_loop()
        face = await loop.run_in_executor(None, self.face_db.get_face_by_id, face_id)
        return face
    
    async def get_faces_by_name_async(self, name: str) -> List[Dict]:
        """
        异步获取指定姓名的人脸
        
        Args:
            name: 人员姓名
            
        Returns:
            人脸信息列表
        """
        loop = asyncio.get_event_loop()
        faces = await loop.run_in_executor(None, self.face_db.query_faces_by_name, name)
        return faces
    
    async def delete_face_async(self, face_id: str) -> bool:
        """
        异步删除人脸
        
        Args:
            face_id: 人脸ID
            
        Returns:
            是否删除成功
        """
        loop = asyncio.get_event_loop()
        success = await loop.run_in_executor(None, self.face_db.delete_face_by_id, face_id)
        return success
    
    async def delete_faces_by_name_async(self, name: str) -> int:
        """
        异步删除指定姓名的所有人脸
        
        Args:
            name: 人员姓名
            
        Returns:
            删除的人脸数量
        """
        loop = asyncio.get_event_loop()
        count = await loop.run_in_executor(None, self.face_db.delete_face_by_name, name)
        return count
    
    async def update_face_async(self, face_id: str, name: Optional[str] = None, metadata: Optional[Dict] = None) -> bool:
        """
        异步更新人脸信息
        
        Args:
            face_id: 人脸ID
            name: 新的人员姓名
            metadata: 新的元数据
            
        Returns:
            是否更新成功
        """
        try:
            face = await self.get_face_by_id_async(face_id)
            if not face:
                logger.warning(f"未找到ID为 {face_id} 的人脸")
                return False
            
            current_name = face.get("name") if face else None
            current_metadata = face.get("metadata", {}) if face else {}
            
            update_name = name if name is not None else current_name
            update_metadata = metadata if metadata is not None else current_metadata
            
            loop = asyncio.get_event_loop()
            
            def update_face_wrapper():
                return self.face_db.update_face(face_id, update_name, update_metadata)
                
            success = await loop.run_in_executor(None, update_face_wrapper)
            return success
        except Exception as e:
            logger.error(f"更新人脸失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
